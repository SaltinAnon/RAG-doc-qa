"""LLM 连通性诊断。

## 为什么需要这个模块

「配了 API 还是连不上」是新手最容易被卡住的一类问题，而它的可能原因非常多：

1. 没网 / 公司网络拦截；
2. **环境变量的代理**（`HTTP_PROXY` / `HTTPS_PROXY`）指向一个不通的地址；
3. DNS 解析不了 API 域名；
4. 防火墙 / SSL 证书问题；
5. base_url 写错（少了 `/v1`、多了斜杠）；
6. 模型名不存在（这个不走网络，但也表现为「用不了」）。

这些东西**光看一句 `APIConnectionError: Connection error.` 完全判断不出来**。
所以这里做一件事：把上面每一条都**实际测一遍**，然后输出**能照着做**的结论。

设计原则（和项目其它地方一致）：
- **不静默**：每一层探测的结果都要报出来，哪怕是通过；
- **不猜测**：代理这类可能存在的干扰，直接读环境变量确认，不靠推理；
- **只诊断，不改配置**：发现代理有问题也只提示，绝不擅自 `os.environ.pop`。
"""

from __future__ import annotations

import os
import socket
import ssl
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

from app.config import settings

# 会影响 httpx / requests 的代理环境变量（大小写都要查）
_PROXY_ENV_KEYS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
)

# ⚠️ 平台差异（真实踩过）：
#   · Windows 的 `os.environ` **大小写不敏感** —— `HTTP_PROXY` 与 `http_proxy`
#     其实是同一个变量，「同时设两个不同值」在 Windows 上根本不可能出现。
#   · Linux 大小写**严格敏感** —— `http_proxy` 和 `HTTP_PROXY` 是两个独立变量，
#     用户完全可能只设了小写那个（很多文档/脚本就是这么写的）。
#   所以检测**必须遍历 `os.environ` 里真实存在的键**、按大写归一去重，
#   而不是只去 `get()` 那三个大写名 —— 后者在 Linux 上会 100% 漏检小写代理。
_PROXY_CANONICAL = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")


@dataclass
class CheckResult:
    """单项检查的结果。"""

    name: str
    ok: bool
    detail: str
    hint: str = ""


@dataclass
class DiagnosisReport:
    """一次完整诊断的结果。"""

    checks: list[CheckResult] = field(default_factory=list)
    conclusion: str = ""
    suggestions: list[str] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def to_dict(self) -> dict:
        return {
            "all_ok": self.all_ok,
            "conclusion": self.conclusion,
            "suggestions": self.suggestions,
            "checks": [
                {"name": c.name, "ok": c.ok, "detail": c.detail, "hint": c.hint}
                for c in self.checks
            ],
        }


def detect_proxies() -> dict[str, str]:
    """返回当前生效的代理环境变量（已按变量名去重）。

    ⚠️ 这是**最容易被忽略**的一个坑：很多人机器上装了代理软件（Clash / v2ray 等），
    或者某个工具往环境变量里写了代理地址，Python 进程会**自动继承**它。
    于是「请求 API 失败」根本不是 API 的问题，而是被导到了一个不通的代理上。

    Windows 上 `os.environ` 大小写不敏感（`HTTP_PROXY` 与 `http_proxy` 是同一个），
    Linux 上则严格区分 —— 因此这里**遍历环境里真实存在的键**、按大写名归一去重：
    既能捞到用户实际写的小写名（Linux），又不会把同一个代理报成好几条。

    Returns:
        {变量名: 值}，只包含真正设置过的；键是**用户实际使用的**那个名字。
    """
    found: dict[str, str] = {}
    for raw_key, value in os.environ.items():
        if not value:
            continue
        canonical = raw_key.upper()
        if canonical not in _PROXY_CANONICAL:
            continue
        # 大小写视为同一个变量：同一 canonical 只保留**首个**命中的实际写法。
        # （Windows 下两者本就是同一个；Linux 下若真被设了两份，只报一份即可 ——
        #   报重复值反而会让界面比配置本身更让人困惑。）
        if canonical in {k.upper() for k in found}:
            continue
        found[raw_key] = value
    return found


def _check_proxy_env() -> CheckResult:
    """检查并**实际验证**代理是否可用。"""
    proxies = detect_proxies()
    if not proxies:
        return CheckResult(
            name="代理环境变量",
            ok=True,
            detail="未检测到 HTTP_PROXY / HTTPS_PROXY 等代理变量",
        )

    # 有代理 → 必须真的测一下通不通，因为「有代理且不通」正是最坑的形态
    # 优先挑 HTTPS_PROXY（API 走 https 时 httpx 主要看它），否则取第一个。
    proxy_url = next(
        (v for k, v in proxies.items() if k.upper() == "HTTPS_PROXY"),
        next(iter(proxies.values())),
    )
    parsed = urlparse(proxy_url)
    host, port = parsed.hostname or "", parsed.port or 0

    reachable = False
    try:
        with socket.create_connection((host, port), timeout=3):
            reachable = True
    except OSError:
        reachable = False

    lines = [f"{k}={v}" for k, v in proxies.items()]
    detail = "；".join(lines)
    if reachable:
        return CheckResult(
            name="代理环境变量",
            ok=True,
            detail=f"{detail}（代理端口可达 ✅）",
            hint="代理是通的，但如果 API 走代理仍然失败，可以试试临时取消代理。",
        )
    return CheckResult(
        name="代理环境变量",
        ok=False,
        detail=f"{detail}（代理端口 {host}:{port} **连不上** ❌）",
        hint=(
            "这就是请求失败的常见原因：Python 进程继承了代理变量，"
            "请求被导到一个不通的代理上。\n"
            "     解决：启动服务前清掉代理变量，或在 .env 里排除该域名。"
        ),
    )


def _check_dns(host: str) -> CheckResult:
    """DNS 能否解析 API 域名。"""
    if not host:
        return CheckResult(name="DNS 解析", ok=False, detail="base_url 为空，无法解析")
    try:
        t0 = time.perf_counter()
        ip = socket.gethostbyname(host)
        ms = (time.perf_counter() - t0) * 1000
        return CheckResult(name=f"DNS 解析 {host}", ok=True, detail=f"→ {ip}（{ms:.0f} ms）")
    except OSError as exc:
        return CheckResult(
            name=f"DNS 解析 {host}",
            ok=False,
            detail=f"解析失败：{exc}",
            hint="域名解析不了：检查网络、DNS 设置，或换个网络环境（如手机热点）再试。",
        )


def _check_tcp(host: str, port: int = 443) -> CheckResult:
    """443 端口能否建立 TCP 连接。"""
    try:
        t0 = time.perf_counter()
        with socket.create_connection((host, port), timeout=5):
            ms = (time.perf_counter() - t0) * 1000
        return CheckResult(name=f"TCP 连接 {host}:{port}", ok=True, detail=f"已连通（{ms:.0f} ms）")
    except OSError as exc:
        return CheckResult(
            name=f"TCP 连接 {host}:{port}",
            ok=False,
            detail=f"连接失败：{exc}",
            hint="端口不通：可能是防火墙、公司网络策略，或代理配置问题。",
        )


def _check_https(host: str) -> CheckResult:
    """HTTPS 握手能否成功（顺带验证证书）。"""
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=5) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                proto = ssock.version()
        return CheckResult(name=f"HTTPS 握手 {host}", ok=True, detail=f"成功（{proto}）")
    except ssl.SSLError as exc:
        return CheckResult(
            name=f"HTTPS 握手 {host}",
            ok=False,
            detail=f"SSL 错误：{exc}",
            hint="证书校验失败：可能是网络中间人（公司代理）或系统时间不对。",
        )
    except OSError as exc:
        return CheckResult(
            name=f"HTTPS 握手 {host}", ok=False, detail=f"失败：{exc}",
        )


def _check_model_name() -> CheckResult:
    """模型名与 provider 是否匹配（不联网，纯配置检查）。"""
    if settings.llm_provider == "offline":
        return CheckResult(name="模型名检查", ok=True, detail="当前为离线模式，跳过")

    if settings.llm_model_mismatch:
        return CheckResult(
            name="模型名检查",
            ok=False,
            detail=(
                f"LLM_MODEL='{settings.llm_model}' 不属于 provider='{settings.llm_provider}'"
                f"（已自动改用 '{settings.resolved_llm_model}'）"
            ),
            hint="建议在 .env 里把 LLM_MODEL 改成该提供商真实提供的模型名，或直接留空。",
        )

    from app.config import PROVIDER_DEFAULT_MODELS

    known = _known_models(settings.llm_provider)
    if known and settings.llm_model not in known:
        return CheckResult(
            name="模型名检查",
            ok=False,
            detail=(
                f"LLM_MODEL='{settings.llm_model}' 不在已知的 {settings.llm_provider} 模型列表里；"
                f"常见可用的有：{' / '.join(known)}"
                f"（该家默认：{PROVIDER_DEFAULT_MODELS.get(settings.llm_provider, '?')}）"
            ),
            hint=(
                "模型名写错时，接口会返回 400/404，表现为「调用失败」。"
                "如果这是你自己确认过存在的模型（如私有部署），可以忽略本条。"
            ),
        )

    return CheckResult(name="模型名检查", ok=True, detail=f"'{settings.llm_model}' 看起来正常")


# 各家常见模型（仅用于「提醒」，不做强制校验 —— 新模型随时会出）
_KNOWN_MODELS: dict[str, tuple[str, ...]] = {
    "deepseek": ("deepseek-chat", "deepseek-reasoner", "deepseek-coder"),
    "openai": ("gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"),
    "zhipu": ("glm-4-flash", "glm-4-plus", "glm-4-air", "glm-4"),
    "moonshot": ("moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"),
}


def _known_models(provider: str) -> tuple[str, ...]:
    return _KNOWN_MODELS.get(provider, ())


def diagnose_llm(do_network: bool = True) -> DiagnosisReport:
    """对当前 LLM 配置做一次完整体检。

    Args:
        do_network: 是否执行联网探测（DNS/TCP/TLS）。测试里可设为 False。

    Returns:
        `DiagnosisReport`，含逐项结果与最终结论。
    """
    report = DiagnosisReport()

    # 1. 配置层
    report.checks.append(_check_model_name())

    # 2. 代理（这一步最关键 —— 它是「看起来像 API 问题」的头号元凶）
    proxy_check = _check_proxy_env()
    report.checks.append(proxy_check)

    if settings.llm_provider == "offline" or not settings.resolved_llm_api_key:
        report.conclusion = "当前未启用在线 LLM（离线模式或未配置 Key），跳过网络探测。"
        return report

    base = settings.resolved_base_url
    if not base:
        report.checks.append(CheckResult(name="base_url", ok=False, detail="解析为空"))
        report.conclusion = "无法确定 API 地址，请检查 LLM_PROVIDER / LLM_BASE_URL。"
        return report

    host = urlparse(base).hostname or ""
    report.checks.append(
        CheckResult(name="API 地址", ok=True, detail=f"{base}（host={host}）")
    )

    if do_network:
        report.checks.append(_check_dns(host))
        report.checks.append(_check_tcp(host))
        report.checks.append(_check_https(host))

    # 3. 汇总结论
    failed = [c for c in report.checks if not c.ok]
    if not failed:
        report.conclusion = (
            f"✅ 配置与网络探测全部通过（{settings.llm_provider} / "
            f"{settings.resolved_llm_model}）。如果仍调用失败，请查看服务端日志里的原始异常。"
        )
        return report

    report.conclusion = f"❌ 发现 {len(failed)} 个问题：" + "；".join(c.name for c in failed)
    for c in failed:
        if c.hint:
            report.suggestions.append(f"[{c.name}] {c.hint}")

    # 代理不通是最常见的，单独点出来
    if not proxy_check.ok:
        report.suggestions.insert(
            0,
            "⭐ 最可能的原因：检测到**不可用的代理**且请求会走代理。\n"
            "     临时验证（Windows PowerShell，在当前终端里设置后再启动服务）：\n"
            "       $env:HTTP_PROXY=\"\"; $env:HTTPS_PROXY=\"\"; python -m uvicorn app.main:app\n"
            "     永久解决：把 API 域名加进 NO_PROXY，或在系统里关掉该代理。",
        )
    return report


def format_report(report: DiagnosisReport) -> str:
    """把诊断结果格式化成适合打印到终端的文本。"""
    lines = ["LLM 连通性诊断", "=" * 52]
    for c in report.checks:
        mark = "✅" if c.ok else "❌"
        lines.append(f"  {mark} {c.name}")
        lines.append(f"      {c.detail}")
        if c.hint and not c.ok:
            lines.append(f"      ↳ {c.hint}")
    lines.append("-" * 52)
    lines.append(f"结论：{report.conclusion}")
    if report.suggestions:
        lines.append("")
        lines.append("建议按顺序尝试：")
        for i, s in enumerate(report.suggestions, 1):
            lines.append(f"  {i}. {s}")
    return "\n".join(lines)
