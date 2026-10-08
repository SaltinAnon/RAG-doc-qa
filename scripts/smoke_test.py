"""端到端冒烟测试：起一个真实后端 → 打接口 → **无论成败都关掉它**。

## 为什么要有这个脚本

手工开一个 uvicorn 再 curl 有几个典型的翻车点：

1. **忘了关**。窗口一关/一转身，进程还在后台跑着占端口；
   下次启动就报 `Address already in use`，或者更糟 ——
   你以为测的是新代码，其实打到的是**上一版还没死的进程**上。
2. **环境变量靠手敲**，容易少设一个（例如 `EMBEDDING_DIM`），
   于是测出来的行为根本不是线上行为。
3. **断言靠肉眼**看输出，看漏一行就当过了。

所以正确的做法是：把「起服务 → 探活 → 断言 → 关服务」写进**一个脚本**，
用 `try/finally` 保证清理一定执行。这就是本脚本。

## 用法

    python scripts/smoke_test.py                # 默认端口 8124
    python scripts/smoke_test.py --port 18123   # 换个端口（端口被占时）
    python scripts/smoke_test.py --keep         # 调错用：测完不关，自己手动看

退出码：0 = 全部通过；1 = 有断言失败（CI 可直接用）。

## ⚠️ 刻意不设 EMBEDDING_DIM

`data/chroma` 里的索引维度是**建索引时**定死的。如果这里强行设一个和索引不一致的
维度（例如索引是 512，这里设 256），向量检索会报
`Collection expecting embedding with dimension of 512, got 256`，
然后**静默降级成纯 BM25** —— 接口照常 200，你会以为一切正常。

这个坑本项目真踩过，所以：
- 脚本默认**不覆盖** `EMBEDDING_DIM`，让它和 `.env` / 索引保持一致；
- 并且会额外检查 `retrieval_debug.vector_hits`，为 0 时**显式告警**。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# 让脚本可以直接 `python scripts/smoke_test.py` 运行
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

API = "/api/v1"


# ============================================================
#  HTTP 小工具（只用标准库，不给项目加依赖）
# ============================================================
def http_get(url: str, timeout: float = 5.0) -> tuple[int, dict]:
    """发 GET，返回 (状态码, JSON)。连不上时状态码为 0。"""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {"raw": body}
    except (urllib.error.URLError, OSError):
        return 0, {}


def http_post_json(url: str, payload: dict, timeout: float = 30.0) -> tuple[int, dict]:
    """发 POST JSON，返回 (状态码, JSON)。连不上时状态码为 0。"""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {"raw": body}
    except (urllib.error.URLError, OSError):
        return 0, {}


# ============================================================
#  服务生命周期
# ============================================================
def wait_ready(base: str, timeout: float) -> dict | None:
    """轮询 /healthz 直到服务起来，返回 /readyz 的内容。超时返回 None。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        code, _ = http_get(f"{base}/healthz", timeout=2.0)
        if code == 200:
            _, ready = http_get(f"{base}/readyz", timeout=5.0)
            return ready
        time.sleep(0.4)
    return None


def start_server(port: int, python_exe: str, log_path: Path) -> subprocess.Popen:
    """以子进程方式启动 uvicorn。

    环境变量在这里**一次性**设定，避免手工敲漏：
      - LLM_PROVIDER=offline      不调用任何大模型，零成本、可离线
      - EMBEDDING_PROVIDER=hash   用内置无依赖 Embedding
      - API_KEY=""                关掉鉴权（本地冒烟测试）
    注意：**不设 EMBEDDING_DIM**，理由见模块 docstring。
    """
    env = os.environ.copy()
    env.setdefault("LLM_PROVIDER", "offline")
    env.setdefault("EMBEDDING_PROVIDER", "hash")
    env["API_KEY"] = ""
    env["PYTHONIOENCODING"] = "utf-8"

    log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115 - 交给子进程持有
    proc = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return proc


def stop_server(proc: subprocess.Popen, log_path: Path) -> None:
    """确保进程一定被杀掉。

    先 `terminate()`（SIGTERM，uvicorn 会优雅退出），
    给 5 秒宽限；还活着就 `kill()`（SIGKILL，不给机会）。
    Windows 上 `terminate()` 等价于 TerminateProcess，同样有效。
    """
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print("  ⚠️  优雅退出超时，强制结束进程")
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"  ❌ 进程仍未结束，日志见 {log_path}")


# ============================================================
#  断言
# ============================================================
class Result:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.warnings: list[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> None:
        if ok:
            self.passed += 1
            print(f"  ✅ {label}")
        else:
            self.failed += 1
            print(f"  ❌ {label}")
            if detail:
                print(f"     实际：{detail}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"  ⚠️  {msg}")


def run_checks(base: str, res: Result) -> None:
    """核心断言：能答的要答出来、答不了的必须拒答。"""
    print("\n【1】存活与就绪")
    code, health = http_get(f"{base}/healthz")
    res.check(code == 200 and health.get("status") == "ok", "/healthz 返回 ok", f"{code} {health}")

    code, ready = http_get(f"{base}/readyz")
    res.check(code == 200 and ready.get("ready") is True, "/readyz 报告就绪", f"{code} {ready}")

    chunk_count = ready.get("chunk_count", 0)
    if not chunk_count:
        res.warn(
            "向量库是空的！请先跑：python scripts/ingest_cli.py --dir data/docs\n"
            "     （空库时 /query 会返回 503，下面的问答断言必然失败）"
        )

    print("\n【2】可回答问题：必须答出来，且带引用")
    q = "员工年假有多少天？"
    code, body = http_post_json(f"{base}{API}/query", {"question": q, "top_k": 5})
    data = body.get("data") or {}
    res.check(code == 200, f"HTTP 200（问题：{q}）", f"{code} {body}")
    res.check(data.get("refused") is False, "refused 为 false", f"refused={data.get('refused')}")
    res.check(bool(data.get("citations")), "返回了至少 1 条引用", f"citations={data.get('citations')}")
    res.check(bool((data.get("answer") or "").strip()), "答案非空", repr(data.get("answer"))[:80])

    # 维度不匹配的显式告警（本项目真踩过的坑）
    debug = data.get("retrieval_debug") or {}
    if debug.get("bm25_hits", 0) > 0 and debug.get("vector_hits", 0) == 0:
        res.warn(
            "vector_hits=0 但 bm25_hits>0 → 向量检索整条腿没工作，"
            "本次结果其实是纯 BM25 的。\n"
            "     最常见原因：EMBEDDING_DIM 与建索引时的维度不一致（看服务日志里的 "
            "'expecting embedding with dimension of ...'）。"
        )

    print("\n【3】不可回答问题：必须拒答，且不挂引用")
    for q in ("公司的股票代码是多少？", "员工食堂午餐菜单有哪些菜？"):
        code, body = http_post_json(f"{base}{API}/query", {"question": q, "top_k": 5})
        data = body.get("data") or {}
        res.check(code == 200, f"HTTP 200（问题：{q}）", f"{code} {body}")
        res.check(data.get("refused") is True, f"refused 为 true（问题：{q}）",
                  f"refused={data.get('refused')}")
        res.check(
            data.get("refusal_reason") in {"no_retrieval", "model_refused", "empty_output"},
            f"refusal_reason 合法（问题：{q}）",
            f"reason={data.get('refusal_reason')}",
        )
        res.check(data.get("citations") == [], f"拒答时不挂引用（问题：{q}）",
                  f"citations={data.get('citations')}")


# ============================================================
#  主流程
# ============================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="端到端 HTTP 冒烟测试（自动关服务）")
    parser.add_argument("--port", type=int, default=8124, help="监听端口（默认 8124）")
    parser.add_argument("--timeout", type=float, default=60.0, help="等待服务就绪的秒数")
    parser.add_argument(
        "--keep", action="store_true", help="测完**不关**服务（调错用），自己记得手动关"
    )
    args = parser.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    log_path = ROOT / f".tmp-smoke-{args.port}.log"
    python_exe = sys.executable

    print("=" * 64)
    print(f"端到端冒烟测试  →  {base}")
    print(f"日志：{log_path}")
    print("=" * 64)

    proc = start_server(args.port, python_exe, log_path)
    res = Result()
    try:
        print(f"\n【0】启动服务（PID {proc.pid}），最多等 {args.timeout:.0f} 秒…")
        ready = wait_ready(base, args.timeout)
        if ready is None:
            print("  ❌ 服务没能在超时内就绪。服务日志尾部：\n")
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
            for line in tail:
                print(f"     {line}")
            return 1
        print(f"  ✅ 服务已就绪（向量库 {ready.get('chunk_count')} 个片段）")

        run_checks(base, res)
    finally:
        # ★ 关键：不管断言是否失败、是否抛异常，清理一定执行
        if args.keep:
            print(f"\n【4】--keep 已指定，服务保留在 {base}（PID {proc.pid}）")
            print(f"     用完请手动关：taskkill /PID {proc.pid} /F   （Windows）")
            print(f"                    kill {proc.pid}              （macOS / Linux）")
        else:
            print("\n【4】关闭服务")
            stop_server(proc, log_path)
            print(f"  ✅ 已关闭 PID {proc.pid}，端口 {args.port} 已释放")

    print("\n" + "=" * 64)
    print(f"结果：{res.passed} 项通过 / {res.failed} 项失败 / {len(res.warnings)} 项告警")
    if res.warnings:
        print("\n告警明细：")
        for msg in res.warnings:
            print(f"  ⚠️  {msg}")
    if res.failed:
        print(f"\n❌ 冒烟测试未通过。服务日志：{log_path}")
        print("   提示：日志文件不会被自动删除，排查完可手动删。")
        return 1
    print("\n✅ 冒烟测试全部通过")
    # 全通过时顺手清理日志，不给项目留垃圾
    if log_path.exists() and not args.keep:
        log_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
