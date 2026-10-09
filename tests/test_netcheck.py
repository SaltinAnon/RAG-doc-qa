"""LLM 连通性诊断（`app/utils/netcheck.py`）的测试。

这一组守的是一个**真实踩过的坑**：

    用户的 .env 里 provider / model / key 全都配好了，却一直报
    `APIConnectionError: Connection error.`。
    排查发现原因是**进程继承了 HTTP_PROXY / HTTPS_PROXY 环境变量**，
    请求被导到一个不通的代理上 —— 这和 API 配置毫无关系，
    但从错误信息里完全看不出来。

所以诊断工具必须做到：**碰到这种情况要明确指出来**，而不是让用户去猜。
"""

from __future__ import annotations

import os

import pytest

from app.config import Settings
from app.utils.netcheck import (
    DiagnosisReport,
    CheckResult,
    detect_proxies,
    diagnose_llm,
    format_report,
)

_PROXY_KEYS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
)


@pytest.fixture
def clean_proxy_env(monkeypatch):
    """把代理环境变量清干净，避免宿主机的代理影响测试结论。"""
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


@pytest.fixture
def case_sensitive_proxy_env(monkeypatch):
    """把 `netcheck.os.environ` 换成一个**严格区分大小写**的映射。

    `monkeypatch.setenv` 在 Windows 上走的是大小写不敏感的 `os.environ`，
    没法构造「只有小写 http_proxy」这种局面 —— 而这正是 Linux CI 上挂掉的原因。
    所以这里直接替换 `os.environ` 对象，**在任意平台上模拟 Linux 语义**。

    返回一个普通 dict（区分大小写），测试往里塞键即可。
    """
    import app.utils.netcheck as nc

    fake_env: dict[str, str] = {}
    monkeypatch.setattr(nc.os, "environ", fake_env)
    return fake_env


class TestDetectProxies:
    """代理检测必须覆盖大小写两种写法，且**对同一个变量去重**。

    ⚠️ 这里有一个**真实踩过的 CI 坑**（本地绿、Linux CI 红）：

        Windows 上 `os.environ` 大小写**不敏感** —— `HTTP_PROXY` 和 `http_proxy`
        是同一个变量，所以「只查大写名」在 Windows 上也能误打误撞查到大写写入的值；
        但 Linux 严格区分大小写，只设了 `https_proxy`（小写）时，
        `os.environ.get("HTTPS_PROXY")` 返回 `None` → **代理被完全漏检**。

    所以本组测试**不能依赖宿主平台的 `os.environ` 语义**来断言，
    必须显式构造「区分大小写」的环境来验证检测逻辑本身。
    """

    def test_none_when_clean(self, clean_proxy_env):
        assert detect_proxies() == {}

    def test_detects_uppercase(self, clean_proxy_env):
        clean_proxy_env.setenv("HTTPS_PROXY", "http://127.0.0.1:9999")
        assert detect_proxies() == {"HTTPS_PROXY": "http://127.0.0.1:9999"}

    def test_detects_lowercase(self, clean_proxy_env):
        """只设小写变量也必须能被检测到（Linux 上曾经漏检 → CI 挂）。"""
        clean_proxy_env.setenv("https_proxy", "http://127.0.0.1:9999")
        found = detect_proxies()
        assert list(found.values()) == ["http://127.0.0.1:9999"]
        assert list(found)[0].lower() == "https_proxy", "应保留用户实际使用的变量名"

    def test_lowercase_detected_under_case_sensitive_semantics(
        self, case_sensitive_proxy_env
    ):
        """⭐ 回归测试：**在区分大小写的环境下**（Linux 语义）小写代理不得漏检。

        这条测试在**任何平台**都用「区分大小写的 dict」跑，
        因此 Windows 本机也能拦住这个「CI 才暴露」的 bug。
        """
        case_sensitive_proxy_env["https_proxy"] = "http://127.0.0.1:9999"
        found = detect_proxies()
        assert list(found.values()) == ["http://127.0.0.1:9999"], (
            f"区分大小写的环境下漏检了小写代理：{found}"
        )

    def test_uppercase_detected_under_case_sensitive_semantics(
        self, case_sensitive_proxy_env
    ):
        case_sensitive_proxy_env["HTTP_PROXY"] = "http://a:1"
        assert detect_proxies() == {"HTTP_PROXY": "http://a:1"}

    def test_single_variable_is_reported_once(self, clean_proxy_env):
        """同一个变量（大小写视为一个）只应报出一次。"""
        clean_proxy_env.setenv("HTTP_PROXY", "http://a:1")
        found = detect_proxies()
        assert len(found) == 1, f"同一代理被重复报告：{found}"
        assert list(found.values()) == ["http://a:1"]

    def test_two_distinct_proxies_are_both_reported(self, clean_proxy_env):
        clean_proxy_env.setenv("HTTP_PROXY", "http://a:1")
        clean_proxy_env.setenv("HTTPS_PROXY", "http://b:2")
        found = detect_proxies()
        assert len(found) == 2
        assert set(found.values()) == {"http://a:1", "http://b:2"}

    def test_case_duplicates_are_deduped_under_case_sensitive_semantics(
        self, case_sensitive_proxy_env
    ):
        """Linux 上若真被写了大小写两份，只报一份即可（避免比配置本身更让人困惑）。"""
        case_sensitive_proxy_env["HTTP_PROXY"] = "http://a:1"
        case_sensitive_proxy_env["http_proxy"] = "http://a:1"
        found = detect_proxies()
        assert len(found) == 1, f"大小写重复未去重：{found}"

    def test_unrelated_env_is_ignored(self, case_sensitive_proxy_env):
        case_sensitive_proxy_env["SOMETHING_PROXY_LIKE"] = "http://x:1"
        case_sensitive_proxy_env["NOT_A_PROXY"] = "1"
        assert detect_proxies() == {}


class TestDiagnoseOffline:
    """离线 / 未配 Key 时不应执行网络探测（否则既慢又无意义）。"""

    def test_offline_short_circuits(self, clean_proxy_env, monkeypatch):
        import app.utils.netcheck as nc

        monkeypatch.setattr(
            nc, "settings", Settings(_env_file=None, llm_provider="offline")
        )
        report = diagnose_llm(do_network=False)
        names = [c.name for c in report.checks]
        assert "DNS 解析" not in " ".join(names)
        assert "离线" in report.conclusion

    def test_no_key_short_circuits(self, clean_proxy_env, monkeypatch):
        import app.utils.netcheck as nc

        monkeypatch.setattr(
            nc,
            "settings",
            Settings(_env_file=None, llm_provider="deepseek", llm_api_key=""),
        )
        report = diagnose_llm(do_network=False)
        assert "离线" in report.conclusion or "未配置" in report.conclusion


class TestModelNameCheck:
    """模型名检查：既不漏报，也不该把自定义模型名误报成错误。"""

    def _diagnose(self, monkeypatch, **kw):
        import app.utils.netcheck as nc

        monkeypatch.setattr(nc, "settings", Settings(_env_file=None, **kw))
        return diagnose_llm(do_network=False)

    def test_unknown_model_is_flagged(self, clean_proxy_env, monkeypatch):
        """⭐ 真实案例：LLM_MODEL=deepseek-flash —— DeepSeek 根本没有这个模型。"""
        report = self._diagnose(
            monkeypatch, llm_provider="deepseek", llm_api_key="sk-x",
            llm_model="deepseek-flash",
        )
        model_check = next(c for c in report.checks if c.name == "模型名检查")
        assert model_check.ok is False
        assert "deepseek-chat" in model_check.detail  # 给出可用的替代
        assert report.all_ok is False

    def test_known_model_passes(self, clean_proxy_env, monkeypatch):
        report = self._diagnose(
            monkeypatch, llm_provider="deepseek", llm_api_key="sk-x",
            llm_model="deepseek-chat",
        )
        model_check = next(c for c in report.checks if c.name == "模型名检查")
        assert model_check.ok is True

    def test_ollama_model_is_not_flagged(self, clean_proxy_env, monkeypatch):
        """本地模型名自由，不该被"已知清单"误判。"""
        report = self._diagnose(
            monkeypatch, llm_provider="ollama", llm_api_key="ollama",
            llm_model="my-local-finetune:latest",
        )
        model_check = next(c for c in report.checks if c.name == "模型名检查")
        assert model_check.ok is True

    def test_mismatched_model_is_flagged(self, clean_proxy_env, monkeypatch):
        report = self._diagnose(
            monkeypatch, llm_provider="deepseek", llm_api_key="sk-x",
            llm_model="gpt-4o-mini",
        )
        model_check = next(c for c in report.checks if c.name == "模型名检查")
        assert model_check.ok is False


class TestProxyCheckIsTheHeadlineIssue:
    """代理不通必须被识别为**失败项**，并被列为第一建议。"""

    def _diagnose(self, monkeypatch, **kw):
        import app.utils.netcheck as nc

        monkeypatch.setattr(nc, "settings", Settings(_env_file=None, **kw))
        return diagnose_llm(do_network=False)

    def test_unreachable_proxy_is_flagged_and_first_suggestion(
        self, clean_proxy_env, monkeypatch
    ):
        # 指向一个几乎不可能有人监听的端口
        clean_proxy_env.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
        report = self._diagnose(
            monkeypatch, llm_provider="deepseek", llm_api_key="sk-x",
            llm_model="deepseek-chat", llm_base_url="https://api.deepseek.com/v1",
        )
        proxy_check = next(c for c in report.checks if c.name == "代理环境变量")
        assert proxy_check.ok is False
        assert "连不上" in proxy_check.detail
        assert report.suggestions, "必须给出建议"
        assert "代理" in report.suggestions[0], "代理问题必须是第一条建议"

    def test_reachable_proxy_is_not_flagged(self, clean_proxy_env, monkeypatch):
        """代理端口可达时不该报错（它可能是正常的公司代理）。"""
        import socket

        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        try:
            clean_proxy_env.setenv("HTTPS_PROXY", f"http://127.0.0.1:{port}")
            report = self._diagnose(
                monkeypatch, llm_provider="deepseek", llm_api_key="sk-x",
                llm_model="deepseek-chat", llm_base_url="https://api.deepseek.com/v1",
            )
            proxy_check = next(c for c in report.checks if c.name == "代理环境变量")
            assert proxy_check.ok is True
        finally:
            srv.close()

    def test_no_proxy_is_ok(self, clean_proxy_env, monkeypatch):
        report = self._diagnose(
            monkeypatch, llm_provider="deepseek", llm_api_key="sk-x",
            llm_model="deepseek-chat", llm_base_url="https://api.deepseek.com/v1",
        )
        proxy_check = next(c for c in report.checks if c.name == "代理环境变量")
        assert proxy_check.ok is True
        assert "未检测到" in proxy_check.detail


class TestReportShape:
    """报告的序列化与格式化要稳定（要给接口用）。"""

    def test_all_ok_property(self):
        r = DiagnosisReport(checks=[CheckResult("a", True, "d"), CheckResult("b", True, "d")])
        assert r.all_ok is True
        r.checks.append(CheckResult("c", False, "d"))
        assert r.all_ok is False

    def test_to_dict_is_json_friendly(self):
        r = DiagnosisReport(
            checks=[CheckResult("a", False, "detail", "hint")],
            conclusion="结论",
            suggestions=["建议1"],
        )
        d = r.to_dict()
        assert d["all_ok"] is False
        assert d["conclusion"] == "结论"
        assert d["checks"][0]["name"] == "a"
        assert d["checks"][0]["hint"] == "hint"
        assert isinstance(d["suggestions"], list)

    def test_format_report_includes_marks_and_conclusion(self):
        r = DiagnosisReport(
            checks=[CheckResult("DNS", True, "ok"), CheckResult("TCP", False, "fail", "试试这个")],
            conclusion="有问题",
            suggestions=["建议A"],
        )
        text = format_report(r)
        assert "✅" in text and "❌" in text
        assert "有问题" in text
        assert "建议A" in text
        assert "试试这个" in text


class TestConnectionErrorExplanation:
    """`_explain_llm_error` 对连接类异常必须提到代理。"""

    def test_mentions_proxy_when_proxy_present(self, clean_proxy_env, monkeypatch):
        import app.core.llm as L

        clean_proxy_env.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
        monkeypatch.setattr(
            L, "settings",
            Settings(_env_file=None, llm_provider="deepseek", llm_api_key="sk-x",
                     llm_base_url="https://api.deepseek.com/v1"),
        )

        class APIConnectionError(Exception):
            pass

        msg = L._explain_llm_error(APIConnectionError("Connection error."))
        assert "代理" in msg
        assert "HTTP_PROXY" in msg or "HTTPS_PROXY" in msg
        assert "doctor.py" in msg

    def test_mentions_plain_network_checks_when_clean(self, clean_proxy_env, monkeypatch):
        import app.core.llm as L

        monkeypatch.setattr(
            L, "settings",
            Settings(_env_file=None, llm_provider="deepseek", llm_api_key="sk-x",
                     llm_base_url="https://api.deepseek.com/v1"),
        )

        class APIConnectionError(Exception):
            pass

        msg = L._explain_llm_error(APIConnectionError("Connection error."))
        assert "不是" in msg and "网络" in msg  # 明确「这不是配置问题」
        assert "doctor.py" in msg
