"""配置层与 LLM 错误提示的测试。

这一组测试守的是一个**真实踩过的 bug**：

    .env 里只把 LLM_PROVIDER 改成 deepseek，忘了改 LLM_MODEL，
    于是程序拿默认的 `gpt-4o-mini` 去请求 api.deepseek.com，
    DeepSeek 返回 400，最终用户看到的是一个**没有任何线索的 HTTP 500**。

修复思路是两条：
1. 模型名由 **provider 说了算**（不匹配就退回该 provider 的默认模型）；
2. 配置不一致时**大声说出来**（启动日志 + 体检接口），别让它烂在 500 里。
"""

from __future__ import annotations

import pytest

from app.config import PROVIDER_BASE_URLS, PROVIDER_DEFAULT_MODELS, Settings
from app.core.llm import _explain_llm_error


def _settings(**kw) -> Settings:
    """构造一个不读 .env 的配置对象，避免被本机环境影响。"""
    return Settings(_env_file=None, **kw)


class TestProviderDefaults:
    """每个内置 provider 都应该有可用的默认 model 与 base_url。"""

    @pytest.mark.parametrize("provider", ["openai", "deepseek", "zhipu", "moonshot"])
    def test_every_online_provider_has_default_model(self, provider):
        assert PROVIDER_DEFAULT_MODELS.get(provider), f"{provider} 缺默认模型"

    @pytest.mark.parametrize("provider", ["openai", "deepseek", "zhipu", "moonshot", "ollama"])
    def test_every_provider_has_default_base_url(self, provider):
        url = PROVIDER_BASE_URLS.get(provider, "")
        assert url.startswith("http"), f"{provider} 缺默认 base_url"

    def test_deepseek_base_url_is_not_openai(self):
        """回归：DeepSeek 绝不能悄悄走 OpenAI 的地址。"""
        s = _settings(llm_provider="deepseek", llm_api_key="sk-x", llm_base_url="")
        assert "deepseek" in s.resolved_base_url
        assert "openai" not in s.resolved_base_url

    def test_explicit_base_url_wins(self):
        s = _settings(
            llm_provider="deepseek",
            llm_api_key="sk-x",
            llm_base_url="https://my-gateway.example.com/v1",
        )
        assert s.resolved_base_url == "https://my-gateway.example.com/v1"


class TestResolvedModel:
    """`resolved_llm_model` 是修复 bug 的核心：模型名必须跟 provider 走。"""

    def test_stale_default_model_is_replaced(self):
        """⭐ 原 bug 场景：provider=deepseek 但 model 还是默认的 gpt-4o-mini。"""
        s = _settings(llm_provider="deepseek", llm_api_key="sk-x", llm_model="gpt-4o-mini")
        assert s.llm_model == "gpt-4o-mini"          # 用户的配置原样保留
        assert s.resolved_llm_model == "deepseek-chat"  # 但发出去的是对的
        assert s.llm_model_mismatch is True
        assert s.llm_config_warnings, "不一致时必须给出告警"

    def test_matching_model_is_kept(self):
        s = _settings(llm_provider="deepseek", llm_api_key="sk-x", llm_model="deepseek-reasoner")
        assert s.resolved_llm_model == "deepseek-reasoner"
        assert s.llm_model_mismatch is False
        assert s.llm_config_warnings == []

    def test_openai_default_is_untouched(self):
        """修 bug 不能误伤默认的 OpenAI 配置。"""
        s = _settings(llm_provider="openai", llm_api_key="sk-x")
        assert s.resolved_llm_model == "gpt-4o-mini"
        assert s.llm_config_warnings == []

    def test_custom_model_name_is_not_second_guessed(self):
        """第三方中转站的私有模型名不该被改掉（宁可不管，也不能猜错）。"""
        s = _settings(llm_provider="openai", llm_api_key="sk-x", llm_model="my-gateway-v2")
        assert s.resolved_llm_model == "my-gateway-v2"
        assert s.llm_model_mismatch is False

    def test_ollama_models_are_never_second_guessed(self):
        s = _settings(llm_provider="ollama", llm_api_key="ollama", llm_model="qwen2.5:7b")
        assert s.resolved_llm_model == "qwen2.5:7b"
        assert s.llm_model_mismatch is False

    def test_offline_provider_has_no_warnings(self):
        s = _settings(llm_provider="offline")
        assert s.llm_config_warnings == []
        assert s.llm_model_mismatch is False

    @pytest.mark.parametrize(
        "provider,bad_model,expected",
        [
            ("deepseek", "gpt-4o-mini", "deepseek-chat"),
            ("zhipu", "gpt-4o-mini", "glm-4-flash"),
            ("moonshot", "deepseek-chat", "moonshot-v1-8k"),
            ("openai", "deepseek-chat", "gpt-4o-mini"),
        ],
    )
    def test_cross_provider_model_is_corrected(self, provider, bad_model, expected):
        s = _settings(llm_provider=provider, llm_api_key="sk-x", llm_model=bad_model)
        assert s.resolved_llm_model == expected


class TestApiKeyHygiene:
    """Key 与 provider 对不上是很隐蔽的一类配置错误。"""

    def test_non_sk_key_for_openai_is_flagged(self):
        s = _settings(llm_provider="openai", llm_api_key="abc.def.ghi")
        assert any("LLM_API_KEY" in w for w in s.llm_config_warnings)

    def test_normal_sk_key_is_not_flagged(self):
        s = _settings(llm_provider="openai", llm_api_key="sk-abcdefg")
        assert not any("LLM_API_KEY" in w for w in s.llm_config_warnings)

    def test_offline_when_no_key(self):
        s = _settings(llm_provider="deepseek", llm_api_key="")
        assert s.is_offline_llm is True

    def test_online_when_key_present(self):
        s = _settings(llm_provider="deepseek", llm_api_key="sk-x")
        assert s.is_offline_llm is False


class TestExplainLlmError:
    """报错必须**能照着做**，而不是把 SDK 的英文原文丢给用户。"""

    class _FakeError(Exception):
        def __init__(self, msg: str, status_code: int | None = None):
            super().__init__(msg)
            self.status_code = status_code

    def test_auth_error_mentions_key(self):
        msg = _explain_llm_error(self._FakeError("Error code: 401 - bad key", 401))
        assert "LLM_API_KEY" in msg
        assert "401" in msg

    def test_model_not_found_mentions_llm_model(self):
        msg = _explain_llm_error(self._FakeError("Model Not Exist", 404))
        assert "LLM_MODEL" in msg
        assert "deepseek-chat" in msg  # 直接给出可用的示例值

    def test_bad_request_mentions_model_and_base_url(self):
        msg = _explain_llm_error(self._FakeError("invalid request", 400))
        assert "LLM_MODEL" in msg
        assert "LLM_BASE_URL" in msg
        assert "/v1" in msg

    def test_rate_limit(self):
        assert "限流" in _explain_llm_error(self._FakeError("rate limit", 429))

    def test_timeout(self):
        assert "网络" in _explain_llm_error(self._FakeError("Connection timed out"))

    def test_unknown_error_falls_back_to_exception_text(self):
        msg = _explain_llm_error(self._FakeError("weird thing happened"))
        assert "weird thing happened" in msg
