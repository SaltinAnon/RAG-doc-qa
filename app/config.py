"""全局配置模块。

本项目所有可调参数**只有一个来源**：环境变量（本地由 `.env` 文件提供）。
业务代码里禁止出现 `os.getenv(...)`，一律 `from app.config import settings`。

这样做的好处：
1. 换环境不用改代码（本地 / Docker / HuggingFace Spaces 用同一份代码）；
2. 敏感信息（API Key）只存在于 `.env`，不会误提交到 GitHub；
3. 面试时能一句话讲清「配置外置」这个工程实践。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（本文件在 <root>/app/config.py，所以向上两级）
BASE_DIR = Path(__file__).resolve().parent.parent

# 各提供商的默认端点（`LLM_BASE_URL` 留空时按 provider 取这里的值）
PROVIDER_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "moonshot": "https://api.moonshot.cn/v1",
    "ollama": "http://localhost:11434/v1",
}

# 各提供商的默认模型名（`LLM_MODEL` 留空、或与实际 provider 明显不匹配时取这里的值）
PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "deepseek": "deepseek-chat",
    "zhipu": "glm-4-flash",
    "moonshot": "moonshot-v1-8k",
    "ollama": "qwen2.5:7b",
}

# 用来判断「模型名看起来是哪家的」—— 只用于体检与默认值兜底，不做强校验
_PROVIDER_MODEL_HINTS: dict[str, tuple[str, ...]] = {
    "openai": ("gpt-", "o1", "o3", "o4", "text-davinci"),
    "deepseek": ("deepseek",),
    "zhipu": ("glm",),
    "moonshot": ("moonshot", "kimi"),
    "ollama": (),  # 本地模型名自由，不做判断
}

# 历史遗留的占位默认值 —— 用户没改过 LLM_MODEL 时 `llm_model` 就是它
_LEGACY_MODEL_PLACEHOLDER = "gpt-4o-mini"


class Settings(BaseSettings):
    """应用配置。字段名不区分大小写，对应 `.env` 里的大写变量。"""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # .env 里有多余变量时不报错
    )

    # ---------------- 应用 ----------------
    app_name: str = "RAG-DocQA"
    app_env: Literal["dev", "prod"] = "dev"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = ""  # 空 = 关闭鉴权

    # ---------------- LLM ----------------
    llm_provider: Literal[
        "openai", "deepseek", "zhipu", "moonshot", "ollama", "offline"
    ] = "offline"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_timeout: int = 60

    # ---------------- Embedding ----------------
    embedding_provider: Literal["openai", "local", "hash"] = "hash"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 512
    embedding_batch_size: int = 64

    # ---------------- 切分 ----------------
    chunk_size: int = Field(default=500, ge=50, le=4000)
    chunk_overlap: int = Field(default=80, ge=0, le=1000)
    max_file_size_mb: int = 50

    # ---------------- 检索 ----------------
    top_k: int = Field(default=5, ge=1, le=50)
    fetch_k: int = Field(default=20, ge=1, le=200)
    hybrid_enabled: bool = True
    rrf_k: int = 60
    rerank_enabled: bool = True
    score_threshold: float = 0.0

    # ---------------- 存储 ----------------
    chroma_dir: str = "data/chroma"
    docs_dir: str = "data/docs"
    collection_name: str = "default"

    # ---------------- 多轮对话 ----------------
    memory_enabled: bool = True
    memory_max_turns: int = 6
    session_ttl_minutes: int = 120

    # ---------------- 派生属性 ----------------

    @property
    def chroma_path(self) -> Path:
        """向量库绝对路径（相对路径按项目根目录解析）。"""
        p = Path(self.chroma_dir)
        return p if p.is_absolute() else (BASE_DIR / p)

    @property
    def docs_path(self) -> Path:
        """默认文档目录绝对路径。"""
        p = Path(self.docs_dir)
        return p if p.is_absolute() else (BASE_DIR / p)

    @property
    def auth_enabled(self) -> bool:
        """是否启用 API Key 鉴权。"""
        return bool(self.api_key.strip())

    @property
    def resolved_llm_api_key(self) -> str:
        """LLM Key：优先专用变量，回退到通用的 OPENAI_API_KEY。"""
        return self.llm_api_key.strip() or os.getenv("OPENAI_API_KEY", "").strip()

    @property
    def resolved_base_url(self) -> str:
        """实际使用的 base_url：显式配置优先，否则按 provider 取默认值。

        空串表示「不传 base_url，由 SDK 自己决定」（等价于 OpenAI 官方地址）。
        """
        if self.llm_base_url.strip():
            return self.llm_base_url.strip()
        return PROVIDER_BASE_URLS.get(self.llm_provider, "")

    @property
    def _model_matches_provider(self) -> bool:
        """`llm_model` 看起来是否属于当前 provider。

        只做**保守**判断：
        - `ollama` / `offline` 不判断（本地模型名自由）；
        - 模型名匹配到**任意其他** provider 的特征词 → 判定为「不属于本家」；
        - 都不匹配 → 视为「自定义模型名」，放行（不误伤第三方中转站/微调模型）。
        """
        if self.llm_provider in ("ollama", "offline"):
            return True
        model = self.llm_model.strip().lower()
        if not model:
            return False

        own = _PROVIDER_MODEL_HINTS.get(self.llm_provider, ())
        if own and any(h in model for h in own):
            return True
        for other, hints in _PROVIDER_MODEL_HINTS.items():
            if other == self.llm_provider:
                continue
            if hints and any(h in model for h in hints):
                return False  # 明显是别家的模型名
        return True

    @property
    def llm_model_mismatch(self) -> bool:
        """模型名是否与 provider 明显不匹配（用于启动告警与体检）。

        这是**真实踩过的 bug**：`.env` 里只改了 `LLM_PROVIDER=deepseek`，
        没有改 `LLM_MODEL`，于是程序拿默认的 `gpt-4o-mini` 去请求
        `api.deepseek.com` —— DeepSeek 上根本没有这个模型，接口返回 400，
        最终暴露给用户的是一个没有任何线索的 **HTTP 500**。
        """
        if self.llm_provider == "offline":
            return False
        return not self._model_matches_provider

    @property
    def resolved_llm_model(self) -> str:
        """实际会发出去的模型名。

        修复上面那个 bug 的关键：**provider 说了算** ——
        如果 `llm_model` 与 provider 明显不匹配（典型就是没改过默认的
        `gpt-4o-mini`），就退回该 provider 的默认模型，而不是把错的名字
        原样发出去、让对方返回一个看不懂的错误。
        """
        if self._model_matches_provider:
            return self.llm_model.strip()
        return PROVIDER_DEFAULT_MODELS.get(self.llm_provider, self.llm_model.strip())

    @property
    def llm_config_warnings(self) -> list[str]:
        """返回配置体检发现的问题（启动时打到日志、体检接口也返回）。

        设计原则：**只提醒，不擅自改用户的配置**。能安全兜底的
        （模型名）自动兜底并说明；兜底不了的（Key 的归属）只提示。
        """
        warnings: list[str] = []
        if self.llm_provider == "offline":
            return warnings

        if self.llm_model_mismatch:
            warnings.append(
                f"LLM_MODEL='{self.llm_model}' 看起来不属于 provider='{self.llm_provider}'，"
                f"已自动改用该提供商的默认模型 '{self.resolved_llm_model}'。"
                f"如果这是你有意配置的，请确认这个模型名在 {self.llm_provider} 上真实存在。"
            )

        # Key 与 provider 的官配前缀不一致 —— 常见于「换了 provider 但没换 Key」
        key = self.resolved_llm_api_key
        prefix_map = {"openai": "sk-", "deepseek": "sk-", "moonshot": "sk-"}
        expected = prefix_map.get(self.llm_provider)
        if expected and key and not key.startswith(expected):
            warnings.append(
                f"LLM_API_KEY 不像 {self.llm_provider} 的 Key（通常以 '{expected}' 开头）。"
                "如果鉴权报 401，请优先检查这里。"
            )

        # 用了 OpenAI 专用变量名去配别家
        if self.llm_provider != "openai" and not self.llm_api_key.strip():
            if os.getenv("OPENAI_API_KEY", "").strip():
                warnings.append(
                    "没有设置 LLM_API_KEY，当前用的是通用的 OPENAI_API_KEY 环境变量。"
                    f"如果它其实是 OpenAI 的 Key，拿去请求 {self.llm_provider} 会鉴权失败。"
                )
        return warnings

    @property
    def is_offline_llm(self) -> bool:
        """是否处于「离线抽取式」降级模式（前端要据此显示黄条提示）。"""
        if self.llm_provider == "offline":
            return True
        # 选了在线提供商但没填 Key → 也会降级
        return not self.resolved_llm_api_key

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_lt_size(cls, v: int, info) -> int:
        size = info.data.get("chunk_size")
        if size is not None and v >= size:
            raise ValueError(
                f"CHUNK_OVERLAP({v}) 必须小于 CHUNK_SIZE({size})，"
                "否则切分会死循环。建议 overlap = size * 0.1~0.2"
            )
        return v

    def ensure_dirs(self) -> None:
        """确保运行时目录存在（启动时调用一次）。"""
        for p in (self.chroma_path, self.docs_path):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取配置单例。

    用 `lru_cache` 保证整个进程只解析一次 `.env`，
    既省 IO 也让「测试里改环境变量」有明确的失效点（`get_settings.cache_clear()`）。
    """
    return Settings()


settings = get_settings()
