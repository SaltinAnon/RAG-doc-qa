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
        """不同提供商的默认 base_url。"""
        if self.llm_base_url.strip():
            return self.llm_base_url.strip()
        return {
            "openai": "https://api.openai.com/v1",
            "deepseek": "https://api.deepseek.com/v1",
            "zhipu": "https://open.bigmodel.cn/api/paas/v4",
            "moonshot": "https://api.moonshot.cn/v1",
            "ollama": "http://localhost:11434/v1",
        }.get(self.llm_provider, "")

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
