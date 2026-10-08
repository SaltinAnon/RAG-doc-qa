"""Embedding 工厂：把文本变成向量。支持在线 API、本地开源模型、零依赖兜底三种模式。

## 为什么要有「零依赖兜底」

现实问题：面试官 clone 你的仓库，大概率**没有 OpenAI Key**，也不想为一个 demo 花钱。
如果项目在 `pip install` 之后还必须配 Key 才能跑，那就是「跑不起来的开源项目」，
GitHub 上一定会有人提 issue 说 broke，你的简历项目反而变成减分项。

所以我们做了三层：

| 模式 | 触发条件 | 效果 | 成本 |
|---|---|---|---|
| `openai` | `EMBEDDING_PROVIDER=openai` + 有 Key | 最好，语义理解强 | 按量付费 |
| `local`  | `EMBEDDING_PROVIDER=local` + 装了 sentence-transformers | 好（BGE 中文模型），完全离线 | 免费，占 500MB 磁盘 |
| `hash`   | 默认 / 上面两种不可用时自动回退 | 一般（字面相似度），但**链路完整可跑** | 免费，零依赖 |

> 面试话术：「我做了 provider 抽象和自动降级，保证项目在无网络/无密钥环境下
> 依然能完整演示检索与引用溯源链路，只是生成质量会下降。
> 这一点在 CI 里也有覆盖——GitHub Actions 跑测试用的就是 hash 模式，不依赖任何外部服务。」
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from functools import lru_cache

import numpy as np
from langchain_core.embeddings import Embeddings

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
#  兜底方案：字符 n-gram 哈希向量（纯 Python + numpy，零外部依赖）
# ============================================================
class LocalHashEmbeddings(Embeddings):
    """基于**符号哈希（signed hashing trick）** 的轻量文本向量化。

    原理（推荐读一遍，面试常问「不用模型怎么算相似度」）：

    1. 把文本拆成字符 n-gram：中文用 unigram + bigram + trigram
       （「年假规定」→ 年/假/规/定/年假/假规/规定/年假规/假规定/规定…）；
    2. 每个 n-gram 用 **稳定哈希**（blake2b，不是 Python 内置 `hash()`，
       后者每进程随机加盐，重启就变）映射到 `dimension` 个桶之一；
    3. 哈希结果的某一位决定符号 ±1 —— 这一步叫 **signed hashing**，
       能让不同 token 的碰撞互相抵消而不是无限累加，显著降低噪声；
    4. 词频做次线性压缩 `1 + log(tf)`（避免高频词支配向量）；
    5. 最后 L2 归一化，于是**余弦相似度 = 点积**，可以直接用向量库的距离度量。

    局限（必须诚实说明）：
    - 它**不理解语义**：「年假」和「带薪休假」在这套表示下几乎不相关；
    - 优势是**字面精确匹配**很强，对专有名词/编号（如「GB/T 1234」）反而比
      小模型更靠谱。

    所以它是「保证项目可跑」的兜底，不是「效果好」的方案。
    要效果请用 `openai` 或 `local` 模式。
    """

    def __init__(self, dimension: int = 512) -> None:
        """初始化。

        Args:
            dimension: 向量维度。越大碰撞越少，512 在「几万条 chunk」规模下够用。
        """
        if dimension <= 0:
            raise ValueError("dimension 必须为正整数")
        self.dimension = dimension

    # ---------- 内部：文本 → 稀疏权重表 ----------
    @staticmethod
    def _ngram_weights(text: str) -> Counter[str]:
        """抽取字符 n-gram 并赋权。

        权重设计（有依据，不是拍脑袋）：
        - unigram(单字) 权重 1.0：提供召回，但区分度低（「的」到处都是）
        - bigram(双字)  权重 2.0：中文里最强的语义单元，权重最高
          （「年假」vs「假年」完全不同；单个「假」字没用）
        - trigram       权重 1.5：更具体但更稀疏，做补充

        Args:
            text: 输入文本。

        Returns:
            `Counter{ngram: 权重}`。
        """
        # 只保留有意义的字符：CJK + 字母数字
        import re

        chunks = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", text.lower())
        weights: Counter[str] = Counter()

        for chunk in chunks:
            is_cjk = "\u4e00" <= chunk[0] <= "\u9fff"
            n = len(chunk)

            for ch in chunk:
                weights[ch] += 1.0
                # 英文整词也作为整体特征，避免 "training" 被拆散后失去意义
            if not is_cjk:
                if n > 1:
                    weights[chunk] += 2.0
                continue

            for i in range(n - 1):
                weights[chunk[i : i + 2]] += 2.0
            for i in range(n - 2):
                weights[chunk[i : i + 3]] += 1.5

        return weights

    def _embed_one(self, text: str) -> np.ndarray:
        """单条文本 → 归一化向量。

        Args:
            text: 输入文本。

        Returns:
            形状 `(dimension,)` 的 float32 单位向量；空文本返回全零向量。
        """
        vec = np.zeros(self.dimension, dtype=np.float32)
        weights = self._ngram_weights(text)
        if not weights:
            return vec

        for ngram, tf in weights.items():
            digest = hashlib.blake2b(ngram.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign * (1.0 + math.log(tf))  # 次线性压缩词频

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    # ---------- LangChain Embeddings 接口 ----------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化（入库时调用）。

        Args:
            texts: 待向量化的文本列表。

        Returns:
            与输入等长的向量列表。
        """
        return [self._embed_one(t).tolist() for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """单条查询向量化（检索时调用）。

        ⚠️ 注意：`embed_query` 和 `embed_documents` 必须用**完全相同的算法**。
        很多教程在这里踩坑——查询用了 A 模型、文档用了 B 模型，
        结果检索永远返回随机结果，还很难排查。

        Args:
            text: 用户问题。

        Returns:
            归一化向量。
        """
        return self._embed_one(text).tolist()

    def __repr__(self) -> str:  # pragma: no cover - 仅用于日志可读性
        return f"LocalHashEmbeddings(dim={self.dimension})"


# ============================================================
#  工厂
# ============================================================
def _build_openai_embeddings() -> Embeddings:
    """构造 OpenAI（或任意 OpenAI 兼容）Embedding。

    Returns:
        LangChain Embeddings 实例。

    Raises:
        RuntimeError: 缺少 API Key。
    """
    from langchain_openai import OpenAIEmbeddings

    key = settings.llm_api_key.strip() or __import__("os").getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("EMBEDDING_PROVIDER=openai 但没有配置 LLM_API_KEY / OPENAI_API_KEY")

    logger.info("Embedding 使用在线接口：%s", settings.embedding_model)
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=key,
        base_url=settings.resolved_base_url or None,
        chunk_size=settings.embedding_batch_size,
    )


def _build_local_embeddings() -> Embeddings:
    """构造本地开源 Embedding（BGE 中文小模型）。

    国内网络建议先设镜像：`export HF_ENDPOINT=https://hf-mirror.com`

    Returns:
        LangChain HuggingFaceEmbeddings 实例。

    Raises:
        RuntimeError: 未安装 sentence-transformers。
    """
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "EMBEDDING_PROVIDER=local 需要先安装本地模型依赖：\n"
            "    pip install -r requirements-local.txt\n"
            "（会下载约 500MB 依赖 + 100MB 模型，国内建议先执行：\n"
            "    set HF_ENDPOINT=https://hf-mirror.com\n"
            " 或 export HF_ENDPOINT=https://hf-mirror.com）"
        ) from exc

    model_name = settings.embedding_model
    if model_name.startswith("text-embedding"):  # 用户没改配置时给个合理默认
        model_name = "BAAI/bge-small-zh-v1.5"

    logger.info("Embedding 使用本地模型：%s（首次运行会下载）", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        # ⚠️ BGE 系列必须归一化 + 查询侧加指令前缀，否则相似度排序会明显变差
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=4)
def get_embeddings(provider: str | None = None) -> Embeddings:
    """获取 Embedding 实例（进程内单例）。

    自动降级顺序：请求的 provider → 失败则逐级回退到 `hash`。
    **绝不因为 Key 没配就让整个应用起不来。**

    Args:
        provider: 覆盖配置的 provider（`openai` / `local` / `hash`）。
            传 `None` 用 `.env` 里的值。

    Returns:
        可用的 Embeddings 实例（保证不抛异常）。

    Raises:
        无。任何异常都会被捕获并降级。
    """
    wanted = (provider or settings.embedding_provider).lower()

    builders = {
        "openai": _build_openai_embeddings,
        "local": _build_local_embeddings,
    }

    if wanted in builders:
        try:
            return builders[wanted]()
        except Exception as exc:
            logger.warning(
                "Embedding provider=%s 初始化失败（%s），自动降级为本地哈希向量。"
                "检索仍可运行，但语义相似度能力有限。",
                wanted, exc,
            )
    elif wanted != "hash":
        logger.warning("未知 EMBEDDING_PROVIDER=%s，回退 hash 模式", wanted)

    logger.info("Embedding 使用兜底模式：LocalHashEmbeddings(dim=%d)", settings.embedding_dim)
    return LocalHashEmbeddings(dimension=settings.embedding_dim)


def describe_embeddings(emb: Embeddings) -> str:
    """返回 Embedding 的可读描述（写进 API 响应，让调用方知道用的什么）。"""
    name = type(emb).__name__
    if isinstance(emb, LocalHashEmbeddings):
        return f"offline-hash(dim={emb.dimension})"
    return getattr(emb, "model", None) or getattr(emb, "model_name", None) or name
