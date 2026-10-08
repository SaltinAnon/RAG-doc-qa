"""检索器：向量检索 + BM25 关键词检索 的混合召回与融合。

## 为什么纯向量检索不够（这是 RAG 最常见的性能瓶颈）

向量检索擅长「语义相近」但**对精确字面匹配不敏感**：

| 用户问 | 向量检索 | BM25 关键词 |
|---|---|---|
| 「年假几天」→ 文档写「带薪年休假 5 天」 | ✅ 语义命中了 | ❌ 字面不重叠 |
| 「GB/T 19001 是什么」→ 文档有该编号 | ❌ 编号被「平均」掉 | ✅ 精确命中 |
| 「张三的入职日期」→ 文档有「张三」 | ⚠️ 人名可能被弱化 | ✅ 精确命中 |
| 「怎么请假」→ 文档写「请假流程」 | ✅ | ✅ |

结论：**两者互补，必须一起用**。

## 融合方法：RRF（Reciprocal Rank Fusion）

两种检索器的分数**量纲不同**（余弦相似度 0~1 vs BM25 无上界），
直接加权平均是错的 —— 权重调参永远调不好。

RRF 只用**排名**不用分数，天然解决量纲问题：

    RRF_score(d) = Σ_over_retrievers  1 / (k + rank_r(d))

- `k` 默认 60（论文经验值），作用是压低「排第 1」的过度优势，
  让「两路都排第 3」的文档胜过「一路第 1、另一路第 50」的文档。
- 无需归一化、无需调参、对异常分数鲁棒。

## 最后一步：启发式重排序（Rerank）

召回阶段拿的是「可能相关」的 20 条，要交给 LLM 的只有 5 条，
所以再做一轮精排。生产环境这里应该换成 Cross-Encoder 重排模型
（如 `BAAI/bge-reranker-base`），但那个要额外 400MB 依赖。

本项目用一套**无需模型的启发式打分**，反而更适合面试讲：
- 查询词覆盖率（query 的 token 有多少出现在 chunk 里）
- 短语命中（连续 2 字以上的查询片段整段出现，加分）
- 数字/编号命中（问「5 天」时含数字的片段更相关）
- 长度惩罚（过短的信息不足，过长的可能是拼接噪声）
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from langchain_core.documents import Document

from app.config import settings
from app.core.embeddings import get_embeddings
from app.core.vectorstore import VectorStore, get_vector_store
from app.utils.logger import get_logger
from app.utils.text import idf_weights, jaccard, tokenize_for_bm25

logger = get_logger(__name__)


# ============================================================
#  数据结构
# ============================================================
@dataclass
class RetrievedChunk:
    """一条检索结果（含各阶段分数，方便调试与展示）。"""

    doc: Document
    final_score: float            # 最终排序分（0~1，用于展示与阈值过滤）
    rrf_score: float = 0.0        # RRF 融合分
    vector_score: float = 0.0     # 向量余弦相似度
    bm25_score: float = 0.0       # BM25 原始分（未归一化）
    vector_rank: int = -1         # 在向量结果里的名次（从 1 开始，-1 表示未命中）
    bm25_rank: int = -1           # 在 BM25 结果里的名次
    rerank_score: float = 0.0     # 重排序分

    @property
    def text(self) -> str:
        return self.doc.page_content

    @property
    def metadata(self) -> dict[str, Any]:
        return self.doc.metadata

    def to_dict(self) -> dict[str, Any]:
        """转成可 JSON 序列化的字典（日志/接口调试用）。"""
        return {
            "text": self.text[:120],
            "source": self.metadata.get("source"),
            "page": self.metadata.get("page"),
            "final_score": round(self.final_score, 4),
            "vector_score": round(self.vector_score, 4),
            "bm25_score": round(self.bm25_score, 3),
            "vector_rank": self.vector_rank,
            "bm25_rank": self.bm25_rank,
        }


@dataclass
class RetrievalResult:
    """一次完整检索的产出。"""

    chunks: list[RetrievedChunk] = field(default_factory=list)
    query: str = ""
    debug: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.chunks)

    def is_empty(self) -> bool:
        return not self.chunks


class RetrieverError(RuntimeError):
    """检索过程出现技术性错误。"""


# ============================================================
#  BM25 索引缓存
# ============================================================
class _BM25Cache:
    """按集合缓存 BM25 索引。

    为什么缓存：BM25Okapi 每次构建都要遍历全部文档做词频统计，
    几千条 chunk 要几百毫秒。而知识库在两次入库之间是不变的，
    所以用「集合名 + chunk 数量」当缓存键，数量变了才重建。
    （用数量而不是哈希，是因为它 O(1) 且足够灵敏——入库必然改变数量。）
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._index: Any = None
        self._corpus: list[Document] = []
        self._key: tuple[str, int] = ("", -1)

    def get(self, collection: str, documents: list[Document]) -> tuple[Any, list[Document]]:
        """获取（或重建）BM25 索引。

        Args:
            collection: 集合名。
            documents: 当前集合全部 chunk。

        Returns:
            `(bm25_index, corpus)`；语料为空时返回 `(None, [])`。
        """
        key = (collection, len(documents))

        with self._lock:
            if key == self._key and self._index is not None:
                return self._index, self._corpus

            if not documents:
                self._index, self._corpus, self._key = None, [], key
                return None, []

            try:
                from rank_bm25 import BM25Okapi
            except ImportError as exc:  # pragma: no cover
                raise RetrieverError(
                    "缺少 rank-bm25 依赖，混合检索不可用。请执行：pip install rank-bm25\n"
                    "（或把 .env 里的 HYBRID_ENABLED 设为 false 只用向量检索）"
                ) from exc

            # 用自定义分词器：中文按字 + bigram，英文按词。
            # 不用 jieba 的理由见 app/utils/text.py 的 tokenize_for_bm25 文档。
            tokenized = [tokenize_for_bm25(d.page_content) for d in documents]

            # BM25Okapi 对空文档会除零告警，塞一个占位 token
            tokenized = [t if t else ["∅"] for t in tokenized]

            try:
                index = BM25Okapi(tokenized)
            except ZeroDivisionError:
                # 全库只有一个文档时 idf 计算会除零，此时 BM25 无意义，退回纯向量
                logger.warning("BM25 索引构建失败（语料过小），本次退化为纯向量检索")
                self._index, self._corpus, self._key = None, documents, key
                return None, documents

            self._index, self._corpus, self._key = index, documents, key
            logger.info("BM25 索引已构建：collection=%s docs=%d", collection, len(documents))
            return index, documents


# ============================================================
#  检索器
# ============================================================
class HybridRetriever:
    """混合检索器：向量 + BM25 → RRF 融合 → 启发式重排。"""

    def __init__(self, store: VectorStore | None = None) -> None:
        """初始化。

        Args:
            store: 向量库实例；None 时用全局单例。
        """
        self.store = store or get_vector_store()
        self.embeddings = get_embeddings()
        self._bm25 = _BM25Cache()

    # ---------- 对外主入口 ----------
    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        fetch_k: int | None = None,
        collection: str | None = None,
        where: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """执行一次混合检索。

        Args:
            query: 用户问题。
            top_k: 最终返回条数，默认配置的 `TOP_K`。
            fetch_k: 每路召回候选数，默认配置的 `FETCH_K`。
            collection: 集合名。
            where: 向量侧的元数据过滤条件。

        Returns:
            `RetrievalResult`。查询为空或知识库为空时返回空结果（不抛异常）。

        Raises:
            RetrieverError: 向量库或索引层面的技术错误。
        """
        query = (query or "").strip()
        if not query:
            return RetrievalResult(query=query, debug={"error": "empty_query"})

        col = collection or self.store.collection_name
        top_k = top_k or settings.top_k
        fetch_k = fetch_k or max(settings.fetch_k, top_k)

        debug: dict[str, Any] = {
            "used_hybrid": False,
            "used_rerank": False,
            "vector_hits": 0,
            "bm25_hits": 0,
            "fused": 0,
            "reranked": 0,
        }

        # ---------- 第 1 路：向量检索 ----------
        vector_hits: list[tuple[Document, float]] = []
        try:
            q_vec = self.embeddings.embed_query(query)
            vector_hits = self.store.similarity_search(
                q_vec, top_k=fetch_k, collection=col, where=where
            )
        except Exception as exc:
            # 向量路失败不应该让整个问答挂掉 —— BM25 还能兜底
            logger.warning("向量检索失败，尝试仅用关键词检索：%s", exc, exc_info=True)
        debug["vector_hits"] = len(vector_hits)

        if not vector_hits and not settings.hybrid_enabled:
            return RetrievalResult(query=query, debug=debug)

        # ---------- 第 2 路：BM25 关键词检索 ----------
        bm25_hits: list[tuple[Document, float]] = []
        if settings.hybrid_enabled:
            try:
                corpus = self.store.get_all_documents(col)
                index, docs = self._bm25.get(col, corpus)
                if index is not None and docs:
                    scores = np.asarray(index.get_scores(tokenize_for_bm25(query)), dtype=np.float32)
                    # 只保留有正分的，取前 fetch_k
                    order = np.argsort(-scores)[:fetch_k]
                    for i in order:
                        s = float(scores[i])
                        if s > 0:
                            bm25_hits.append((docs[int(i)], s))
                    debug["used_hybrid"] = True
            except RetrieverError:
                raise
            except Exception as exc:
                logger.warning("BM25 检索失败，退化为纯向量检索：%s", exc, exc_info=True)
        debug["bm25_hits"] = len(bm25_hits)

        # ---------- 融合 ----------
        if bm25_hits and vector_hits:
            fused = self._rrf_fuse(vector_hits, bm25_hits, k=settings.rrf_k)
        elif vector_hits:
            fused = self._wrap_single_source(vector_hits, source="vector")
        elif bm25_hits:
            fused = self._wrap_single_source(bm25_hits, source="bm25")
        else:
            return RetrievalResult(query=query, debug={**debug, "note": "no_hits"})

        debug["fused"] = len(fused)

        # ---------- 重排序 ----------
        if settings.rerank_enabled:
            fused = self._rerank(query, fused)
            debug["used_rerank"] = True

        # ---------- 多样性去重：避免 5 条都是同一页的近似内容 ----------
        fused = self._dedupe_similar(fused)

        # ---------- 阈值过滤 + 截断 ----------
        if settings.score_threshold > 0:
            fused = [c for c in fused if c.final_score >= settings.score_threshold]

        result = fused[:top_k]
        debug["reranked"] = len(result)

        logger.info(
            "检索完成 query=%r 向量命中=%d BM25命中=%d 融合=%d 最终=%d",
            query[:30], debug["vector_hits"], debug["bm25_hits"],
            debug["fused"], len(result),
        )
        return RetrievalResult(chunks=result, query=query, debug=debug)

    # ---------- 融合算法 ----------
    @staticmethod
    def _rrf_fuse(
        vector_hits: list[tuple[Document, float]],
        bm25_hits: list[tuple[Document, float]],
        k: int = 60,
    ) -> list[RetrievedChunk]:
        """RRF 加权融合两路检索结果。

        同一份 chunk 可能在两路都出现（用 chunk_id 或文本哈希对齐），
        这时它的 RRF 分会叠加 —— 这正是我们想要的「双路共识」信号。

        Args:
            vector_hits: `[(Document, 相似度)]`，已按相似度降序。
            bm25_hits: `[(Document, BM25 分)]`，已按分数降序。
            k: RRF 平滑系数，论文默认 60。

        Returns:
            按 RRF 分降序的 `RetrievedChunk` 列表。

        Raises:
            RetrieverError: 两路都为空。
        """
        if not vector_hits and not bm25_hits:
            raise RetrieverError("RRF 融合收到两路空结果集")

        merged: dict[str, RetrievedChunk] = {}

        def _key(doc: Document) -> str:
            """chunk 的唯一键：优先用 chunk_id，回退用 (doc_id, 页码, 文本哈希)。"""
            cid = doc.metadata.get("chunk_id")
            if cid:
                return str(cid)
            import hashlib

            raw = f"{doc.metadata.get('doc_id')}|{doc.metadata.get('page')}|{doc.page_content}"
            return hashlib.sha256(raw.encode()).hexdigest()[:16]

        for rank, (doc, score) in enumerate(vector_hits, start=1):
            key = _key(doc)
            item = merged.setdefault(key, RetrievedChunk(doc=doc, final_score=0.0))
            item.vector_score = float(score)
            item.vector_rank = rank
            item.rrf_score += 1.0 / (k + rank)

        for rank, (doc, score) in enumerate(bm25_hits, start=1):
            key = _key(doc)
            item = merged.setdefault(key, RetrievedChunk(doc=doc, final_score=0.0))
            item.bm25_score = float(score)
            item.bm25_rank = rank
            item.rrf_score += 1.0 / (k + rank)

        chunks = list(merged.values())

        # RRF 分的理论最大值：两路都排第一 = 2/(k+1)
        max_possible = 2.0 / (k + 1)
        for c in chunks:
            c.final_score = min(1.0, c.rrf_score / max_possible)

        chunks.sort(key=lambda x: x.rrf_score, reverse=True)
        return chunks

    @staticmethod
    def _wrap_single_source(
        hits: list[tuple[Document, float]], source: str
    ) -> list[RetrievedChunk]:
        """只有一路有结果时，包装成 RetrieverChunk（分数归一化到 0~1）。

        Args:
            hits: 检索结果。
            source: `"vector"` 或 `"bm25"`。

        Returns:
            统一结构的列表。
        """
        if not hits:
            return []

        scores = [float(s) for _, s in hits]
        lo, hi = min(scores), max(scores)
        span = hi - lo

        out: list[RetrievedChunk] = []
        for rank, (doc, score) in enumerate(hits, start=1):
            # 分数全相同时统一给 1.0，避免除零
            norm = 1.0 if span == 0 else (float(score) - lo) / span
            # 名次本身也带信息：名次靠前的最终分更高
            final = 0.6 * norm + 0.4 * (1.0 - (rank - 1) / max(len(hits), 1))
            c = RetrievedChunk(doc=doc, final_score=round(final, 4))
            if source == "vector":
                c.vector_score = float(score)
                c.vector_rank = rank
            else:
                c.bm25_score = float(score)
                c.bm25_rank = rank
            out.append(c)
        return out

    # ---------- 重排序 ----------
    @staticmethod
    def _length_score(text_len: int, ideal: int | None = None) -> float:
        """片段长度合理性打分（0~1）。

        为什么要有这一项：检索结果里经常混进两种垃圾：
        - **过短的碎片**（如「详见附件」「如下：」），检索分可能很高（因为字面匹配），
          但根本没有信息量，塞给 LLM 纯属浪费；
        - **过长的拼接块**（表格被硬切、多段落粘在一起），噪声比例高。

        曲线设计（以 `ideal` 为理想长度）：
            ratio < 0.2   → 0.20        （极短，强烈惩罚）
            0.2 ~ 0.5     → 线性升到 1.0
            0.5 ~ 1.5     → 1.00        （甜区，不惩罚）
            > 1.5         → 缓慢衰减到 0.3

        Args:
            text_len: 片段字符数。
            ideal: 理想长度，默认取配置的 `CHUNK_SIZE`。

        Returns:
            0.0 ~ 1.0 的分数。
        """
        ideal = ideal or settings.chunk_size
        if ideal <= 0:
            return 1.0

        ratio = text_len / ideal
        if ratio < 0.2:
            return 0.20
        if ratio < 0.5:
            return 0.20 + (ratio - 0.2) / 0.3 * 0.80
        if ratio <= 1.5:
            return 1.0
        return max(0.30, 1.0 - (ratio - 1.5) / 3.0)

    @staticmethod
    def _rerank(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """启发式重排序：用「与查询的字面贴合度」给候选精排。

        打分公式（每项都归一化到 0~1 后加权）：

            rerank = 0.40 * 查询词覆盖率      ← 最重要的信号
                   + 0.25 * 短语命中率        ← 连续片段整体命中，强信号
                   + 0.20 * 数字/编号命中     ← 「5 天」「GB/T 19001」这类
                   + 0.15 * 长度合理性        ← 太短信息不足、太长噪声多

        最终分 = 0.4 * RRF 分 + 0.6 * rerank 分（重排结果占主导，
        但保留一部分召回阶段的共识信息）。

        Args:
            query: 用户问题。
            chunks: 融合后的候选。

        Returns:
            重排序后的列表（原地更新 `final_score` 与 `rerank_score`）。
        """
        if not chunks:
            return chunks

        q_tokens = set(tokenize_for_bm25(query))
        # 查询里的「实词」：去掉单字功能词后的 token，用来算覆盖率
        q_content = {t for t in q_tokens if len(t) > 1} or q_tokens
        q_numbers = set(re.findall(r"\d+(?:\.\d+)?", query))
        # 查询的连续片段（2~6 字），用于短语命中判断
        q_phrases = [
            query[i : i + n]
            for n in (2, 3, 4, 6)
            for i in range(max(0, len(query) - n + 1))
            if not query[i : i + n].isspace()
        ]
        q_phrases = list(dict.fromkeys(q_phrases))[:40]  # 去重 + 限量，控制耗时

        # ⭐ IDF 加权：让「公司」「员工」这类到处都是的词不再虚抬覆盖率。
        #    详细原理见 app/utils/text.py 的 idf_weights 文档。
        cand_token_sets = [set(tokenize_for_bm25(c.text)) for c in chunks]
        q_weights = idf_weights(cand_token_sets, q_content)
        total_weight = sum(q_weights.values()) or 1.0

        for c, t_tokens in zip(chunks, cand_token_sets):
            text = c.text

            # 1) 查询词覆盖率（IDF 加权）
            coverage = (
                sum(w for t, w in q_weights.items() if t in t_tokens) / total_weight
                if q_content
                else 0.0
            )

            # 2) 短语命中率（同时考虑短语长度加权）
            phrase_hit, phrase_weight = 0.0, 0.0
            for p in q_phrases:
                w = len(p)
                phrase_weight += w
                if p in text:
                    phrase_hit += w
            phrase_score = (phrase_hit / phrase_weight) if phrase_weight else 0.0

            # 3) 数字/编号命中 —— 问「几天」时含数字的片段更可能给出答案
            if q_numbers:
                num_score = len(q_numbers & set(re.findall(r"\d+(?:\.\d+)?", text))) / len(q_numbers)
            else:
                # 查询没有数字时，给「含具体数字的片段」一点小加分
                num_score = 0.5 if re.search(r"\d", text) else 0.0

            # 4) 长度合理性
            length_score = HybridRetriever._length_score(len(text))

            c.rerank_score = round(
                float(
                    0.40 * coverage
                    + 0.25 * phrase_score
                    + 0.20 * num_score
                    + 0.15 * length_score
                ),
                4,
            )
            c.final_score = round(0.4 * c.final_score + 0.6 * c.rerank_score, 4)

        chunks.sort(key=lambda x: x.final_score, reverse=True)
        return chunks

    # ---------- 多样性去重 ----------
    @staticmethod
    def _dedupe_similar(chunks: list[RetrievedChunk], threshold: float = 0.85) -> list[RetrievedChunk]:
        """剔除内容高度重叠的候选（同一个 PDF 的相邻 chunk 常有 60%~90% 重叠）。

        为什么必要：`CHUNK_OVERLAP` 让相邻 chunk 天生共享一段文本，
        如果不处理，top_k=5 可能全是同一段话的五个变体，
        白白浪费上下文窗口，还让 LLM 以为「多处都这么说」。

        实现用 **MMR 的简化版**：按分数从高到低遍历，
        只有与所有已选中的 chunk 相似度都低于阈值才保留。

        Args:
            chunks: 已按分数降序的候选。
            threshold: Jaccard 相似度阈值。

        Returns:
            去重后的列表。
        """
        kept: list[RetrievedChunk] = []
        token_sets: list[set[str]] = []

        for c in chunks:
            ts = set(tokenize_for_bm25(c.text))
            if any(jaccard(ts, prev) > threshold for prev in token_sets):
                continue
            kept.append(c)
            token_sets.append(ts)

        return kept


# ============================================================
#  单例
# ============================================================
_retriever: HybridRetriever | None = None
_retriever_lock = threading.Lock()


def get_retriever(store: VectorStore | None = None, force_new: bool = False) -> HybridRetriever:
    """获取检索器单例（避免重复加载 Embedding 模型）。

    Args:
        store: 指定向量库（测试用）。
        force_new: 强制新建。

    Returns:
        HybridRetriever 实例。
    """
    global _retriever
    if force_new or store is not None:
        return HybridRetriever(store)

    with _retriever_lock:
        if _retriever is None:
            _retriever = HybridRetriever()
        return _retriever


def reset_retriever_singleton() -> None:
    """清空单例（测试用）。"""
    global _retriever
    with _retriever_lock:
        _retriever = None
