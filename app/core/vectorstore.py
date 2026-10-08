"""向量库封装：ChromaDB 持久化存储与相似度检索。

## 为什么选 ChromaDB

- **零运维**：`PersistentClient(path=...)` 直接落盘，不需要起服务、不需要 Docker；
- **元数据过滤**：支持 `where={"doc_id": ...}`，删文档/多租户隔离都靠它；
- **API 干净**：`upsert` 天然幂等，重复入库不会产生重复向量。

生产环境要换 Milvus / Qdrant / pgvector，只需要替换本文件的实现，
上层 `retriever.py` 完全不用改 —— 这就是「把向量库藏在接口后面」的价值。

## 关键设计点

1. **用 `upsert` 而不是 `add`**：`add` 遇到重复 ID 会抛异常。
   我们的 chunk_id 是内容哈希，同一文档重复上传时 ID 相同，
   用 upsert 就自动幂等，不需要先查再删。
2. **显式指定余弦距离**：Chroma 默认是 L2 欧氏距离。对文本向量来说
   余弦相似度更合适（我们关心方向不关心模长），而且余弦距离
   `1 - cos` 可以直接转成「相似度分数」展示给用户。
3. **批次写入**：Chroma 单次 upsert 有大小限制（约 5461 条），
   超了会静默丢数据。必须分批。这是实测踩到的坑。
"""

from __future__ import annotations

import threading
from collections import Counter
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Chroma 单次写入上限。官方文档写的 5461（对应 SQLite 变量上限），
# 我们保守取 1000，避免某些平台编译参数不同导致的边界问题。
MAX_BATCH_SIZE = 1000


class VectorStoreError(RuntimeError):
    """向量库操作失败。"""


class VectorStore:
    """ChromaDB 的薄封装。

    所有对 Chroma 的直接调用都收敛在这个类里，
    这样将来换向量库（Milvus/Qdrant/pgvector）只改这一个文件。

    线程安全：Chroma 自身的 client 是线程安全的，
    我们额外用锁保护「集合句柄缓存」，避免并发首次访问时重复创建。
    """

    def __init__(self, persist_dir: Path | str | None = None, collection: str | None = None) -> None:
        """初始化并连接持久化向量库。

        Args:
            persist_dir: 落盘目录，默认取配置的 `CHROMA_DIR`。
            collection: 集合名，默认取配置的 `COLLECTION_NAME`。
                「集合」≈ 数据库里的「表」，可以按租户/项目分集合做隔离。

        Raises:
            VectorStoreError: 目录无法创建或 Chroma 初始化失败。
        """
        self.persist_dir = Path(persist_dir) if persist_dir else settings.chroma_path
        self.collection_name = collection or settings.collection_name
        self._lock = threading.Lock()
        self._collections: dict[str, Any] = {}

        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            # anonymized_telemetry=False：关掉遥测，容器里没网也不会卡住启动
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
            logger.info("ChromaDB 已连接：path=%s collection=%s",
                        self.persist_dir, self.collection_name)
        except Exception as exc:
            raise VectorStoreError(
                f"ChromaDB 初始化失败（目录：{self.persist_dir}）：{exc}\n"
                "排查建议：\n"
                "  1. 目录是否有写权限？\n"
                "  2. 是否装了 sqlite3？（Linux 最小镜像常见问题：apt install libsqlite3-dev）\n"
                "  3. 依赖版本是否匹配？pip install -r requirements.txt"
            ) from exc

    # ---------------- 集合管理 ----------------
    def _get_collection(self, name: str | None = None, create: bool = True) -> Any:
        """获取集合句柄（带缓存）。

        Args:
            name: 集合名，None 用实例默认值。
            create: 不存在时是否创建。

        Returns:
            Chroma Collection 对象。

        Raises:
            VectorStoreError: 集合不存在且 create=False。
        """
        name = name or self.collection_name
        with self._lock:
            if name in self._collections:
                return self._collections[name]

            try:
                if create:
                    col = self._create_or_get(name)
                else:
                    col = self._client.get_collection(name)
            except Exception as exc:
                raise VectorStoreError(
                    f"获取集合「{name}」失败：{exc}。"
                    "如果提示 collection does not exist，说明还没入库任何文档。"
                ) from exc

            self._collections[name] = col
            return col

    def _create_or_get(self, name: str) -> Any:
        """创建或获取集合，并确保使用**余弦距离**。

        为什么必须显式指定：Chroma 默认是 L2 欧氏距离。
        对文本向量来说余弦更合适（我们关心方向不关心模长），
        而且余弦距离 `1 - cos` 可以直接转成「相似度分数」展示给用户。
        用默认的 L2 会让相似度分数失去直观意义（范围是 0~∞ 而不是 0~1）。

        不同 chromadb 版本的参数写法不一样，这里按优先级依次尝试：

        - **chromadb 1.x**：`configuration={"hnsw": {"space": "cosine"}}`
        - **chromadb 0.4~0.5.x**：`metadata={"hnsw:space": "cosine"}`

        如果两种都失败（未来版本又改 API），**不会静默降级** ——
        会打一条 WARNING 明确说明「当前使用默认距离度量，相似度分数不可比」，
        因为静默改变距离度量会让检索质量悄悄变差，是最难排查的问题之一。

        Args:
            name: 集合名。

        Returns:
            Collection 对象。
        """
        # 方案 1：chromadb 1.x 的 configuration 参数
        try:
            return self._client.get_or_create_collection(
                name=name,
                configuration={"hnsw": {"space": "cosine"}},
                embedding_function=None,  # 我们自己算向量，不用 Chroma 内置的
            )
        except (TypeError, ValueError, KeyError) as exc:
            logger.debug("configuration 参数不被支持（%s），尝试 metadata 写法", exc)

        # 方案 2：0.4~0.5.x 的 metadata 写法
        try:
            return self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,
            )
        except (TypeError, ValueError, KeyError) as exc:
            logger.debug("metadata 的 hnsw:space 不被支持（%s），使用默认参数", exc)

        # 方案 3：兜底 —— 但明确告警，绝不静默
        logger.warning(
            "集合「%s」无法设置余弦距离度量（当前 chromadb 版本 %s 的参数写法不匹配），"
            "已使用默认距离度量。这会导致相似度分数不可直接解释，"
            "建议检查 chromadb 版本或改用受支持的版本。",
            name, getattr(__import__("chromadb"), "__version__", "unknown"),
        )
        return self._client.get_or_create_collection(name=name, embedding_function=None)

    def list_collections(self) -> list[str]:
        """列出所有集合名。"""
        return [c.name for c in self._client.list_collections()]

    def drop_collection(self, name: str | None = None) -> None:
        """删除整个集合（危险操作，用于重置）。

        Args:
            name: 集合名，None 用默认集合。
        """
        name = name or self.collection_name
        with self._lock:
            try:
                self._client.delete_collection(name)
                self._collections.pop(name, None)
                logger.warning("已删除集合：%s", name)
            except Exception as exc:
                logger.warning("删除集合 %s 失败（可能本来就不存在）：%s", name, exc)

    # ---------------- 写入 ----------------
    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
        collection: str | None = None,
    ) -> int:
        """把切分好的 chunk 及其向量写入向量库。

        用 `upsert` 保证幂等：同一文档重复入库会覆盖而不是新增。

        Args:
            documents: chunk 列表（metadata 里必须有 `chunk_id`）。
            embeddings: 与 documents 一一对应的向量列表。
            collection: 目标集合名。

        Returns:
            实际写入的条数。

        Raises:
            VectorStoreError: 长度不一致或写入失败。
        """
        if len(documents) != len(embeddings):
            raise VectorStoreError(
                f"documents({len(documents)}) 与 embeddings({len(embeddings)}) 数量不一致，"
                "说明向量化过程有丢数据，请检查 embeddings 实现。"
            )
        if not documents:
            return 0

        col = self._get_collection(collection)

        ids: list[str] = []
        texts: list[str] = []
        metas: list[dict[str, Any]] = []

        for i, d in enumerate(documents):
            cid = d.metadata.get("chunk_id") or f"auto-{i}-{abs(hash(d.page_content))}"
            ids.append(str(cid))
            texts.append(d.page_content)
            # Chroma 的 metadata 只接受 str/int/float/bool，None 会报错，必须过滤
            metas.append(
                {k: v for k, v in d.metadata.items() if isinstance(v, (str, int, float, bool))}
            )

        written = 0
        for start in range(0, len(ids), MAX_BATCH_SIZE):
            end = start + MAX_BATCH_SIZE
            try:
                col.upsert(
                    ids=ids[start:end],
                    embeddings=embeddings[start:end],
                    documents=texts[start:end],
                    metadatas=metas[start:end],
                )
                written += end - start if end < len(ids) else len(ids) - start
            except Exception as exc:
                raise VectorStoreError(
                    f"向量写入失败（第 {start}~{min(end, len(ids))} 条）：{exc}\n"
                    "常见原因：向量维度与集合已有数据不一致（换过 EMBEDDING_PROVIDER 或 "
                    "EMBEDDING_DIM）。解决办法：删除 data/chroma 目录重建，或换一个集合名。"
                ) from exc

        logger.info("写入向量库：collection=%s count=%d", col.name, written)
        return written

    # ---------------- 检索 ----------------
    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        *,
        collection: str | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """向量相似度检索。

        Args:
            query_embedding: 查询向量。
            top_k: 返回条数。
            collection: 集合名。
            where: 元数据过滤条件，如 `{"doc_id": "abc123"}`。

        Returns:
            `[(Document, score)]`，score 为余弦相似度（0~1，越大越相关）。
            查询不到结果时返回空列表（**不抛异常**，让上层决定怎么提示）。

        Raises:
            VectorStoreError: 集合不存在等技术性错误。
        """
        col = self._get_collection(collection, create=False)
        if col.count() == 0:
            return []

        try:
            res = col.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, col.count()),
                where=where or None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"向量检索失败：{exc}") from exc

        docs: list[tuple[Document, float]] = []
        texts = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        for text, meta, dist in zip(texts, metas, dists):
            # 余弦距离 → 相似度。Chroma 的 cosine 距离 = 1 - cos，范围 [0, 2]
            score = 1.0 - float(dist)
            score = max(0.0, min(1.0, score))
            docs.append((Document(page_content=text, metadata=dict(meta or {})), score))

        return docs

    # ---------------- 读取与统计 ----------------
    def get_all_documents(
        self, collection: str | None = None, limit: int | None = None
    ) -> list[Document]:
        """取出集合内全部 chunk（BM25 索引构建需要）。

        Args:
            collection: 集合名。
            limit: 最多取多少条；None 表示全部。

        Returns:
            Document 列表。
        """
        try:
            col = self._get_collection(collection, create=False)
        except VectorStoreError:
            return []

        if col.count() == 0:
            return []

        res = col.get(include=["documents", "metadatas"], limit=limit)
        texts = res.get("documents") or []
        metas = res.get("metadatas") or []

        return [
            Document(page_content=t, metadata=dict(m or {}))
            for t, m in zip(texts, metas)
        ]

    def count(self, collection: str | None = None) -> int:
        """集合内 chunk 总数。"""
        try:
            return self._get_collection(collection, create=False).count()
        except Exception:
            return 0

    def list_documents(self, collection: str | None = None) -> list[dict[str, Any]]:
        """按文档聚合，返回知识库里的文档清单。

        Args:
            collection: 集合名。

        Returns:
            每项含 `doc_id / filename / chunks / source_type`。
        """
        docs = self.get_all_documents(collection)
        agg: dict[str, dict[str, Any]] = {}

        for d in docs:
            did = str(d.metadata.get("doc_id", "unknown"))
            entry = agg.setdefault(
                did,
                {
                    "doc_id": did,
                    "filename": str(d.metadata.get("source", "未知文件")),
                    "chunks": 0,
                    "source_type": str(d.metadata.get("source_type", "unknown")),
                },
            )
            entry["chunks"] += 1

        return sorted(agg.values(), key=lambda x: x["filename"])

    def delete_document(self, doc_id: str, collection: str | None = None) -> int:
        """删除一篇文档的所有 chunk。

        Args:
            doc_id: 文档 ID（内容哈希）。
            collection: 集合名。

        Returns:
            被删除的 chunk 数（0 表示没找到该文档）。

        Raises:
            VectorStoreError: 删除失败。
        """
        try:
            col = self._get_collection(collection, create=False)
        except VectorStoreError:
            return 0

        before = col.count()
        try:
            col.delete(where={"doc_id": doc_id})
        except Exception as exc:
            raise VectorStoreError(f"删除文档 {doc_id} 失败：{exc}") from exc

        removed = before - col.count()
        logger.info("删除文档 doc_id=%s 移除 chunk=%d", doc_id, removed)
        return removed

    def health(self, collection: str | None = None) -> tuple[bool, str]:
        """健康检查（供 /readyz 使用）。

        Returns:
            `(是否正常, 说明文字)`。
        """
        try:
            n = self.count(collection)
            return True, f"ok, {n} chunks"
        except Exception as exc:
            return False, str(exc)


# ============================================================
#  进程内单例
# ============================================================
_store: VectorStore | None = None
_store_lock = threading.Lock()


def get_vector_store(
    persist_dir: Path | str | None = None, collection: str | None = None, force_new: bool = False
) -> VectorStore:
    """获取向量库单例。

    为什么是单例：Chroma 的 PersistentClient 每次创建都会打开 SQLite 连接，
    在 FastAPI 里每个请求建一个会导致连接数爆炸（实测 50 并发就报
    `database is locked`）。

    Args:
        persist_dir: 覆盖默认落盘目录（测试用）。
        collection: 覆盖默认集合。
        force_new: True 时强制新建（测试里隔离用）。

    Returns:
        VectorStore 实例。
    """
    global _store
    if force_new or persist_dir is not None or collection is not None:
        return VectorStore(persist_dir, collection)

    with _store_lock:
        if _store is None:
            _store = VectorStore()
        return _store


def reset_vector_store_singleton() -> None:
    """清空单例（测试 teardown 用）。"""
    global _store
    with _store_lock:
        _store = None


def collection_summary(store: VectorStore, collection: str | None = None) -> dict[str, Any]:
    """统计集合的文档/类型分布（README 里放数据用）。"""
    docs = store.get_all_documents(collection)
    by_type = Counter(str(d.metadata.get("source_type", "unknown")) for d in docs)
    return {
        "total_chunks": len(docs),
        "by_type": dict(by_type),
        "documents": len({str(d.metadata.get("doc_id")) for d in docs}),
    }
