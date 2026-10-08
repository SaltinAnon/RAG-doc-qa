"""入库管道：把原始文件变成向量库里的 chunk。

串起 `loaders` → `splitter` → `embeddings` → `vectorstore` 四步，
是 RAG 的「写路径」。

## 幂等性设计（企业场景的刚需）

同一份 500 页的《员工手册》被 3 个部门重复上传，会怎样？

- 如果每份都新增一遍 → 知识库里 3 份重复内容，检索结果全是重复片段，
  top_k 名额直接被吃掉，**还会互相挤掉真正相关的其他文档**；
- 用户看到的引用是「员工手册.pdf 第 3 页」出现三次，观感极差。

解决：`doc_id = 全文内容哈希`。内容不变 → ID 不变 → `upsert` 覆盖。
内容改了 → ID 变了 → 会自动变成新文档（旧的需要显式删除）。

`skip_existing=True` 时还能在入库前先查一次，直接跳过，省掉向量化的钱。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from app.config import settings
from app.core.embeddings import get_embeddings
from app.core.loaders import load_directory, load_document
from app.core.splitter import split_documents
from app.core.vectorstore import VectorStore, get_vector_store
from app.models.schemas import DocumentListResponse, DocumentInfo, IngestResult
from app.utils.logger import get_logger
from app.utils.text import content_hash, safe_filename

logger = get_logger(__name__)


class IngestError(RuntimeError):
    """入库失败。"""


class DocumentIngestor:
    """文档入库服务。"""

    def __init__(self, store: VectorStore | None = None) -> None:
        """初始化。

        Args:
            store: 向量库；None 用全局单例。
        """
        self.store = store or get_vector_store()
        self.embeddings = get_embeddings()
        # 同一时刻只允许一个入库任务：Chroma 并发写同一集合会锁表
        self._write_lock = threading.Lock()

    # ---------- 单文件 ----------
    def ingest_path(
        self,
        path: Path | str,
        *,
        collection: str | None = None,
        skip_existing: bool = False,
    ) -> IngestResult:
        """把本地文件入库。

        Args:
            path: 文件路径。
            collection: 集合名。
            skip_existing: True 时若同内容文档已存在则跳过（省向量化开销）。

        Returns:
            入库结果。

        Raises:
            IngestError: 加载/切分/写入环节的任何失败。
        """
        started = time.perf_counter()
        path = Path(path)
        col = collection or self.store.collection_name

        try:
            docs = load_document(path)
        except Exception as exc:
            raise IngestError(f"加载文件「{path.name}」失败：{exc}") from exc

        doc_id = str(docs[0].metadata.get("doc_id", ""))
        full_chars = sum(len(d.page_content) for d in docs)

        if skip_existing and self._doc_exists(doc_id, col):
            logger.info("文档已存在，跳过：%s (doc_id=%s)", path.name, doc_id)
            return IngestResult(
                doc_id=doc_id,
                filename=safe_filename(path.name),
                pages=len(docs),
                chunks=0,
                chars=full_chars,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )

        chunks = self._split_or_raise(docs, path.name)
        written = self._embed_and_store(chunks, url=path.name, collection=col)

        elapsed = int((time.perf_counter() - started) * 1000)
        logger.info(
            "入库完成 file=%s pages=%d chunks=%d chars=%d 耗时=%dms",
            path.name, len(docs), written, full_chars, elapsed,
        )
        return IngestResult(
            doc_id=doc_id,
            filename=safe_filename(path.name),
            pages=len(docs),
            chunks=written,
            chars=full_chars,
            elapsed_ms=elapsed,
        )

    # ---------- 纯文本 ----------
    def ingest_text(
        self,
        text: str,
        *,
        title: str = "未命名文本",
        collection: str | None = None,
    ) -> IngestResult:
        """直接把一段文本入库（前端「粘贴文本」功能用）。

        Args:
            text: 正文。
            title: 文档标题（会作为引用里的 source 展示）。
            collection: 集合名。

        Returns:
            入库结果。

        Raises:
            IngestError: 文本为空或入库失败。
        """
        from app.utils.text import normalize_text

        started = time.perf_counter()
        cleaned = normalize_text(text)
        if not cleaned:
            raise IngestError("文本内容为空（清洗后没有任何有效字符）")

        col = collection or self.store.collection_name
        doc_id = content_hash(cleaned)

        doc = Document(
            page_content=cleaned,
            metadata={
                "doc_id": doc_id,
                "source": safe_filename(title),
                "source_type": "text",
                "page": 1,
                "file_hash": doc_id,
            },
        )
        chunks = self._split_or_raise([doc], title)
        written = self._embed_and_store(chunks, url=title, collection=col)

        return IngestResult(
            doc_id=doc_id,
            filename=safe_filename(title),
            pages=1,
            chunks=written,
            chars=len(cleaned),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    # ---------- 批量目录 ----------
    def ingest_directory(
        self, directory: Path | str, *, collection: str | None = None, skip_existing: bool = True
    ) -> dict[str, Any]:
        """批量入库目录下所有支持的文档。

        Args:
            directory: 目录路径。
            collection: 集合名。
            skip_existing: 已存在的文档是否跳过。

        Returns:
            `{ "succeeded": [...], "failed": [...], "total_chunks": int }`。
        """
        directory = Path(directory)
        col = collection or self.store.collection_name

        raw_docs, load_failures = load_directory(directory)
        failed: list[dict[str, str]] = [
            {"filename": name, "error": err} for name, err in load_failures
        ]

        # 按 doc_id 分组，逐文档处理（这样才能拿到每个文档的准确统计）
        grouped: dict[str, list[Document]] = {}
        for d in raw_docs:
            grouped.setdefault(str(d.metadata.get("doc_id")), []).append(d)

        succeeded: list[dict[str, Any]] = []
        total_chunks = 0

        for doc_id, docs in grouped.items():
            name = str(docs[0].metadata.get("source", "unknown"))
            if skip_existing and self._doc_exists(doc_id, col):
                logger.info("跳过已存在文档：%s", name)
                continue
            try:
                chunks = self._split_or_raise(docs, name)
                written = self._embed_and_store(chunks, url=name, collection=col)
                total_chunks += written
                succeeded.append({"filename": name, "doc_id": doc_id, "chunks": written})
            except Exception as exc:
                logger.error("入库失败 %s：%s", name, exc, exc_info=True)
                failed.append({"filename": name, "error": str(exc)})

        logger.info(
            "批量入库完成 dir=%s 成功=%d 失败=%d 总chunk=%d",
            directory, len(succeeded), len(failed), total_chunks,
        )
        return {"succeeded": succeeded, "failed": failed, "total_chunks": total_chunks}

    # ---------- 查询/删除 ----------
    def list_documents(self, collection: str | None = None) -> DocumentListResponse:
        """列出知识库中的所有文档。

        Args:
            collection: 集合名。

        Returns:
            文档清单（含总数统计）。
        """
        col = collection or self.store.collection_name
        items = self.store.list_documents(col)
        return DocumentListResponse(
            total_docs=len(items),
            total_chunks=self.store.count(col),
            collection=col,
            documents=[DocumentInfo(**it) for it in items],
        )

    def delete_document(self, doc_id: str, collection: str | None = None) -> int:
        """删除一篇文档。

        Args:
            doc_id: 文档 ID。
            collection: 集合名。

        Returns:
            删除的 chunk 数。
        """
        return self.store.delete_document(doc_id, collection)

    # ---------- 内部 ----------
    def _doc_exists(self, doc_id: str, collection: str | None) -> bool:
        """检查文档是否已在库中。"""
        if not doc_id:
            return False
        try:
            # 直接用一次 count 查询代替全量拉取，成本低得多
            col = self.store._get_collection(collection, create=False)  # noqa: SLF001
            res = col.get(where={"doc_id": doc_id}, limit=1, include=["metadatas"])
            return bool(res.get("ids"))
        except Exception:
            return False

    @staticmethod
    def _split_or_raise(docs: list[Document], name: str) -> list[Document]:
        """切分并包装异常，让报错能定位到具体文件。

        Args:
            docs: 加载结果。
            name: 文件名（用于报错信息）。

        Returns:
            切分后的 chunk 列表。

        Raises:
            IngestError: 切分失败或产出为空。
        """
        try:
            chunks = split_documents(docs)
        except Exception as exc:
            raise IngestError(
                f"切分文档「{name}」失败：{exc}\n"
                f"提示：文档是否过短？当前 CHUNK_SIZE={settings.chunk_size}，"
                "正文少于该长度时可能全部被判定为噪声。"
            ) from exc
        return chunks

    def _embed_and_store(
        self, chunks: list[Document], *, url: str, collection: str | None
    ) -> int:
        """向量化并写入向量库（带写锁）。

        Args:
            chunks: 待入库片段。
            url: 文件名/标题（仅用于日志）。
            collection: 集合名。

        Returns:
            写入条数。

        Raises:
            IngestError: 向量化或写入失败。
        """
        texts = [c.page_content for c in chunks]

        try:
            # 分批向量化：一次 1000 条既能跑满吞吐，又不会撑爆内存
            vectors: list[list[float]] = []
            batch = max(1, settings.embedding_batch_size)
            for i in range(0, len(texts), batch):
                vectors.extend(self.embeddings.embed_documents(texts[i : i + batch]))
        except Exception as exc:
            raise IngestError(
                f"向量化失败（{url}）：{exc}\n"
                "常见原因：\n"
                "  1. EMBEDDING_PROVIDER=openai 但 Key 无效/额度用尽；\n"
                "  2. 网络不通；\n"
                "  3. 文本过长超出模型限制（可调小 CHUNK_SIZE）。"
            ) from exc

        with self._write_lock:
            try:
                return self.store.add_documents(chunks, vectors, collection)
            except Exception as exc:
                raise IngestError(f"写入向量库失败（{url}）：{exc}") from exc


# ============================================================
#  单例
# ============================================================
_ingestor: DocumentIngestor | None = None
_ingestor_lock = threading.Lock()


def get_ingestor() -> DocumentIngestor:
    """获取入库服务单例。"""
    global _ingestor
    with _ingestor_lock:
        if _ingestor is None:
            _ingestor = DocumentIngestor()
        return _ingestor


def reset_ingestor_singleton() -> None:
    """清空单例（测试用）。"""
    global _ingestor
    with _ingestor_lock:
        _ingestor = None
