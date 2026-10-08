"""文档切分（Chunking）。

## 为什么切分是 RAG 里最容易被忽视、却最影响效果的一环

- **切太大**：一个 chunk 里混了好几个主题，向量被「平均」掉，检索精度下降，
  而且塞给 LLM 的上下文里一半是无关内容（费 token、还容易带偏答案）。
- **切太小**：一句话被拆成两半，检索到了却「读不懂」，答案缺前提。
- **按固定长度硬切英文教程的做法**：`\n\n` 分不开中文段落，
  结果就是拿「字数」当刀，把「公司规定的年假为」/「5 天」切成两块。

## 本项目的中文切分策略

三级切分符优先级（从「语义强」到「语义弱」）：

1. **段落级**：`\n\n`、`\n`
2. **中文句级**：`。！？；` + 英文 `. ! ? ;`（注意中文标点必须显式列出，
   否则 `.` 匹配不到「。」）
3. **中文子句级**：`，、` 和空格

配合 `keep_separator=True` 保留标点（否则句号会被吃掉，
拼接回上下文时句子黏在一起），以及 `add_start_index=True`
记录每个 chunk 在原文中的偏移量（引用溯源时可以精确定位）。
"""

from __future__ import annotations

import hashlib
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.utils.logger import get_logger
from app.utils.text import jaccard, normalize_text, tokenize_for_bm25

logger = get_logger(__name__)

# 中文优先的切分符表。顺序 = 优先级，切分器会依次尝试。
# ⚠️ 中文标点必须显式写出：Python 的 str.split 不会把「，」当成「,」。
CHINESE_SEPARATORS: list[str] = [
    "\n\n",     # 段落
    "\n",       # 换行
    "。", "！", "？", "；",  # 中文句末
    ". ", "! ", "? ", "; ",  # 英文句末（带空格，避免切散小数 3.14）
    "，", "、",   # 中文逗号、顿号
    ", ",        # 英文逗号
    " ",
    "",         # 最后兜底：硬切
]


def _make_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """构造切分器（参数默认从全局配置读）。

    Args:
        chunk_size: 每块最大字符数。None 用配置值。
        chunk_overlap: 相邻块重叠字符数。None 用配置值。

    Returns:
        已配置好的 LangChain 切分器实例。
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size if chunk_size is not None else settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
        # length_function=len 表示按**字符**计长度。
        # 中文场景按字符更直观（一个汉字 ≈ 1 字符 ≈ 0.6 token），
        # 想按 token 精确控制可以换成 tiktoken 的 encoder，但要多一个依赖。
        length_function=len,
        separators=CHINESE_SEPARATORS,
        keep_separator=True,   # 保留标点，否则「。」被吃掉，语义会断
        add_start_index=True,  # 记录在原文中的偏移，引用溯源要用
        strip_whitespace=True,
    )


def _chunk_id(doc_id: str, index: int, text: str) -> str:
    """生成 chunk 的稳定唯一 ID。

    用「doc_id + 序号 + 内容」做哈希，保证：
    - 同一文档重复入库时 chunk_id 不变（幂等，不会产生重复向量）；
    - 不同文档的同名片段不会撞 ID。

    Args:
        doc_id: 所属文档 ID。
        index: chunk 在文档内的序号。
        text: chunk 文本。

    Returns:
        16 位十六进制 ID。
    """
    return hashlib.sha256(f"{doc_id}:{index}:{text}".encode()).hexdigest()[:16]


def _is_noise(text: str) -> bool:
    """判断 chunk 是否是噪声（页眉页脚、纯符号、目录省略号等）。

    规则：
    - 去掉空白后少于 15 个字符；
    - 中文字符 + 字母数字的占比低于 30%（说明符号/乱码为主）。

    Args:
        text: chunk 文本。

    Returns:
        True 表示噪声，应当丢弃。
    """
    stripped = re.sub(r"\s", "", text)
    if len(stripped) < 15:
        return True
    meaningful = len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", stripped))
    return meaningful / len(stripped) < 0.3


def _dedupe(chunks: list[Document], threshold: float = 0.9) -> list[Document]:
    """对切分结果做近似去重。

    为什么需要：PDF 的页眉页脚会在每一页重复，切分后就是一堆近似重复的 chunk。
    它们会占满 top_k 名额，把真正有用的片段挤掉。

    用 bigram token 集合的 Jaccard 相似度判断；复杂度 O(n²)，
    但单文档 chunk 数通常几十到几百，完全可接受。

    Args:
        chunks: 切分后的 Document 列表。
        threshold: 相似度超过它就算重复。

    Returns:
        去重后的列表（保留先出现的那个）。
    """
    kept: list[Document] = []
    token_sets: list[set[str]] = []

    for c in chunks:
        ts = set(tokenize_for_bm25(c.page_content))
        if any(jaccard(ts, prev) > threshold for prev in token_sets):
            continue
        kept.append(c)
        token_sets.append(ts)

    if len(kept) != len(chunks):
        logger.info("切分去重：%d → %d（移除 %d 个近似重复片段）",
                    len(chunks), len(kept), len(chunks) - len(kept))
    return kept


def split_documents(
    documents: list[Document],
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    dedupe: bool = True,
) -> list[Document]:
    """把加载好的 Document 切成适合检索的片段。

    对每个 chunk 会补充这些 metadata：
        - `chunk_id`    : 全局唯一 ID（向量库主键）
        - `chunk_index` : 在所属文档内的序号，从 0 开始（引用时展示「第 n 段」）
        - `char_count`  : 字符数（用于统计与调试）

    Args:
        documents: `loaders.load_document()` 的返回值。
        chunk_size: 覆盖配置里的切分长度。
        chunk_overlap: 覆盖配置里的重叠长度。
        dedupe: 是否做近似重复剔除。

    Returns:
        切分后的 Document 列表（已过滤噪声、已补全元数据）。

    Raises:
        ValueError: 输入为空列表。
    """
    if not documents:
        raise ValueError("split_documents 收到空列表：请先用 load_document() 加载文档。")

    splitter = _make_splitter(chunk_size, chunk_overlap)

    # 先按 doc_id 分组：保证 chunk_index 在文档内连续，
    # 否则一个文档被多页拆开时会得到一堆重复序号。
    grouped: dict[str, list[Document]] = {}
    for d in documents:
        grouped.setdefault(d.metadata.get("doc_id", "unknown"), []).append(d)

    result: list[Document] = []

    for doc_id, docs in grouped.items():
        # 先把同一文档的各页/各段文本拼起来再切。
        # 为什么要拼：一句话可能跨页（PDF 分页是排版行为，不是语义边界），
        # 单独切每一页会把跨页的句子切成两半。
        merged_parts: list[str] = []
        page_map: list[tuple[int, int, int]] = []  # (起始偏移, 结束偏移, 页码)

        cursor = 0
        for d in docs:
            page = int(d.metadata.get("page", 1))
            part = d.page_content.strip()
            if not part:
                continue
            if merged_parts:
                merged_parts.append("\n\n")
                cursor += 2
            start = cursor
            merged_parts.append(part)
            cursor += len(part)
            page_map.append((start, cursor, page))

        merged_text = "".join(merged_parts)
        if not merged_text.strip():
            continue

        raw_chunks = splitter.create_documents([merged_text])
        if not raw_chunks:
            continue

        # 后处理：过滤噪声 → 补元数据 → 回填页码
        cleaned: list[Document] = []
        for c in raw_chunks:
            text = normalize_text(c.page_content)
            if _is_noise(text):
                continue

            offset = int(c.metadata.get("start_index", 0))
            page = _locate_page(offset, page_map)

            meta = dict(docs[0].metadata)  # 继承 source / source_type / file_hash
            meta.update(
                {
                    "doc_id": doc_id,
                    "page": page,
                    "char_count": len(text),
                    "start_index": offset,
                }
            )
            cleaned.append(Document(page_content=text, metadata=meta))

        if dedupe:
            cleaned = _dedupe(cleaned)

        for i, c in enumerate(cleaned):
            c.metadata["chunk_index"] = i
            c.metadata["chunk_id"] = _chunk_id(doc_id, i, c.page_content)
            result.append(c)

    if not result:
        raise ValueError(
            "切分后没有任何有效片段。可能原因：文档太短、或内容全是符号/表格线。"
            f"（当前 CHUNK_SIZE={chunk_size or settings.chunk_size}）"
        )

    logger.info(
        "切分完成：输入 %d 个单元 → 输出 %d 个 chunk（平均 %d 字符）",
        len(documents), len(result),
        sum(len(c.page_content) for c in result) // max(len(result), 1),
    )
    return result


def _locate_page(offset: int, page_map: list[tuple[int, int, int]]) -> int:
    """根据 chunk 在合并文本中的偏移量，反查它属于哪一页。

    Args:
        offset: chunk 的 start_index。
        page_map: `[(起始偏移, 结束偏移, 页码)]` 列表。

    Returns:
        页码（从 1 开始）；找不到时返回最后一页。
    """
    for start, end, page in page_map:
        if start <= offset < end:
            return page
    return page_map[-1][2] if page_map else 1
