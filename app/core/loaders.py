"""文档加载：把 PDF / Word / TXT / Markdown 统一成 LangChain 的 `Document` 列表。

## 为什么自己写 loader，而不是直接用 `langchain_community.document_loaders`

LangChain 社区版确实提供了 `PyPDFLoader` / `Docx2txtLoader`，但它们有两个问题：

1. **额外重依赖**：`UnstructuredWordDocumentLoader` 会拖进 `unstructured` + `nltk`，
   光新词表就几十 MB，Docker 镜像直接胖一圈。
2. **元数据不可控**：我们需要**页码级引用溯源**（答案里要能标出「员工手册.pdf 第 3 页」），
   自己解析才能精确控制 page 从 1 开始、页眉页脚要不要留。

所以我们用 pypdf / python-docx 直接解析，**但输出的仍然是 LangChain 的 `Document`**，
下游的切分器、向量库、链全都按 LangChain 生态走。

> 面试时可以这么讲：「我评估过 community loader，因为元数据粒度和镜像体积的原因
> 自己实现了薄封装，但保持了 LangChain Document 抽象，所以生态工具链完全不受影响。」
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from langchain_core.documents import Document

from app.utils.logger import get_logger
from app.utils.text import content_hash, file_hash, normalize_text, safe_filename

logger = get_logger(__name__)

# 支持的文件类型 → 后缀集合
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".markdown", ".csv"}

# 单页/单段最少有效字符数：低于这个数视为噪声（空白页、纯页眉页脚）
_MIN_PAGE_CHARS = 10


class UnsupportedFileTypeError(ValueError):
    """文件类型不支持。"""


class EmptyDocumentError(ValueError):
    """文件解析成功但没有任何有效文本（可能是扫描版 PDF）。"""


# ============================================================
#  各类型解析器
# ============================================================
def _load_pdf(path: Path) -> list[Document]:
    """解析 PDF，**按页**产出 Document（页码从 1 开始）。

    注意：pypdf 只能提取文本层。如果是扫描件（图片型 PDF），
    提取结果会是空的 —— 这种情况必须明确报错，提示用户走 OCR，
    而不是返回一个空文档让后面默默失败。

    Args:
        path: PDF 文件路径。

    Returns:
        每页一个 Document。

    Raises:
        EmptyDocumentError: 全文提取不到文本（扫描件）。
    """
    from pypdf import PdfReader  # 延迟导入：不加载 PDF 时不付出这个开销

    reader = PdfReader(str(path))
    docs: list[Document] = []
    empty_pages = 0

    for i, page in enumerate(reader.pages):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:  # 单页损坏不应该毁掉整个文件
            logger.warning("PDF 第 %d 页解析失败，已跳过：%s", i + 1, exc)
            continue

        text = normalize_text(raw)
        if len(text) < _MIN_PAGE_CHARS:
            empty_pages += 1
            continue

        docs.append(
            Document(page_content=text, metadata={"page": i + 1, "total_pages": len(reader.pages)})
        )

    if not docs:
        raise EmptyDocumentError(
            f"PDF「{path.name}」提取不到任何文本（共 {len(reader.pages)} 页全部为空）。"
            "最常见原因是**扫描版 PDF**（内容是图片不是文字）。"
            "解决办法：先用 OCR 工具（如 ocrmypdf / 微信「提取文字」）转成文本型 PDF 再上传。"
        )

    if empty_pages:
        logger.info("PDF %s：跳过 %d 个空白/低信息页", path.name, empty_pages)

    return docs


def _load_docx(path: Path) -> list[Document]:
    """解析 Word 文档，**按段落累积成逻辑块**产出 Document。

    Word 没有「页」的概念（分页由渲染器决定），所以这里不伪造页码，
    改为记录段落序号，引用时展示为「段落 N」。

    同时会**抽取表格内容**（很多企业文档的关键信息在表格里，
    只读 paragraphs 会漏掉一半内容——这是踩过的坑）。

    Args:
        path: .docx 文件路径。

    Returns:
        整个文档一个 Document（长文档由下游切分器负责切）。
    """
    import docx  # python-docx

    document = docx.Document(str(path))
    parts: list[str] = []

    for para in document.paragraphs:
        t = normalize_text(para.text)
        if t:
            parts.append(t)

    # 表格：转成「表头: 值」的行文本，比 Markdown 表格更好被 LLM 理解
    table_count = 0
    for table in document.tables:
        table_count += 1
        rows: list[list[str]] = []
        for row in table.rows:
            cells = [normalize_text(c.text) for c in row.cells]
            # 去掉合并单元格造成的重复列
            deduped: list[str] = []
            for c in cells:
                if not deduped or deduped[-1] != c:
                    deduped.append(c)
            if any(deduped):
                rows.append(deduped)

        if not rows:
            continue

        header = rows[0]
        parts.append(f"\n【表格 {table_count}】")
        for r in rows[1:] if len(rows) > 1 else rows:
            pairs = [
                f"{header[i] if i < len(header) else f'列{i + 1}'}：{v}"
                for i, v in enumerate(r)
                if v
            ]
            if pairs:
                parts.append("；".join(pairs))
        # 只有一行时表头本身也是数据，补上
        if len(rows) == 1:
            parts.append("；".join(f"{header[i]}：{v}" for i, v in enumerate(rows[0]) if v))

    text = normalize_text("\n\n".join(parts))
    if not text:
        raise EmptyDocumentError(
            f"Word 文档「{path.name}」没有提取到文本。可能内容全部在文本框/图片里。"
        )
    return [Document(page_content=text, metadata={"paragraphs": len(parts)})]


def _load_text_file(path: Path) -> list[Document]:
    """解析纯文本 / Markdown / CSV。

    按空行分隔的「段」切分成多个 Document，这样页码语义可以近似为「段号」。

    Args:
        path: 文本文件路径。

    Returns:
        每段一个 Document。
    """
    raw = None
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030", "latin-1"):
        try:
            raw = path.read_text(encoding=enc)
            if enc not in ("utf-8", "utf-8-sig"):
                logger.info("文件 %s 以 %s 编码读取成功（非 UTF-8）", path.name, enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if raw is None:
        raise EmptyDocumentError(f"无法解码文件「{path.name}」，请另存为 UTF-8 编码后重试。")

    text = normalize_text(raw)
    if not text:
        raise EmptyDocumentError(f"文件「{path.name}」内容为空。")

    # 按 Markdown 标题切段，让「段号」更贴合人类理解的章节
    import re

    segments = re.split(r"\n(?=#{1,6}\s)", text)
    docs: list[Document] = []
    for i, seg in enumerate(segments):
        seg = seg.strip()
        if seg:
            docs.append(Document(page_content=seg, metadata={"page": i + 1}))
    return docs or [Document(page_content=text, metadata={"page": 1})]


# 后缀 → 解析器 的注册表（新增类型只要加一行）
_PARSERS: dict[str, Callable[[Path], list[Document]]] = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".doc": _load_docx,  # .doc 需要先另存为 .docx，见下方提示
    ".txt": _load_text_file,
    ".md": _load_text_file,
    ".markdown": _load_text_file,
    ".csv": _load_text_file,
}


# ============================================================
#  对外统一入口
# ============================================================
def load_document(path: Path | str) -> list[Document]:
    """加载单个文件，返回带完整元数据的 Document 列表。

    每个 Document 的 metadata 都包含：
        - `doc_id`      : 内容哈希，同一文件重复上传会得到相同 ID（天然去重）
        - `source`      : 原始文件名
        - `source_type` : pdf / docx / txt / md
        - `page`        : 页码或段号（从 1 开始）
        - `file_hash`   : 文件字节哈希（用于判断文件是否被改过）

    Args:
        path: 文件路径。

    Returns:
        Document 列表；调用方负责切分。

    Raises:
        FileNotFoundError: 文件不存在。
        UnsupportedFileTypeError: 后缀不在支持列表里。
        EmptyDocumentError: 解析成功但没有有效文本。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")

    ext = path.suffix.lower()
    parser = _PARSERS.get(ext)
    if parser is None:
        raise UnsupportedFileTypeError(
            f"不支持的文件类型「{ext}」。当前支持：{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if ext == ".doc":
        # python-docx 读不了老的二进制 .doc 格式，提前给出可操作的建议
        with open(path, "rb") as f:
            if f.read(2) != b"PK":  # .docx 是 zip（PK 开头）
                raise UnsupportedFileTypeError(
                    f"「{path.name}」是旧版 .doc 二进制格式，无法直接解析。"
                    "请用 Word / WPS 打开后「另存为 → .docx」再上传。"
                )

    docs = parser(path)

    # 统一补全元数据
    fh = file_hash(path)
    full_text = "\n".join(d.page_content for d in docs)
    doc_id = content_hash(full_text)
    source = safe_filename(path.name)

    for d in docs:
        d.metadata.update(
            {
                "doc_id": doc_id,
                "source": source,
                "source_type": ext.lstrip("."),
                "file_hash": fh,
            }
        )
        d.metadata.setdefault("page", 1)

    logger.info(
        "加载完成 file=%s type=%s units=%d chars=%d doc_id=%s",
        source, ext, len(docs), len(full_text), doc_id,
    )
    return docs


def load_directory(
    directory: Path | str, *, recursive: bool = True, skip_errors: bool = True
) -> tuple[list[Document], list[tuple[str, str]]]:
    """批量加载目录下的所有受支持文档。

    Args:
        directory: 目录路径。
        recursive: 是否递归子目录。
        skip_errors: True 时单个文件失败只记录不中断（批量入库场景更实用）。

    Returns:
        `(documents, failures)`，其中 failures 是 `[(文件名, 错误信息)]`。
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"不是目录：{directory}")

    pattern = "**/*" if recursive else "*"
    files = sorted(p for p in directory.glob(pattern) if p.is_file() and p.suffix.lower() in _PARSERS)

    all_docs: list[Document] = []
    failures: list[tuple[str, str]] = []

    for f in files:
        try:
            all_docs.extend(load_document(f))
        except Exception as exc:
            if not skip_errors:
                raise
            logger.warning("跳过 %s：%s", f.name, exc)
            failures.append((f.name, str(exc)))

    logger.info("目录扫描完成 dir=%s 命中=%d 成功=%d 失败=%d",
                directory, len(files), len(files) - len(failures), len(failures))
    return all_docs, failures


def get_supported_extensions() -> list[str]:
    """返回支持的文件后缀（前端上传组件用它做过滤）。"""
    return sorted(SUPPORTED_EXTENSIONS)
