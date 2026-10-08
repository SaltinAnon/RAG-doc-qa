"""入库管道与加载器测试。"""

from __future__ import annotations

import pytest

from app.core.ingest import DocumentIngestor, IngestError
from app.core.loaders import (
    EmptyDocumentError,
    UnsupportedFileTypeError,
    get_supported_extensions,
    load_directory,
    load_document,
)


# ============================================================
#  加载器
# ============================================================
class TestLoadDocument:
    """各种格式的加载。"""

    def test_load_txt(self, tmp_path):
        f = tmp_path / "测试.txt"
        f.write_text("这是一段中文测试内容，用于验证文本加载器。" * 10, encoding="utf-8")

        docs = load_document(f)
        assert len(docs) == 1
        assert "中文测试内容" in docs[0].page_content
        assert docs[0].metadata["source"] == "测试.txt"
        assert docs[0].metadata["source_type"] == "txt"
        assert docs[0].metadata["doc_id"]

    def test_load_markdown_splits_by_heading(self, tmp_path):
        f = tmp_path / "手册.md"
        f.write_text(
            "# 第一章\n" + "内容甲。" * 20 + "\n\n## 第二章\n" + "内容乙。" * 20,
            encoding="utf-8",
        )
        docs = load_document(f)
        assert len(docs) >= 2, "Markdown 应该按标题切段（这样段号更贴合章节语义）"
        assert all("page" in d.metadata for d in docs)

    def test_load_gbk_encoded_file(self, tmp_path):
        """中文 Windows 上大量 TXT 是 GBK 编码，必须能自动探测。"""
        f = tmp_path / "gbk文本.txt"
        content = "这是用 GBK 编码保存的中文内容，用来测试编码自动探测。" * 8
        f.write_bytes(content.encode("gbk"))

        docs = load_document(f)
        assert "GBK 编码" in docs[0].page_content

    def test_load_docx_with_table(self, tmp_path):
        """Word 表格必须被提取 —— 企业文档的关键信息常在表格里。"""
        docx = pytest.importorskip("docx", reason="需要 python-docx")

        path = tmp_path / "制度.docx"
        document = docx.Document()
        document.add_paragraph("这是正文段落，包含一些说明文字。" * 3)
        table = document.add_table(rows=3, cols=2)
        table.rows[0].cells[0].text = "级别"
        table.rows[0].cells[1].text = "标准"
        table.rows[1].cells[0].text = "一线城市"
        table.rows[1].cells[1].text = "600 元"
        table.rows[2].cells[0].text = "其他城市"
        table.rows[2].cells[1].text = "350 元"
        document.save(str(path))

        docs = load_document(path)
        text = "\n".join(d.page_content for d in docs)
        assert "600 元" in text, "表格内容丢失了"
        assert "一线城市" in text
        assert docs[0].metadata["source_type"] == "docx"

    def test_unsupported_extension(self, tmp_path):
        f = tmp_path / "图片.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n")
        with pytest.raises(UnsupportedFileTypeError, match="不支持"):
            load_document(f)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_document(tmp_path / "不存在.txt")

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / "空.txt"
        f.write_text("   \n  \n", encoding="utf-8")
        with pytest.raises(EmptyDocumentError):
            load_document(f)

    def test_doc_id_stable(self, tmp_path):
        """同一内容两次加载必须得到相同 doc_id（幂等入库的基础）。"""
        content = "稳定的内容，用于验证 doc_id 的确定性。" * 10
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text(content, encoding="utf-8")
        f2.write_text(content, encoding="utf-8")

        assert load_document(f1)[0].metadata["doc_id"] == load_document(f2)[0].metadata["doc_id"]

    def test_source_type_recorded(self, tmp_path):
        f = tmp_path / "x.md"
        f.write_text("内容" * 50, encoding="utf-8")
        assert load_document(f)[0].metadata["source_type"] == "md"


class TestLoadDirectory:
    """批量加载。"""

    def test_loads_supported_only(self, tmp_path):
        (tmp_path / "a.md").write_text("内容甲。" * 30, encoding="utf-8")
        (tmp_path / "b.txt").write_text("内容乙。" * 30, encoding="utf-8")
        (tmp_path / "c.png").write_bytes(b"not a doc")

        docs, failures = load_directory(tmp_path)
        assert len(docs) >= 2
        assert failures == []

    def test_recursive(self, tmp_path):
        sub = tmp_path / "子目录"
        sub.mkdir()
        (sub / "deep.md").write_text("深层文件内容。" * 30, encoding="utf-8")

        docs, _ = load_directory(tmp_path, recursive=True)
        assert any(d.metadata["source"] == "deep.md" for d in docs)

    def test_non_recursive(self, tmp_path):
        sub = tmp_path / "子目录"
        sub.mkdir()
        (sub / "deep.md").write_text("深层内容。" * 30, encoding="utf-8")
        (tmp_path / "top.md").write_text("顶层内容。" * 30, encoding="utf-8")

        docs, _ = load_directory(tmp_path, recursive=False)
        sources = {d.metadata["source"] for d in docs}
        assert "top.md" in sources
        assert "deep.md" not in sources

    def test_failures_recorded_not_raised(self, tmp_path):
        """单个文件失败不能中断批量导入（skip_errors=True）。"""
        (tmp_path / "good.md").write_text("正常内容。" * 30, encoding="utf-8")
        (tmp_path / "bad.md").write_text("   ", encoding="utf-8")  # 空文件

        docs, failures = load_directory(tmp_path, skip_errors=True)
        assert len(failures) == 1
        assert failures[0][0] == "bad.md"
        assert docs

    def test_not_a_directory(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(NotADirectoryError):
            load_directory(f)


class TestSupportedExtensions:
    def test_core_formats_present(self):
        exts = get_supported_extensions()
        for e in (".pdf", ".docx", ".txt", ".md"):
            assert e in exts


# ============================================================
#  入库管道
# ============================================================
class TestDocumentIngestor:
    """端到端入库（真实向量库，离线 Embedding）。"""

    def test_ingest_text(self):
        ingestor = DocumentIngestor()
        result = ingestor.ingest_text(
            "员工入职满一年后每年享有 5 天带薪年假。" * 10, title="年假制度"
        )

        assert result.chunks > 0
        assert result.doc_id
        assert result.filename == "年假制度"

    def test_ingest_file(self, tmp_path):
        f = tmp_path / "制度.md"
        f.write_text("# 报销制度\n" + "差旅费用报销标准说明。" * 40, encoding="utf-8")

        ingestor = DocumentIngestor()
        result = ingestor.ingest_path(f)

        assert result.chunks > 0
        assert result.filename == "制度.md"

    def test_idempotent_reingest(self, tmp_path):
        """同一文件入库两次，chunk 数不能翻倍（幂等性）。

        这是企业场景的刚需：三个部门重复上传同一份《员工手册》，
        知识库里不能出现三份重复内容。
        """
        f = tmp_path / "手册.md"
        f.write_text("# 手册\n" + "公司规章制度说明内容。" * 40, encoding="utf-8")

        ingestor = DocumentIngestor()
        first = ingestor.ingest_path(f)
        count_after_first = ingestor.store.count()

        second = ingestor.ingest_path(f)
        count_after_second = ingestor.store.count()

        assert first.doc_id == second.doc_id
        assert count_after_first == count_after_second, "重复入库产生了重复向量"

    def test_skip_existing(self, tmp_path):
        f = tmp_path / "x.md"
        f.write_text("内容。" * 100, encoding="utf-8")

        ingestor = DocumentIngestor()
        ingestor.ingest_path(f)
        again = ingestor.ingest_path(f, skip_existing=True)

        assert again.chunks == 0, "skip_existing 应该直接跳过"

    def test_empty_text_raises(self):
        ingestor = DocumentIngestor()
        with pytest.raises(IngestError, match="为空"):
            ingestor.ingest_text("   \n  ")

    def test_list_documents(self, tmp_path):
        ingestor = DocumentIngestor()
        ingestor.ingest_text("文档一的内容。" * 50, title="文档一")
        ingestor.ingest_text("文档二的内容。" * 50, title="文档二")

        listing = ingestor.list_documents()
        assert listing.total_docs == 2
        assert listing.total_chunks > 0
        names = {d.filename for d in listing.documents}
        assert names == {"文档一", "文档二"}

    def test_delete_document(self, tmp_path):
        ingestor = DocumentIngestor()
        result = ingestor.ingest_text("待删除的文档内容。" * 50, title="待删")
        assert ingestor.store.count() > 0

        removed = ingestor.delete_document(result.doc_id)
        assert removed == result.chunks
        assert ingestor.store.count() == 0

    def test_ingest_directory(self, tmp_path):
        (tmp_path / "a.md").write_text("文档甲内容。" * 50, encoding="utf-8")
        (tmp_path / "b.md").write_text("文档乙内容。" * 50, encoding="utf-8")
        (tmp_path / "c.txt").write_text("文档丙内容。" * 50, encoding="utf-8")

        ingestor = DocumentIngestor()
        report = ingestor.ingest_directory(tmp_path)

        assert len(report["succeeded"]) == 3
        assert report["failed"] == []
        assert report["total_chunks"] > 0

    def test_ingest_directory_skip_second_run(self, tmp_path):
        (tmp_path / "a.md").write_text("文档内容。" * 50, encoding="utf-8")
        ingestor = DocumentIngestor()

        ingestor.ingest_directory(tmp_path, skip_existing=True)
        report = ingestor.ingest_directory(tmp_path, skip_existing=True)

        assert report["succeeded"] == []
