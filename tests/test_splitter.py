"""切分器测试。"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.core.splitter import CHINESE_SEPARATORS, _is_noise, split_documents


def make_doc(text: str, page: int = 1, doc_id: str = "doc1", source: str = "测试.md") -> Document:
    """构造一个带完整元数据的测试 Document。"""
    return Document(
        page_content=text,
        metadata={
            "doc_id": doc_id,
            "source": source,
            "source_type": "md",
            "page": page,
            "file_hash": "abc",
        },
    )


class TestChineseSeparators:
    """切分符表本身的正确性。"""

    def test_chinese_punctuation_present(self):
        # 这是最容易翻车的地方：照抄英文教程会漏掉中文标点
        for p in ["。", "！", "？", "；", "，", "、"]:
            assert p in CHINESE_SEPARATORS, f"缺少中文标点「{p}」"

    def test_priority_order(self):
        # 段落必须在句子之前，句子必须在逗号之前
        assert CHINESE_SEPARATORS.index("\n\n") < CHINESE_SEPARATORS.index("。")
        assert CHINESE_SEPARATORS.index("。") < CHINESE_SEPARATORS.index("，")

    def test_empty_string_is_last(self):
        # "" 必须是兜底项，否则超长无标点文本会被整块留下
        assert CHINESE_SEPARATORS[-1] == ""


class TestSplitDocuments:
    """主流程。"""

    def test_produces_multiple_chunks(self):
        text = "。".join(f"这是第{i}句测试内容，用于验证切分逻辑是否正常工作" for i in range(40))
        chunks = split_documents([make_doc(text)])
        assert len(chunks) > 1

    def test_respects_chunk_size(self):
        from app.config import settings

        text = "。".join(f"第{i}句" + "内容" * 20 for i in range(30))
        chunks = split_documents([make_doc(text)])
        # 允许少量超出（切分器会尽量在分隔符处断开，不做硬截断）
        assert all(len(c.page_content) <= settings.chunk_size * 1.5 for c in chunks)

    def test_metadata_completed(self):
        text = "。".join(f"第{i}句中包含足够多的文字以便通过噪声过滤" for i in range(20))
        chunks = split_documents([make_doc(text)])

        for i, c in enumerate(chunks):
            assert c.metadata["chunk_id"], "缺少 chunk_id（向量库主键）"
            assert c.metadata["chunk_index"] == i, "chunk_index 必须连续"
            assert c.metadata["doc_id"] == "doc1"
            assert c.metadata["source"] == "测试.md"
            assert "char_count" in c.metadata
            assert "start_index" in c.metadata

    def test_chunk_id_is_stable(self):
        """同一份文档切两次，chunk_id 必须完全一致（保证幂等入库）。"""
        text = "。".join(f"第{i}句测试文本内容足够长以通过过滤" for i in range(20))
        a = split_documents([make_doc(text)])
        b = split_documents([make_doc(text)])
        assert [c.metadata["chunk_id"] for c in a] == [c.metadata["chunk_id"] for c in b]

    def test_page_number_tracked(self):
        """跨页文档的 chunk 必须能定位到正确的页码（引用溯源的基础）。"""
        docs = [
            make_doc("第一页的内容。" * 30, page=1),
            make_doc("第二页的内容。" * 30, page=2),
            make_doc("第三页的内容。" * 30, page=3),
        ]
        chunks = split_documents(docs)
        pages = {c.metadata["page"] for c in chunks}
        assert pages <= {1, 2, 3}
        assert len(pages) >= 2, "页码没有正确区分，引用溯源会失效"

    def test_multiple_documents_isolated(self):
        """两个文档的 chunk_index 各自从 0 开始，不能混在一起编号。"""
        docs = [
            make_doc("文档A的内容。" * 30, doc_id="A", source="A.md"),
            make_doc("文档B的内容。" * 30, doc_id="B", source="B.md"),
        ]
        chunks = split_documents(docs)
        a_idx = [c.metadata["chunk_index"] for c in chunks if c.metadata["doc_id"] == "A"]
        b_idx = [c.metadata["chunk_index"] for c in chunks if c.metadata["doc_id"] == "B"]
        assert a_idx == list(range(len(a_idx)))
        assert b_idx == list(range(len(b_idx)))

    def test_noise_filtered(self):
        """纯符号块必须被过滤掉，不能产生「只有分割线」的 chunk。

        注意断言写法：不能断言「结果里不含 -」，因为切分器会把正常文本和
        下一页的分割线合并到同一个 chunk 里（这是合理的）。
        要断言的是「没有任何一个 chunk 本身是噪声」。
        """
        text = (
            "正常的一段内容，包含足够多的有效文字用于通过噪声过滤检查。"
            + "\n\n"
            + "-" * 400  # 模拟表格线/分割线，长度超过 chunk_size 一定会产生独立块
        )
        chunks = split_documents([make_doc(text)])

        assert chunks, "正常内容不应该被全部过滤掉"
        for c in chunks:
            assert not _is_noise(c.page_content), f"产生了纯噪声 chunk：{c.page_content[:40]!r}"

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="空列表"):
            split_documents([])

    def test_all_noise_raises(self):
        # 全是符号，切完没有任何有效片段，必须明确报错而不是返回空列表
        with pytest.raises(ValueError):
            split_documents([make_doc("！！！？？？。。。")])

    def test_dedupe_removes_repeats(self):
        """页眉页脚导致的重复 chunk 必须被去掉。

        构造方式：同一段文字重复 10 次（模拟每页都有的页眉）。
        """
        body = "星辰科技员工手册内部资料，请勿外传，本文档受公司保密制度保护。" * 3
        text = "\n\n".join([body] * 10)
        chunks = split_documents([make_doc(text)], dedupe=True)
        # 去重后应该只剩极少数（理想是 1~2 个）
        assert len(chunks) <= 3

    def test_dedupe_can_be_disabled(self):
        body = "星辰科技员工手册内部资料，请勿外传，本文档受公司保密制度保护。" * 3
        text = "\n\n".join([body] * 10)
        chunks = split_documents([make_doc(text)], dedupe=False)
        assert len(chunks) >= 3


class TestNoiseDetection:
    """单条噪声判定规则。"""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("。", True),                      # 太短
            ("第 3 页", True),                  # 页眉，字符太少
            ("---------", True),               # 纯符号
            (".....", True),                   # 纯符号
            ("这是一段有足够长度的正常中文内容", False),
            ("Annual leave policy is 5 days for all employees", False),
        ],
    )
    def test_cases(self, text, expected):
        assert _is_noise(text) is expected

    def test_symbol_heavy_is_noise(self):
        assert _is_noise("a" + "!@#$%^&*()" * 5) is True
