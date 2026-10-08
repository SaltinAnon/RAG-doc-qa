"""文本工具测试。"""

from __future__ import annotations

from app.utils.text import (
    chinese_ratio,
    content_hash,
    idf_weights,
    jaccard,
    normalize_text,
    safe_filename,
    tokenize_for_bm25,
    truncate,
)


class TestNormalizeText:
    """清洗逻辑。"""

    def test_removes_zero_width_chars(self):
        # PDF 提取最常见的脏数据：零宽空格、BOM、软连字符
        raw = "年\u200b假\u200c\u200d规\ufeff定\u00ad"
        assert normalize_text(raw) == "年假规定"

    def test_merges_hyphen_break(self):
        # 英文 PDF 的断词
        assert normalize_text("docu-\nment") == "document"

    def test_compresses_blank_lines(self):
        assert normalize_text("a\n\n\n\n\nb") == "a\n\nb"

    def test_compresses_spaces(self):
        # 全角空格也要处理
        assert normalize_text("a　 　 b") == "a b"

    def test_collapses_lone_symbol_lines(self):
        """PDF 表格错位产生的单符号行应该被清掉。

        ⚠️ 但只清**符号**，不能清文字 —— 这条边界是本项目真实踩过的坑：
        第一版的规则是 `^\\s*\\S\\s*$`（任意单字符行），
        结果把正文里独立成行的单个字母/汉字也删掉了，
        表现为「清洗后文档内容凭空少了一部分」，非常难查。
        """
        assert "|" not in normalize_text("正常段落\n|\n另一段落")
        assert "·" not in normalize_text("正常段落\n·\n另一段落")

    def test_keeps_lone_alphanumeric_lines(self):
        """单个字母/汉字成行时必须保留（防止过度过滤）。"""
        assert "x" in normalize_text("正常段落\nx\n另一段落")

    def test_empty_input(self):
        assert normalize_text("") == ""

    def test_none_safe(self):
        # 虽然类型标注是 str，但实际调用方可能传 None，不该崩
        assert normalize_text(None) == ""  # type: ignore[arg-type]

    def test_nfkc_normalization(self):
        # 全角数字/字母转半角
        assert normalize_text("１２３ＡＢＣ") == "123ABC"


class TestContentHash:
    """哈希稳定性 —— 这个测试保护的是「幂等入库」这个核心特性。"""

    def test_deterministic_across_calls(self):
        # Python 内置 hash() 每进程加盐，重启后结果不同；
        # 我们的实现必须跨进程稳定，否则重复入库会失效
        assert content_hash("同样的内容") == content_hash("同样的内容")

    def test_different_content_different_hash(self):
        assert content_hash("内容A") != content_hash("内容B")

    def test_length(self):
        assert len(content_hash("任意内容")) == 16


class TestTokenizeForBm25:
    """BM25 分词（中文按字 + bigram）。"""

    def test_chinese_unigram_and_bigram(self):
        tokens = tokenize_for_bm25("年假")
        assert "年" in tokens
        assert "假" in tokens
        assert "年假" in tokens

    def test_english_word_kept_whole(self):
        tokens = tokenize_for_bm25("RAG system")
        assert "rag" in tokens
        assert "system" in tokens

    def test_mixed_content(self):
        tokens = tokenize_for_bm25("使用 RAG 系统查询 5 天年假")
        assert "rag" in tokens
        assert "年假" in tokens
        assert "5" in tokens

    def test_punctuation_dropped(self):
        tokens = tokenize_for_bm25("你好，世界！")
        assert "，" not in tokens
        assert "！" not in tokens

    def test_single_letter_english_dropped(self):
        # 单个字母噪声太大，去掉了能提升 BM25 质量
        assert "a" not in tokenize_for_bm25("a b c")

    def test_empty(self):
        assert tokenize_for_bm25("") == []


class TestSimilarity:
    """Jaccard 相似度。"""

    def test_identical(self):
        assert jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint(self):
        assert jaccard({"a"}, {"b"}) == 0.0

    def test_partial(self):
        assert abs(jaccard({"a", "b"}, {"a", "c"}) - 1 / 3) < 1e-9

    def test_empty_sets(self):
        assert jaccard(set(), {"a"}) == 0.0
        assert jaccard(set(), set()) == 0.0


class TestHelpers:
    """截断、中文字符占比、文件名安全。"""

    def test_truncate_short_text_unchanged(self):
        assert truncate("短文本", 10) == "短文本"

    def test_truncate_long_text(self):
        result = truncate("一二三四五六七八九十", 5)
        assert len(result) == 5
        assert result.endswith("…")

    def test_chinese_ratio_pure_chinese(self):
        assert chinese_ratio("全部中文") == 1.0

    def test_chinese_ratio_mixed(self):
        assert 0 < chinese_ratio("中文abc") < 1

    def test_chinese_ratio_empty(self):
        assert chinese_ratio("") == 0.0

    def test_safe_filename_strips_path_traversal(self):
        # 关键安全测试：防止 ../../etc/passwd 这类路径穿越
        assert safe_filename("../../../etc/passwd") == "passwd"
        assert "/" not in safe_filename("a/b/c.txt")
        assert "\\" not in safe_filename("a\\b\\c.txt")

    def test_safe_filename_keeps_chinese(self):
        assert safe_filename("员工手册.pdf") == "员工手册.pdf"

    def test_safe_filename_replaces_illegal_chars(self):
        assert "*" not in safe_filename('a*b?c"d.txt')

    def test_safe_filename_empty(self):
        assert safe_filename("") == "unnamed"


class TestIdfWeights:
    """IDF 加权 —— 修复「常见词污染」的关键，直接决定拒答准确率。

    背景：朴素覆盖率会让「公司」「员工」这类无处不在的词虚抬相关度，
    导致「知识库里没有答案」时系统硬答（幻觉）。
    """

    def test_absent_terms_get_max_weight(self):
        """没出现在任何候选片段里的词，区分度最高，应给最大权重。

        这类词恰恰说明「知识库里根本没有相关内容」——
        把它权重压低会让系统误以为资料够用。
        """
        corpus = [{"年假", "天"}, {"病假", "工资"}]
        w = idf_weights(corpus, {"年假", "股票"})

        assert w["股票"] > w["年假"]

    def test_common_term_gets_lower_weight_than_rare_term(self):
        """出现在全部片段里的词 → 权重最低；只出现在个别片段里的词 → 权重更高。"""
        corpus = [{"公司", "年假"}, {"公司", "病假"}, {"公司", "股票"}]
        w = idf_weights(corpus, {"公司", "股票", "年假"})

        assert w["公司"] < w["年假"]
        assert w["公司"] < w["股票"]

    def test_every_query_term_present_in_result(self):
        """返回的字典必须覆盖所有查询词，否则调用方 KeyError。"""
        w = idf_weights([{"a"}], {"a", "b", "c"})
        assert set(w) == {"a", "b", "c"}

    def test_empty_corpus_is_neutral(self):
        """语料为空时全部权重为 1.0（不放大也不缩小）。"""
        assert idf_weights([], {"x", "y"}) == {"x": 1.0, "y": 1.0}

    def test_empty_query_returns_empty(self):
        assert idf_weights([{"a"}], set()) == {}

    def test_weights_are_positive(self):
        """权重必须为正 —— 负数会让覆盖率出现诡异的抵消。"""
        w = idf_weights([{"a", "b"}, {"b"}], {"a", "b", "zzz"})
        assert all(v > 0 for v in w.values())

    def test_single_block_corpus_absent_term_heavier(self):
        """只有一个候选片段时，也应当满足「未命中的词更重」。

        这是最容易写错的边界：N=1 时若公式写错，未命中词的权重
        会反而低于命中词，导致拒答逻辑完全失效。
        """
        w = idf_weights([{"年假"}], {"年假", "股票"})
        assert w["股票"] > w["年假"]
