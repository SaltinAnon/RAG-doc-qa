"""检索器测试：RRF 融合、重排序、混合检索端到端。

这些测试保护的是「检索召回率」这个核心指标背后的算法正确性。
RRF 写错一个符号，召回率会掉一大截，而且**不会报错**，
所以必须有单元测试钉死。
"""

from __future__ import annotations

from langchain_core.documents import Document

from app.core.retriever import (
    HybridRetriever,
    RetrievedChunk,
    _BM25Cache,
)


def doc(text: str, cid: str, source: str = "测试.md") -> Document:
    """构造带 chunk_id 的测试文档。"""
    return Document(
        page_content=text,
        metadata={
            "chunk_id": cid,
            "doc_id": f"doc-{cid}",
            "source": source,
            "page": 1,
        },
    )


class TestRRFFusion:
    """RRF 融合算法。"""

    def test_single_source_each_side(self):
        v = [(doc("向量命中的内容", "a"), 0.9)]
        b = [(doc("BM25 命中的内容", "b"), 12.5)]
        fused = HybridRetriever._rrf_fuse(v, b, k=60)

        assert len(fused) == 2
        keys = {c.metadata["chunk_id"] for c in fused}
        assert keys == {"a", "b"}

    def test_both_ranks_accumulate(self):
        """两路都命中的文档，RRF 分必须高于只命中一路的文档。

        这就是 RRF 的核心价值：『双路共识』应该被奖励。
        """
        shared = doc("两路都命中", "shared")
        only_v = doc("只有向量命中", "only_v")
        only_b = doc("只有BM25命中", "only_b")

        v = [(shared, 0.9), (only_v, 0.8)]
        b = [(shared, 10.0), (only_b, 9.0)]
        fused = HybridRetriever._rrf_fuse(v, b, k=60)

        by_id = {c.metadata["chunk_id"]: c for c in fused}
        assert by_id["shared"].rrf_score > by_id["only_v"].rrf_score
        assert by_id["shared"].rrf_score > by_id["only_b"].rrf_score
        assert fused[0].metadata["chunk_id"] == "shared"

    def test_rank_first_beats_rank_second(self):
        """同一路里名次靠前的应该分更高。"""
        v = [(doc("第一条", "first"), 0.9), (doc("第二条", "second"), 0.5)]
        fused = HybridRetriever._rrf_fuse(v, [], k=60)
        assert fused[0].metadata["chunk_id"] == "first"

    def test_rrf_score_normalized(self):
        """RRF 分归一化到 0~1，方便展示和设阈值。"""
        v = [(doc("内容一", "a"), 0.9)]
        b = [(doc("内容一", "a"), 10.0)]
        fused = HybridRetriever._rrf_fuse(v, b, k=60)
        assert 0.0 <= fused[0].final_score <= 1.0
        # 两路都排第一 = 理论最大值
        assert fused[0].final_score > 0.99

    def test_k_parameter_effect(self):
        """k 越小，名次差异被放大得越明显。"""
        v = [(doc("第一", "a"), 0.9), (doc("第二", "b"), 0.5)]

        small_k = HybridRetriever._rrf_fuse(v, [], k=1)
        large_k = HybridRetriever._rrf_fuse(v, [], k=1000)

        ratio_small = small_k[0].rrf_score / small_k[1].rrf_score
        ratio_large = large_k[0].rrf_score / large_k[1].rrf_score
        assert ratio_small > ratio_large

    def test_different_chunk_ids_not_merged(self):
        """不同 chunk_id 不能因为文本相同就被合并。"""
        v = [(doc("一样的内容", "a"), 0.9)]
        b = [(doc("一样的内容", "b"), 10.0)]
        fused = HybridRetriever._rrf_fuse(v, b, k=60)
        assert len(fused) == 2

    def test_same_chunk_merged(self):
        """相同 chunk_id 必须被合并成一条（分数叠加）。"""
        v = [(doc("内容", "same"), 0.9)]
        b = [(doc("内容", "same"), 10.0)]
        fused = HybridRetriever._rrf_fuse(v, b, k=60)
        assert len(fused) == 1
        assert fused[0].vector_rank == 1
        assert fused[0].bm25_rank == 1

    def test_fallback_key_without_chunk_id(self):
        """没有 chunk_id 时要能用 (doc_id, page, 文本) 生成稳定 key。"""
        d1 = Document(page_content="无ID内容", metadata={"doc_id": "x", "page": 1})
        v = [(d1, 0.9)]
        b = [(d1, 5.0)]
        fused = HybridRetriever._rrf_fuse(v, b, k=60)
        assert len(fused) == 1


class TestSingleSourceWrapping:
    """只有一路有结果时的分数归一化。"""

    def test_normalized_to_range(self):
        hits = [(doc(f"内容{i}", f"c{i}"), 10.0 - i) for i in range(5)]
        wrapped = HybridRetriever._wrap_single_source(hits, "vector")
        assert len(wrapped) == 5
        assert all(0.0 <= c.final_score <= 1.0 for c in wrapped)
        # 分数降序
        scores = [c.final_score for c in wrapped]
        assert scores == sorted(scores, reverse=True)

    def test_identical_scores_no_div_zero(self):
        """所有分数相同时不能除零崩溃。"""
        hits = [(doc(f"内容{i}", f"c{i}"), 5.0) for i in range(3)]
        wrapped = HybridRetriever._wrap_single_source(hits, "bm25")
        assert all(0.0 <= c.final_score <= 1.0 for c in wrapped)

    def test_empty(self):
        assert HybridRetriever._wrap_single_source([], "vector") == []


class TestRerank:
    """启发式重排序。"""

    def test_keyword_coverage_matters(self):
        """查询词覆盖更多的片段应该排更前。"""
        relevant = RetrievedChunk(
            doc=doc("员工年假规定：入职满一年后每年享有 5 天带薪年假", "r"), final_score=0.5
        )
        irrelevant = RetrievedChunk(
            doc=doc("公司楼下咖啡厅今日供应美式与拿铁，第二杯半价", "i"), final_score=0.5
        )

        result = HybridRetriever._rerank("年假有几天", [irrelevant, relevant])
        assert result[0].metadata["chunk_id"] == "r"
        assert result[0].rerank_score > result[1].rerank_score

    def test_number_hit_boost(self):
        """问句里含数字时，含相同数字的片段应该加分。

        对应真实场景：「年假是 5 天还是 10 天？」
        """
        with_num = RetrievedChunk(doc=doc("年假为 5 天", "n"), final_score=0.5)
        without = RetrievedChunk(doc=doc("年假天数按照工龄确定", "w"), final_score=0.5)
        HybridRetriever._rerank("年假是 5 天吗", [with_num, without])
        assert with_num.rerank_score > without.rerank_score

    def test_length_score_curve(self):
        """长度打分曲线：甜区满分、极短重罚、超长缓降。"""
        ideal = 200

        assert HybridRetriever._length_score(10, ideal) == 0.20        # 极短碎片
        assert HybridRetriever._length_score(40, ideal) < 0.5          # 偏短
        assert HybridRetriever._length_score(100, ideal) == 1.0        # 甜区
        assert HybridRetriever._length_score(200, ideal) == 1.0        # 正好理想
        assert HybridRetriever._length_score(300, ideal) == 1.0        # 甜区上界
        assert HybridRetriever._length_score(2000, ideal) <= 0.4       # 超长衰减

    def test_length_score_monotonic_in_short_range(self):
        """短区间内长度越短惩罚越重。"""
        scores = [HybridRetriever._length_score(n, 200) for n in (10, 30, 60, 90, 120)]
        assert scores == sorted(scores), "长度分数必须单调递增"

    def test_short_chunk_penalized(self):
        """过短片段（无信息量的碎片）应该被惩罚。

        用同一句话造出「短」和「长」两个版本，长版本包含短版本的全部内容，
        这样覆盖率/短语/数字三项信号一致，差异只来自长度项。
        """
        sentence = "员工入职满一年后每年享有 5 天带薪年假。"
        short = RetrievedChunk(doc=doc(sentence, "s"), final_score=0.5)
        long = RetrievedChunk(doc=doc(sentence * 6, "l"), final_score=0.5)

        HybridRetriever._rerank("年假 5 天", [short, long])
        assert long.rerank_score > short.rerank_score

    def test_rerank_score_in_range(self):
        chunks = [
            RetrievedChunk(doc=doc(f"测试内容{i}" * 20, f"c{i}"), final_score=0.5)
            for i in range(5)
        ]
        HybridRetriever._rerank("测试内容", chunks)
        assert all(0.0 <= c.rerank_score <= 1.0 for c in chunks)

    def test_empty_input(self):
        assert HybridRetriever._rerank("查询", []) == []


class TestDedupe:
    """相似片段去重。"""

    def test_removes_near_duplicates(self):
        body = "公司的年假政策规定员工入职满一年后每年享有带薪年休假，具体天数按工龄计算。" * 3
        a = RetrievedChunk(doc=doc(body, "a"), final_score=0.9)
        b = RetrievedChunk(doc=doc(body, "b"), final_score=0.8)
        c = RetrievedChunk(doc=doc("完全不同的另一段内容，讲的是报销流程和发票要求。", "c"), final_score=0.7)

        kept = HybridRetriever._dedupe_similar([a, b, c])
        ids = [x.metadata["chunk_id"] for x in kept]
        assert "a" in ids
        assert "b" not in ids
        assert "c" in ids

    def test_keeps_dissimilar(self):
        chunks = [
            RetrievedChunk(doc=doc("年假规定内容。", "a"), final_score=0.9),
            RetrievedChunk(doc=doc("报销流程说明，涉及发票与审批。", "b"), final_score=0.8),
        ]
        assert len(HybridRetriever._dedupe_similar(chunks)) == 2


class TestBM25Cache:
    """BM25 索引缓存。"""

    def test_empty_corpus(self):
        cache = _BM25Cache()
        index, docs = cache.get("test", [])
        assert index is None
        assert docs == []

    def test_builds_and_caches(self):
        cache = _BM25Cache()
        corpus = [doc("年假政策说明内容", f"c{i}") for i in range(10)]

        index1, _ = cache.get("test", corpus)
        index2, _ = cache.get("test", corpus)
        assert index1 is not None
        assert index1 is index2, "相同语料必须复用缓存的索引"

    def test_rebuilds_on_count_change(self):
        cache = _BM25Cache()
        corpus = [doc("年假政策说明内容", f"c{i}") for i in range(10)]
        index1, _ = cache.get("test", corpus)

        corpus.append(doc("新增的报销制度内容", "new"))
        index2, _ = cache.get("test", corpus)
        assert index1 is not index2, "语料变化后必须重建索引"


class TestEndToEndRetrieval:
    """端到端检索（用真实向量库）。"""

    def test_hybrid_retrieval_finds_relevant(self, seeded_store):
        retriever = HybridRetriever(seeded_store)
        result = retriever.retrieve("年假有多少天", top_k=2)

        assert len(result.chunks) > 0
        top = result.chunks[0]
        assert "年假" in top.text
        assert result.debug["used_hybrid"] is True

    def test_scores_descending(self, seeded_store):
        retriever = HybridRetriever(seeded_store)
        result = retriever.retrieve("住房公积金比例", top_k=3)
        scores = [c.final_score for c in result.chunks]
        assert scores == sorted(scores, reverse=True)

    def test_empty_query_returns_empty(self, seeded_store):
        retriever = HybridRetriever(seeded_store)
        result = retriever.retrieve("   ", top_k=3)
        assert result.is_empty()

    def test_no_results_on_empty_store(self, tmp_path):
        from app.core.vectorstore import VectorStore

        empty = VectorStore(persist_dir=tmp_path / "empty", collection="empty_col")
        retriever = HybridRetriever(empty)
        result = retriever.retrieve("任何问题", top_k=3)
        assert result.is_empty()

    def test_debug_info_populated(self, seeded_store):
        retriever = HybridRetriever(seeded_store)
        result = retriever.retrieve("离职需要提前多久", top_k=2)
        for key in ("vector_hits", "bm25_hits", "fused", "reranked"):
            assert key in result.debug, f"调试信息缺少 {key}"
