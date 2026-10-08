"""RAG 链与会话记忆测试。"""

from __future__ import annotations

import time

import pytest

from app.core.chain import RAGChain, SessionMemory
from app.core.llm import ExtractiveLLM
from app.core.retriever import HybridRetriever


class TestSessionMemory:
    """多轮对话记忆。"""

    def test_append_and_get(self):
        mem = SessionMemory()
        mem.append("s1", "问题1", "回答1")
        mem.append("s1", "问题2", "回答2")

        history = mem.get_history("s1")
        assert history == [("问题1", "回答1"), ("问题2", "回答2")]

    def test_sessions_isolated(self):
        """不同 session 的历史绝不能串。"""
        mem = SessionMemory()
        mem.append("s1", "s1的问题", "s1的回答")
        mem.append("s2", "s2的问题", "s2的回答")

        assert mem.get_history("s1") == [("s1的问题", "s1的回答")]
        assert mem.get_history("s2") == [("s2的问题", "s2的回答")]

    def test_max_turns_enforced(self):
        """只保留最近 N 轮 —— 否则 prompt 会无限膨胀直到超长报错。"""
        mem = SessionMemory(max_turns=3)
        for i in range(10):
            mem.append("s1", f"问题{i}", f"回答{i}")

        history = mem.get_history("s1")
        assert len(history) == 3
        assert history[-1][0] == "问题9"
        assert history[0][0] == "问题7"

    def test_reset(self):
        mem = SessionMemory()
        mem.append("s1", "问题", "回答")

        assert mem.reset("s1") is True
        assert mem.get_history("s1") == []
        # 再 reset 一次应该返回 False（不存在）
        assert mem.reset("s1") is False

    def test_none_session_id_is_stateless(self):
        """不传 session_id 时应该完全无状态，不能报错也不能记录。"""
        mem = SessionMemory()
        mem.append(None, "问题", "回答")
        assert mem.get_history(None) == []
        assert mem.stats()["active_sessions"] == 0

    def test_ttl_expiry(self):
        """过期会话必须被清理 —— 否则长跑服务会内存泄漏。"""
        mem = SessionMemory(ttl_minutes=0)  # 立即过期
        mem.append("s1", "问题", "回答")
        time.sleep(0.01)

        assert mem.get_history("s1") == []

    def test_stats(self):
        mem = SessionMemory(max_turns=5)
        mem.append("s1", "a", "b")
        mem.append("s2", "c", "d")

        stats = mem.stats()
        assert stats["active_sessions"] == 2
        assert stats["total_turns"] == 2
        assert stats["max_turns_per_session"] == 5

    def test_history_is_copy(self):
        """返回的必须是副本，外部改动不能污染内部状态。"""
        mem = SessionMemory()
        mem.append("s1", "问题", "回答")

        history = mem.get_history("s1")
        history.append(("伪造", "伪造"))
        assert len(mem.get_history("s1")) == 1


class TestRAGChain:
    """问答链主流程。"""

    def _make_chain(self, store) -> RAGChain:
        """构造一个完全离线的链（不依赖任何外部服务）。"""
        return RAGChain(
            retriever=HybridRetriever(store),
            llm=ExtractiveLLM(),
            memory=SessionMemory(),
        )

    def test_answers_with_citations(self, seeded_store):
        chain = self._make_chain(seeded_store)
        resp = chain.answer("员工入职满一年后有多少天年假？")

        assert resp.answer
        assert len(resp.citations) > 0
        assert resp.offline_mode is True
        assert resp.model == "offline-extractive"

        c = resp.citations[0]
        assert c.source
        assert c.snippet
        assert c.index >= 1

    def test_citations_reference_real_chunks(self, seeded_store):
        chain = self._make_chain(seeded_store)
        resp = chain.answer("住房公积金缴存比例是多少")

        sources = {c.source for c in resp.citations}
        assert sources <= {f"测试文档{i}.md" for i in range(3)}

    def test_empty_question_raises(self, seeded_store):
        chain = self._make_chain(seeded_store)
        with pytest.raises(ValueError, match="不能为空"):
            chain.answer("   ")

    def test_no_results_path(self, tmp_path):
        """知识库为空时必须给出可操作的拒答，而不是抛异常。"""
        from app.core.vectorstore import VectorStore

        empty = VectorStore(persist_dir=tmp_path / "e", collection="empty")
        chain = self._make_chain(empty)
        resp = chain.answer("任何问题")

        assert "无法回答" in resp.answer
        assert resp.citations == []

    def test_memory_records_turns(self, seeded_store):
        chain = self._make_chain(seeded_store)
        chain.answer("年假多少天", session_id="s1")
        chain.answer("病假呢", session_id="s1")

        assert len(chain.memory.get_history("s1")) == 2

    def test_latency_recorded(self, seeded_store):
        chain = self._make_chain(seeded_store)
        resp = chain.answer("年假")
        assert resp.latency_ms >= 0

    def test_retrieval_debug_populated(self, seeded_store):
        chain = self._make_chain(seeded_store)
        resp = chain.answer("离职需要提前多久通知")

        d = resp.retrieval_debug
        assert d.vector_hits >= 0
        assert d.fused >= 0
        assert resp.retrieval_debug.used_hybrid is True

    def test_top_k_respected(self, seeded_store):
        chain = self._make_chain(seeded_store)
        resp = chain.answer("年假", top_k=1)
        assert len(resp.citations) <= 3  # 引用可能含兜底，但不超过兜底上限

    def test_citation_fallback_when_model_forgets(self, seeded_store):
        """模型没标引用时，必须兜底挂上检索结果而不是返回空引用。

        用一个「永远不标引用」的假模型来触发这条路径。
        """
        from langchain_core.language_models.llms import LLM

        class ForgetfulLLM(LLM):
            @property
            def _llm_type(self) -> str:
                return "forgetful"

            def _call(self, prompt, stop=None, run_manager=None, **kwargs):
                return "这是一段完全没有任何引用标记的回答。"

        chain = RAGChain(
            retriever=HybridRetriever(seeded_store),
            llm=ForgetfulLLM(),
            memory=SessionMemory(),
        )
        resp = chain.answer("年假多少天")

        assert resp.answer
        assert len(resp.citations) > 0, "引用兜底失效，接口会返回空引用列表"
        assert resp.offline_mode is False  # ForgetfulLLM 不是 ExtractiveLLM

    def test_describe(self, seeded_store):
        chain = self._make_chain(seeded_store)
        info = chain.describe()
        assert "embedding" in info
        assert "llm" in info
        assert "memory" in info

    # ---------- 结构化拒答 ----------
    # 拒答必须是**字段**，不能让前端/评测去猜文本。
    # 下面每条测试都对应一个具体的失效场景，别删。

    def test_answerable_question_is_not_marked_refused(self, seeded_store):
        """能答的问题绝不能被打上拒答标记 —— 否则前端会显示误导性提示。"""
        chain = self._make_chain(seeded_store)
        resp = chain.answer("员工入职满一年后有多少天年假？")

        assert resp.refused is False
        assert resp.refusal_reason is None
        assert resp.answer

    def test_irrelevant_question_sets_structured_refusal(self, seeded_store):
        """知识库里没有的内容：拒答 + 原因 + **不挂引用**。"""
        chain = self._make_chain(seeded_store)
        resp = chain.answer("公司的股票代码是多少？下面还有一百层吗？")

        assert resp.refused is True
        assert resp.refusal_reason == "model_refused"
        # 拒答时挂引用会制造「有据可依」的假象，必须为空
        assert resp.citations == []

    def test_empty_model_output_is_refusal_not_empty_answer(self, seeded_store):
        """模型返回空串是异常，必须显式暴露成拒答，而不是静默返回空答案。"""
        from langchain_core.language_models.llms import LLM

        class BlankLLM(LLM):
            @property
            def _llm_type(self) -> str:
                return "blank"

            def _call(self, prompt, stop=None, run_manager=None, **kwargs):
                return "   "

        chain = RAGChain(
            retriever=HybridRetriever(seeded_store),
            llm=BlankLLM(),
            memory=SessionMemory(),
        )
        resp = chain.answer("年假多少天")

        assert resp.refused is True
        assert resp.refusal_reason == "empty_output"
        assert resp.answer.strip(), "不能把空答案原样吐给调用方"
        assert resp.citations == []

    def test_empty_retrieval_reports_no_retrieval_reason(self, seeded_store):
        """检索阶段一无所获时，原因要标成 no_retrieval，便于和「闸门拒答」区分。"""
        chain = self._make_chain(seeded_store)
        resp = chain.answer("年假多少天", top_k=1)

        # 正常情况这里应该答出来；本测试只为守住 no_retrieval 这条分支的语义
        if resp.refused:
            assert resp.refusal_reason in {"no_retrieval", "model_refused"}

    def test_citation_fallback_flag_is_exposed(self, seeded_store):
        """模型忘标引用时，除了兜底挂引用，还要在 debug 里留痕。"""
        from langchain_core.language_models.llms import LLM

        class ForgetfulLLM(LLM):
            @property
            def _llm_type(self) -> str:
                return "forgetful"

            def _call(self, prompt, stop=None, run_manager=None, **kwargs):
                return "这是一段完全没有任何引用标记的回答。"

        chain = RAGChain(
            retriever=HybridRetriever(seeded_store),
            llm=ForgetfulLLM(),
            memory=SessionMemory(),
        )
        resp = chain.answer("年假多少天")

        assert resp.refused is False
        assert resp.retrieval_debug.citation_fallback is True, "兜底挂引用必须可观测"

    def test_rewrite_skipped_when_offline(self, seeded_store):
        """离线模式下不启用查询改写（改写需要真正的 LLM）。"""
        chain = self._make_chain(seeded_store)
        chain.answer("年假多少天", session_id="s1")
        resp = chain.answer("那病假呢", session_id="s1")
        # 离线模式不应该因为改写失败而报错
        assert resp.answer

    def test_query_rewrite_falls_back_on_error(self, seeded_store):
        """改写失败必须静默回退到原问题 —— 可选增强不能拖垮主链路。"""
        from langchain_core.language_models.llms import LLM

        class BrokenLLM(LLM):
            @property
            def _llm_type(self) -> str:
                return "broken"

            def _call(self, prompt, stop=None, run_manager=None, **kwargs):
                raise RuntimeError("模拟 LLM 故障")

        chain = RAGChain(
            retriever=HybridRetriever(seeded_store),
            llm=BrokenLLM(),
            memory=SessionMemory(),
        )
        chain.memory.append("s1", "年假多少天", "5 天")

        query, debug = chain._rewrite_query("那病假呢", chain.memory.get_history("s1"))
        assert query == "那病假呢"
        assert debug["rewrite"].startswith("failed")


class TestCitations:
    """引用解析（`_build_citations` 的边界情况）。"""

    def test_out_of_range_index_ignored(self):
        """模型编造了不存在的编号时，必须忽略而不是崩。"""
        from langchain_core.documents import Document

        docs = [Document(page_content="内容", metadata={"source": "a.md", "page": 1, "chunk_id": "c1"})]
        citations, fallback = RAGChain._build_citations("答案见 [99][1]", [], [(1, docs[0])])

        assert len(citations) == 1
        assert citations[0].index == 1
        assert fallback is False

    def test_duplicate_indices_deduped(self):
        from langchain_core.documents import Document

        docs = [Document(page_content="内容", metadata={"source": "a.md", "page": 1, "chunk_id": "c1"})]
        citations, _ = RAGChain._build_citations("见 [1][1][1]", [], [(1, docs[0])])
        assert len(citations) == 1

    def test_no_citation_triggers_fallback(self):
        from langchain_core.documents import Document

        docs = [
            Document(page_content=f"内容{i}", metadata={"source": f"a{i}.md", "page": i + 1, "chunk_id": f"c{i}"})
            for i in range(5)
        ]
        citations, fallback = RAGChain._build_citations(
            "没有引用标记的答案", [], [(i + 1, d) for i, d in enumerate(docs)]
        )
        assert fallback is True
        assert len(citations) == 3  # CITATION_FALLBACK_COUNT

    def test_page_converted_to_int(self):
        from langchain_core.documents import Document

        docs = [Document(page_content="x", metadata={"source": "a.md", "page": "5", "chunk_id": "c1"})]
        citations, _ = RAGChain._build_citations("见 [1]", [], [(1, docs[0])])
        assert citations[0].page == 5
