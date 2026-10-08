"""离线抽取式 LLM 与 Prompt 模板测试。

这个模块是项目「零成本可跑」承诺的技术底座，必须有测试保护。
"""

from __future__ import annotations

from langchain_core.documents import Document

from app.core.llm import OFFLINE_ANSWER_PREFIX, ExtractiveLLM, describe_llm, is_offline
from app.core.prompts import (
    NO_RESULT_ANSWER,
    REFUSAL_MARKERS,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    format_context,
    format_history,
    is_refusal_text,
)


def build_prompt(question: str, docs: list[Document], history: str = "（无）") -> str:
    """按生产链路的同样方式组装 Prompt。"""
    context, _ = format_context(docs)
    return (
        SYSTEM_PROMPT
        + "\n\n"
        + USER_TEMPLATE.format(context=context, history=history, question=question)
    )


class TestFormatContext:
    """参考资料编号格式化 —— 引用绑定准确率的基础。"""

    def test_numbering_starts_at_one(self):
        docs = [Document(page_content=f"内容{i}", metadata={"source": f"f{i}.md", "page": i + 1})
                for i in range(3)]
        text, indexed = format_context(docs)

        assert "[1]" in text
        assert "[2]" in text
        assert "[3]" in text
        assert [i for i, _ in indexed] == [1, 2, 3]

    def test_page_included(self):
        docs = [Document(page_content="内容", metadata={"source": "手册.pdf", "page": 3})]
        text, _ = format_context(docs)
        assert "手册.pdf" in text
        assert "第 3 页" in text

    def test_index_matches_document_order(self):
        """编号必须与传入顺序严格对应 —— 错位会导致引用张冠李戴。"""
        docs = [Document(page_content=t, metadata={"source": f"{t}.md", "page": 1})
                for t in ("甲", "乙", "丙")]
        _, indexed = format_context(docs)
        for expected_idx, (idx, d) in enumerate(indexed, start=1):
            assert idx == expected_idx
            assert d.metadata["source"] == f"{docs[expected_idx - 1].page_content}.md"

    def test_empty_documents(self):
        text, indexed = format_context([])
        assert indexed == []
        assert "没有检索到" in text


class TestFormatHistory:
    """对话历史格式化。"""

    def test_no_history_placeholder(self):
        assert "无历史对话" in format_history([])

    def test_formats_turns(self):
        history = [("年假多少天", "5 天[1]"), ("病假呢", "80% 工资[2]")]
        text = format_history(history)
        assert "年假多少天" in text
        assert "5 天[1]" in text
        assert "病假呢" in text

    def test_truncates_long_answers(self):
        history = [("问题", "很长很长的回答" * 100)]
        text = format_history(history)
        assert len(text) < 400
        assert "…" in text

    def test_keeps_only_recent_turns(self):
        history = [(f"问题{i}", f"回答{i}") for i in range(20)]
        text = format_history(history, max_turns=3)
        assert "问题19" in text
        assert "问题0" not in text


class TestExtractiveLLM:
    """离线抽取实现。"""

    def test_extracts_relevant_sentence(self):
        docs = [
            Document(
                page_content="员工入职满一年后开始享受带薪年休假。工龄一年以上不满十年的，"
                             "每年享有 5 天年假。工龄十年以上不满二十年的，每年享有 10 天年假。",
                metadata={"source": "员工手册.md", "page": 8},
            ),
            Document(
                page_content="公司为全体正式员工缴纳五险一金，其中住房公积金缴存比例为 12%。",
                metadata={"source": "员工手册.md", "page": 12},
            ),
        ]
        llm = ExtractiveLLM()
        answer = llm.invoke(build_prompt("员工入职满一年后有多少天年假？", docs))

        assert OFFLINE_ANSWER_PREFIX in answer
        assert "5 天" in answer
        assert "[1]" in answer

    def test_citation_points_to_correct_block(self):
        """抽出来的句子必须标注它来自哪一号资料。"""
        docs = [
            Document(page_content="完全无关的内容，讲的是公司楼下的咖啡厅。", metadata={"source": "a.md", "page": 1}),
            Document(page_content="住房公积金缴存比例为 12%。", metadata={"source": "b.md", "page": 2}),
        ]
        llm = ExtractiveLLM()
        answer = llm.invoke(build_prompt("住房公积金比例是多少", docs))
        assert "[2]" in answer
        assert "12%" in answer

    def test_no_context_returns_refusal(self):
        llm = ExtractiveLLM()
        answer = llm.invoke(build_prompt("任意问题", []))
        assert answer  # 不能返回空串
        assert "无法回答" in answer

    def test_q_type_returns_answer_not_refusal(self):
        """普通问题（无资料）也必须有明确输出，不能是 None 或空。

        这是「绝不静默失败」原则的体现：宁可给拒答话术，也不能返回空串
        让前端显示一片空白。
        """
        llm = ExtractiveLLM()
        answer = llm.invoke("请直接回答问题，没有任何参考资料。")
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_sentence_limit_respected(self):
        """最多抽取 top_sentences 条，不能把整篇文档吐出来。"""
        long_text = "。".join(f"这是第{i}个句子，包含年假相关的说明内容" for i in range(30))
        docs = [Document(page_content=long_text, metadata={"source": "长文档.md", "page": 1})]

        llm = ExtractiveLLM(top_sentences=2)
        answer = llm.invoke(build_prompt("年假说明", docs))
        # 减去前缀行，正文行数不超过配置值
        body_lines = [ln for ln in answer.split("\n") if ln.startswith("- ")]
        assert len(body_lines) <= 2

    def test_answer_length_bounded(self):
        long_text = "这是一段很长的内容用来测试答案长度限制。" * 200
        docs = [Document(page_content=long_text, metadata={"source": "a.md", "page": 1})]
        llm = ExtractiveLLM(max_answer_chars=300)
        answer = llm.invoke(build_prompt("测试长度限制", docs))
        assert len(answer) <= 400  # 300 + 前缀余量

    def test_handles_malformed_prompt(self):
        """有人直接把任意字符串丢进来时不能崩。"""
        llm = ExtractiveLLM()
        answer = llm.invoke("这是一段完全没有按模板格式化的文本")
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_llm_type(self):
        assert ExtractiveLLM()._llm_type == "offline-extractive"

    def test_lcel_chain_compatible(self):
        """必须能被 LCEL 的 `prompt | llm | parser` 使用。

        这是架构上的关键约束：在线/离线两条路径共用同一套编排代码。
        """
        from langchain_core.output_parsers import StrOutputParser

        docs = [Document(page_content="年假为 5 天。", metadata={"source": "a.md", "page": 1})]
        chain = ExtractiveLLM() | StrOutputParser()
        result = chain.invoke(build_prompt("年假多少天", docs))
        assert isinstance(result, str)
        assert "5 天" in result


class TestOfflineDetection:
    """降级状态探测。"""

    def test_is_offline_for_extractive(self):
        assert is_offline(ExtractiveLLM()) is True

    def test_describe(self):
        assert describe_llm(ExtractiveLLM()) == "offline-extractive"

    def test_is_offline_without_instance_uses_settings(self, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "llm_provider", "offline")
        assert is_offline() is True

        monkeypatch.setattr(settings, "llm_provider", "openai")
        monkeypatch.setattr(settings, "llm_api_key", "sk-fake")
        assert is_offline() is False

        # 选了在线但没 Key，仍然算离线（因为工厂会降级）
        monkeypatch.setattr(settings, "llm_api_key", "")
        assert is_offline() is True


class TestGetLLM:
    """工厂降级。"""

    def test_offline_provider_returns_extractive(self, monkeypatch):
        from app.config import settings
        from app.core.llm import get_llm, reset_llm_singleton

        monkeypatch.setattr(settings, "llm_provider", "offline")
        reset_llm_singleton()

        llm = get_llm()
        assert isinstance(llm, ExtractiveLLM)
        reset_llm_singleton()

    def test_online_without_key_falls_back(self, monkeypatch):
        """选了 openai 但没 Key —— 必须降级而不是抛异常。

        这条测试保护的是「别人 clone 下来不带 Key 也能跑通问答接口」这个卖点。
        """
        from app.config import settings
        from app.core.llm import get_llm, reset_llm_singleton

        monkeypatch.setattr(settings, "llm_provider", "openai")
        monkeypatch.setattr(settings, "llm_api_key", "")
        reset_llm_singleton()

        llm = get_llm()
        assert isinstance(llm, ExtractiveLLM)
        reset_llm_singleton()

    def test_prompt_template_variables(self):
        """Prompt 模板变量必须与链传入的 key 一致。

        这是最容易出错的地方：模板里写 {question}，链里传 {"query": ...}，
        运行时才报 KeyError，而且小模型可能静默忽略。
        """
        import re

        vars_in_template = set(re.findall(r"\{(\w+)\}", USER_TEMPLATE + SYSTEM_PROMPT))
        assert {"context", "history", "question"} <= vars_in_template


class TestRefusalGate:
    """拒答闸门 —— 离线模式抑制幻觉的最后一道防线。

    设计要点（详见 ExtractiveLLM 类文档字符串）：
    「孤证不算证据」—— 只命中**一个**查询词的句子不被采信，
    除非它的相关度非常高。因为实测发现所有「答不出来」的问题，
    最高分句子都恰好只命中一个巧合的词（如问「股票代码」命中「代码仓库」）。
    """

    def test_refuses_on_lone_coincidental_match(self):
        """只命中一个巧合词 → 必须拒答，而不是硬抽句子。

        场景：问「股票代码」，知识库里只有「代码仓库禁止提交密钥」。
        「代码」字面命中，但语义完全无关 —— 纯词面阈值拦不住，
        靠「只命中 1 个词」这个结构性信号拦住。
        """
        docs = [
            Document(
                page_content="公司为全体正式员工缴纳五险一金，其中住房公积金缴存比例为 12%。"
                             "代码仓库禁止提交明文密钥、证书或生产环境配置。",
                metadata={"source": "员工手册.md", "page": 12},
            ),
            Document(
                page_content="员工入职满一年后开始享受带薪年休假，每年 5 天。",
                metadata={"source": "员工手册.md", "page": 8},
            ),
            Document(
                page_content="平台采用微服务架构，按业务域拆分为八个核心服务。",
                metadata={"source": "技术架构说明.md", "page": 1},
            ),
        ]
        llm = ExtractiveLLM()
        answer = llm.invoke(build_prompt("公司的股票代码是多少？", docs))

        assert "无法回答" in answer, f"应当拒答，实际：{answer}"

    def test_answers_when_multiple_terms_corroborate(self):
        """两个以上独立词命中 → 有印证，正常作答。"""
        docs = [
            Document(
                page_content="员工出差期间每天可领取餐补 120 元，凭发票报销。"
                             "住宿费按城市等级设上限。",
                metadata={"source": "差旅与费用报销制度.md", "page": 2},
            ),
            Document(
                page_content="公司为员工缴纳五险一金。",
                metadata={"source": "员工手册.md", "page": 12},
            ),
        ]
        llm = ExtractiveLLM()
        answer = llm.invoke(build_prompt("出差餐补每天多少钱？", docs))

        assert "无法回答" not in answer
        assert "120" in answer

    def test_short_answer_sentence_is_not_gated_away(self):
        """回归测试：短句惩罚不能影响拒答判定。

        曾经的 bug：「年假为 5 天。」这种 5 个字的正确答案，
        被「短句惩罚 0.5」压到 0 分，触发拒答。
        短句惩罚只应影响**排序**，不该参与**拒答**。
        """
        docs = [Document(page_content="年假为 5 天。", metadata={"source": "a.md", "page": 1})]
        llm = ExtractiveLLM()
        answer = llm.invoke(build_prompt("年假多少天", docs))

        assert "无法回答" not in answer
        assert "5 天" in answer

    def test_single_term_query_with_strong_match_passes(self):
        """短查询词只有一个内容词时，只要相关度高就应该作答。

        否则「年假」这种两个字的查询将永远无法作答（命中数上限就是 1）。
        """
        docs = [
            Document(
                page_content="员工入职满一年后享受带薪年假，每年 5 天。",
                metadata={"source": "员工手册.md", "page": 8},
            )
        ]
        llm = ExtractiveLLM()
        report = llm.evidence_report(build_prompt("年假", docs))

        assert report["gate_pass"] is True
        assert report["best_score"] >= ExtractiveLLM().lone_evidence_relevance

    def test_gate_thresholds_are_configurable(self):
        """把闸门阈值调到不可能满足，任何问题都应拒答（证明闸门真的在起作用）。"""
        docs = [Document(page_content="年假为 5 天。", metadata={"source": "a.md", "page": 1})]
        llm = ExtractiveLLM(min_relevance=0.99, lone_evidence_relevance=0.99)

        assert "无法回答" in llm.invoke(build_prompt("年假多少天", docs))

    def test_lone_evidence_survives_sentence_split(self):
        """回归测试：关键术语被句号拆到相邻两句时，不能误拒答。

        这是**端到端冒烟测试**发现的真实失败：
        文档写「第八条 年假。员工入职满一年后开始享受带薪年休假。」，
        问「员工年假有多少天？」时，「年假」与「员工」被拆进两个句子，
        逐句统计只剩 1 个命中 → 被当成孤证误拒。
        修法是把孤证门槛从 0.15 降到 0.13（见 lone_evidence_relevance 的注释）。

        ⚠️ 这条用例的价值在于：它**不在**最初的评测集里 ——
        26 条用例全绿、拒答 100%，但真实用户一开口就翻车。
        教训：**评测集全绿 ≠ 系统可用**，必须拿自然说法去手测。
        """
        docs = [
            Document(
                page_content="第八条 年假。员工入职满一年后开始享受带薪年休假。\n"
                             "工龄一年以上不满十年的，每年享有 5 天年假。\n"
                             "工龄十年以上不满二十年的，每年享有 10 天年假。",
                metadata={"source": "员工手册.md", "page": 4},
            ),
            Document(
                page_content="公司为全体正式员工缴纳五险一金，住房公积金缴存比例 12%。",
                metadata={"source": "员工手册.md", "page": 12},
            ),
        ]
        llm = ExtractiveLLM()
        answer = llm.invoke(build_prompt("员工年假有多少天？", docs))

        assert "无法回答" not in answer, f"不应误拒答，实际：{answer}"

    def test_evidence_report_shape(self):
        """可观测接口的返回结构（标定脚本与线上排查依赖它）。"""
        docs = [
            Document(
                page_content="员工出差期间每天可领取餐补 120 元。",
                metadata={"source": "差旅制度.md", "page": 2},
            )
        ]
        report = ExtractiveLLM().evidence_report(build_prompt("出差餐补每天多少钱", docs))

        for key in ("best_score", "best_matched", "max_matched", "gate_pass", "required", "sentences"):
            assert key in report, f"缺少字段 {key}"
        assert isinstance(report["gate_pass"], bool)
        assert report["sentences"] >= 1

    def test_evidence_report_without_context(self):
        """没有参考资料时也不能崩，应返回全零画像。"""
        report = ExtractiveLLM().evidence_report(build_prompt("任意问题", []))

        assert report["best_score"] == 0.0
        assert report["gate_pass"] is False


class TestIsRefusalText:
    """拒答识别必须是**契约**，而不是「看起来像」。

    这组测试守住 prompt 契约与识别逻辑之间的一致性：
    Prompt 里告诉模型该说什么，识别函数就必须认得出来。
    """

    def test_offline_no_result_answer_is_refusal(self):
        """离线闸门拒答的话术必须被识别。"""
        assert is_refusal_text(NO_RESULT_ANSWER) is True

    def test_online_prompt_contract_phrase_is_refusal(self):
        """Prompt 里约定的在线拒答话术必须被识别（否则会漏判）。"""
        assert is_refusal_text("根据现有资料无法回答该问题。缺少 2027 年的数据。") is True

    def test_empty_or_blank_is_refusal(self):
        """空产出等于「答不上来」，不能当成正常答案。"""
        assert is_refusal_text("") is True
        assert is_refusal_text("   \n  ") is True
        assert is_refusal_text(None) is True

    def test_normal_answer_is_not_refusal(self):
        assert is_refusal_text("工龄一年以上不满十年的，每年享有 5 天年假。[1]") is False

    def test_marker_in_body_does_not_trigger(self):
        """特征串只应看开头 —— 正文里偶然提到「无法回答」不算拒答。"""
        long_prefix = "本制度适用于全体正式员工，详见后文说明。" * 6
        text = long_prefix + "无法回答的情形需报备。"
        assert is_refusal_text(text) is False

    def test_markers_are_conservative(self):
        """标记词必须是契约话术，不能混进「没有」「不知道」这类会误伤的通用词。"""
        for bad in ("没有", "不知道", "无法", "不能"):
            assert bad not in REFUSAL_MARKERS, f"「{bad}」太宽泛，会造成误判"
