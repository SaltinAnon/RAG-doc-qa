"""RAG 问答链：检索 → 组装 Prompt → 生成 → 解析引用 → 写入记忆。

## 关于多轮对话记忆的一个「坑」

你可能会看到大量教程写：

```python
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
```

**这在 LangChain 0.3 里已经被标记为 deprecated 了。** 官方推荐改用
`RunnableWithMessageHistory` + `BaseChatMessageHistory`。

那为什么本项目两个都没用、而是自己写了 `SessionMemory`？两个原因：

1. `RunnableWithMessageHistory` 要求链的第一个节点能接受 **message 列表**，
   而我们的离线降级实现 `ExtractiveLLM` 是 `LLM`（文本进文本出），
   不是 `ChatModel`，接上去会直接报类型错误。
2. 我们需要**按 session_id 隔离 + TTL 过期 + 轮数上限**这三点定制行为，
   LangChain 的内置 history 需要额外包一层才做得到，代码量反而更多。

> 面试话术：「我知道 `ConversationBufferMemory` 已废弃、官方推荐
> `RunnableWithMessageHistory`，也评估过。但它要求 ChatModel 类型的节点，
> 而我的架构里保留了无依赖离线降级路径（LLM 类型），
> 两者不兼容。所以我实现了一个同语义的 `SessionMemory`，
> 并保留了迁移路径：只要换回 ChatModel，链结构一行不用改。」

## 引用溯源是怎么做到的

1. `prompts.format_context()` 给每个检索片段**编号**（从 1 开始）；
2. Prompt 里强制要求「每个事实后标注 [编号]」；
3. 生成完成后，用正则从答案里抠出所有 `[n]`；
4. 按编号回查 chunk，把 `source / page / snippet / score` 一起返回。

第 3 步之后还有一道**兜底**：如果模型忘了标引用（小模型经常忘），
我们不是返回空引用，而是把检索到的 top-N 挂上去，
并在 `retrieval_debug` 里记一笔 `citation_fallback: true`。
这样接口层面「永远有引用可查」，同时又不隐瞒「这不是模型主动标的」。
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.core.embeddings import describe_embeddings, get_embeddings
from app.core.llm import ExtractiveLLM, _explain_llm_error, describe_llm, get_llm, is_offline
from app.core.prompts import (
    REFUSAL_EMPTY_OUTPUT,
    REFUSAL_MODEL,
    REFUSAL_NO_RETRIEVAL,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    build_query_rewrite_prompt,
    format_context,
    format_history,
    is_refusal_text,
)
from app.core.retriever import HybridRetriever, get_retriever
from app.models.schemas import Citation, QueryResponse, RetrievalDebug
from app.utils.logger import get_logger
from app.utils.text import truncate

logger = get_logger(__name__)

# 从答案中提取 [1] [2] 这类引用标记
_CITATION_RE = re.compile(r"\[(\d{1,2})\]")
# 引用没标上时，兜底挂几个片段
CITATION_FALLBACK_COUNT = 3


# ============================================================
#  会话记忆
# ============================================================
@dataclass
class _Session:
    """单个会话的状态。"""

    turns: list[tuple[str, str]] = field(default_factory=list)
    last_access: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_access = time.time()


class SessionMemory:
    """按 session_id 隔离的多轮对话记忆。

    特性：
    - **隔离**：每个 session_id 一份历史，互不干扰；
    - **有界**：只保留最近 `MEMORY_MAX_TURNS` 轮（防止 prompt 无限膨胀）；
    - **会过期**：超过 `SESSION_TTL_MINUTES` 未访问的会话自动清理
      （否则长时间运行的服务会把内存吃光——这是真实的生产问题）；
    - **线程安全**：FastAPI 默认多线程处理请求，字典操作必须加锁。
    """

    def __init__(self, max_turns: int | None = None, ttl_minutes: int | None = None) -> None:
        """初始化。

        Args:
            max_turns: 保留轮数，默认读配置。
            ttl_minutes: 会话过期分钟数，默认读配置。

        Note:
            ⚠️ 这里必须用 `if x is None` 而不是 `x or 默认值`。
            `0 or 默认值` 的结果是**默认值**，所以 `ttl_minutes=0`
            会被静默当成"未传参数"，导致"立即过期"这种测试场景永远失败。
            这是 Python 里非常常见的一个坑：**`or` 只能用于布尔语义的默认值。**
        """
        self.max_turns = max_turns if max_turns is not None else settings.memory_max_turns
        self.ttl_seconds = (
            ttl_minutes if ttl_minutes is not None else settings.session_ttl_minutes
        ) * 60
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    def get_history(self, session_id: str | None) -> list[tuple[str, str]]:
        """读取会话历史。

        Args:
            session_id: 会话 ID；None 或空串返回空历史（无状态问答）。

        Returns:
            `[(问题, 回答)]`，按时间正序。
        """
        if not session_id:
            return []
        with self._lock:
            self._purge_locked()
            s = self._sessions.get(session_id)
            if s is None:
                return []
            s.touch()
            return list(s.turns)

    def append(self, session_id: str | None, question: str, answer: str) -> None:
        """追加一轮对话。

        Args:
            session_id: 会话 ID；为空则忽略（无状态模式）。
            question: 用户问题。
            answer: 助手回答。
        """
        if not session_id:
            return
        with self._lock:
            s = self._sessions.setdefault(session_id, _Session())
            s.turns.append((question, answer))
            # 只保留最近 max_turns 轮
            if len(s.turns) > self.max_turns:
                s.turns = s.turns[-self.max_turns :]
            s.touch()

    def reset(self, session_id: str) -> bool:
        """清空某个会话。

        Args:
            session_id: 会话 ID。

        Returns:
            True 表示该会话之前存在。
        """
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def stats(self) -> dict[str, Any]:
        """返回记忆使用情况（供 /readyz 或调试接口展示）。"""
        with self._lock:
            self._purge_locked()
            return {
                "active_sessions": len(self._sessions),
                "total_turns": sum(len(s.turns) for s in self._sessions.values()),
                "max_turns_per_session": self.max_turns,
                "ttl_minutes": self.ttl_seconds // 60,
            }

    def _purge_locked(self) -> int:
        """清理过期会话。**必须在持锁状态下调用。**"""
        now = time.time()
        expired = [k for k, s in self._sessions.items() if now - s.last_access > self.ttl_seconds]
        for k in expired:
            del self._sessions[k]
        if expired:
            logger.info("清理过期会话 %d 个", len(expired))
        return len(expired)


# ============================================================
#  RAG 链
# ============================================================
class RAGChain:
    """检索增强生成链。

    对外只暴露一个方法 `answer()`，内部完成全部编排。
    这种「胖服务、瘦接口」的设计让 API 层只负责协议转换，业务逻辑可单测。
    """

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        llm: Any | None = None,
        memory: SessionMemory | None = None,
    ) -> None:
        """初始化。

        Args:
            retriever: 检索器；None 用全局单例。
            llm: 语言模型；None 用全局单例（含自动降级）。
            memory: 会话记忆；None 新建一个。
        """
        self.retriever = retriever or get_retriever()
        self._llm = llm
        self.memory = memory or SessionMemory()

        # LangChain LCEL 链：Prompt → LLM → 纯文本
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", USER_TEMPLATE)]
        )
        self._parser = StrOutputParser()
        self._compiled: Any = None

    # ---------- 属性 ----------
    @property
    def llm(self) -> Any:
        """延迟获取 LLM（避免构造 RAGChain 时就加载模型）。"""
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    @property
    def chain(self) -> Any:
        """延迟编译 LCEL 链。

        为什么延迟：`ChatPromptTemplate | llm` 在导入时构建会触发
        模型的加载/校验，而我们希望 **import 不产生副作用**（CLAUDE.md §5.6）。
        """
        if self._compiled is None:
            self._compiled = self._prompt | self.llm | self._parser
        return self._compiled

    # ---------- 对外主入口 ----------
    def answer(
        self,
        question: str,
        *,
        session_id: str | None = None,
        top_k: int | None = None,
        collection: str | None = None,
        enable_rewrite: bool | None = None,
    ) -> QueryResponse:
        """回答一个问题。

        Args:
            question: 用户问题。
            session_id: 会话 ID，传了才启用多轮记忆。
            top_k: 检索片段数，None 用配置值。
            collection: 集合名。
            enable_rewrite: 是否启用多轮查询改写。
                None 表示「在线模式且有历史时自动启用」。

        Returns:
            结构化问答结果（含引用、耗时、调试信息）。

        Raises:
            ValueError: 问题为空。
        """
        started = time.perf_counter()
        question = (question or "").strip()
        if not question:
            raise ValueError("问题不能为空")

        history = self.memory.get_history(session_id) if settings.memory_enabled else []
        offline = is_offline(self._llm)

        # 自动决定是否改写查询
        if enable_rewrite is None:
            enable_rewrite = bool(history) and not offline

        search_query = question
        rewrite_debug: dict[str, Any] = {}
        if enable_rewrite and history:
            search_query, rewrite_debug = self._rewrite_query(question, history)

        # ---------- 1. 检索 ----------
        retrieval = self.retriever.retrieve(
            search_query, top_k=top_k, collection=collection
        )
        debug = RetrievalDebug(**{k: v for k, v in retrieval.debug.items() if k in RetrievalDebug.model_fields})

        # ---------- 2. 无结果：直接给结构化拒答，不浪费一次 LLM 调用 ----------
        if retrieval.is_empty():
            from app.core.prompts import NO_RESULT_ANSWER

            response = QueryResponse(
                answer=NO_RESULT_ANSWER,
                citations=[],
                latency_ms=int((time.perf_counter() - started) * 1000),
                model=describe_llm(self._llm),
                refused=True,
                refusal_reason=REFUSAL_NO_RETRIEVAL,
                offline_mode=offline,
                retrieval_debug=debug,
            )
            self.memory.append(session_id, question, response.answer)
            return response

        # ---------- 3. 组装 Prompt ----------
        documents = [c.doc for c in retrieval.chunks]
        context_text, indexed = format_context(documents)
        history_text = format_history(history, self.memory.max_turns)

        # ---------- 4. 生成 ----------
        try:
            raw_answer = self.chain.invoke(
                {"context": context_text, "history": history_text, "question": question}
            )
        except Exception as exc:
            logger.error("LLM 调用失败：%s", exc, exc_info=True)
            raise LLMInvocationError(_explain_llm_error(exc)) from exc

        answer_text = (raw_answer or "").strip()

        # ---------- 5. 判定是否拒答（结构化，不让上层去猜文本） ----------
        refused = False
        refusal_reason: str | None = None
        if not answer_text:
            # 模型返回空串属于**异常**，绝不能静默当成一条正常答案返回
            logger.error("模型返回空答案（question=%s），按拒答处理", question)
            from app.core.prompts import NO_RESULT_ANSWER

            answer_text = NO_RESULT_ANSWER
            refused = True
            refusal_reason = REFUSAL_EMPTY_OUTPUT
        elif is_refusal_text(answer_text):
            refused = True
            refusal_reason = REFUSAL_MODEL

        # ---------- 6. 解析引用 ----------
        # 拒答时**不挂引用**：答案正文里没有 [n]，硬把检索结果挂上去会制造
        # 「看起来有据可依」的假象 —— 这正是我们要避免的幻觉。
        if refused:
            citations: list[Citation] = []
            fallback_used = False
        else:
            citations, fallback_used = self._build_citations(answer_text, retrieval.chunks, indexed)
        if fallback_used:
            # 兑现模块 docstring 里的承诺：兜底挂引用这件事要可观测。
            debug.citation_fallback = True

        # ---------- 7. 写记忆 ----------
        self.memory.append(session_id, question, answer_text)

        elapsed = int((time.perf_counter() - started) * 1000)
        logger.info(
            "问答完成 session=%s 耗时=%dms 引用=%d 离线=%s 拒答=%s",
            session_id or "-", elapsed, len(citations), offline, refused,
        )

        return QueryResponse(
            answer=answer_text,
            citations=citations,
            latency_ms=elapsed,
            model=describe_llm(self._llm),
            refused=refused,
            refusal_reason=refusal_reason,
            offline_mode=offline,
            retrieval_debug=debug,
        )

    # ---------- 内部工具 ----------
    @staticmethod
    def _build_citations(
        answer: str,
        chunks: list[Any],
        indexed: list[tuple[int, Document]],
    ) -> tuple[list[Citation], bool]:
        """从答案里解析引用编号并回查出处。

        Args:
            answer: 模型生成的答案文本。
            chunks: 检索到的 `RetrievedChunk` 列表（用于取相关性分数）。
            indexed: `[(编号, Document)]`，来自 `format_context`。

        Returns:
            `(引用列表, 是否使用了兜底)`。
        """
        by_index = {i: doc for i, doc in indexed}
        score_by_key: dict[str, float] = {}
        for c in chunks:
            key = str(c.metadata.get("chunk_id") or c.metadata.get("doc_id"))
            score_by_key[key] = max(score_by_key.get(key, 0.0), c.final_score)

        # 保持答案中出现的顺序，去重
        seen: OrderedDict[int, None] = OrderedDict()
        for m in _CITATION_RE.finditer(answer):
            num = int(m.group(1))
            if num in by_index:
                seen[num] = None

        fallback = False
        if not seen:
            # 模型忘了标引用：兜底挂上相关性最高的前 N 个片段
            fallback = True
            for i, _doc in indexed[:CITATION_FALLBACK_COUNT]:
                seen[i] = None

        citations: list[Citation] = []
        for num in seen:
            doc = by_index[num]
            key = str(doc.metadata.get("chunk_id") or doc.metadata.get("doc_id"))
            page = doc.metadata.get("page")
            citations.append(
                Citation(
                    index=num,
                    doc_id=str(doc.metadata.get("doc_id", "unknown")),
                    source=str(doc.metadata.get("source", "未知来源")),
                    page=int(page) if isinstance(page, (int, str)) and str(page).isdigit() else None,
                    score=round(score_by_key.get(key, 0.0), 4),
                    snippet=truncate(doc.page_content, 180),
                )
            )
        return citations, fallback

    def _rewrite_query(
        self, question: str, history: list[tuple[str, str]]
    ) -> tuple[str, dict[str, Any]]:
        """多轮场景下把指代词还原成实体，提升检索命中率。

        失败时**静默回退到原问题**——这是一个可选增强，
        绝不能因为它出问题就让整个问答挂掉。

        Args:
            question: 本轮原始问题。
            history: 历史对话。

        Returns:
            `(用于检索的查询, 调试信息)`。
        """
        try:
            prompt = build_query_rewrite_prompt(question, history)
            rewritten = (self.llm.invoke(prompt) or "").strip()

            # 合理性校验：太短、太长、或明显是解释性文字，都判为改写失败
            if (
                not rewritten
                or len(rewritten) > 200
                or len(rewritten) < 2
                or rewritten.startswith(("改写", "好的", "根据"))
            ):
                return question, {"rewrite": "skipped_invalid"}

            if rewritten != question:
                logger.info("查询改写：%r → %r", question[:40], rewritten[:60])
            return rewritten, {"rewrite": "ok", "original": question, "rewritten": rewritten}
        except Exception as exc:
            logger.warning("查询改写失败，使用原问题检索：%s", exc)
            return question, {"rewrite": f"failed: {exc}"}

    # ---------- 状态自检 ----------
    def describe(self) -> dict[str, Any]:
        """返回链路的运行时状态（供 /readyz 与调试用）。"""
        emb = get_embeddings()
        return {
            "embedding": describe_embeddings(emb),
            "llm": describe_llm(self._llm),
            "llm_offline": is_offline(self._llm),
            # —— LLM 配置体检：让「配置错了」在接口层面可见，而不是只藏在日志里 ——
            "llm_provider": settings.llm_provider,
            "llm_model_configured": settings.llm_model,
            "llm_model_effective": settings.resolved_llm_model,
            "llm_base_url": settings.resolved_base_url or "(SDK 默认)",
            "llm_config_ok": not settings.llm_config_warnings,
            "llm_config_warnings": settings.llm_config_warnings,
            "hybrid_enabled": settings.hybrid_enabled,
            "rerank_enabled": settings.rerank_enabled,
            "top_k": settings.top_k,
            "chunk_size": settings.chunk_size,
            "memory": self.memory.stats(),
        }


class LLMInvocationError(RuntimeError):
    """调用大模型失败（网络/鉴权/额度）。"""


# ============================================================
#  单例
# ============================================================
_chain: RAGChain | None = None
_chain_lock = threading.Lock()


def get_rag_chain() -> RAGChain:
    """获取 RAG 链单例。"""
    global _chain
    with _chain_lock:
        if _chain is None:
            _chain = RAGChain()
        return _chain


def reset_rag_chain_singleton() -> None:
    """清空单例（测试用）。"""
    global _chain
    with _chain_lock:
        _chain = None


# 便于测试替换的别名
ConversationMemory = SessionMemory
"""`SessionMemory` 的别名，语义上等价于 LangChain 的 ConversationBufferMemory。"""

# 供外部（如 ExtractiveLLM 单测）使用的导出
__all__ = [
    "SessionMemory",
    "ConversationMemory",
    "RAGChain",
    "LLMInvocationError",
    "get_rag_chain",
    "reset_rag_chain_singleton",
    "ExtractiveLLM",
]
