"""LLM 工厂 + 离线抽取式降级实现。

## 两个诉求的冲突

- 项目要能展示**真正的生成式 RAG** → 需要大模型 API；
- 项目要能被人**零成本跑起来** → 不能强依赖 API Key。

解决办法：`get_llm()` 返回统一的 LangChain `LLM` 接口，内部有两条实现：

- **在线**：`ChatOpenAI`（可指向 OpenAI / DeepSeek / 智谱 / Ollama 等任何
  OpenAI 兼容端点，只改 `base_url`）；
- **离线**：`ExtractiveLLM` —— 一个把 LangChain `LLM` 接口实现出来的
  「假模型」，它从 Prompt 里解析出参考资料，按问题相关度**抽取原文句子**拼成答案。

> ⚠️ 重要原则：离线模式**绝不假装自己是生成式回答**。
> 它的输出第一行就明确标注「离线抽取模式」，API 响应里也有 `offline_mode: true`，
> 前端顶部显示黄条。诚实比好看重要 —— 面试官发现你在造数据，整个项目就废了。
"""

from __future__ import annotations

import re
from functools import lru_cache

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

from app.config import settings
from app.core.prompts import NO_CONTEXT_PLACEHOLDER, NO_RESULT_ANSWER, PROMPT_VERSION
from app.utils.logger import get_logger
from app.utils.text import idf_weights, tokenize_for_bm25

logger = get_logger(__name__)

# 离线抽取答案的标识前缀
OFFLINE_ANSWER_PREFIX = "（离线抽取模式·未调用大模型）"

# 句子切分：中文句末标点 + 英文句末 + 换行
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？；!?;])|\n+")
# 参考资料块的正则：`[编号] 来源：xxx` 开头
_CONTEXT_BLOCK = re.compile(r"^\[(\d+)\]\s*来源：.*$", re.MULTILINE)
# 从 prompt 里抠出「本次问题」
_QUESTION_BLOCK = re.compile(r"【本次问题】\s*(.*?)\s*===== 问题结束 =====", re.DOTALL)
# 少于这么多个「有效字符」的片段视为小标题（如「第八条」「第三章 假期管理」），
# 不与正文并列，而是并入相邻句子 —— 见 _split_sentences()
_MIN_FRAGMENT_CHARS = 6
# 计算「有效字符」时要去掉的：空白、标点、下划线
_NON_CONTENT = re.compile(r"[\s\W_]", re.UNICODE)


def _effective_len(text: str) -> int:
    """返回「有效字符数」—— 去掉空白与标点后的长度。

    ⚠️ 这里必须去标点，不能只去空白。踩过的坑：
    「第八条 年假。」只去空白是 6 个字符，刚好躲过「少于 6 视为小标题」的判定，
    于是它被当成一个完整句子、与下一句拆开，
    导致关键词「年假」与「员工」分属两句 → 印证数只有 1 → 误拒答。
    去掉标点后是 5 个字符，正确识别为小标题。

    Args:
        text: 任意文本。

    Returns:
        有效字符数。
    """
    return len(_NON_CONTENT.sub("", text))


def _split_sentences(text: str) -> list[str]:
    """把一段正文切成句子，并把**过短的片段并入相邻句子**。

    ## 为什么需要「并入」这一步

    企业文档里大量存在小标题：

        第八条 年假。员工入职满一年后开始享受带薪年休假。

    如果只按句末标点切分，会得到两句：「第八条 年假」和
    「员工入职满一年后开始享受带薪年休假」——**「年假」这个关键词
    被孤立在第一句里，而真正的解释在第二句**。

    后果很严重：整套「引用溯源」和「孤证判定」都建立在「一句里命中了几个词」
    之上。关键词被句号切开后，本来明显可答的问题会退化成「孤证」。

    实测踩到的真实案例：问「员工年假有多少天？」时，
    「年假」与「员工」分属两句 → 逐句命中数只有 1 → 被拒答网关误杀。
    （教训：评测集全绿不等于系统可用，一定要起真实服务手测几轮。）

    Args:
        text: 一段正文（通常是一个检索片段）。

    Returns:
        合并后的句子列表。过短的片段会被并到**后一句**前面；
        若它是最后一段，则并到前一句后面。
    """
    fragments = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]

    merged: list[str] = []
    pending = ""
    for frag in fragments:
        if _effective_len(frag) < _MIN_FRAGMENT_CHARS:
            # 太短 → 先攒着（多半是「第八条」这种小标题），并给下一句
            pending += frag
            continue
        merged.append(pending + frag)
        pending = ""

    if pending:
        if merged:
            merged[-1] += pending
        else:
            merged.append(pending)

    return merged


# ============================================================
#  离线抽取式「LLM」
# ============================================================
class ExtractiveLLM(LLM):
    """不调用任何外部服务、从参考资料中抽取句子作答的兜底「模型」。

    它实现了 LangChain 的 `LLM` 接口，所以可以无缝接入
    `prompt | llm | StrOutputParser()` 这样的 LCEL 链 ——
    这意味着**在线/离线两条路径的编排代码是完全一样的**，
    只有模型实例不同。这是本项目能同时在两种模式下跑通 CI 的关键。

    算法（三步）：
    1. 从 Prompt 里解析出「本次问题」和带编号的参考资料块；
    2. 把每块拆成句子，按与问题的词汇重合度打分，选出最相关的几句；
    3. 按原文顺序拼接，并在每句后标注它来自哪一号资料 `[n]`。

    ## 拒答闸门（抑制幻觉的关键设计）

    离线模式没有语言模型，不会「理解」问题，所以知识库里**确实没有答案**时，
    它照样会从无关段落里抽句子。为此加了一道闸门，用**两个互补信号**判定：

    | 信号 | 含义 | 性质 |
    |---|---|---|
    | 句内相关度 | IDF 加权词汇覆盖率 + 短语 + 数字 | 命中了多少「信息量」 |
    | **独立命中词数** | 该句命中了几个**互不相同**的查询词 | 有几个**独立**证据 |

    ⭐ 核心洞察：**孤证不算证据**。实测发现所有「答不出来」的问题，
    其最高分句子都只命中了**恰好一个**词，而且是巧合：

        「公司的股票代码是多少？」 → 只命中「代码」（文档讲的是「代码仓库禁止提交密钥」）
        「公司 CEO 的生日是哪一天？」→ 只命中「生日」（文档讲的是「员工生日礼金」）

    这两个词在各自的文档里**真实存在**，所以任何纯词面阈值都拦不住它们
    （见 `scripts/calibrate_gate.py` 打出来的两条分布几乎完全重叠）。
    但它们的**独立命中词数都是 1**，而所有可答用例都 ≥ 2——
    这是一个结构性差异，比任何分数阈值都稳。

    于是门槛按印证数量分档：

    - **有印证**（命中 ≥ `min_matched_terms` 个词）→ 相关度只需过 `min_relevance`（0.08）；
    - **孤证**（只命中更少的词）→ 要求很高的相关度 `lone_evidence_relevance`（0.15）。

    两支各取最高分，**任一达标即放行**。

    局限（诚实写出来）：这条规则用中文 **bigram** 计数，所以

    - 「一个长词」会贡献多个 bigram、可能被当成「多词印证」；
    - 「年假」与「年休假」这类**同义不同形**的写法匹配不上，会被误判为孤证；
    - 关键术语被**句号拆到相邻两句**时（「第八条 年假。员工入职满一年后…」），
      逐句统计会只剩 1 个命中、退化成孤证 —— 这个问题由
      `_split_sentences()` 把「小标题并入下一句」解决（实测踩到的真实误拒场景）。

    这几类问题都需要**真正的语义理解** —— 这正是**在线模式（真 LLM）存在的价值**，
    也是本项目保留双后端的原因。离线模式是一个「零成本可跑」的诚实降级，
    不是语义理解的替代品。

    > 阈值不是拍脑袋：全部在自建评测集上打分布后定的，见
    > `scripts/calibrate_gate.py`（该脚本会把两个信号的分布打出来，
    > 并给出「阈值必须落在哪个区间」的结论）。
    > ⚠️ 换 Embedding / 换语料后必须重跑标定脚本。
    """

    # 最多抽取几句
    top_sentences: int = 4
    # 答案长度上限（防止把整个 chunk 都吐出来）
    max_answer_chars: int = 900
    # 每句话在答案里的最大展示长度
    sentence_max_chars: int = 220

    # ---------- 拒答闸门参数（标定脚本：scripts/calibrate_gate.py）----------
    # 有「多词印证」时要求的最低相关度。
    # 实测有印证分支的可答用例最低 0.126，故取 0.08 —— 余量约 37%，对取值不敏感。
    min_relevance: float = 0.08
    # 「孤证」（只命中 < min_matched_terms 个查询词）时要求的相关度。
    #
    # 实测（27 条评测用例 + 单术语短查询）：
    #     孤证分支·拒答用例 ：0.085 / 0.091 / 0.124       ← 最高 0.124
    #     孤证分支·单术语查询：0.860（「年假」「密码」「SLA」）← 最低 0.860
    # 窗口宽达 (0.124, 0.860]，取值空间极大、**不敏感** —— 取 0.15 既有
    # 21% 的拒答余量，又离真实短查询很远。
    #
    # ⚠️ 这条门槛之所以存在，是因为「孤证不算证据」：
    #    「生日」「代码」这类单个词的巧合命中（文档讲的是「员工生日礼金」「代码仓库」）
    #    在字面上是命中的，纯分数阈值拦不住，但它们的独立命中词数只有 1。
    #
    # ⚠️ 注意：一旦关键术语被句号拆进相邻两句，印证数会退化成 1。
    #    这个问题由 `_split_sentences()` 的「小标题并入下一句」解决，
    #    而不是靠调这个阈值 —— 详见该函数的说明。
    lone_evidence_relevance: float = 0.15
    # 达到几个「互不相同的查询词命中」才算有印证（而非孤证）
    min_matched_terms: int = 2

    @property
    def _llm_type(self) -> str:
        """LangChain 要求的标识符。"""
        return "offline-extractive"

    @property
    def _identifying_params(self) -> dict[str, object]:
        """供 LangChain 打印/缓存用。"""
        return {"top_sentences": self.top_sentences, "version": PROMPT_VERSION}

    # ---------- 解析 ----------
    @staticmethod
    def _parse(prompt: str) -> tuple[str, list[tuple[int, str]]]:
        """从 Prompt 文本中解析出问题与参考资料。

        Args:
            prompt: 完整的（已填充模板的）Prompt。

        Returns:
            `(question, [(编号, 正文)])`。
        """
        m = _QUESTION_BLOCK.search(prompt)
        question = m.group(1).strip() if m else ""

        # 找到所有 `[n] 来源：...` 的起始位置，按顺序切出正文
        blocks: list[tuple[int, str]] = []
        matches = list(_CONTEXT_BLOCK.finditer(prompt))

        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt)
            body = prompt[start:end]
            # 砍掉模板尾部（后面的历史/问题段不属于资料内容）
            for marker in ("===== 参考资料结束 =====", "【历史对话】"):
                if marker in body:
                    body = body.split(marker)[0]
            idx = int(match.group(1))
            blocks.append((idx, body.strip()))

        # 没有按模板格式化时（比如有人直接把任意 prompt 丢进来），
        # 退化为「整段当作 1 号资料」
        if not blocks and NO_CONTEXT_PLACEHOLDER not in prompt:
            blocks = [(1, prompt)]

        return question, blocks

    # ---------- 打分 ----------
    @staticmethod
    def _score_sentence(
        sentence: str,
        q_weights: dict[str, float],
        total_weight: float,
        q_numbers: set[str],
        q_phrases: list[str],
        s_tokens: set[str] | None = None,
    ) -> float:
        """给单个句子打分（与问题的相关度）。

        打分信号（参照 retriever 的重排逻辑，保持一致的设计思路）：
        - 0.55 **IDF 加权的**词汇覆盖率
        - 0.25 短语命中：问题里的连续词组整体出现在句中
        - 0.20 数字命中：问题问「几天」时，含数字的句子更可能是答案

        ⭐ 为什么覆盖率必须做 IDF 加权：
        朴素覆盖率会被「公司」「员工」这类高频词污染 —— 问「公司的股票代码是多少」时，
        随便一个含「公司」的句子都能拿到 1/3 的覆盖率，于是系统硬答而不是拒答。
        IDF 加权后，这类零信息量的词权重降到 1.0，而「股票」「代码」这类
        未在任何候选中出现的词拿到最高权重，覆盖率能迅速掉到接近 0，
        从而正确触发拒答。

        ⚠️ 这里**不做短句惩罚**。短句惩罚（「如下：」这种噪声）属于「排序」层面的事，
        在 `_call()` 里单独施加；相关度本身必须保持「纯」的语义，
        否则「年假为 5 天。」这种 5 个字的正确答案会被罚掉、导致误拒答。

        Args:
            sentence: 候选句子。
            q_weights: `{查询词: IDF 权重}`（由 `idf_weights()` 计算）。
            total_weight: 所有查询词的权重之和（用于归一化）。
            q_numbers: 问题里的数字。
            q_phrases: 问题的连续词组。
            s_tokens: 该句的 token 集合（调用方已算好，避免重复分词；
                None 时自行计算）。

        Returns:
            0.0 ~ 1.0 的分数。
        """
        if s_tokens is None:
            s_tokens = set(tokenize_for_bm25(sentence))

        if q_weights and total_weight > 0:
            coverage = sum(w for t, w in q_weights.items() if t in s_tokens) / total_weight
        else:
            coverage = 0.0

        if q_phrases:
            hit = sum(len(p) for p in q_phrases if p in sentence)
            total = sum(len(p) for p in q_phrases) or 1
            phrase = hit / total
        else:
            phrase = 0.0

        if q_numbers:
            num = (
                len(q_numbers & set(re.findall(r"\d+(?:\.\d+)?", sentence))) / len(q_numbers)
            )
        else:
            num = 0.3 if re.search(r"\d", sentence) else 0.0

        return max(0.0, 0.55 * coverage + 0.25 * phrase + 0.20 * num)

    # ---------- 证据分析（_call 与 evidence_report 共用） ----------
    def _score_all(
        self, prompt: str
    ) -> tuple[str, list[tuple[float, float, int, int, str, int]], bool]:
        """把 Prompt 里的参考资料逐句打分，返回全部证据。

        Args:
            prompt: 完整 Prompt。

        Returns:
            `(question, scored, has_blocks)`：
            - `scored` 为 `[(排序键, 原始相关度, 块编号, 句内序号, 句子, 独立命中词数)]`
              按排序键降序。排序键 = 原始相关度 - 短句惩罚；
              原始相关度单独保留给「拒答闸门」用（短句惩罚不该影响拒答）。
            - `has_blocks` 表示 Prompt 里是否真的带了参考资料。
        """
        question, blocks = self._parse(prompt)
        if not blocks or (len(blocks) == 1 and blocks[0][1].strip() == ""):
            return question, [], False

        q_tokens = set(tokenize_for_bm25(question))
        q_content = {t for t in q_tokens if len(t) > 1} or q_tokens
        q_numbers = set(re.findall(r"\d+(?:\.\d+)?", question))
        q_phrases = [
            question[i : i + n]
            for n in (2, 3, 4)
            for i in range(max(0, len(question) - n + 1))
            if not question[i : i + n].isspace()
        ][:30]

        # ⭐ IDF 加权：用检索到的候选片段作为语料统计词频。
        #    让「公司」「员工」这类无处不在的词不再虚抬相关度，
        #    从而在「知识库里确实没有答案」时正确拒答（而不是硬从无关段落里抽句子）。
        block_token_sets = [set(tokenize_for_bm25(body)) for _, body in blocks]
        q_weights = idf_weights(block_token_sets, q_content)
        total_weight = sum(q_weights.values()) or 1.0

        scored: list[tuple[float, float, int, int, str, int]] = []

        for block_idx, body in blocks:
            for pos, sent in enumerate(_split_sentences(body)):
                # ⚠️ 这里的门槛（4）**故意低于** _split_sentences 的合并门槛（6）。
                # 合并已经把「第八条 年假」这类小标题并进了相邻句子；
                # 合并后仍然很短的，只可能是「整块本来就短」（例如整篇就是
                # 「年假为 5 天。」）—— 那是**唯一**的正文，不能丢。
                # 这一档只用来滤掉「如下：」「详见附件」这类纯噪声。
                if _effective_len(sent) < 4:
                    continue
                s_tokens = set(tokenize_for_bm25(sent))
                raw = self._score_sentence(
                    sent, q_weights, total_weight, q_numbers, q_phrases, s_tokens
                )
                # 独立命中词数：这一句命中了几个**互不相同**的查询词
                matched = sum(1 for t in q_weights if t in s_tokens)
                # 短句惩罚：少于 8 个有效字符基本没信息（「如下：」「详见附件」）。
                # 只影响排序、不影响拒答 —— 正确答案本身可能就是短句。
                penalty = 0.0 if _effective_len(sent) >= 8 else 0.5
                scored.append((max(0.0, raw - penalty), raw, block_idx, pos, sent, matched))

        scored.sort(key=lambda x: (-x[0], x[2], x[3]))
        return question, scored, True

    def _score_branches(
        self, scored: list[tuple[float, float, int, int, str, int]]
    ) -> tuple[float, float]:
        """把候选句子按「印证数量」分成两支，各自取最高相关度。

        Args:
            scored: `_score_all` 的输出。

        Returns:
            `(有印证分支的最高相关度, 孤证分支的最高相关度)`。
            某一支没有句子时该值为 0.0。
        """
        corroborated = max(
            (raw for _rk, raw, _b, _p, _s, m in scored if m >= self.min_matched_terms),
            default=0.0,
        )
        lone = max(
            (raw for _rk, raw, _b, _p, _s, m in scored if m < self.min_matched_terms),
            default=0.0,
        )
        return corroborated, lone

    def _gate_passes(self, scored: list[tuple[float, float, int, int, str, int]]) -> bool:
        """拒答闸门：判断检索到的资料里是否存在**足以支撑回答**的证据。

        规则（两个互补信号，详见类文档字符串里那张表）：

            有印证分支（命中 ≥ N 个不同查询词）→ 只需过 `min_relevance`（低门槛）
            孤证分支（只命中 < N 个词）      → 必须过 `lone_evidence_relevance`（高门槛）

        理由：**孤证不算证据**。「生日」「代码」这类单个词的巧合命中
        （文档讲的是「员工生日礼金」「代码仓库」）不足以支撑回答。

        Args:
            scored: `_score_all` 的输出。

        Returns:
            True 表示证据充分（不拒答）。
        """
        corroborated, lone = self._score_branches(scored)
        return (
            corroborated >= self.min_relevance
            or lone >= self.lone_evidence_relevance
        )

    # ---------- 主逻辑 ----------
    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> str:
        """LangChain LLM 接口的实现。

        Args:
            prompt: 完整 Prompt。
            stop: 停止词（离线模式忽略）。
            run_manager: LangChain 回调管理器（离线模式忽略）。
            **kwargs: 其他参数（忽略）。

        Returns:
            抽取式答案文本，带 `[n]` 引用标记。
        """
        _question, scored, has_blocks = self._score_all(prompt)

        if not has_blocks:
            return NO_RESULT_ANSWER

        if not scored:
            # 连一句「够长到能打分」的句子都没有 —— 说明检索到的内容全是
            # 「如下：」这类碎片，无法评估相关性。
            # ⚠️ 这里**必须拒答，不能兜底把碎片拼出来当答案**：
            #    拼出来的东西没经过任何相关性判定，属于「静默幻觉」。
            #    （早期版本就是这么兜底的，结果让「拒答闸门」形同虚设。）
            logger.debug("离线抽取：没有可打分的句子（内容全是碎片），判定为资料不足")
            return NO_RESULT_ANSWER

        # ---------- 拒答闸门 ----------
        # 离线抽取模式没有语言模型，无法"理解"问题；
        # 但可以在**证据不足**时明确拒答，而不是硬从无关段落里抽句子。
        # 这是抑制幻觉的最后一道防线，也保证了「拒答准确率」这个指标有意义。
        if not self._gate_passes(scored):
            best = max(s[1] for s in scored)
            logger.debug(
                "离线抽取：证据不足（最高相关度 %.3f、最多命中 %d 词），判定为资料不足并拒答",
                best, max(s[5] for s in scored),
            )
            return NO_RESULT_ANSWER

        # 排序按含惩罚的排序键；输出时再按 (块编号, 句内序号) 恢复阅读顺序
        picked = scored[: self.top_sentences]
        picked.sort(key=lambda x: (x[2], x[3]))

        # 同一块里最多取 2 句，避免答案被单一来源垄断
        per_block: dict[int, int] = {}
        lines: list[str] = []
        total = 0

        for _rank, _raw, bidx, _pos, sent, _m in picked:
            if per_block.get(bidx, 0) >= 2:
                continue
            per_block[bidx] = per_block.get(bidx, 0) + 1

            # 句末去重标点：句子已含标点就不要再加
            text = sent.rstrip()
            if text and text[-1] not in "。！？.!?;；":
                text += "。"
            line = f"- {text[: self.sentence_max_chars]}[{bidx}]"
            if total + len(line) > self.max_answer_chars:
                break
            lines.append(line)
            total += len(line)

        if not lines:
            return NO_RESULT_ANSWER

        body = "\n".join(lines)
        return f"{OFFLINE_ANSWER_PREFIX}\n{body}"

    # ---------- 可观测性 ----------
    def evidence_report(self, prompt: str) -> dict[str, object]:
        """返回该 Prompt 下的**证据画像**（供标定与线上排查）。

        暴露这个是为了：

        - `scripts/calibrate_gate.py` 能在评测集上把「可答 / 拒答」两条分布打出来，
          从而**用数据**定闸门阈值，而不是拍脑袋；
        - 线上排查「为什么这题被拒答了」时，可以直接看到分数与命中词数。

        Args:
            prompt: 完整 Prompt。

        Returns:
            含以下字段的字典：

            - `best_score`   最高原始相关度（不含短句惩罚）
            - `best_matched` 取得最高相关度那一句的独立命中词数
            - `max_matched`  全文最多的独立命中词数（决定走哪一支）
            - `corroborated_score` 有印证分支的最高相关度
            - `lone_score`         孤证分支的最高相关度
            - `required`     实际适用的门槛
            - `gate_pass`    闸门是否放行
            - `sentences`    参与打分的句子数
        """
        _question, scored, has_blocks = self._score_all(prompt)
        if not has_blocks or not scored:
            return {
                "best_score": 0.0,
                "best_matched": 0,
                "max_matched": 0,
                "corroborated_score": 0.0,
                "lone_score": 0.0,
                "gate_pass": False,
                "required": self.lone_evidence_relevance,
                "sentences": 0,
            }

        # 取「原始相关度最高」的那一句作为代表
        best = max(scored, key=lambda x: x[1])
        corroborated, lone = self._score_branches(scored)
        corroborated_wins = corroborated >= self.min_relevance

        return {
            "best_score": round(best[1], 4),
            "best_matched": best[5],
            "max_matched": max(s[5] for s in scored),
            "corroborated_score": round(corroborated, 4),
            "lone_score": round(lone, 4),
            "gate_pass": corroborated_wins or lone >= self.lone_evidence_relevance,
            "required": (
                self.min_relevance if corroborated_wins else self.lone_evidence_relevance
            ),
            "sentences": len(scored),
        }

    def best_relevance(self, prompt: str) -> float:
        """返回该 Prompt 下**全文最高原始相关度**（不含短句惩罚）。

        Args:
            prompt: 完整 Prompt。

        Returns:
            最高原始相关度（0.0 ~ 1.0）。没有可用句子时返回 0.0。
        """
        return float(self.evidence_report(prompt)["best_score"])


# ============================================================
#  工厂
# ============================================================
def _build_chat_model() -> LLM:
    """构造在线 Chat 模型（OpenAI 兼容协议）。

    之所以用 `ChatOpenAI` 而不是各家 SDK：
    DeepSeek、智谱、Moonshot、Ollama、vLLM 全都提供
    **OpenAI 兼容端点**，一个类 + 改 `base_url` 就能通吃，
    不用为每家写一套适配代码。

    Returns:
        配置好的 ChatOpenAI 实例。

    Raises:
        RuntimeError: 缺少 API Key。
    """
    from langchain_openai import ChatOpenAI

    key = settings.resolved_llm_api_key
    if not key:
        raise RuntimeError("未配置任何 API Key")

    # ⚠️ 必须用 resolved_llm_model，不能直接用 settings.llm_model：
    #    后者在「只改了 LLM_PROVIDER、没改 LLM_MODEL」时会残留 gpt-4o-mini，
    #    拿去请求 DeepSeek/智谱会直接 400，最后表现为一个没有线索的 500。
    model = settings.resolved_llm_model
    if model != settings.llm_model:
        logger.warning(
            "LLM_MODEL='%s' 与 provider='%s' 不匹配，已自动改用 '%s'。"
            "建议在 .env 里显式写成：LLM_MODEL=%s",
            settings.llm_model, settings.llm_provider, model, model,
        )

    logger.info(
        "LLM 使用在线模型：provider=%s model=%s base_url=%s",
        settings.llm_provider, model, settings.resolved_base_url or "(SDK 默认)",
    )
    return ChatOpenAI(
        model=model,
        api_key=key,
        base_url=settings.resolved_base_url or None,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
        max_retries=2,  # 网络抖动自动重试，失败两次就不再拖时间
    )


def _explain_llm_error(exc: Exception) -> str:
    """把 SDK 抛出的异常翻译成**能照着做**的排查建议。

    这一步的价值：OpenAI SDK 的异常信息对没排查过的人来说几乎无用
    （比如 `Error code: 400 - {'error': {'message': 'Model Not Exist'}}`），
    用户看到的是一个 500。这里按 HTTP 状态码分流，直接告诉他该改哪一行。

    Args:
        exc: 调用 LLM 时捕获到的异常。

    Returns:
        多行中文排查建议（不含原始堆栈，堆栈在日志里）。
    """
    text = str(exc)
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    model = settings.resolved_llm_model
    provider = settings.llm_provider

    if status in (401, 403) or "AuthenticationError" in name or "invalid_api_key" in text:
        return (
            f"鉴权失败（{status or '401'}）：{provider} 不认这个 API Key。\n"
            "  → 检查 .env 的 LLM_API_KEY 是否填错/过期/多打了引号或空格；\n"
            "  → 确认这个 Key 是**从该提供商**申请的，不是别家的（DeepSeek 和 OpenAI 的 Key 不通用）。"
        )
    if status == 404 or "model_not_found" in text or "Model Not Exist" in text or "does not exist" in text:
        return (
            f"模型不存在：{provider} 上没有名为 '{model}' 的模型。\n"
            f"  → 在 .env 里把 LLM_MODEL 改成 {provider} 真实提供的模型名\n"
            "     （DeepSeek: deepseek-chat / deepseek-reasoner；"
            "智谱: glm-4-flash；Moonshot: moonshot-v1-8k）；\n"
            "  → 或者留空 LLM_MODEL，让程序按 LLM_PROVIDER 自动选默认模型。"
        )
    if status == 400:
        return (
            f"请求被拒绝（400）：{text[:200]}\n"
            f"  → 最常见的原因是 LLM_MODEL 与实际提供商不匹配（当前 model='{model}'）；\n"
            f"  → 其次是 LLM_BASE_URL 写错（当前 '{settings.resolved_base_url or 'SDK 默认'}'），"
            "注意多数提供商要带 /v1 后缀。"
        )
    if status == 402 or "insufficient" in text.lower() or "quota" in text.lower():
        return "账户余额/额度不足。请到对应平台充值或更换 Key。"
    if status == 429 or "rate limit" in text.lower():
        return "触发限流（429）。稍后重试，或降低请求频率 / 升级套餐。"
    if "timeout" in text.lower() or "timed out" in text.lower() or "ConnectError" in name:
        return _explain_connection_error()
    if "Connection error" in text or "APIConnectionError" in name:
        return _explain_connection_error()
    return f"调用失败：{type(exc).__name__}: {text[:300]}"


def _explain_connection_error() -> str:
    """「连不上」是最容易被误判为「配置错」的一类问题，这里专门处理。

    真实踩过的坑：用户配好了 `LLM_PROVIDER` / `LLM_MODEL` / `LLM_API_KEY`，
    仍然报 `APIConnectionError: Connection error.`。原因是**进程继承了
    HTTP_PROXY / HTTPS_PROXY 环境变量**，请求被导到一个不通的代理上 ——
    这和 API 配置毫无关系，但错误信息里完全看不出来。

    Returns:
        多行排查建议。
    """
    from app.utils.netcheck import detect_proxies

    base = settings.resolved_base_url or "(SDK 默认)"
    lines = [
        f"网络连接失败：无法访问 {base}。",
        "  注意：这**不是** API Key 或模型名的问题，是网络层到不了。",
    ]
    proxies = detect_proxies()
    if proxies:
        shown = "；".join(f"{k}={v}" for k, v in proxies.items())
        lines += [
            f"  ⭐ 检测到进程里有代理环境变量：{shown}",
            "     如果你的代理不通，请求就会被导到那里去，表现为 Connection error。",
            "     验证（Windows PowerShell，设置后再启动服务）：",
            '       $env:HTTP_PROXY=""; $env:HTTPS_PROXY=""; $env:ALL_PROXY=""',
            "     或把 API 域名加进 NO_PROXY。",
        ]
    else:
        lines += [
            "  → 未检测到代理变量。请检查：DNS 能否解析该域名、",
            "     公司网络 / 防火墙是否拦截、能否访问外网（试试手机热点）。",
        ]
    lines.append("  → 一键体检：python scripts/doctor.py")
    return "\n".join(lines)


@lru_cache(maxsize=2)
def get_llm() -> LLM:
    """获取 LLM 实例（进程内单例）。

    降级逻辑：只要在线模型构建失败（没 Key / 包没装 / base_url 写错），
    立刻回退到 `ExtractiveLLM`，**保证问答接口永远可用**。

    Returns:
        实现了 LangChain `LLM` 接口的实例。
    """
    if settings.llm_provider != "offline":
        try:
            return _build_chat_model()
        except Exception as exc:
            logger.warning(
                "在线 LLM 初始化失败（provider=%s）：%s —— 已降级为离线抽取模式。"
                "问答仍可用，但答案不经过语言模型生成。",
                settings.llm_provider, exc,
            )
    else:
        logger.info("LLM 配置为 offline，使用离线抽取模式")

    return ExtractiveLLM(top_sentences=4, max_answer_chars=900)


def is_offline(llm: LLM | None = None) -> bool:
    """判断当前是否处于离线抽取模式。

    Args:
        llm: 指定的 LLM 实例；None 时检查全局单例（不触发构建）。

    Returns:
        True 表示离线模式。
    """
    if llm is None:
        # 不调用 get_llm()，避免「只是问一下状态」就把模型拉起来
        return settings.is_offline_llm
    return isinstance(llm, ExtractiveLLM)


def describe_llm(llm: LLM | None = None) -> str:
    """返回 LLM 的可读描述（写进 API 响应的 `model` 字段）。"""
    if llm is None or isinstance(llm, ExtractiveLLM):
        return "offline-extractive"
    return getattr(llm, "model_name", None) or getattr(llm, "model", None) or type(llm).__name__


def reset_llm_singleton() -> None:
    """清空 LLM 单例（测试用）。"""
    get_llm.cache_clear()
