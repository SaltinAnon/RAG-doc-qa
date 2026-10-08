"""RAG 效果评测：把「简历上写的指标」变成「可以复现的数字」。

## 为什么必须做评测（而不是拍脑袋写 95%）

简历上写「检索召回率 Recall@5 达到 95%」这句话，面试官极可能追问：

> 「这个 95% 是怎么算的？测试集多大？金标准怎么标的？用哪个 Embedding？」
> 「换个 Embedding 会掉多少？」

**答不上来 = 简历注水。** 而如果你能说「27 条用例、金标准是答案来源文档、
Recall@5 = X%、MRR = Y、引用绑定准确率 = Z%，脚本在 `scripts/evaluate.py`，
`python scripts/evaluate.py` 就能复现」—— 这是完全不同量级的可信度。

> ⚠️ 诚实提醒：本项目示例文档只有 6 篇，任一种向量化方式都容易拿到很高的
> Recall。所以**不要把这个数字当成模型能力的证明**，而应该强调
> 「我建立了可复现的评测流程」。真正有说服力的是「换不同 Embedding 的对比」
> 和「引入混合检索前后 Recall 的提升幅度」——这正是本脚本 `--compare` 做的事。

## 指标定义（写清楚，面试能直接背）

| 指标 | 定义 | 考察什么 |
|---|---|---|
| **Recall@k** | top-k 检索结果里出现「答案来源文档」的比例 | 检索有没有把答案找回来（文件级） |
| **AnswerHit@k** | top-k 里存在「同时包含全部答案关键词」的片段的比例 | 检索有没有把**装着答案的片段**找回来（片段级，更严格） |
| **Precision@k** | top-k 里相关片段占的比例 | 检索结果里有多少是噪音 |
| **MRR** | 第一条相关结果名次的倒数的平均 | 相关结果排得够不够靠前 |
| **引用绑定准确率** | 返回的引用里命中金标准来源的比例 | 引用有没有张冠李戴 |
| **答案关键词命中率** | 答案里出现预期关键信息的比例 | 生成有没有答到点上 |
| **拒答准确率** | 知识库确实没有答案时，正确拒答的比例 | 幻觉抑制能力 |

> 💡 为什么要同时有 Recall@k 和 AnswerHit@k 两个召回指标：
> 小语料下文件级 Recall 会**饱和**（只要对的文件里随便哪个片段进了 top-k 就算命中），
> 掩盖真实差异。片段级指标才是有区分度、也更贴近 RAG 实际价值的那个。
> 这个发现过程见 README「效果数据」一节的说明。

## 用法

    # 全流程评测（检索 + 生成 + 引用）
    python scripts/evaluate.py

    # 只测检索（快，不花钱）
    python scripts/evaluate.py --mode retrieve

    # 对比「纯向量」vs「混合检索+重排」——面试时这个对比最能打
    python scripts/evaluate.py --compare
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BASE_DIR, settings  # noqa: E402
from app.core.chain import get_rag_chain  # noqa: E402
from app.core.retriever import get_retriever  # noqa: E402
from app.utils.logger import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

# 判定「拒答」的**兜底**关键词。
# ⚠️ 正常情况下不该用到它：`QueryResponse.refused` 已经是结构化字段。
#    这里保留是为了让评测脚本在「拿旧版本代码跑」时依旧能工作。
#    注意这些词偏宽（如「不包含」「未提及」），宽是有意的 —— 兜底宁可宽松，
#    真正的判定交给结构化字段。
REFUSAL_MARKERS = [
    "无法回答", "无法根据", "没有找到", "未找到", "没有相关",
    "资料中没有", "未提及", "不包含", "知识库中没有", "没有检索到",
]


@dataclass
class CaseResult:
    """单条用例的评测结果。"""

    question: str
    gold_source: str
    expect_refusal: bool

    retrieved_sources: list[str] = field(default_factory=list)
    hit_rank: int = -1                 # 第一个相关片段的名次（1 开始），-1 表示未命中
    recall_at_k: bool = False
    precision_at_k: float = 0.0
    reciprocal_rank: float = 0.0

    # ---- 片段级（比文件级更严格、更贴近 RAG 实际价值）----
    # 文件名级召回只要"对的文件里随便哪个片段"进了 top-k 就算命中；
    # 而 RAG 真正需要的是"**装着答案的那个片段**"被检索到。
    # 示例语料只有 6 篇、文件名级指标会饱和，所以这个指标才是区分度所在。
    answer_chunk_rank: int = -1        # 含全部 gold_keywords 的片段名次，-1 未命中
    answer_hit_at_k: bool = False
    answer_rr: float = 0.0

    answer: str = ""
    citation_sources: list[str] = field(default_factory=list)
    citation_hit: bool = False         # 引用里是否包含金标准来源
    keyword_hit: bool = False          # 答案里是否出现预期关键词
    refused: bool = False              # 答案是否表现为拒答
    latency_ms: int = 0
    error: str = ""


@dataclass
class Metrics:
    """汇总指标。"""

    total_cases: int = 0
    answerable_cases: int = 0
    refusal_cases: int = 0

    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    answer_hit_at_k: float = 0.0       # 片段级答案命中率
    answer_mrr: float = 0.0            # 答案片段 MRR
    citation_accuracy: float = 0.0
    keyword_hit_rate: float = 0.0
    refusal_accuracy: float = 0.0
    avg_latency_ms: float = 0.0
    errors: int = 0

    k: int = 5
    mode: str = "full"
    embedding: str = ""
    llm: str = ""


# ============================================================
#  单条评测
# ============================================================
def evaluate_case(case: dict, k: int, mode: str, collection: str) -> CaseResult:
    """评测单条用例。

    Args:
        case: 评测集里的一条。
        k: top-k。
        mode: `retrieve`（只测检索）或 `full`（含生成与引用）。
        collection: 集合名。

    Returns:
        `CaseResult`。**任何异常都记进 `error` 字段，不中断整个评测**
        —— 一条用例失败不应该让 27 条全部白跑。
    """
    question = case["question"]
    gold = case.get("gold_source") or ""
    expect_refusal = bool(case.get("expect_refusal"))
    result = CaseResult(question=question, gold_source=gold, expect_refusal=expect_refusal)

    # 一条用例可能有**多个合法来源**：同一段内容同时存在于 PDF 和 Markdown 里，
    # 这是真实企业知识库里的常见现象。只认一个来源会得出"检索失败"的错误结论。
    gold_list: list[str] = list(case.get("gold_sources") or ([gold] if gold else []))

    started = time.perf_counter()

    # ---------- 检索指标 ----------
    try:
        retrieval = get_retriever().retrieve(question, top_k=k, collection=collection)
        result.retrieved_sources = [
            str(c.metadata.get("source", "")) for c in retrieval.chunks
        ]

        # ---- 文件级：金标准文件有没有进 top-k ----
        if gold_list:
            for rank, src in enumerate(result.retrieved_sources, start=1):
                if src in gold_list:
                    result.hit_rank = rank
                    break

            result.recall_at_k = result.hit_rank > 0
            result.reciprocal_rank = 1.0 / result.hit_rank if result.hit_rank > 0 else 0.0
            relevant = sum(1 for s in result.retrieved_sources if s in gold_list)
            result.precision_at_k = relevant / max(len(result.retrieved_sources), 1)

        # ---- 片段级：装着答案的那个片段有没有进 top-k ----
        # 判据：某个片段同时包含该用例的**全部** gold_keywords（这些词就是答案本身）。
        # 关键词去空格后比较，避免「5 天」/「5天」这种排版差异造成漏判。
        kws = {re.sub(r"\s", "", kw).lower() for kw in (case.get("gold_keywords") or [])}
        kws.discard("")
        if kws:
            for rank, chunk in enumerate(retrieval.chunks, start=1):
                text = re.sub(r"\s", "", chunk.text).lower()
                if all(kw in text for kw in kws):
                    result.answer_chunk_rank = rank
                    break
            result.answer_hit_at_k = result.answer_chunk_rank > 0
            result.answer_rr = (
                1.0 / result.answer_chunk_rank if result.answer_chunk_rank > 0 else 0.0
            )
    except Exception as exc:
        result.error = f"检索失败：{exc}"
        logger.error("用例检索失败：%s", exc, exc_info=True)
        return result

    # ---------- 生成与引用指标 ----------
    if mode == "full":
        try:
            response = get_rag_chain().answer(question, top_k=k, collection=collection)
            result.answer = response.answer
            result.citation_sources = [c.source for c in response.citations]
            result.citation_hit = gold in result.citation_sources if gold else False

            keywords = case.get("gold_keywords") or []
            # 任一关键词命中即算命中（避免因表述差异误判）
            result.keyword_hit = (
                any(kw in response.answer for kw in keywords) if keywords else False
            )
            # 拒答判定：优先读结构化的 refused 字段；
            # 文本特征串只作为**降级兜底**（例如拿旧版本代码跑评测时）。
            structured = getattr(response, "refused", None)
            if structured is not None:
                result.refused = bool(structured)
            else:
                result.refused = any(m in response.answer for m in REFUSAL_MARKERS)
        except Exception as exc:
            result.error = f"问答失败：{exc}"
            logger.error("用例问答失败：%s", exc, exc_info=True)

    result.latency_ms = int((time.perf_counter() - started) * 1000)
    return result


# ============================================================
#  汇总
# ============================================================
def summarize(results: list[CaseResult], k: int, mode: str) -> Metrics:
    """把单条结果汇总成总体指标。

    Args:
        results: 所有用例结果。
        k: top-k。
        mode: 评测模式。

    Returns:
        `Metrics`。
    """
    m = Metrics(total_cases=len(results), k=k, mode=mode)

    answerable = [r for r in results if not r.expect_refusal]
    refusals = [r for r in results if r.expect_refusal]
    m.answerable_cases = len(answerable)
    m.refusal_cases = len(refusals)
    m.errors = sum(1 for r in results if r.error)

    if answerable:
        m.recall_at_k = sum(r.recall_at_k for r in answerable) / len(answerable)
        m.precision_at_k = sum(r.precision_at_k for r in answerable) / len(answerable)
        m.mrr = sum(r.reciprocal_rank for r in answerable) / len(answerable)
        m.answer_hit_at_k = sum(r.answer_hit_at_k for r in answerable) / len(answerable)
        m.answer_mrr = sum(r.answer_rr for r in answerable) / len(answerable)

    if mode == "full":
        if answerable:
            m.citation_accuracy = sum(r.citation_hit for r in answerable) / len(answerable)
            m.keyword_hit_rate = sum(r.keyword_hit for r in answerable) / len(answerable)
        if refusals:
            # 拒答用例：答案里出现拒答话术 = 正确
            m.refusal_accuracy = sum(r.refused for r in refusals) / len(refusals)

    latencies = [r.latency_ms for r in results if not r.error]
    m.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

    # 从链路里取实际使用的模型信息
    try:
        desc = get_rag_chain().describe()
        m.embedding = desc.get("embedding", "")
        m.llm = desc.get("llm", "")
    except Exception:
        pass

    return m


# ============================================================
#  输出
# ============================================================
def print_report(m: Metrics, results: list[CaseResult]) -> None:
    """在终端打印可读报告。"""
    line = "=" * 78
    print(f"\n{line}")
    print(f"  RAG 评测报告  ｜  模式={m.mode} ｜  top_k={m.k}  ｜  用例数={m.total_cases}")
    print(f"  Embedding={m.embedding}  ｜  LLM={m.llm}")
    print(line)

    print(f"\n【核心指标】")
    print(f"  Recall@{m.k}（检索召回率）        : {m.recall_at_k * 100:6.2f}%"
          f"   ← 简历上可以写的数字（文件级）")
    print(f"  Precision@{m.k}（检索准确率）     : {m.precision_at_k * 100:6.2f}%")
    print(f"  MRR（平均倒数名次）              : {m.mrr:6.4f}")
    print(f"  AnswerHit@{m.k}（片段级命中率）    : {m.answer_hit_at_k * 100:6.2f}%"
          f"   ← 更严格：答案片段本身有没有被检索到")
    print(f"  答案片段 MRR                     : {m.answer_mrr:6.4f}")

    if m.mode == "full":
        print(f"  引用绑定准确率                   : {m.citation_accuracy * 100:6.2f}%"
              f"   ← 引用有没有张冠李戴")
        print(f"  答案关键词命中率                 : {m.keyword_hit_rate * 100:6.2f}%")
        print(f"  拒答准确率（幻觉抑制）           : {m.refusal_accuracy * 100:6.2f}%")

    print(f"\n【运行信息】")
    print(f"  可回答用例 / 拒答用例            : {m.answerable_cases} / {m.refusal_cases}")
    print(f"  平均单次耗时                     : {m.avg_latency_ms:.0f} ms")
    print(f"  失败用例数                       : {m.errors}")

    # ---- 逐条明细 ----
    print(f"\n【逐条明细】")
    print(f"  {'结果':<6}{'RR':>6}{'片段':>6}{'引用':>6}{'关键词':>8}  {'问题'}")
    print("  " + "-" * 78)

    for r in results:
        if r.error:
            flag = "ERR"
        elif r.expect_refusal:
            flag = "PASS" if r.refused else "FAIL"
        else:
            flag = "PASS" if r.recall_at_k else "MISS"

        rr = f"{r.reciprocal_rank:.2f}" if not r.expect_refusal else "—"
        chunk = f"#{r.answer_chunk_rank}" if r.answer_chunk_rank > 0 else "✗"
        if r.expect_refusal:
            chunk = "—"
        cite = ("✓" if r.citation_hit else "✗") if not r.expect_refusal and m.mode == "full" else "—"
        kw = ("✓" if r.keyword_hit else "✗") if r.gold_source and m.mode == "full" else "—"
        q = r.question if len(r.question) <= 34 else r.question[:32] + ".."
        print(f"  {flag:<6}{rr:>6}{chunk:>6}{cite:>6}{kw:>8}  {q}")

    # ---- 失败原因 ----
    failures = [r for r in results if not r.expect_refusal and not r.recall_at_k]
    if failures:
        print(f"\n【文件级召回失败】（{len(failures)} 条）")
        for r in failures:
            print(f"  ✗ {r.question}")
            print(f"    期望来源：{r.gold_source}")
            print(f"    实际返回：{', '.join(r.retrieved_sources[:5]) or '(空)'}")
            if r.error:
                print(f"    错误：{r.error[:160]}")

    # 文件级命中了、但"装着答案的片段"没进 top-k —— 比召回失败更隐蔽
    partial = [
        r
        for r in results
        if not r.expect_refusal and not r.error and r.recall_at_k and not r.answer_hit_at_k
    ]
    if partial:
        print(f"\n【片段级命中失败】（{len(partial)} 条：文件对了，但答案片段没进 top-{m.k}）")
        for r in partial:
            print(f"  ⚠ {r.question}")
            print(f"    金标准文件：{r.gold_source}（已在检索结果里）")
            print(f"    含答案关键词的片段未进入 top-{m.k}，实际返回：{', '.join(r.retrieved_sources[:5])}")

    print(f"\n{line}\n")


def save_reports(m: Metrics, results: list[CaseResult], out_dir: Path) -> Path:
    """把结果写成 JSON（机器可读）与 Markdown（贴 README 用）两份。

    Args:
        m: 汇总指标。
        results: 逐条结果。
        out_dir: 输出目录。

    Returns:
        生成的 JSON 报告路径。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "metrics": asdict(m),
        "cases": [asdict(r) for r in results],
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# RAG 评测报告",
        "",
        f"- 生成时间：{payload['generated_at']}",
        f"- 模式：`{m.mode}` ｜ top_k = {m.k} ｜ 用例数 = {m.total_cases}",
        f"- Embedding：`{m.embedding}` ｜ LLM：`{m.llm}`",
        "",
        "## 核心指标",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| Recall@{m.k} | **{m.recall_at_k * 100:.2f}%** |",
        f"| Precision@{m.k} | {m.precision_at_k * 100:.2f}% |",
        f"| MRR | {m.mrr:.4f} |",
        f"| AnswerHit@{m.k}（片段级） | **{m.answer_hit_at_k * 100:.2f}%** |",
        f"| 答案片段 MRR | {m.answer_mrr:.4f} |",
    ]
    if m.mode == "full":
        md += [
            f"| 引用绑定准确率 | **{m.citation_accuracy * 100:.2f}%** |",
            f"| 答案关键词命中率 | {m.keyword_hit_rate * 100:.2f}% |",
            f"| 拒答准确率 | {m.refusal_accuracy * 100:.2f}% |",
        ]
    md += [
        f"| 平均耗时 | {m.avg_latency_ms:.0f} ms |",
        "",
        "## 逐条明细",
        "",
        "| 结果 | RR | 答案片段名次 | 引用 | 关键词 | 问题 |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        if r.error:
            flag = "ERR"
        elif r.expect_refusal:
            flag = "PASS" if r.refused else "FAIL"
        else:
            flag = "PASS" if r.recall_at_k else "MISS"
        rr = f"{r.reciprocal_rank:.2f}" if not r.expect_refusal else "—"
        chunk = ("—" if r.expect_refusal else (f"#{r.answer_chunk_rank}" if r.answer_chunk_rank > 0 else "✗"))
        cite = ("✓" if r.citation_hit else "✗") if not r.expect_refusal and m.mode == "full" else "—"
        kw = ("✓" if r.keyword_hit else "✗") if r.gold_source and m.mode == "full" else "—"
        md.append(f"| {flag} | {rr} | {chunk} | {cite} | {kw} | {r.question} |")

    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    logger.info("报告已写入：%s / %s", json_path, md_path)
    return json_path


# ============================================================
#  对比实验
# ============================================================
def run_compare(cases: list[dict], k: int, collection: str) -> None:
    """对比「纯向量检索」与「混合检索 + 重排序」的召回差异。

    这是整个评测脚本里**最有说服力**的部分：
    它证明了「我加的 BM25 和重排不是炫技，而是带来了可量化的提升」。

    Args:
        cases: 评测集用例。
        k: top-k。
        collection: 集合名。
    """
    from app.config import get_settings

    answerable = [c for c in cases if not c.get("expect_refusal")]
    configs = [
        ("纯向量检索", {"hybrid_enabled": False, "rerank_enabled": False}),
        ("向量 + BM25（RRF 融合）", {"hybrid_enabled": True, "rerank_enabled": False}),
        ("向量 + BM25 + 重排序（本项目默认）", {"hybrid_enabled": True, "rerank_enabled": True}),
    ]

    s = get_settings()
    original = (s.hybrid_enabled, s.rerank_enabled)
    rows: list[tuple[str, float, float, float, float]] = []

    try:
        for name, cfg in configs:
            s.hybrid_enabled = cfg["hybrid_enabled"]
            s.rerank_enabled = cfg["rerank_enabled"]
            # 清掉检索器单例，让它用新配置重建
            from app.core.retriever import reset_retriever_singleton

            reset_retriever_singleton()

            results = [evaluate_case(c, k, "retrieve", collection) for c in answerable]
            m = summarize(results, k, "retrieve")
            rows.append((name, m.recall_at_k, m.mrr, m.answer_hit_at_k, m.answer_mrr))
            print(
                f"  {name:<32} "
                f"文件级 Recall@{k}={m.recall_at_k * 100:6.2f}%  "
                f"片段级 Hit@{k}={m.answer_hit_at_k * 100:6.2f}%  "
                f"片段MRR={m.answer_mrr:.4f}"
            )
    finally:
        # 恢复配置，避免影响后续调用
        s.hybrid_enabled, s.rerank_enabled = original
        from app.core.retriever import reset_retriever_singleton

        reset_retriever_singleton()

    print("\n【对比结论】")
    print("  说明：示例语料（6 篇）下**文件级**召回会饱和，区分度体现在**片段级**指标上。")
    if rows:
        base = rows[0]
        for name, recall, mrr, hit, amrr in rows[1:]:
            print(
                f"  {name}：\n"
                f"      文件级 Recall@{k} 变化 {(recall - base[1]) * 100:+.2f} 个百分点\n"
                f"      片段级 AnswerHit@{k} 变化 {(hit - base[3]) * 100:+.2f} 个百分点\n"
                f"      答案片段 MRR 变化 {amrr - base[4]:+.4f}"
            )
    print(
        f"\n  ⚠️ 说明：示例文档只有 6 篇、可答用例 {len(answerable)} 条，"
        "绝对数值偏高且区分度有限。\n"
        "     要让这个对比有说服力，请换成你自己的真实文档（几十篇以上）和评测集。"
    )


# ============================================================
#  入口
# ============================================================
def main() -> int:
    """脚本入口。"""
    parser = argparse.ArgumentParser(
        description="RAG 系统效果评测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--eval-set", default="data/eval/eval_set.json", help="评测集路径")
    parser.add_argument("--mode", choices=["retrieve", "full"], default="full", help="评测模式")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="top-k，默认 5")
    parser.add_argument("--collection", default=None, help="集合名")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（调试用）")
    parser.add_argument("--compare", action="store_true", help="跑多配置对比实验")
    parser.add_argument("--out", default="data/eval", help="报告输出目录")
    args = parser.parse_args()

    setup_logging("WARNING")  # 评测时压掉 INFO 日志，让报告更干净

    eval_path = BASE_DIR / args.eval_set
    if not eval_path.exists():
        print(
            f"❌ 找不到评测集：{eval_path}\n"
            "请先运行：python scripts/make_sample_docs.py"
        )
        return 1

    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    cases: list[dict] = payload.get("cases", [])
    if args.limit:
        cases = cases[: args.limit]

    collection = args.collection or settings.collection_name

    # 前置检查：知识库为空时直接提示，而不是跑出 0%
    from app.core.vectorstore import get_vector_store

    store = get_vector_store()
    count = store.count(collection)
    if count == 0:
        print(
            f"❌ 集合「{collection}」是空的，无法评测。\n"
            "请先执行：\n"
            "   python scripts/make_sample_docs.py\n"
            "   python scripts/ingest_cli.py --dir data/docs --reset"
        )
        return 1

    print(f"\n📊 开始评测：{len(cases)} 条用例 ｜ 集合={collection}（{count} 个片段）")

    # ---- 对比模式 ----
    if args.compare:
        print(f"\n【对比实验】top_k={args.top_k}")
        run_compare(cases, args.top_k, collection)
        return 0

    # ---- 常规评测 ----
    print(f"模式={args.mode}，正在逐条执行……\n")
    results: list[CaseResult] = []
    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case['question'][:44]:<46}", end="\r")
        results.append(evaluate_case(case, args.top_k, args.mode, collection))

    m = summarize(results, args.top_k, args.mode)
    print_report(m, results)
    save_reports(m, results, BASE_DIR / args.out)

    # ---- 判定是否达标（供 CI 使用）----
    # 这里设两道门禁。为什么不止一道：
    #   召回率低 → 「答不出来」（体验差）
    #   拒答率低 → 「答错了还不知道」（**幻觉，更严重**）
    # 幻觉是 RAG 系统最致命的风险，必须单独守。
    failures: list[str] = []

    recall_threshold = 0.6
    if m.recall_at_k < recall_threshold:
        failures.append(
            f"Recall@{args.top_k} = {m.recall_at_k * 100:.2f}% < "
            f"{recall_threshold * 100:.0f}%（请检查入库数据与检索配置）"
        )

    # 只在「完整模式 + 有拒答用例」时才检查 —— 检索模式不产出答案，无从判定
    refusal_threshold = 0.6
    if m.mode == "full" and m.refusal_cases > 0 and m.refusal_accuracy < refusal_threshold:
        failures.append(
            f"拒答准确率 = {m.refusal_accuracy * 100:.2f}% < "
            f"{refusal_threshold * 100:.0f}%（知识库无答案时出现幻觉，"
            f"请检查 ExtractiveLLM 的拒答闸门参数，可跑 scripts/calibrate_gate.py 重新标定）"
        )

    if failures:
        for f in failures:
            print(f"⚠️  {f}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
