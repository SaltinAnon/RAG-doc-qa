"""为离线抽取模式标定「拒答闸门」参数。

## 为什么需要这个脚本

`ExtractiveLLM` 是「没有 API Key 也能跑通问答」的兜底实现。它不做生成，
只从检索到的片段里抽句子。问题是：**知识库里确实没有答案时，它也照样抽**——
这就是幻觉。所以加了一道闸门，用两个互补信号判定「资料够不够」：

| 信号 | 含义 |
|---|---|
| `best_score` | 句内 IDF 加权相关度（命中了多少「信息量」） |
| `best_matched` | 独立命中词数（有几个**互不相同**的查询词被命中） |

## 本脚本做的事

1. 在评测集上逐条算出 `(best_score, best_matched)`；
2. 按「可答 / 拒答」两类把两条分布打出来；
3. 输出**在哪些参数下能完全分开**，并给出建议值。

## 用法

    python scripts/make_sample_docs.py
    python scripts/ingest_cli.py --dir data/docs --reset
    python scripts/calibrate_gate.py

> ⚠️ 换 Embedding、换切分参数、换语料之后都应该重跑 —— 分布会变。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BASE_DIR, settings  # noqa: E402
from app.core.llm import ExtractiveLLM  # noqa: E402
from app.core.prompts import (  # noqa: E402
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    format_context,
    format_history,
)
from app.core.retriever import get_retriever  # noqa: E402
from app.utils.logger import setup_logging  # noqa: E402


def probe(case: dict, k: int, collection: str, llm: ExtractiveLLM) -> dict[str, object]:
    """算出一条用例的证据画像。"""
    retrieval = get_retriever().retrieve(case["question"], top_k=k, collection=collection)
    context, _ = format_context([c.doc for c in retrieval.chunks])
    prompt = SYSTEM_PROMPT + "\n\n" + USER_TEMPLATE.format(
        context=context, history=format_history([]), question=case["question"]
    )
    rep = llm.evidence_report(prompt)
    rep["label"] = "refusal" if case.get("expect_refusal") else "answerable"
    rep["question"] = case["question"]
    return rep


def _stats(xs: list[float]) -> str:
    """一组数的 min / mean / max 描述。"""
    if not xs:
        return "（无）"
    return f"min={min(xs):.3f}  mean={sum(xs) / len(xs):.3f}  max={max(xs):.3f}"


def main() -> int:
    """脚本入口。"""
    parser = argparse.ArgumentParser(description="标定离线抽取的拒答闸门")
    parser.add_argument("--eval-set", default="data/eval/eval_set.json")
    parser.add_argument("-k", "--top-k", type=int, default=5)
    parser.add_argument("--collection", default=None)
    args = parser.parse_args()

    setup_logging("ERROR")

    eval_path = BASE_DIR / args.eval_set
    if not eval_path.exists():
        print(f"❌ 找不到评测集：{eval_path}，请先运行 python scripts/make_sample_docs.py")
        return 1

    cases = json.loads(eval_path.read_text(encoding="utf-8")).get("cases", [])
    collection = args.collection or settings.collection_name

    from app.core.vectorstore import get_vector_store

    if get_vector_store().count(collection) == 0:
        print(f"❌ 集合「{collection}」为空，请先入库：python scripts/ingest_cli.py --dir data/docs --reset")
        return 1

    llm = ExtractiveLLM()
    rows = [probe(c, args.top_k, collection, llm) for c in cases]
    ans = [r for r in rows if r["label"] == "answerable"]
    ref = [r for r in rows if r["label"] == "refusal"]

    print(f"\n📐 标定：{len(cases)} 条用例 ｜ top_k={args.top_k} ｜ 集合={collection}\n")
    print(f"  {'score':>7} {'命中间':>5}  {'类型':<10}  问题")
    print("  " + "-" * 74)
    for r in sorted(rows, key=lambda x: (x["best_matched"], x["best_score"])):
        q = r["question"]
        q = q if len(q) <= 38 else q[:36] + ".."
        print(f"  {r['best_score']:7.3f} {r['best_matched']:5d}  {r['label']:<10}  {q}")

    # ---------- 两个信号各自的分布 ----------
    print("\n【信号 1：最高相关度 best_score】")
    print(f"  可答（{len(ans)} 条）: {_stats([r['best_score'] for r in ans])}")
    print(f"  拒答（{len(ref)} 条）: {_stats([r['best_score'] for r in ref])}")
    s_gap = min((r["best_score"] for r in ans), default=0) - max(
        (r["best_score"] for r in ref), default=0
    )
    print(f"  可分间隔 = {s_gap:+.3f}  {'✅ 可分' if s_gap > 0 else '❌ 重叠'}")

    print("\n【信号 2：独立命中词数 max_matched】")
    print(f"  可答命中数分布: {sorted(r['max_matched'] for r in ans)}")
    print(f"  拒答命中数分布: {sorted(r['max_matched'] for r in ref)}")
    m_gap = min((r["max_matched"] for r in ans), default=0) - max(
        (r["max_matched"] for r in ref), default=0
    )
    print(f"  可分间隔 = {m_gap:+d}  {'✅ 可分' if m_gap > 0 else '❌ 重叠'}")

    # ---------- 关键：两个分支各自的分数分布 ----------
    # 闸门规则是「两支各取最高分，任一达标即放行」，所以必须分别标定两个门槛。
    cur = ExtractiveLLM()
    n = cur.min_matched_terms

    cor_ans = sorted(r["corroborated_score"] for r in ans if r["max_matched"] >= n)
    cor_ref = sorted(r["corroborated_score"] for r in ref if r["max_matched"] >= n)
    lone_ans = sorted(r["lone_score"] for r in ans if r["max_matched"] < n)
    lone_ref = sorted(r["lone_score"] for r in ref if r["max_matched"] < n)

    print(f"\n【有印证分支（命中数 ≥ {n}）—— min_relevance 的标定依据】")
    print(f"  可答分支得分: [{cor_ans[0]:.3f} … {cor_ans[-1]:.3f}]  共 {len(cor_ans)} 条"
          if cor_ans else "  可答：无")
    print(f"  拒答分支得分: {[round(x, 3) for x in cor_ref] or '无'}")
    if cor_ans:
        print(f"  → min_relevance 应 ≤ {min(cor_ans):.3f}（当前 {cur.min_relevance}）")

    print(f"\n【孤证分支（命中数 < {n}）—— lone_evidence_relevance 的标定依据】")
    print(f"  可答分支得分: {[round(x, 3) for x in lone_ans] or '无'}")
    print(f"  拒答分支得分: {[round(x, 3) for x in lone_ref] or '无'}")
    if lone_ans and lone_ref:
        print(f"  → 阈值必须落在 ({max(lone_ref):.3f}, {min(lone_ans):.3f}] 之间"
              f"  ｜ 窗口宽度 {min(lone_ans) - max(lone_ref):.3f}")
        print(f"  → 建议取中点 {(max(lone_ref) + min(lone_ans)) / 2:.3f}"
              f"，当前设置 {cur.lone_evidence_relevance}")
    elif lone_ans:
        print("  ⚠️ 没有「孤证型拒答用例」→ 该阈值无从标定，只能凭经验设低一点以保安全。")
    elif lone_ref:
        print("  ⚠️ 没有「孤证型可答用例」→ 阈值越高越安全（当前 "
              f"{cur.lone_evidence_relevance}，孤证拒答最高 {max(lone_ref):.3f}），"
              "但要当心把「关键术语被句号拆开」的问题误拒。")
    else:
        print("  （两类都没有孤证用例，该分支未被触发）")

    # ---------- 模拟闸门 ----------
    def simulate(min_rel: float, lone_rel: float) -> tuple[float, float]:
        """模拟闸门，返回 (可答放行率, 拒答放行率)。

        与 `ExtractiveLLM._gate_passes()` 完全同构：
        任一支达标即放行。`min_matched_terms` 不参与模拟（它的取值是结构性的
        —— 「至少两个词互相印证」—— 所以分档分数已经按它算好了）。
        """
        def one(r: dict) -> bool:
            return r["corroborated_score"] >= min_rel or r["lone_score"] >= lone_rel

        ok = sum(1 for r in ans if one(r)) / max(len(ans), 1)
        bad = sum(1 for r in ref if one(r)) / max(len(ref), 1)
        return ok, bad

    print("\n【闸门模拟】规则：有印证分支只需低门槛，孤证分支需要高门槛，任一达标即放行")
    print(f"  当前参数：min_relevance={cur.min_relevance}  "
          f"lone_evidence_relevance={cur.lone_evidence_relevance}  "
          f"min_matched_terms={cur.min_matched_terms}")
    ok, bad = simulate(cur.min_relevance, cur.lone_evidence_relevance)
    print(f"    → 可答放行率 {ok * 100:.1f}%   拒答放行率（越小越好）{bad * 100:.1f}%")

    print("\n  不同参数组合（min_matched_terms 固定为 "
          f"{cur.min_matched_terms}）：")
    print(f"  {'min_rel':>8} {'lone_rel':>9}  {'可答放行':>8}  {'拒答放行':>8}")
    best_combo = None
    for min_rel in (0.08, 0.10, 0.12):
        for lone_rel in (0.12, 0.13, 0.14, 0.15, 0.18):
            ok, bad = simulate(min_rel, lone_rel)
            star = ""
            if ok == 1.0 and bad == 0.0:
                star = "  ✅"
                if best_combo is None:
                    best_combo = (min_rel, lone_rel)
            print(f"  {min_rel:>8.2f} {lone_rel:>9.2f}  "
                  f"{ok * 100:>7.1f}%  {bad * 100:>7.1f}%{star}")

    if best_combo:
        print(f"\n👉 完整可分的一组参数：min_relevance={best_combo[0]}、"
              f"lone_evidence_relevance={best_combo[1]}")
    else:
        print("\n⚠️ 没有一组参数能做到「可答全放行 + 拒答全拦住」——")
        print("   说明这两个信号在该语料上仍未完全分开，需要靠扩大语料或换 Embedding 解决。")

    print(
        "\n⚠️ 诚实提醒：示例文档只有 6 篇。这种规模下任何阈值都容易过拟合，\n"
        "   请务必换成自己的真实语料后重跑本脚本再定参。\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
