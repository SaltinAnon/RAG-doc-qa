"""命令行批量入库工具。

用法：
    # 入库 data/docs 下所有文档
    python scripts/ingest_cli.py --dir data/docs

    # 入库单个文件，指定集合
    python scripts/ingest_cli.py --file 员工手册.pdf --collection hr

    # 先清空该集合再入库（重建索引）
    python scripts/ingest_cli.py --dir data/docs --reset

    # 只看看会入库什么，不实际写入
    python scripts/ingest_cli.py --dir data/docs --dry-run

这个脚本存在的意义：**让「入库」这个动作可以脱离 Web 界面独立执行**。
在真实项目里，批量导入通常发生在数据初始化阶段或由定时任务触发，
不该需要人点按钮。这也是面试时可以强调的「接口分层」意识。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.core.ingest import DocumentIngestor, IngestError  # noqa: E402
from app.core.loaders import SUPPORTED_EXTENSIONS, load_document  # noqa: E402
from app.core.splitter import split_documents  # noqa: E402
from app.core.vectorstore import get_vector_store  # noqa: E402
from app.utils.logger import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


def dry_run(paths: list[Path]) -> None:
    """预演：只做加载和切分，不写向量库，也不调用 Embedding 接口。

    这个模式非常有用：
    - **不花钱**（不调用 Embedding API）；
    - **快**（省掉向量化）；
    - 能提前发现「切分粒度是否合理」「有没有文档解析失败」。

    Args:
        paths: 待预演的文件列表。
    """
    print("\n" + "=" * 78)
    print(f"{'文件名':<32}{'单元':>6}{'chunk':>7}{'字符数':>10}{'平均长度':>10}")
    print("-" * 78)

    total_chunks = total_chars = 0
    failures: list[tuple[str, str]] = []

    for p in paths:
        try:
            docs = load_document(p)
            chunks = split_documents(docs)
            chars = sum(len(c.page_content) for c in chunks)
            avg = chars // max(len(chunks), 1)
            total_chunks += len(chunks)
            total_chars += chars
            name = p.name if len(p.name) <= 30 else p.name[:28] + ".."
            print(f"{name:<32}{len(docs):>6}{len(chunks):>7}{chars:>10}{avg:>10}")
        except Exception as exc:
            failures.append((p.name, str(exc)))
            name = p.name if len(p.name) <= 30 else p.name[:28] + ".."
            print(f"{name:<32}{'—':>6}{'FAIL':>7}  {str(exc)[:30]}")

    print("-" * 78)
    print(f"合计：{len(paths)} 个文件，{total_chunks} 个 chunk，{total_chars} 字符")
    if failures:
        print(f"\n⚠️  {len(failures)} 个文件解析失败：")
        for name, err in failures:
            print(f"   - {name}: {err}")

    # 给出切分质量建议
    if total_chunks:
        avg_size = total_chars // total_chunks
        print(f"\n平均 chunk 长度 {avg_size} 字符（配置值 CHUNK_SIZE={settings.chunk_size}）")
        if avg_size < settings.chunk_size * 0.5:
            print("   建议：平均长度明显偏小，说明文档里短段落很多，"
                  "可以适当调大 CHUNK_SIZE 或减少分隔符优先级。")
        elif avg_size > settings.chunk_size * 1.1:
            print("   建议：平均长度超过配置值，说明存在无法按分隔符切分的超长文本块"
                  "（如无标点的长表格），可考虑开启按 token 硬切。")
        else:
            print("   切分粒度合理。")
    print("=" * 78 + "\n")


def main() -> int:
    """脚本入口。"""
    parser = argparse.ArgumentParser(
        description="RAG 项目命令行入库工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir", help="要入库的目录")
    src.add_argument("--file", help="要入库的单个文件")

    parser.add_argument("--collection", default=None, help="集合名（默认读 .env）")
    parser.add_argument("--reset", action="store_true", help="入库前先清空该集合（危险）")
    parser.add_argument("--force", action="store_true", help="已存在的文档也重新入库")
    parser.add_argument("--dry-run", action="store_true", help="只预演，不实际写入")
    args = parser.parse_args()

    setup_logging(settings.log_level)
    settings.ensure_dirs()

    collection = args.collection or settings.collection_name

    # 收集待处理文件
    if args.file:
        paths = [Path(args.file).resolve()]
        if not paths[0].is_file():
            print(f"❌ 文件不存在：{paths[0]}")
            return 1
    else:
        directory = Path(args.dir).resolve()
        if not directory.is_dir():
            print(f"❌ 目录不存在：{directory}")
            return 1
        paths = sorted(
            p for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    if not paths:
        print(f"⚠️  没有找到任何受支持的文件（支持：{', '.join(sorted(SUPPORTED_EXTENSIONS))}）")
        return 1

    print(f"\n📂 待处理 {len(paths)} 个文件 → 集合「{collection}」")

    # ---- 预演模式 ----
    if args.dry_run:
        dry_run(paths)
        return 0

    # ---- 重置集合 ----
    store = get_vector_store()
    if args.reset:
        existed = store.count(collection)
        if existed:
            ans = input(
                f"⚠️  即将清空集合「{collection}」中的 {existed} 个 chunk，此操作不可撤销。"
                "确认请输入 yes："
            ).strip().lower()
            if ans != "yes":
                print("已取消。")
                return 0
        store.drop_collection(collection)
        print(f"🧹 集合「{collection}」已重置")

    # ---- 执行入库 ----
    ingestor = DocumentIngestor(store)
    started = time.perf_counter()
    ok = fail = skipped = 0
    total_chunks = 0
    failures: list[tuple[str, str]] = []

    print("-" * 78)
    for i, p in enumerate(paths, 1):
        prefix = f"[{i}/{len(paths)}]"
        try:
            if p.suffix.lower() in {".doc", ".docx"} and args.file:
                # 单文件模式走 ingest_path（批量模式已按 doc_id 分组处理）
                res = ingestor.ingest_path(p, collection=collection, skip_existing=not args.force)
            else:
                res = ingestor.ingest_path(p, collection=collection, skip_existing=not args.force)

            if res.chunks == 0:
                skipped += 1
                print(f"{prefix} ⏭️  {p.name}（内容未变，跳过）")
            else:
                ok += 1
                total_chunks += res.chunks
                print(f"{prefix} ✅ {p.name} → {res.chunks} 个片段（{res.elapsed_ms}ms）")
        except IngestError as exc:
            fail += 1
            failures.append((p.name, str(exc)))
            print(f"{prefix} ❌ {p.name}：{str(exc)[:100]}")
        except Exception as exc:
            fail += 1
            failures.append((p.name, str(exc)))
            print(f"{prefix} ❌ {p.name}：{type(exc).__name__}: {str(exc)[:100]}")

    elapsed = time.perf_counter() - started
    print("-" * 78)
    print(
        f"\n📊 入库结果：成功 {ok} ｜ 跳过 {skipped} ｜ 失败 {fail} ｜ "
        f"新增 {total_chunks} 个片段 ｜ 耗时 {elapsed:.1f}s"
    )
    print(f"📚 集合「{collection}」当前共 {store.count(collection)} 个片段，"
          f"{len(store.list_documents(collection))} 篇文档")

    if failures:
        print("\n失败明细：")
        for name, err in failures:
            print(f"  - {name}\n    {err[:300]}")

    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
