"""LLM 连通性诊断命令行工具。

用法：

    python scripts/doctor.py              # 完整体检（含联网探测）
    python scripts/doctor.py --no-network # 只查配置，不联网

什么时候用它：

    当问答报「调用失败 / APIConnectionError / HTTP 500」时，
    先跑这个脚本，它会逐层告诉你问题出在**配置、代理、DNS、还是网络**，
    而不是让你对着一句英文异常干瞪眼。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 让脚本能直接 `python scripts/doctor.py` 运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.utils.netcheck import diagnose_llm, format_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM 配置与连通性诊断")
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="跳过 DNS/TCP/TLS 联网探测，只检查配置与环境变量",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 输出（便于脚本消费）",
    )
    args = parser.parse_args()

    report = diagnose_llm(do_network=not args.no_network)

    if args.json:
        import json

        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print()
        print(f"provider     : {settings.llm_provider}")
        print(f"model（配置） : {settings.llm_model or '(未填)'}")
        print(f"model（生效） : {settings.resolved_llm_model}")
        print(f"base_url     : {settings.resolved_base_url or '(未填)'}")
        print()
        print(format_report(report))
        print()

    return 0 if report.all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
