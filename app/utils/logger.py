"""结构化日志。

为什么不用 `print`：
- 生产环境日志要能按级别过滤、能带请求 ID 串起一次调用链；
- 容器里 stdout 结构化输出可以直接被采集系统吃掉。

用法：
    from app.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("入库完成 doc_id=%s chunks=%d", doc_id, chunks)
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar

# 请求级上下文：中间件里 set，日志格式化器里读。
# 用 ContextVar 而不是全局变量，是为了在异步/多线程下互不干扰。
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | rid=%(request_id)s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _RequestIdFilter(logging.Filter):
    """把 ContextVar 里的 request_id 注入 LogRecord。"""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.request_id = request_id_ctx.get()
        return True


_configured = False


def setup_logging(level: str = "INFO") -> None:
    """初始化根 logger。重复调用是安全的（幂等）。"""
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    handler.addFilter(_RequestIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # 第三方库太吵，压到 WARNING
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "openai", "watchfiles"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """获取带请求上下文的 logger。"""
    setup_logging()
    return logging.getLogger(name)


def new_request_id() -> str:
    """生成短请求 ID（8 位足够，日志里更好读）。"""
    return uuid.uuid4().hex[:8]
