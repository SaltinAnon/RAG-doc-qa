"""FastAPI 依赖注入。

依赖注入（DI）解决的问题：路由函数不应该关心「服务是怎么来的」。
它只声明「我需要一个 DocumentIngestor」，由框架负责提供。

好处：
- **可测试**：测试里用 `app.dependency_overrides` 换成假实现，不用改业务代码；
- **不重复创建**：服务是单例，避免每次请求都重新加载模型/建数据库连接。
"""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from app.config import settings
from app.core.chain import RAGChain, get_rag_chain
from app.core.ingest import DocumentIngestor, get_ingestor
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
#  鉴权
# ============================================================
async def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> None:
    """校验 API Key。

    支持两种传法（选一种即可）：
    - 请求头 `X-API-Key: <key>`
    - 请求头 `Authorization: Bearer <key>`

    当 `.env` 里 `API_KEY` 为空时**自动关闭鉴权**（本地开发方便），
    此时会在启动日志里打一条 WARNING 提醒。

    Args:
        x_api_key: `X-API-Key` 头。
        authorization: `Authorization` 头。

    Raises:
        HTTPException: 401，密钥缺失或不匹配。
    """
    if not settings.auth_enabled:
        return

    provided = (x_api_key or "").strip()
    if not provided and authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()

    # 用 compare_digest 做**常量时间**比较，防止时序攻击侧信道泄露密钥
    if not provided or not secrets.compare_digest(provided, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key 无效或缺失。请在请求头带上 X-API-Key（或 Authorization: Bearer <key>）。",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ============================================================
#  服务实例
# ============================================================
def get_ingestor_dep() -> DocumentIngestor:
    """提供入库服务实例。"""
    return get_ingestor()


def get_chain_dep() -> RAGChain:
    """提供 RAG 链实例。"""
    return get_rag_chain()


# ============================================================
#  依赖别名
# ============================================================
# ⚠️ 这里有一个容易踩的坑：
#
#   不能写成 `IngestorDep = Depends(get_ingestor_dep)` 然后用于类型标注
#   `async def f(ingestor: IngestorDep)`。FastAPI 会把这个 Depends 实例
#   当成「Pydantic 字段类型」去解析，直接抛：
#       FastAPIError: Invalid args for response field!
#       Hint: check that Depends(get_ingestor_dep) is a valid Pydantic field type
#
#   正确做法是用 `Annotated[类型, Depends(...)]`：
#   类型信息给 IDE 和 Pydantic 看，依赖信息给 FastAPI 看，两者互不干扰。
#
#   （而路由级的 `dependencies=[...]` 参数仍然接受裸的 Depends 对象，
#     所以 AuthDep 保持原样。）
IngestorDep = Annotated[DocumentIngestor, Depends(get_ingestor_dep)]
ChainDep = Annotated[RAGChain, Depends(get_chain_dep)]

# 路由级依赖：用于 `APIRouter(dependencies=[AuthDep])`
AuthDep = Depends(verify_api_key)
