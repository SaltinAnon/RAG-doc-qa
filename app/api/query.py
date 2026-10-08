"""问答相关路由。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import AuthDep, ChainDep
from app.core.chain import LLMInvocationError
from app.models.schemas import ApiResponse, QueryRequest
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["问答"], dependencies=[AuthDep])


@router.post(
    "/query",
    response_model=ApiResponse,
    summary="提问（RAG 问答）",
    description=(
        "基于已入库的文档回答问题，返回**带引用溯源**的答案。\n\n"
        "- 传 `session_id` 启用多轮对话记忆（同一 session 内会引用历史）；\n"
        "- 未配置大模型 Key 时自动进入**离线抽取模式**，响应里 `offline_mode=true`；\n"
        "- 响应中的 `retrieval_debug` 展示了混合检索各阶段的命中数，便于调优。"
    ),
)
async def query(payload: QueryRequest, chain: ChainDep) -> ApiResponse:
    """执行一次 RAG 问答。

    Args:
        payload: 提问请求。
        chain: RAG 链（依赖注入）。

    Returns:
        统一响应，`data` 为 `QueryResponse`。

    Raises:
        HTTPException: 400（问题为空）、500（模型调用失败）、503（知识库为空）。
    """
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    # 知识库为空时给出明确指导，而不是返回一个「无法回答」的黑盒
    from app.core.vectorstore import get_vector_store

    if get_vector_store().count(payload.collection) == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"集合「{payload.collection}」中没有任何文档，无法回答。\n"
                "请先调用 POST /api/v1/ingest/file 上传文档，"
                "或运行 `python scripts/ingest_cli.py --dir data/docs` 批量入库示例文档。"
            ),
        )

    try:
        result = chain.answer(
            question,
            session_id=payload.session_id,
            top_k=payload.top_k,
            collection=payload.collection,
        )
    except LLMInvocationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ApiResponse(data=result.model_dump())


@router.post(
    "/sessions/{session_id}/reset",
    response_model=ApiResponse,
    summary="清空会话记忆",
    description="把指定 session 的对话历史清空，用于开始一个全新的话题。",
)
async def reset_session(session_id: str, chain: ChainDep) -> ApiResponse:
    """清空某个会话的历史。

    Args:
        session_id: 会话 ID。
        chain: RAG 链。

    Returns:
        统一响应，`data.existed` 表示该会话此前是否存在。
    """
    existed = chain.memory.reset(session_id)
    return ApiResponse(data={"session_id": session_id, "existed": existed})


@router.get(
    "/sessions/stats",
    response_model=ApiResponse,
    summary="查看会话记忆使用情况",
)
async def session_stats(chain: ChainDep) -> ApiResponse:
    """查看当前活跃会话数与总轮数（运维观测用）。"""
    return ApiResponse(data=chain.memory.stats())


@router.get(
    "/config/status",
    response_model=ApiResponse,
    summary="查看当前运行时配置与降级状态",
    description="前端用它决定是否显示「离线模式」提示条。",
)
async def config_status(chain: ChainDep) -> ApiResponse:
    """返回 Embedding / LLM / 检索参数的运行时状态。"""
    return ApiResponse(data=chain.describe())
