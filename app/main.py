"""FastAPI 应用装配入口。

启动方式：
    uvicorn app.main:app --reload
或：
    python -m app.main

访问 http://127.0.0.1:8000/docs 可以看到自动生成的交互式 API 文档
（这是 FastAPI 相对 Flask/Django 最大的开发体验优势，
面试演示时直接用它点几下就能跑通全流程）。
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api import ingest as ingest_routes
from app.api import query as query_routes
from app.config import settings
from app.models.schemas import ApiResponse, HealthResponse, ReadyResponse
from app.utils.logger import get_logger, new_request_id, request_id_ctx, setup_logging

logger = get_logger(__name__)

API_PREFIX = "/api/v1"


# ============================================================
#  生命周期
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用启动与关闭时的钩子（替代已废弃的 @app.on_event）。

    启动时做的事：
    1. 创建运行时目录（向量库、文档目录）；
    2. **探测**依赖是否可用，但**不因为探测失败就拒绝启动** ——
       服务能起来、能返回清晰的错误信息，比直接崩掉对使用者友好得多。
    """
    setup_logging(settings.log_level)
    settings.ensure_dirs()

    logger.info("=" * 64)
    logger.info("启动 %s v%s（env=%s）", settings.app_name, __version__, settings.app_env)
    logger.info("向量库目录：%s", settings.chroma_path)

    # 探测向量库
    try:
        from app.core.vectorstore import get_vector_store

        store = get_vector_store()
        ok, detail = store.health()
        logger.info("向量库：%s（%d chunks）", "正常" if ok else "异常", store.count())
        if not ok:
            logger.warning("向量库健康检查未通过：%s", detail)
    except Exception as exc:
        logger.error("向量库初始化失败：%s —— 服务仍会启动，但入库/检索会报错。", exc)

    # 提示降级状态
    if settings.is_offline_llm:
        logger.warning(
            "未配置大模型 API Key，当前为【离线抽取模式】：答案从原文抽取，"
            "不会经过大模型生成。配置 LLM_API_KEY 后自动切换。"
        )

    # 配置体检：把「provider 与 model/base_url/Key 对不上」这类问题
    # 在**启动时**就喊出来，而不是等用户提问时收到一个没有线索的 500
    for warn in settings.llm_config_warnings:
        logger.warning("LLM 配置体检：%s", warn)
    if not settings.auth_enabled:
        logger.warning(
            "API_KEY 为空，接口鉴权已关闭。**部署到公网前必须设置 API_KEY**，"
            "否则任何人都能上传文档并消耗你的模型额度。"
        )

    logger.info("就绪：接口文档 http://127.0.0.1:%d/docs", settings.api_port)
    logger.info("=" * 64)

    yield

    logger.info("服务关闭中……")


# ============================================================
#  应用实例
# ============================================================
app = FastAPI(
    title="企业级 RAG 智能文档问答系统",
    description=(
        "基于 **LangChain + ChromaDB + FastAPI** 的私有文档问答服务。\n\n"
        "**核心能力**：文档加载与切分 → 向量化入库 → 混合检索（向量 + BM25 + RRF）"
        " → 生成带引用溯源的答案 → 多轮对话记忆。\n\n"
        "**零成本可跑**：未配置大模型 API Key 时自动降级为离线抽取模式，"
        "完整链路依然可演示。"
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "入库", "description": "上传文档、查看与删除知识库内容"},
        {"name": "问答", "description": "RAG 问答、会话管理、运行时状态"},
        {"name": "运维", "description": "健康检查与就绪探针"},
    ],
)

# CORS：前端 Streamlit 与后端不同端口，必须显式放行。
# 生产环境请把 allow_origins 收窄成具体域名 —— 通配符 + 允许携带凭证是不安全的组合。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
#  中间件
# ============================================================
@app.middleware("http")
async def request_context_middleware(request: Request, call_next: Any) -> Any:
    """为每个请求分配 ID 并记录耗时。

    请求 ID 会写进日志的 `rid=` 字段，排查线上问题时
    可以用它把所有相关日志串起来（这是可观测性最基础的一环）。
    """
    rid = request.headers.get("X-Request-ID") or new_request_id()
    token = request_id_ctx.set(rid)
    started = time.perf_counter()

    try:
        response = await call_next(request)
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "%s %s → 耗时 %.0fms",
            request.method, request.url.path, elapsed_ms,
        )
        request_id_ctx.reset(token)

    response.headers["X-Request-ID"] = rid
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.0f}"
    return response


# ============================================================
#  统一异常处理
# ============================================================
@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """把 Pydantic 校验错误转成统一响应格式（默认是 FastAPI 的裸 422 结构）。"""
    errors = [
        {
            "field": ".".join(str(x) for x in e.get("loc", [])[1:]) or "(body)",
            "message": e.get("msg", ""),
        }
        for e in exc.errors()
    ]
    logger.warning("请求参数校验失败 path=%s errors=%s", request.url.path, errors)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ApiResponse(code=422, message="请求参数校验失败", data={"errors": errors}).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    """兜底异常处理：**绝不把 Python 堆栈直接吐给客户端**（泄露内部结构）。"""
    logger.error("未捕获异常 path=%s：%s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ApiResponse(
            code=500,
            message=f"服务器内部错误：{type(exc).__name__}",
            data={"hint": "详细堆栈请查看服务端日志（已带 request id，可据此检索）"},
        ).model_dump(),
    )


# ============================================================
#  路由
# ============================================================
app.include_router(ingest_routes.router, prefix=API_PREFIX)
app.include_router(ingest_routes.documents_router, prefix=API_PREFIX)
app.include_router(query_routes.router, prefix=API_PREFIX)


@app.get("/", tags=["运维"], summary="服务信息")
async def root() -> ApiResponse:
    """返回服务基本信息，方便快速确认「服务活着」。"""
    return ApiResponse(
        data={
            "name": settings.app_name,
            "version": __version__,
            "env": settings.app_env,
            "docs": "/docs",
            "health": "/healthz",
            "api_prefix": API_PREFIX,
        }
    )


@app.get("/healthz", tags=["运维"], response_model=HealthResponse, summary="存活探针")
async def healthz() -> HealthResponse:
    """存活探针：只判断进程是否活着，不探测依赖。

    为什么要和 `/readyz` 分开：容器编排里 liveness 探针失败会**重启容器**，
    如果 liveness 里检查数据库，数据库抖动会导致容器被反复重启（雪崩）。
    """
    return HealthResponse(app_name=settings.app_name, env=settings.app_env, version=__version__)


@app.get("/readyz", tags=["运维"], response_model=ReadyResponse, summary="就绪探针")
async def readyz() -> ReadyResponse:
    """就绪探针：检查向量库等关键依赖是否可用。"""
    chroma_ok = False
    chunk_count = 0
    detail = ""

    try:
        from app.core.vectorstore import get_vector_store

        store = get_vector_store()
        chroma_ok, detail = store.health()
        chunk_count = store.count()
    except Exception as exc:
        detail = str(exc)

    return ReadyResponse(
        ready=chroma_ok,
        chroma_ok=chroma_ok,
        chunk_count=chunk_count,
        embedding_provider=settings.embedding_provider,
        llm_provider=settings.llm_provider,
        llm_offline=settings.is_offline_llm,
        detail=detail,
        # 配置体检：探针返回里带一份，排查时不用再去翻日志
        llm_model_configured=settings.llm_model,
        llm_model_effective=settings.resolved_llm_model,
        llm_base_url=settings.resolved_base_url or "(SDK 默认)",
        llm_config_ok=not settings.llm_config_warnings,
        llm_config_warnings=settings.llm_config_warnings,
    )


# ============================================================
#  直接运行
# ============================================================
def main() -> None:
    """`python -m app.main` 的入口。"""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.app_env == "dev",
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
