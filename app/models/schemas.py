"""Pydantic 数据模型：所有 API 的入参与出参都在这里定义。

好处：
- FastAPI 自动生成 OpenAPI 文档（/docs 页面）；
- 请求体自动校验，前端传错立刻返回 422 而不是 500；
- 前后端对接有「唯一事实来源」，改字段只改这里。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ============================================================
#  通用响应外壳
# ============================================================
class ApiResponse(BaseModel):
    """统一响应格式：{ code, message, data }。"""

    code: int = Field(default=0, description="0=成功，非0=业务错误")
    message: str = Field(default="ok", description="给人看的说明")
    data: Any | None = Field(default=None, description="业务数据")


# ============================================================
#  入库
# ============================================================
class IngestTextRequest(BaseModel):
    """直接提交一段文本入库（不经过文件上传）。"""

    text: str = Field(..., min_length=1, description="要入库的正文")
    title: str = Field(default="未命名文本", description="文档标题")
    collection: str = Field(default="default", description="集合名（多租户隔离）")


class IngestResult(BaseModel):
    """入库结果。"""

    doc_id: str = Field(..., description="文档唯一 ID（内容哈希）")
    filename: str
    pages: int = Field(default=0, description="原始页数/段落数")
    chunks: int = Field(..., description="切分后的片段数")
    chars: int = Field(default=0, description="正文字符数")
    elapsed_ms: int = Field(default=0, description="耗时（毫秒）")


class DocumentInfo(BaseModel):
    """知识库中一篇文档的概要。"""

    doc_id: str
    filename: str
    chunks: int
    source_type: str = Field(default="unknown", description="pdf/docx/txt/md")


class DocumentListResponse(BaseModel):
    total_docs: int
    total_chunks: int
    collection: str
    documents: list[DocumentInfo]


# ============================================================
#  问答
# ============================================================
class QueryRequest(BaseModel):
    """提问请求。"""

    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    session_id: str | None = Field(
        default=None, description="会话 ID；传了才会启用多轮对话记忆"
    )
    top_k: int | None = Field(default=None, ge=1, le=50, description="返回片段数")
    collection: str = Field(default="default")
    stream: bool = Field(default=False, description="是否流式返回（预留）")


class Citation(BaseModel):
    """答案中的一条引用（溯源信息）。"""

    index: int = Field(..., description="答案正文里 [n] 对应的序号")
    doc_id: str
    source: str = Field(..., description="文件名")
    page: int | None = Field(default=None, description="页码，从 1 开始")
    score: float = Field(default=0.0, description="综合相关性得分，越大越相关")
    snippet: str = Field(default="", description="被引用的原文片段")


class RetrievalDebug(BaseModel):
    """检索过程的可观测性数据——面试官问「混合检索怎么做的」时直接甩这个。"""

    vector_hits: int = 0
    bm25_hits: int = 0
    fused: int = 0
    reranked: int = 0
    used_hybrid: bool = False
    used_rerank: bool = False
    citation_fallback: bool = Field(
        default=False,
        description="true 表示模型忘了标注引用，系统兜底挂上了检索结果（引用不是模型主动标的）",
    )


class QueryResponse(BaseModel):
    """问答结果。"""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    latency_ms: int = 0
    model: str = Field(default="unknown", description="实际使用的模型名；离线模式为 offline-extractive")
    refused: bool = Field(
        default=False,
        description=(
            "true 表示系统判定「答不上来」并拒答，**没有编造答案**。"
            "调用方应读这个字段，而不是去匹配 answer 里的文字。"
        ),
    )
    refusal_reason: str | None = Field(
        default=None,
        description=(
            "拒答原因，仅在 refused=true 时有值："
            "no_retrieval（检索为空）/ model_refused（模型或离线闸门判定资料不足）"
            "/ empty_output（模型返回空，属异常）"
        ),
    )
    offline_mode: bool = Field(
        default=False, description="true 表示没接大模型，答案是抽取式拼接"
    )
    retrieval_debug: RetrievalDebug = Field(default_factory=RetrievalDebug)


# ============================================================
#  健康检查
# ============================================================
class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    app_name: str
    env: str
    version: str


class ReadyResponse(BaseModel):
    """就绪探针：容器编排用它决定是否把流量切进来。"""

    ready: bool
    chroma_ok: bool
    chunk_count: int
    embedding_provider: str
    llm_provider: str
    llm_offline: bool
    detail: str = ""

    # ---- LLM 配置体检 ----
    # 目的：让「provider 与 model 对不上」这类配置错误**在探针上就可见**，
    # 而不是等用户提问时收到一个没有线索的 HTTP 500。
    llm_model_configured: str = Field(default="", description=".env 里写的模型名")
    llm_model_effective: str = Field(default="", description="实际会发出去的模型名")
    llm_base_url: str = Field(default="", description="实际使用的 base_url")
    llm_config_ok: bool = Field(default=True, description="LLM 配置是否自洽")
    llm_config_warnings: list[str] = Field(default_factory=list, description="配置体检发现的问题")
