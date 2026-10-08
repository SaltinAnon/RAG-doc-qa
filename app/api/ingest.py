"""入库相关路由：上传文件、提交文本、查看/删除文档。"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status

from app.api.deps import AuthDep, IngestorDep
from app.config import settings
from app.core.ingest import IngestError
from app.core.loaders import SUPPORTED_EXTENSIONS, UnsupportedFileTypeError, EmptyDocumentError
from app.models.schemas import (
    ApiResponse,
    DocumentListResponse,
    IngestResult,
    IngestTextRequest,
)
from app.utils.logger import get_logger
from app.utils.text import safe_filename

logger = get_logger(__name__)

router = APIRouter(prefix="/ingest", tags=["入库"], dependencies=[AuthDep])


@router.post(
    "/file",
    response_model=ApiResponse,
    summary="上传文件并入库",
    description=(
        "支持 PDF / Word(.docx) / TXT / Markdown / CSV。"
        "同一份文件重复上传是**幂等**的（按内容哈希去重），不会产生重复向量。"
    ),
)
async def ingest_file(
    ingestor: IngestorDep,
    file: UploadFile = File(..., description="要入库的文件"),
    collection: str = Form(default="default", description="集合名（多租户隔离）"),
    skip_existing: bool = Form(default=False, description="已存在时跳过（省向量化开销）"),
) -> ApiResponse:
    """上传一个文件并入库。

    Args:
        ingestor: 入库服务（依赖注入）。
        file: 上传的文件。
        collection: 目标集合。
        skip_existing: 是否跳过已存在的同内容文档。

    Returns:
        统一响应，`data` 为 `IngestResult`。

    Raises:
        HTTPException: 400（文件类型/内容问题）、413（文件过大）、500（内部错误）。
    """
    filename = safe_filename(file.filename or "unnamed")
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"不支持的文件类型「{ext or '无后缀'}」。"
                f"当前支持：{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    # 读入内存并检查大小。用 tempfile 而不是直接存路径，
    # 是为了兼容「上传文件本来就不在本地磁盘」的场景（如云存储直传）。
    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"读取上传文件失败：{exc}") from exc

    size_mb = len(content) / 1024 / 1024
    if size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"文件 {size_mb:.1f}MB 超过上限 {settings.max_file_size_mb}MB。"
                "请拆分后再上传，或调大 .env 里的 MAX_FILE_SIZE_MB。"
            ),
        )
    if not content:
        raise HTTPException(status_code=400, detail="上传的文件是空的。")

    suffix = ext or ".txt"
    tmp_path: Path | None = None
    try:
        # delete=False：Windows 上文件被 open 时无法删除，必须手动清理
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        result = ingestor.ingest_path(tmp_path, collection=collection, skip_existing=skip_existing)
        # 让返回的 filename 保持用户原始文件名，而不是临时文件名
        result.filename = filename
        return ApiResponse(data=result.model_dump())

    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except IngestError as exc:
        logger.error("入库失败 %s：%s", filename, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                # Windows 偶尔会因为句柄未释放删不掉，留个临时文件不影响功能
                logger.warning("临时文件清理失败（可忽略）：%s", tmp_path)


@router.post(
    "/text",
    response_model=ApiResponse,
    summary="提交一段文本入库",
    description="不走文件上传，直接把文本内容入库，适合从网页/剪贴板来的内容。",
)
async def ingest_text(payload: IngestTextRequest, ingestor: IngestorDep) -> ApiResponse:
    """把一段文本入库。

    Args:
        payload: 文本与标题。
        ingestor: 入库服务。

    Returns:
        统一响应，`data` 为 `IngestResult`。

    Raises:
        HTTPException: 400（内容为空）、500（内部错误）。
    """
    try:
        result = ingestor.ingest_text(
            payload.text, title=payload.title, collection=payload.collection
        )
        return ApiResponse(data=result.model_dump())
    except IngestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/batch",
    response_model=ApiResponse,
    summary="批量入库服务端目录",
    description=(
        "扫描服务器上的一个目录，把里面所有支持的文档入库。"
        "默认只允许扫描项目内的 `data/docs`，防止被用来读取任意路径（路径穿越）。"
    ),
)
async def ingest_batch(
    ingestor: IngestorDep,
    directory: str = Query(default="data/docs", description="相对项目根目录的路径"),
    collection: str = Query(default="default"),
) -> ApiResponse:
    """批量入库一个目录。

    Args:
        ingestor: 入库服务。
        directory: 目录路径（限制在项目目录内）。
        collection: 目标集合。

    Returns:
        统一响应，`data` 含成功/失败清单。

    Raises:
        HTTPException: 400（路径越界或目录不存在）。
    """
    from app.config import BASE_DIR

    target = (BASE_DIR / directory).resolve()
    try:
        target.relative_to(BASE_DIR.resolve())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"目录必须位于项目目录内。收到：{directory}",
        ) from None

    if not target.is_dir():
        raise HTTPException(status_code=400, detail=f"目录不存在：{directory}")

    report = ingestor.ingest_directory(target, collection=collection)
    return ApiResponse(
        code=0 if not report["failed"] else 1,
        message="ok" if not report["failed"] else f"部分文件失败：{len(report['failed'])} 个",
        data=report,
    )


# ============================================================
#  文档管理（挂在 /documents 下）
# ============================================================
documents_router = APIRouter(prefix="/documents", tags=["文档管理"], dependencies=[AuthDep])


@documents_router.get(
    "",
    response_model=ApiResponse,
    summary="列出知识库中的文档",
)
async def list_documents(
    ingestor: IngestorDep,
    collection: str = Query(default="default"),
) -> ApiResponse:
    """查看当前知识库里有哪些文档。"""
    result: DocumentListResponse = ingestor.list_documents(collection)
    return ApiResponse(data=result.model_dump())


@documents_router.delete(
    "/{doc_id}",
    response_model=ApiResponse,
    summary="删除一篇文档",
    description="按 doc_id 删除该文档的全部 chunk。doc_id 可从文档列表接口获取。",
)
async def delete_document(
    doc_id: str,
    ingestor: IngestorDep,
    collection: str = Query(default="default"),
) -> ApiResponse:
    """删除文档。

    Args:
        doc_id: 文档 ID。
        ingestor: 入库服务。
        collection: 集合名。

    Returns:
        统一响应，`data.deleted_chunks` 为删除条数。

    Raises:
        HTTPException: 404（文档不存在）。
    """
    removed = ingestor.delete_document(doc_id, collection)
    if removed == 0:
        raise HTTPException(
            status_code=404,
            detail=f"没有找到 doc_id={doc_id} 的文档。可以先调用 GET /api/v1/documents 查看现有文档。",
        )
    return ApiResponse(data={"doc_id": doc_id, "deleted_chunks": removed})
