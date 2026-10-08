# ============================================================
#  多阶段构建 —— 让镜像尽可能小
#
#  为什么分阶段：
#  - builder 阶段要编译器（gcc）来装某些包，但运行阶段根本不需要；
#  - 把编译产物复制到干净的运行镜像，能把体积从 ~1.5GB 压到 ~600MB。
# ============================================================

# ---------- 阶段 1：安装依赖 ----------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build-essential 只在编译期用到，不会进最终镜像
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# 先只复制依赖清单 —— 只要依赖没变，这一层就能命中 Docker 缓存，
# 改代码不会触发重新装包（这是构建速度的关键优化）
COPY requirements.txt .

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt


# ---------- 阶段 2：运行镜像 ----------
FROM python:3.12-slim AS runtime

# curl 用于健康检查探针
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# 从 builder 复制装好的虚拟环境
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    TZ=Asia/Shanghai

WORKDIR /app

# 以非 root 用户运行 —— 容器安全的基本要求
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/data/chroma /app/data/docs \
    && chown -R appuser:appuser /app

COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser frontend ./frontend
COPY --chown=appuser:appuser scripts ./scripts
COPY --chown=appuser:appuser data ./data

USER appuser

# 向量库持久化目录（compose 里挂 volume）
VOLUME ["/app/data/chroma"]

EXPOSE 8000 8501

# 健康检查：用 /readyz 而不是 /healthz，因为它会真正探测向量库
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/readyz || exit 1

# 默认启动后端；前端由 docker-compose 用同一个镜像换 command 启动
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
