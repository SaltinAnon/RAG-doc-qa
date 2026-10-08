# ============================================================
#  Makefile —— 一键命令（Windows 用户用 Git Bash 或 `make` via choco）
#  没有 make 也没关系，每个目标下面都写了等价的原始命令
# ============================================================

.PHONY: help install install-local run-api run-ui ingest test eval eval-compare calibrate smoke clean docker-up docker-down

help:  ## 显示所有可用命令
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## 创建虚拟环境并安装依赖
	python -m venv .venv
	.venv/Scripts/pip install -U pip
	.venv/Scripts/pip install -r requirements.txt

install-local:  ## 额外安装本地开源模型（体积大）
	.venv/Scripts/pip install -r requirements-local.txt

run-api:  ## 启动 FastAPI 后端（http://127.0.0.1:8000/docs）
	.venv/Scripts/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

run-ui:  ## 启动 Streamlit 前端（http://127.0.0.1:8501）
	.venv/Scripts/python -m streamlit run frontend/streamlit_app.py

ingest:  ## 把 data/docs 下的文档全部入库
	.venv/Scripts/python scripts/ingest_cli.py --dir data/docs

test:  ## 跑单元测试
	.venv/Scripts/python -m pytest tests -v

eval:  ## 跑检索/引用评测，输出 Recall@K
	.venv/Scripts/python scripts/evaluate.py

eval-compare:  ## 对比「纯向量 / +BM25 / +重排」的片段级召回差异
	.venv/Scripts/python scripts/evaluate.py --compare

calibrate:  ## 标定离线模式的拒答闸门阈值（换语料后必跑）
	.venv/Scripts/python scripts/calibrate_gate.py

smoke:  ## 端到端冒烟测试：自动起服务 → 打真实接口 → **自动关服务**
	.venv/Scripts/python scripts/smoke_test.py

docker-up:  ## 用 docker-compose 起后端 + 前端
	docker compose -f docker/docker-compose.yml up --build

docker-down:  ## 停掉容器
	docker compose -f docker/docker-compose.yml down

clean:  ## 清理缓存与向量库（会清空已入库的文档！）
	rm -rf data/chroma .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
