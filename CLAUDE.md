# CLAUDE.md — 项目协作契约（AI 必须遵守）

> 这个文件的作用：**让 AI 每次都先读它，再动手。** 没有它，AI 会自由发挥，
> 每个模块的命名、返回结构、错误处理都不一样，最后整合时你会调试到崩溃。
>
> 使用方式：每次让 AI 写代码前，说一句「请先读 CLAUDE.md，然后实现 XXX 模块」。

---

## 0. 一句话目标

做一个**企业级 RAG 智能文档问答系统**：上传私有文档（PDF/Word/TXT/MD）→ 向量化入库
→ 提问 → 返回**带引用溯源**的答案 → 支持多轮对话记忆。

这是为「AI 应用开发 / 大模型应用开发」岗位准备的**作品集项目（Portfolio Project）**，
所以除了「能跑」，还必须满足：**可测试、可部署、可复现、能讲清楚**。

---

## 1. 技术栈（锁定版本，不要随意更换）

| 层 | 选型 | 说明 |
|---|---|---|
| 语言 | Python 3.11 ~ 3.12 | 3.13 部分依赖轮子不全，**别用 3.13** |
| Web 框架 | FastAPI + Uvicorn | 后端 API |
| 编排 | LangChain（`langchain` + `langchain-community` + `langchain-core`） | 文档加载、切分、链 |
| 向量库 | ChromaDB（`chromadb`，PersistentClient 落盘） | 向量存储与相似度检索 |
| Embedding | OpenAI `text-embedding-3-small` / **本地降级** | 见 §4「双后端」 |
| LLM | OpenAI 兼容接口（OpenAI / DeepSeek / 智谱 / Ollama 均可） | 靠 `base_url` 切换 |
| 前端 | Streamlit | 上传 + 聊天 |
| 部署 | Docker + docker-compose | |
| 测试 | pytest + httpx TestClient | |

**统一用 `requirements.txt` 锁版本**，不写 `pip install xxx`（无版本号）到文档里。

---

## 2. 目录结构与职责（不得擅自改名）

```
Newstart/
├── CLAUDE.md                  # 本文件，AI 协作契约
├── README.md                  # 给 HR / 面试官看的门面
├── process.txt                # 开发过程记录（仅本地留存，见下方「不进仓库的文件」）
├── requirements.txt
├── .env.example               # 环境变量模板（真 .env 不进 git）
├── Makefile                   # 一键命令
├── app/
│   ├── config.py              # 唯一配置入口，全部读 .env
│   ├── main.py                # FastAPI 应用装配
│   ├── api/
│   │   ├── deps.py            # 依赖注入（鉴权、获取服务单例）
│   │   ├── ingest.py          # 入库路由
│   │   └── query.py           # 问答路由
│   ├── core/
│   │   ├── loaders.py         # 文档加载（PDF/Word/TXT/MD）
│   │   ├── splitter.py        # 中文感知切分
│   │   ├── embeddings.py      # Embedding 工厂（在线/离线）
│   │   ├── vectorstore.py     # Chroma 封装
│   │   ├── ingest.py          # 入库管道（加载→切分→向量化→落库）
│   │   ├── retriever.py       # 混合检索 + RRF 融合 + 重排
│   │   ├── llm.py             # LLM 工厂 + 无 Key 降级
│   │   ├── prompts.py         # 所有 Prompt 模板集中放这里
│   │   └── chain.py           # RAG 链（检索-生成-引用）+ 会话记忆
│   ├── models/
│   │   └── schemas.py         # 全部 Pydantic 模型
│   └── utils/
│       ├── logger.py          # 结构化日志
│       └── text.py            # 文本清洗、相似度工具
├── frontend/streamlit_app.py
├── scripts/
│   ├── make_sample_docs.py    # 生成演示文档
│   ├── ingest_cli.py          # 命令行批量入库
│   └── evaluate.py            # 评测 Recall@K / MRR / 引用准确率
├── data/
│   ├── docs/                  # 待入库文档
│   └── eval/eval_set.json     # 评测集（问题 + 金标准）
├── tests/                     # pytest
├── docker/                    # Dockerfile + compose
├── deploy/hf_spaces/          # Hugging Face Spaces 部署文件
└── docs/                      # 教程与说明文档
```

> 📌 **不进仓库的文件**（判断标准：**别人跑这个项目用不到它**）
>
> `process.txt`、`docs/03-简历怎么写.md`、`docs/04-GitHub零基础上传教程.md`、
> `docs/06-面试问答准备.md` 属于**个人学习材料**，已在 `.gitignore` 中排除，
> 只保留在本地。**不要**让代码或公开文档依赖它们的内容。
>
> 仓库里保留的文档是 `docs/01 快速开始`、`docs/02 架构设计`、`docs/05 排错`
> 和本文件 —— 即「别人看懂 / 跑起来这个项目所必需」的那部分。

---

## 3. 接口契约（**改这里必须同步改前后端**）

统一前缀 `/api/v1`。统一响应外壳：

```json
{ "code": 0, "message": "ok", "data": { ... } }
```

`code=0` 表示成功，非 0 表示业务错误（配合 HTTP 状态码）。

### POST `/api/v1/ingest/file`  (multipart/form-data)
- 入参：`file`（必填）、`collection`（可选，默认 default）
- 出参 `data`：`{ doc_id, filename, pages, chunks, elapsed_ms }`

### POST `/api/v1/ingest/text`  (application/json)
- 入参：`{ "text": "...", "title": "...", "collection": "..." }`
- 出参同上

### POST `/api/v1/query`  (application/json)
```json
{
  "question": "公司的年假有多少天？",
  "session_id": "可选，用于多轮记忆",
  "top_k": 5,
  "collection": "default",
  "stream": false
}
```
出参 `data`：
```json
{
  "answer": "...",
  "citations": [
    { "index": 1, "doc_id": "a1b2", "source": "员工手册.pdf",
      "page": 3, "score": 0.83, "snippet": "..." }
  ],
  "latency_ms": 1234,
  "model": "gpt-4o-mini",
  "retrieval_debug": { "vector_hits": 10, "bm25_hits": 10, "fused": 8, "reranked": 5 }
}
```

### GET `/api/v1/documents` → 文档列表
### DELETE `/api/v1/documents/{doc_id}` → 删除某文档全部 chunk
### POST `/api/v1/sessions/{session_id}/reset` → 清空该会话记忆
### GET `/healthz`（存活） / GET `/readyz`（依赖就绪，含向量库与模型探测）

---

## 4. ⭐ 双后端设计（本项目最重要的工程决策）

**问题**：面试官/你自己不一定有 OpenAI API Key；就算有，跑 demo 也要花钱。
一个「没 Key 就跑不起来」的项目，在 GitHub 上是劝退的。评分和复现都会挂。

**方案**：`app/core/embeddings.py` 和 `app/core/llm.py` 都实现**工厂 + 自动降级**：

| 组件 | 有 Key | 无 Key（自动降级） |
|---|---|---|
| Embedding | OpenAI / BGE / 任意 OpenAI 兼容 | `LocalHashEmbeddings`：字符 n-gram 哈希 + TF-IDF 加权，纯 Python 零依赖 |
| LLM | OpenAI 兼容 Chat | `ExtractiveLLM`：从检索到的原文里抽取/拼接答案，**明确标注「离线抽取模式」** |

**铁律**：
1. 降级**必须显式暴露**——API 响应里 `model` 字段要写 `offline-extractive`，
   前端顶部显示黄条提示。**绝不静默假装是 AI 生成的答案。**
2. 降级只影响「生成质量」，**不影响链路正确性**（检索、引用溯源、接口全都在跑）。
3. 所有降级逻辑必须被 pytest 覆盖（无网络也能跑 CI）。

---

## 5. 编码规范（AI 生成代码必须遵守）

1. **类型注解全量**：所有函数签名带类型；Pydantic 模型不用裸 `dict`。
2. **不许静默吞异常**：`except Exception: pass` 一律禁止。要么抛出带上下文的异常，
   要么 `logger.warning(..., exc_info=True)` 后降级。
3. **配置只从 `app/config.py` 读**，不许在业务代码里散落 `os.getenv`。
4. **路径一律 `pathlib.Path`**，禁止用字符串拼 `/`。
5. **中文文本处理必须考虑中文标点**（切分、句号、问号），不许照抄英文教程。
6. **每个模块都能单独 import 而不触发副作用**（不在 import 时建连接、不下模型）。
7. 所有对外函数写 docstring（中文），说明「做什么 / 参数 / 返回 / 抛什么异常」。
8. **不引入重量级依赖**（torch / transformers）作为默认安装；需要时放 `requirements-local.txt`。

---

## 6. 分阶段实现顺序（**必须按序，不许跳步**）

| 阶段 | 模块 | 验收标准 |
|---|---|---|
| P0 | 骨架 + config + schemas | `python -c "from app.config import settings"` 通过 |
| P1 | loaders → splitter | 能读 PDF/DOCX/TXT 并切出 chunk，附带 source/page 元数据 |
| P2 | embeddings → vectorstore | 文本能入库、能按相似度检索出结果并落盘 |
| P3 | retriever（混合+RRF） | 检索结果含向量分与 BM25 分 |
| P4 | llm → prompts → chain | 得到 `answer + citations` 结构化结果 |
| P5 | FastAPI 路由 | `curl` 能跑通 ingest → query |
| P6 | Streamlit 前端 | 浏览器能上传文件并对话，引用可见 |
| P7 | 测试 + 评测 | `pytest` 全绿；`evaluate.py` 输出 Recall@5 |
| P8 | Docker + CI | `docker compose up` 起得来；GitHub Actions 绿 |
| P9 | 文档整理 | README 与 docs/ 完整可读；个人材料（process.txt 等）本地留存、不入库 |

**每完成一个阶段，先跑一次手动验证，再进入下一阶段。** 不要攒到最后一起调。

---

## 7. 每阶段的交付物自查（AI 自问自答）

- [ ] 这个模块有没有被 `tests/` 覆盖？
- [ ] 依赖注入/单例是否避免重复加载模型（`@lru_cache`）？
- [ ] 报错信息能不能让**第一次接触项目的人**看懂？（要带文件路径和解决建议）
- [ ] 有没有在文档里留下「已知限制」？（诚实是加分项，不是减分项）
- [ ] 这一阶段在 `process.txt` 里记录了吗？

---

## 8. 禁止事项

- ❌ 不要在代码里硬编码 API Key、绝对路径 `C:\Users\xxx\...`。
- ❌ 不要把 `data/chroma/`、`*.env`、`__pycache__/` 提交到 git。
- ❌ 不要声称未经测量的指标（比如「召回率 95%」必须是 `evaluate.py` 跑出来的）。
- ❌ 不要为了好看而编造功能；没实现的就写进「未来工作」。
- ❌ 不要一次性重写多个模块——**一次只改一个，改完验证**。
