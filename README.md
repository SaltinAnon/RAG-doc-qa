# 📚 企业级 RAG 智能文档问答系统

> 上传私有文档 → 向量化入库 → 提问 → 得到**带引用溯源**的答案。
>
> 基于 **LangChain + ChromaDB + FastAPI + Streamlit**，支持多轮对话记忆。
>
> **不需要任何 API Key 也能完整跑通全链路。**

![CI](https://github.com/SaltinAnon/rag-doc-qa/actions/workflows/ci.yml/badge.svg)

![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)



![License](https://img.shields.io/badge/license-MIT-green)

---

## 为什么值得看

大多数 RAG 示例项目的问题是：**clone 下来跑不起来**（要配 API Key、要起向量库服务），

或者**只有一个 notebook**（没有工程结构、没有测试、没有部署）。

这个项目针对这两点做了取舍：

| 痛点              | 本项目的做法                                                     |
| --------------- | ---------------------------------------------------------- |
| 没 API Key 就跑不了  | **三层 Embedding + 自动降级**，零成本也能完整演示检索与引用溯源                   |
| 只在 notebook 里能跑 | 分模块工程结构（config / core / api / models / utils），全部带类型注解      |
| 没有质量意识          | 9 个测试文件 / 259 个用例 + 可复现的评测脚本（Recall@K / MRR / 引用准确率 / 拒答率） |
| 「在我机器上能跑」       | Docker 多阶段构建 + GitHub Actions 里真跑 `docker run` 验证健康检查      |
| 中文文档效果差         | 切分符、分词器、Prompt 全部针对中文重新设计（不是翻译英文教程）                        |
| 引用张冠李戴          | 编号化上下文 + 强制引用 Prompt + 正则回查 + 兜底机制                         |

---

## 核心特性

- **多格式文档解析**：PDF（按页）/ Word（含表格）/ Markdown / TXT / CSV，自动编码探测
- **中文感知切分**：按「段落 → 中文句末 → 中文逗号」三级降级，跨页句子自动拼接
- **混合检索**：向量检索 + BM25 关键词检索并行，**RRF 融合**（无需调参、不受量纲影响）
- **启发式重排序**：查询词覆盖率 / 短语命中 / 数字命中 / 长度合理性 四维打分
- **引用溯源**：每条答案自带 `[n]` 标记，可展开查看**文件名 + 页码 + 原文片段 + 相关度**
- **结构化拒答（幻觉抑制）**：`refused` / `refusal_reason` 字段显式表达「答不上来」，

  且**拒答时不返回任何引用** —— 不编答案，也不给否定结论伪造依据
- **多轮对话记忆**：按 `session_id` 隔离，带 TTL 自动清理与轮数上限
- **查询改写**：多轮场景下自动把「那病假呢」还原成「公司的病假有多少天」
- **可观测性**：结构化日志 + 请求 ID 串联 + `retrieval_debug` 暴露检索各阶段命中数
- **工程化**：pytest 测试 + Docker + GitHub Actions CI + 一键 Makefile

---

## 效果数据

> 所有数字都可由 `python scripts/evaluate.py` 复现，不是估计值。

**运行环境**：`LLM_PROVIDER=offline`（不调用任何大模型、不需要 API Key）、

Embedding 走内置无依赖兜底 `offline-hash(dim=512)`、`top_k=5`。

| 指标                   | 数值          | 说明                       |
| -------------------- | ----------- | ------------------------ |
| **Recall@5**（文件级）    | **100.00%** | 检索结果中包含「答案来源文档」的比例       |
| **AnswerHit@5**（片段级） | **95.83%**  | 检索结果中存在「装有答案的片段」的比例（更严格） |
| Precision@5          | 71.67%      | 检索结果中相关片段的比例             |
| MRR                  | 1.0000      | 第一条相关结果名次的倒数均值           |
| 答案片段 MRR             | 0.9375      | 答案片段的平均倒数名次              |
| 引用绑定准确率              | 95.83%      | 返回的引用命中金标准来源的比例          |
| 答案关键词命中率             | 83.33%      | 答案中出现预期关键信息的比例           |
| **拒答准确率**            | **100.00%** | 知识库确实无答案时正确拒答的比例（幻觉抑制）   |
| 平均单次耗时               | 50 ms       | 检索 + 抽取式作答全链路            |

### 模块贡献对比（`--compare`，最能说明设计价值的一组实验）

| 配置                  | 文件级 Recall@5 | 片段级 AnswerHit@5     | 答案片段 MRR            |
| ------------------- | ------------ | ------------------- | ------------------- |
| 纯向量检索               | 100.00%      | 87.50%              | 0.6993              |
| 向量 + BM25（RRF 融合）   | 100.00%      | **95.83%**（+8.33pp） | 0.8750（+0.1757）     |
| 向量 + BM25 + 重排序（默认） | 100.00%      | 95.83%              | **0.9375**（+0.2382） |

> 🔍 **一个值得讲的方法论细节**：一开始我用「金标准文件是否进 top-5」作为召回率，
>
> 结果加了重排之后 Recall **反而下降** 4.35pp。追查后发现**不是检索变差**，
>
> 而是两个问题叠加：
>
> ① 示例语料把同一份技术架构文本同时写成了 `.md` 和 `.pdf`，
>
> 文件名级标注只认其中一个 → 命中另一个反而算失败（评测集标注歧义）；
>
> ② 文件名级指标在小语料上**会饱和**（只要对的文件里随便哪个片段进了 top-5 就算命中），
>
> 掩盖了真正的差异。
>
> 于是我补了**片段级指标 `AnswerHit@k`**：要求 top-k 里存在一个**同时包含全部答案关键词**
>
> 的片段。换用这个指标后，混合检索与重排序的价值立刻显现（+8.33pp / +0.2382 MRR）。
>
> 这就是「先怀疑指标，再怀疑系统」的实践。

> 🔍 **另一个更值得讲的细节**：上面这张表全绿之后，我随手用更自然的说法
>
> 「员工年假有多少天？」手测了一下 —— **被拒答了**。原因是文档写的是
>
> 「第八条 年假。员工入职满一年后开始享受带薪年休假。」，
>
> 「年假」和「员工」被句号拆进了两个句子，逐句统计只剩 1 个命中。
>
> 修法不是调阈值，而是修**句子切分**：把「第八条 年假」这类小标题并入下一句。
>
> 这条用例已经补进评测集（27 条）和单元测试。
>
> 教训：**评测集全绿 ≠ 系统可用。**

测试集：`data/eval/eval_set.json`，27 条用例（24 条可回答 + 3 条应拒答），

金标准为「答案来源文档 + 必须出现的关键词」。

其中 1 条是**端到端手测发现的真实误拒用例**（见上面的第二个「方法论细节」）。

> ⚠️ **诚实说明**：示例文档只有 6 篇，因此各项绝对数值偏高、区分度有限。
>
> 这组数字的意义是**证明我建立了可复现的评测流程**，而不是证明模型能力。
>
> 真正有说服力的实验是 `python scripts/evaluate.py --compare`
>
> ——它对比「纯向量检索 → 加 BM25 → 加重排序」的 Recall 变化，
>
> 量化了每个模块的贡献。**换成你自己的真实文档（几十篇以上）重跑，数字才可信。**

---

## 架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         用户界面层                                    │
│   Streamlit 前端 (8501)        FastAPI 自动文档 /docs (8000)          │
│   文件上传 / 聊天 / 引用展开     交互式 API 调试                       │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ HTTP  (X-API-Key 鉴权)
┌───────────────────────────▼─────────────────────────────────────────┐
│                        API 层  app/main.py                           │
│   中间件：请求ID注入 / 耗时统计 / 统一异常处理 / CORS                   │
│   路由：/ingest/file  /ingest/text  /ingest/batch                     │
│         /query  /documents  /sessions/{id}/reset  /healthz  /readyz   │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│                       业务核心层  app/core/                           │
│                                                                      │
│  【写路径】loaders → splitter → embeddings → vectorstore               │
│    多格式解析     中文感知切分   三层可降级     Chroma 持久化           │
│                                                                      │
│  【读路径】retriever → prompts → llm → chain                           │
│    向量+BM25        强制引用   在线/离线  引用解析+会话记忆             │
│    → RRF 融合                   自动降级                              │
│    → 启发式重排                                                       │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   ChromaDB            OpenAI 兼容 API      可选本地模型
  (持久化向量库)      (LLM + Embedding)   (BGE-small-zh)
```

### 一次问答的完整链路

```
用户提问 "员工年假有多少天？"
   │
   ├─ 1. [可选] 查询改写：结合历史对话把指代词还原成实体
   │
   ├─ 2. 混合检索（并行两路，各取 20 条候选）
   │      ├─ 向量检索：Embedding 后查 ChromaDB，按余弦相似度排序
   │      └─ BM25 检索：字符 unigram+bigram 分词后按关键词打分
   │          ↓
   │      RRF 融合：score = Σ 1/(60 + rank)，按排名而非分数融合
   │          ↓
   │      启发式重排：覆盖率40% + 短语25% + 数字20% + 长度15%
   │          ↓
   │      相似去重（剔除相邻 chunk 的 85% 重叠）→ 取 top-5
   │
   ├─ 3. 组装 Prompt：[1] 来源：员工手册.md | 第 8 页 + 正文…
   │                 强制要求「每个事实后标注 [编号]」
   │
   ├─ 4. 生成答案（在线 LLM / 离线抽取式降级）
   │
   ├─ 5. 解析引用：正则抠出 [n] → 回查 chunk → 返回 source/page/snippet/score
   │      （模型忘标引用时，兜底挂上 top-3 并标记 fallback）
   │
   └─ 6. 写入会话记忆（带 TTL 与轮数上限）
```

---

## 快速开始

### 方式一：零成本跑通（不需要任何 API Key）⭐ 推荐先试这个

```bash
# 1. 创建虚拟环境（Python 3.11 或 3.12，不要用 3.13）
python -m venv .venv

# Windows
.venv\Scripts\python.exe -m pip install -U pip
.venv\Scripts\python.exe -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# macOS / Linux
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt
```

```bash
# 2. 生成示例文档与评测集
python scripts/make_sample_docs.py

# 3. 批量入库（会自动显示切分质量分析）
python scripts/ingest_cli.py --dir data/docs

# 4. 启动后端（新开一个终端）
python -m uvicorn app.main:app --port 8000

# 5. 启动前端（再开一个终端）
python -m streamlit run frontend/streamlit_app.py
```

打开 <http://localhost:8501> 就能上传文档、提问、看引用了。

> 此时系统处于**离线抽取模式**：检索和引用溯源完全正常，
>
> 答案从原文抽取拼接，未经语言模型生成。界面上会有醒目的黄条提示。

### 方式二：接入真正的 LLM（推荐，效果提升明显）

```bash
cp .env.example .env
```

编辑 `.env`，填任一家（都走 OpenAI 兼容协议，只需改 4 行）：

```ini
# DeepSeek（性价比高，中文好）
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat          # ⚠️ 必须与 PROVIDER 匹配！
LLM_API_KEY=sk-你的key
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
```

```ini
# 完全本地免费（需要先装 requirements-local.txt + Ollama）
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b
LLM_BASE_URL=http://localhost:11434/v1
EMBEDDING_PROVIDER=local          # 用 BAAI/bge-small-zh-v1.5
```

重启后端即可。

> ⚠️ **一个很容易踩的坑：`LLM_MODEL` 必须跟着 `LLM_PROVIDER` 一起改。**
>
> 如果只改了 `LLM_PROVIDER=deepseek` 而 `LLM_MODEL` 还留着默认的
> `gpt-4o-mini`，程序会拿 OpenAI 的模型名去请求 DeepSeek 的接口，
> 结果是 **HTTP 500**（而且看不到原因）。
>
> **最省事的做法：把 `LLM_MODEL` 留空** —— 程序会按 provider 自动选默认模型，
> 从根上不会配错。配置是否自洽可以直接问接口：
>
> ```bash
> curl http://localhost:8000/readyz
> # llm_model_configured / llm_model_effective / llm_config_ok / llm_config_warnings
> ```
>
> 各家默认模型：`openai → gpt-4o-mini`、`deepseek → deepseek-chat`、
> `zhipu → glm-4-flash`、`moonshot → moonshot-v1-8k`、`ollama → qwen2.5:7b`。
> 详细排查见 [`docs/05-常见问题与排错.md`](docs/05-常见问题与排错.md)。

### 方式三：Docker 一键启动

```bash
cd docker
docker compose up --build
```

- 前端 <http://localhost:8501>
- 后端文档 <http://localhost:8000/docs>

不需要 `.env` 也能起来（自动进入离线抽取模式）。

---

## 常用命令

```bash
# 预演入库（不写向量库、不调 Embedding，用来检查切分质量）
python scripts/ingest_cli.py --dir data/docs --dry-run

# 清空集合后重新入库
python scripts/ingest_cli.py --dir data/docs --reset

# 跑单元测试
pytest tests -v

# 跑效果评测
python scripts/evaluate.py --mode full -k 5

# 对比实验：纯向量 vs +BM25 vs +重排（文件级 + 片段级两个指标）
python scripts/evaluate.py --compare

# 标定离线模式的拒答闸门阈值（换语料后必跑）
python scripts/calibrate_gate.py

# 端到端冒烟测试：自动起服务 → 打真实接口 → **自动关服务**（用完不留后台进程）
python scripts/smoke_test.py
```

有 `make` 的话还能用 `make install / run-api / run-ui / ingest / test / eval / smoke / docker-up`。

---

## API 速查

统一前缀 `/api/v1`，统一响应外壳 `{ code, message, data }`。

| 方法     | 路径                     | 说明                   |
| ------ | ---------------------- | -------------------- |
| POST   | `/ingest/file`         | 上传文件入库（multipart，幂等） |
| POST   | `/ingest/text`         | 提交一段文本入库             |
| POST   | `/ingest/batch`        | 批量入库服务端目录（限项目内路径）    |
| GET    | `/documents`           | 列出知识库文档              |
| DELETE | `/documents/{doc_id}`  | 删除一篇文档               |
| POST   | `/query`               | 提问（RAG 问答，返回答案 + 引用） |
| POST   | `/sessions/{id}/reset` | 清空会话记忆               |
| GET    | `/sessions/stats`      | 会话记忆使用情况             |
| GET    | `/config/status`       | 运行时配置与降级状态           |
| GET    | `/healthz` `/readyz`   | 存活 / 就绪探针            |

示例：

```bash
# 上传文档
curl -X POST http://localhost:8000/api/v1/ingest/file \
  -F "file=@员工手册.pdf" -F "collection=default"

# 提问
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question":"年假有多少天？","session_id":"demo","top_k":5}'
```

响应示例：

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "answer": "员工入职满一年后每年享有 5 天带薪年假[1]。",
    "citations": [
      {
        "index": 1,
        "doc_id": "a1b2c3d4e5f6g7h8",
        "source": "员工手册.md",
        "page": 8,
        "score": 0.8734,
        "snippet": "员工入职满一年后开始享受带薪年休假…"
      }
    ],
    "latency_ms": 412,
    "model": "deepseek-chat",
    "refused": false,
    "refusal_reason": null,
    "offline_mode": false,
    "retrieval_debug": {
      "vector_hits": 20, "bm25_hits": 11,
      "fused": 24, "reranked": 5,
      "used_hybrid": true, "used_rerank": true,
      "citation_fallback": false
    }
  }
}
```


**拒答是一个字段，不是一段特殊文案。** 当知识库里没有相关内容时，系统不会编造答案：

```json
{
  "code": 0, "message": "ok",
  "data": {
    "answer": "根据现有知识库无法回答该问题：没有检索到与该问题相关的文档片段。…",
    "citations": [],
    "refused": true,
    "refusal_reason": "model_refused",
    "offline_mode": true
  }
}
```

> 为什么要单独做 `refused` 字段：早期版本让调用方去 `answer` 里匹配「无法回答」
>   
> 这几个字来判断是否拒答 —— 文案改一个字判断就失灵，「拒答准确率」这个指标
>   
> 也就不再可信。现在链路里判定一次，作为字段一路传出去。
>   
> `refusal_reason` 取值：`no_retrieval`（检索为空）/ `model_refused`（模型或离线闸门


> 判定资料不足）/ `empty_output`（模型返回空，属异常）。
> **约定：`refused=true` 时 `citations` 一定为空** —— 拒答却挂着引用会制造
> 「有据可依」的假象。

---

## 项目结构

```
Newstart/
├── CLAUDE.md                  # AI 协作契约（项目目标/规范/实现顺序）
├── README.md
├── requirements.txt           # 锁定版本
├── requirements-local.txt     # 可选：本地开源模型
├── .env.example               # 环境变量模板
├── Makefile
├── app/
│   ├── config.py              # 唯一配置入口（pydantic-settings）
│   ├── main.py                # FastAPI 装配 + 中间件 + 异常处理
│   ├── api/                   # deps / ingest / query
│   ├── core/                  # loaders / splitter / embeddings / vectorstore
│   │                          # ingest / retriever / llm / prompts / chain
│   ├── models/schemas.py      # 全部 Pydantic 模型
│   └── utils/                 # logger / text
├── frontend/streamlit_app.py  # 前端（约 500 行）
├── scripts/
│   ├── make_sample_docs.py    # 生成示例文档 + 评测集（同源）
│   ├── ingest_cli.py          # 命令行批量入库 + 切分质量分析
│   ├── evaluate.py            # 评测 Recall@K / MRR / 引用准确率 / 拒答率
│   ├── calibrate_gate.py      # 标定离线模式的拒答闸门阈值（用数据，不拍脑袋）
│   └── smoke_test.py          # 端到端冒烟测试（自动起服务 + try/finally 自动关服务）
├── tests/                     # 9 个测试文件 / 259 个用例
├── docker/                    # Dockerfile + docker-compose.yml
├── data/
│   ├── docs/                  # 待入库文档
│   ├── eval/eval_set.json     # 评测集
│   └── chroma/                # 向量库（已 gitignore）
└── docs/                      # 项目文档（见下）
```

### 文档索引

| 文档 | 内容 |
|---|---|
| `docs/01-快速开始.md` | 环境准备、四种运行方式、常见启动问题 |
| `docs/02-架构设计.md` | 每个模块的设计理由与关键代码路径 |
| `docs/05-常见问题与排错.md` | 报错速查表 |

---

## 技术栈

| 层次 | 选型 |
|---|---|
| 语言 | Python 3.12 |
| Web 框架 | FastAPI + Uvicorn |
| 编排 | LangChain 0.3（Document / TextSplitter / LCEL） |
| 向量库 | ChromaDB（PersistentClient） |
| Embedding | OpenAI / BGE-small-zh / 本地哈希向量（可降级） |
| LLM | 任意 OpenAI 兼容端点（OpenAI / DeepSeek / 智谱 / Ollama） |
| 前端 | Streamlit |
| 检索增强 | rank-bm25 + 自实现 RRF 融合与启发式重排 |
| 部署 | Docker 多阶段构建 + docker-compose |
| 质量保障 | pytest + GitHub Actions（含镜像构建与启动验证） |

---

## 已知限制

1. **不支持扫描版 PDF**——只用 pypdf 提取文本层，未集成 OCR。
   遇到扫描件会明确报错并给出转文本的方法，而不是静默返回空结果。
2. **向量库为单机 ChromaDB**——`VectorStore` 已做抽象，换 Milvus/Qdrant 只改一个文件。
3. **重排序为启发式**——效果上限低于 Cross-Encoder，`_rerank()` 可独立替换。
4. **无用户体系**——仅靠 `collection` 做粗粒度隔离，没有用户级权限。
5. **评测集规模小**——27 条用例，用于验证流程而非证明模型能力。

---

## License

MIT © 2026 Saltin
