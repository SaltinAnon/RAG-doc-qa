"""
Hugging Face Spaces 单进程版前端。

## 为什么需要单独一份

本项目的标准架构是「FastAPI 后端 + Streamlit 前端」两个进程。
但 Hugging Face Spaces 的 Streamlit SDK **只跑一个进程**，
没法同时起 uvicorn。

所以这一版把 HTTP 调用换成**直接 import 业务模块**，
功能完全一致：上传 → 入库 → 提问 → 引用溯源。

## 部署步骤

1. 在 https://huggingface.co/new-space 创建一个 Space：
   - SDK 选 **Streamlit**
   - Hardware 用免费的 CPU basic（2 vCPU / 16GB）就够

2. 把本目录的 4 个文件推到 Space 仓库：
   ```
   app.py
   requirements.txt
   README.md
   README_HF_HEADER.md   ← 内容要合并到 README.md 顶部
   ```
   以及项目源码目录（`app/`、`scripts/`）。

   最省事的方式（在项目根目录执行）：
   ```bash
   git clone https://huggingface.co/spaces/你的用户名/rag-doc-qa hf-space
   cd hf-space
   cp -r ../app ../scripts ../data .
   cp ../deploy/hf_spaces/app.py ./app.py
   cp ../deploy/hf_spaces/requirements.txt ./requirements.txt
   cp ../deploy/hf_spaces/README_HF_HEADER.md ./README.md
   git add . && git commit -m "init" && git push
   ```

3. Space 会自动构建。第一次启动时会自动生成示例文档并入库
   （见下方 `_bootstrap()`），大约需要 30~60 秒。

4. 在 Space 的 **Settings → Variables and secrets** 里可以配置
   `LLM_API_KEY` / `LLM_PROVIDER` 等环境变量来接入真实大模型。
   **注意：公开的 Space 不要填自己的付费 Key**，会被人白嫖额度。
   建议保持离线模式，或者在设置里限定访问权限。
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

import streamlit as st

# ---- 让 Space 能找到项目模块（HF Spaces 的工作目录是仓库根）----
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# HF Spaces 默认可用磁盘较小，且不允许写太多东西。
# 把向量库指到工作目录下的 data/ 里。
os.environ.setdefault("CHROMA_DIR", str(ROOT / "data" / "chroma"))
os.environ.setdefault("DOCS_DIR", str(ROOT / "data" / "docs"))
# 默认离线，避免公开 Space 上被人白嫖付费 Key
os.environ.setdefault("LLM_PROVIDER", "offline")
os.environ.setdefault("EMBEDDING_PROVIDER", "hash")
os.environ.setdefault("LOG_LEVEL", "WARNING")

st.set_page_config(
    page_title="RAG 智能文档问答",
    page_icon="📚",
    layout="wide",
)


# ============================================================
#  首次启动时准备示例数据
# ============================================================
@st.cache_resource(show_spinner="首次启动，正在生成示例文档并建立索引……")
def _bootstrap() -> dict:
    """生成示例文档并入库（只在首次访问时执行）。

    用 `@st.cache_resource` 保证整个 Space 生命周期内只跑一次 —— 否则
    每次用户交互都会重新入库，Space 会被拖垮。

    Returns:
        初始化状态信息。
    """
    from app.config import settings
    from app.core.ingest import DocumentIngestor
    from app.core.vectorstore import get_vector_store

    settings.ensure_dirs()
    docs_dir = settings.docs_path

    # 1. 没有示例文档就生成
    if not any(docs_dir.glob("*")):
        try:
            sys.path.insert(0, str(ROOT / "scripts"))
            import make_sample_docs  # noqa: PLC0415

            make_sample_docs.main()
        except Exception as exc:
            return {"ok": False, "error": f"生成示例文档失败：{exc}"}

    # 2. 没有索引就入库
    store = get_vector_store()
    chunks = store.count()
    if chunks == 0:
        try:
            ingestor = DocumentIngestor(store)
            report = ingestor.ingest_directory(docs_dir, skip_existing=True)
            chunks = store.count()
            return {"ok": True, "ingested": report, "chunks": chunks}
        except Exception as exc:
            return {"ok": False, "error": f"入库失败：{exc}"}

    return {"ok": True, "chunks": chunks, "cached": True}


# ============================================================
#  懒加载业务服务（避免在每个脚本重跑周期里重复构建）
# ============================================================
@st.cache_resource(show_spinner=False)
def _get_services():
    """构建链与入库服务（进程内单例）。"""
    from app.core.chain import get_rag_chain
    from app.core.ingest import get_ingestor

    return get_ingestor(), get_rag_chain()


def init_state() -> None:
    """初始化会话状态。"""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("session_id", uuid.uuid4().hex[:12])
    st.session_state.setdefault("top_k", 5)


init_state()

boot = _bootstrap()
if not boot.get("ok"):
    st.error(f"❌ 初始化失败：{boot.get('error')}")
    st.stop()

ingestor, chain = _get_services()


# ============================================================
#  侧边栏
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ 设置")

    info = chain.describe()
    if info.get("llm_offline"):
        st.warning("LLM：离线抽取模式")
    else:
        st.success(f"LLM：{info.get('llm')}")
    st.caption(f"Embedding：`{info.get('embedding')}`")

    st.session_state.top_k = st.slider("检索片段数 (top_k)", 1, 10, st.session_state.top_k)

    st.divider()
    st.caption(f"会话 ID：`{st.session_state.session_id}`")
    if st.button("🆕 新会话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = uuid.uuid4().hex[:12]
        st.rerun()
    if st.button("🧹 清空界面", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    with st.expander("📊 运行时状态"):
        st.json(info)


# ============================================================
#  主区域
# ============================================================
st.title("📚 企业级 RAG 智能文档问答系统")
st.caption("LangChain · ChromaDB · Streamlit ｜ 上传私有文档，获得**带引用溯源**的答案")

if info.get("llm_offline"):
    st.warning(
        "**当前为离线抽取模式**：未配置大模型 API Key，"
        "答案直接从检索到的原文中抽取拼接，**未经语言模型生成**。"
        "检索、引用溯源等功能完全正常。"
    )

tab_chat, tab_kb, tab_about = st.tabs(["💬 智能问答", "📁 知识库", "❓ 说明"])


# ------------------------------------------------------------
with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            meta = msg.get("meta") or {}
            citations = meta.get("citations") or []
            if citations:
                with st.expander(f"📎 引用来源（{len(citations)} 条）"):
                    for c in citations:
                        page = f" · 第 {c['page']} 页" if c.get("page") else ""
                        st.markdown(
                            f"**[{c['index']}]** `{c['source']}`{page} "
                            f"　相关度 `{c.get('score', 0):.3f}`"
                        )
                        st.caption(c.get("snippet", ""))
            dbg = meta.get("retrieval_debug")
            if dbg:
                with st.expander("🔬 检索调试信息"):
                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("向量命中", dbg.get("vector_hits", 0))
                    d2.metric("BM25 命中", dbg.get("bm25_hits", 0))
                    d3.metric("融合后", dbg.get("fused", 0))
                    d4.metric("最终片段", dbg.get("reranked", 0))

    prompt = st.chat_input("请输入关于已上传文档的问题……")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("检索并生成中……"):
                try:
                    resp = chain.answer(
                        prompt,
                        session_id=st.session_state.session_id,
                        top_k=st.session_state.top_k,
                    )
                    st.markdown(resp.answer)

                    citations = [c.model_dump() for c in resp.citations]
                    if citations:
                        with st.expander(f"📎 引用来源（{len(citations)} 条）"):
                            for c in citations:
                                page = f" · 第 {c['page']} 页" if c.get("page") else ""
                                st.markdown(
                                    f"**[{c['index']}]** `{c['source']}`{page} "
                                    f"　相关度 `{c.get('score', 0):.3f}`"
                                )
                                st.caption(c.get("snippet", ""))

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": resp.answer,
                            "meta": {
                                "citations": citations,
                                "retrieval_debug": resp.retrieval_debug.model_dump(),
                                "latency_ms": resp.latency_ms,
                                "model": resp.model,
                            },
                        }
                    )
                except Exception as exc:
                    st.error(f"❌ 提问失败：{exc}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"❌ 提问失败：{exc}"}
                    )
        st.rerun()


# ------------------------------------------------------------
with tab_kb:
    st.subheader("📤 上传文件")
    files = st.file_uploader(
        "支持 PDF / Word(.docx) / TXT / Markdown / CSV，可多选",
        type=["pdf", "docx", "txt", "md", "markdown", "csv"],
        accept_multiple_files=True,
    )

    if st.button("⬆️ 开始入库", type="primary", disabled=not files):
        import tempfile

        for f in files or []:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix) as tmp:
                    tmp.write(f.getvalue())
                    tmp_path = Path(tmp.name)

                result = ingestor.ingest_path(tmp_path, skip_existing=True)
                if result.chunks == 0:
                    st.info(f"⏭️ `{f.name}` 已在知识库中（内容未变），跳过。")
                else:
                    st.success(f"✅ `{f.name}` 入库成功：{result.chunks} 个片段")
            except Exception as exc:
                st.error(f"❌ `{f.name}` 入库失败：{exc}")
            finally:
                if tmp_path and tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass

    st.divider()
    st.subheader("📚 知识库内容")

    listing = ingestor.list_documents()
    c1, c2 = st.columns(2)
    c1.metric("文档数", listing.total_docs)
    c2.metric("片段总数", listing.total_chunks)

    for d in listing.documents:
        st.markdown(f"**{d.filename}** ｜ {d.chunks} 段 ｜ `{d.source_type}`")


# ------------------------------------------------------------
with tab_about:
    st.markdown(
        """
### 这是什么

基于 **RAG（检索增强生成）** 架构的私有文档问答系统。
上传文档后，系统会：**加载 → 切分 → 向量化 → 存入 ChromaDB**，
提问时通过**混合检索（向量 + BM25 + RRF 融合 + 启发式重排）**
找出最相关的片段，交给语言模型生成**带引用溯源**的答案。

### 核心能力

| 能力 | 说明 |
|---|---|
| 多格式解析 | PDF（按页）/ Word（含表格）/ Markdown / TXT / CSV，自动编码探测 |
| 中文感知切分 | 按「段落 → 中文句末 → 中文逗号」三级降级，跨页句子自动拼接 |
| 混合检索 | 向量检索 + BM25 并行召回，RRF 融合（无需调参） |
| 引用溯源 | 答案自带 `[n]` 标记，可展开查看文件名 + 页码 + 原文片段 + 相关度 |
| 多轮对话 | 同一会话内会记住上下文（含查询改写） |
| 零成本可跑 | 无 API Key 时自动降级为离线抽取模式，链路完整 |

### 源码

完整项目（含 FastAPI 后端、Docker 部署、pytest 测试、评测脚本）见 GitHub 仓库。
        """
    )
