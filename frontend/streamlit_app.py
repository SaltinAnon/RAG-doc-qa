"""Streamlit 前端：文件上传 + 带对话记忆的问答界面。

启动：
    streamlit run frontend/streamlit_app.py

设计要点（这些细节决定了 demo 的「专业感」）：
1. **不隐藏降级状态**：离线抽取模式下顶部显示醒目提示条，
   绝不让用户误以为看到了大模型生成的答案；
2. **引用可展开**：每条引用都能点开看原文片段和相似度分数，
   这是「引用溯源」这个卖点的可视化呈现；
3. **检索调试面板**：把混合检索各阶段命中数展示出来，
   面试演示时可以当场解释「向量召回了 20 条、BM25 命中了 8 条、融合后取 5 条」；
4. **错误可行动**：后端不通时给出「怎么启动后端」的具体命令，而不是一句 ConnectionError。
"""

from __future__ import annotations

import os
import time
import uuid

import requests
import streamlit as st

# ============================================================
#  配置
# ============================================================
DEFAULT_API = os.getenv("RAG_API_BASE", "http://127.0.0.1:8000")
DEFAULT_API_KEY = os.getenv("RAG_API_KEY", "")
API_PREFIX = "/api/v1"
TIMEOUT = 180  # 长文档入库可能较慢，超时给宽一点

st.set_page_config(
    page_title="RAG 智能文档问答",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
#  后端调用封装
# ============================================================
class ApiClient:
    """极简 API 客户端。

    所有请求统一走这里，好处是「认认证头、错误处理、超时」只写一遍。
    """

    def __init__(self, base_url: str, api_key: str = "") -> None:
        self.base = base_url.rstrip("/")
        self.headers = {"X-API-Key": api_key} if api_key else {}

    # ---------- 基础 ----------
    def _url(self, path: str) -> str:
        return f"{self.base}{API_PREFIX}{path}"

    def _handle(self, resp: requests.Response) -> dict:
        """统一处理响应。

        Args:
            resp: requests 响应对象。

        Returns:
            响应里的 `data` 部分。

        Raises:
            RuntimeError: HTTP 错误或业务 code 非 0。
        """
        try:
            payload = resp.json()
        except ValueError:
            raise RuntimeError(
                f"后端返回了非 JSON 内容（HTTP {resp.status_code}）：{resp.text[:200]}"
            ) from None

        if resp.status_code >= 400:
            detail = payload.get("detail") or payload.get("message") or str(payload)
            if isinstance(detail, list):  # FastAPI 原生校验错误
                detail = "；".join(str(d.get("message", d)) for d in detail)
            raise RuntimeError(f"HTTP {resp.status_code}：{detail}")

        if isinstance(payload, dict) and payload.get("code", 0) != 0:
            raise RuntimeError(payload.get("message") or str(payload))

        return payload.get("data", payload)

    # ---------- 接口 ----------
    def health(self) -> dict:
        r = requests.get(f"{self.base}/readyz", timeout=10)
        return self._handle(r)

    def status(self) -> dict:
        return self._handle(requests.get(self._url("/config/status"), headers=self.headers, timeout=15))

    def list_documents(self, collection: str) -> dict:
        return self._handle(
            requests.get(
                self._url("/documents"), params={"collection": collection},
                headers=self.headers, timeout=30,
            )
        )

    def upload(self, filename: str, content: bytes, collection: str) -> dict:
        return self._handle(
            requests.post(
                self._url("/ingest/file"),
                files={"file": (filename, content)},
                data={"collection": collection, "skip_existing": "true"},
                headers=self.headers,
                timeout=TIMEOUT,
            )
        )

    def ingest_text(self, text: str, title: str, collection: str) -> dict:
        return self._handle(
            requests.post(
                self._url("/ingest/text"),
                json={"text": text, "title": title, "collection": collection},
                headers=self.headers,
                timeout=TIMEOUT,
            )
        )

    def query(self, question: str, session_id: str, top_k: int, collection: str) -> dict:
        return self._handle(
            requests.post(
                self._url("/query"),
                json={
                    "question": question,
                    "session_id": session_id,
                    "top_k": top_k,
                    "collection": collection,
                },
                headers=self.headers,
                timeout=TIMEOUT,
            )
        )

    def delete_document(self, doc_id: str, collection: str) -> dict:
        return self._handle(
            requests.delete(
                self._url(f"/documents/{doc_id}"),
                params={"collection": collection}, headers=self.headers, timeout=30,
            )
        )

    def reset_session(self, session_id: str) -> dict:
        return self._handle(
            requests.post(
                self._url(f"/sessions/{session_id}/reset"), headers=self.headers, timeout=15,
            )
        )


# ============================================================
#  会话状态初始化
# ============================================================
def init_state() -> None:
    """初始化 `st.session_state` 的默认值。"""
    defaults = {
        "messages": [],          # [{"role": "user"/"assistant", "content": str, "meta": dict}]
        "session_id": uuid.uuid4().hex[:12],
        "collection": "default",
        "top_k": 5,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


init_state()


# ============================================================
#  侧边栏
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ 连接设置")
    api_base = st.text_input("后端地址", value=DEFAULT_API, help="FastAPI 服务的地址")
    api_key = st.text_input(
        "API Key", value=DEFAULT_API_KEY, type="password",
        help="后端 .env 里设置了 API_KEY 时才需要填；留空表示后端未开启鉴权",
    )

    client = ApiClient(api_base, api_key)

    # ---- 连通性检查 ----
    ready: dict | None = None
    conn_error = ""
    try:
        ready = client.health()
    except Exception as exc:
        conn_error = str(exc)

    if conn_error:
        st.error("❌ 无法连接后端")
        with st.expander("怎么解决？", expanded=True):
            st.markdown(
                f"""
**错误信息**：`{conn_error[:200]}`

请先在项目根目录启动后端：

```bash
# Windows
.venv\\Scripts\\python -m uvicorn app.main:app --port 8000

# macOS / Linux
.venv/bin/python -m uvicorn app.main:app --port 8000
```

然后确认地址填对了（当前填的是 `{api_base}`）。
                """
            )
    else:
        st.success("✅ 后端已连接")
        c1, c2 = st.columns(2)
        c1.metric("知识库片段", ready.get("chunk_count", 0))
        c2.metric("可回答", "是" if ready.get("chunk_count", 0) > 0 else "否")

    st.divider()

    # ---- 检索参数 ----
    st.markdown("### 🔍 检索设置")
    st.session_state.collection = st.text_input(
        "集合名", value=st.session_state.collection,
        help="不同集合的文档互相隔离，可用于区分不同项目/租户",
    )
    st.session_state.top_k = st.slider(
        "检索片段数 (top_k)", min_value=1, max_value=15, value=st.session_state.top_k,
        help="交给模型的参考资料条数。太小会漏信息，太大会引入噪声并增加成本",
    )

    st.divider()

    # ---- 会话 ----
    st.markdown("### 💬 会话")
    st.caption(f"当前会话 ID：`{st.session_state.session_id}`")
    b1, b2 = st.columns(2)
    if b1.button("🆕 新会话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = uuid.uuid4().hex[:12]
        st.rerun()
    if b2.button("🧹 清空界面", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔄 同步清空后端记忆", use_container_width=True):
        try:
            client.reset_session(st.session_state.session_id)
            st.toast("后端会话记忆已清空")
        except Exception as exc:
            st.error(f"清空失败：{exc}")

    st.divider()

    # ---- 运行时状态 ----
    st.markdown("### 🧩 运行时状态")
    if ready and not conn_error:
        if ready.get("llm_offline"):
            st.warning("LLM：离线抽取模式")
        else:
            st.success(f"LLM：{ready.get('llm_provider')}")
        st.caption(f"Embedding：`{ready.get('embedding_provider')}`")

    with st.expander("全部参数"):
        if ready and not conn_error:
            st.json(ready)
        else:
            st.caption("后端未连接")


# ============================================================
#  主区域
# ============================================================
st.title("📚 企业级 RAG 智能文档问答系统")
st.caption(
    "LangChain · ChromaDB · FastAPI · Streamlit ｜ "
    "上传私有文档，获得**带引用溯源**的答案"
)

# ---- 离线模式提示条（醒目，不可忽略）----
if ready and ready.get("llm_offline"):
    st.warning(
        "**当前为离线抽取模式**：未配置大模型 API Key，"
        "答案直接从检索到的原文中抽取拼接，**未经语言模型生成**。"
        "在项目根目录的 `.env` 里填入 `LLM_API_KEY` 并重启后端即可切换为真正的生成式问答。"
    )

tab_chat, tab_kb, tab_help = st.tabs(["💬 智能问答", "📁 知识库管理", "❓ 使用说明"])


# ------------------------------------------------------------
#  Tab 1：问答
# ------------------------------------------------------------
with tab_chat:
    if not ready:
        st.info("请先在左侧确认后端已启动并连接成功。")

    # 渲染历史消息
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
                    st.caption(
                        f"混合检索：{'开' if dbg.get('used_hybrid') else '关'} ｜ "
                        f"重排序：{'开' if dbg.get('used_rerank') else '关'} ｜ "
                        f"耗时：{meta.get('latency_ms', 0)} ms ｜ "
                        f"模型：`{meta.get('model', 'unknown')}`"
                    )

    # 输入框
    prompt = st.chat_input("请输入关于已上传文档的问题……")
    if prompt:
        if not ready:
            st.error("后端未连接，无法提问。")
        elif ready.get("chunk_count", 0) == 0:
            st.error("知识库为空。请先切换到「📁 知识库管理」标签页上传并入库文档。")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("检索文档并生成回答中……"):
                    try:
                        data = client.query(
                            prompt,
                            session_id=st.session_state.session_id,
                            top_k=st.session_state.top_k,
                            collection=st.session_state.collection,
                        )
                        # 结构化拒答：明确告诉用户「这是拒答，不是答案」，
                        # 别让用户把「我答不上来」误读成「文档里就是这么写的」。
                        if data.get("refused"):
                            hint = {
                                "no_retrieval": "知识库里没有检索到相关片段",
                                "model_refused": "检索到了片段，但判定资料不足以回答",
                                "empty_output": "模型返回了空内容（异常）",
                            }.get(data.get("refusal_reason"), "")
                            st.info(
                                "🙅 系统判定 **无法回答**，未编造答案。"
                                + (f"原因：{hint}" if hint else "")
                            )

                        st.markdown(data["answer"])

                        citations = data.get("citations") or []
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
                                "content": data["answer"],
                                "meta": {
                                    "citations": citations,
                                    "retrieval_debug": data.get("retrieval_debug"),
                                    "latency_ms": data.get("latency_ms"),
                                    "model": data.get("model"),
                                    "refused": data.get("refused"),
                                    "refusal_reason": data.get("refusal_reason"),
                                },
                            }
                        )
                    except Exception as exc:
                        err = f"❌ 提问失败：{exc}"
                        st.error(err)
                        st.session_state.messages.append({"role": "assistant", "content": err})

            st.rerun()


# ------------------------------------------------------------
#  Tab 2：知识库管理
# ------------------------------------------------------------
with tab_kb:
    col_up, col_txt = st.columns(2)

    # ---- 文件上传 ----
    with col_up:
        st.subheader("📤 上传文件")
        files = st.file_uploader(
            "支持 PDF / Word(.docx) / TXT / Markdown / CSV，可多选",
            type=["pdf", "docx", "txt", "md", "markdown", "csv"],
            accept_multiple_files=True,
            key="uploader",
        )
        if st.button("⬆️ 开始入库", type="primary", disabled=not files):
            progress = st.progress(0.0, text="准备中……")
            log_box = st.container()

            for i, f in enumerate(files or []):
                progress.progress((i + 0.1) / len(files), text=f"处理 {f.name} ……")
                started = time.perf_counter()
                try:
                    res = client.upload(f.name, f.getvalue(), st.session_state.collection)
                    cost = time.perf_counter() - started
                    if res.get("chunks", 0) == 0:
                        log_box.info(
                            f"⏭️ `{f.name}` 已在知识库中（内容未变），跳过。"
                            "如需更新请先在下方删除再上传。"
                        )
                    else:
                        log_box.success(
                            f"✅ `{f.name}` 入库成功：{res['chunks']} 个片段，"
                            f"{res.get('chars', 0)} 字，耗时 {cost:.1f}s"
                        )
                except Exception as exc:
                    log_box.error(f"❌ `{f.name}` 入库失败：{exc}")
                progress.progress((i + 1) / len(files), text=f"{i + 1}/{len(files)} 完成")

            progress.empty()
            st.rerun()

    # ---- 粘贴文本 ----
    with col_txt:
        st.subheader("📝 粘贴文本")
        title = st.text_input("文档标题", value="临时笔记")
        text = st.text_area("正文内容", height=220, placeholder="把要入库的文字粘贴到这里……")
        if st.button("⬆️ 入库这段文本", disabled=not text.strip()):
            try:
                res = client.ingest_text(text, title, st.session_state.collection)
                st.success(f"✅ 入库成功：{res['chunks']} 个片段")
                st.rerun()
            except Exception as exc:
                st.error(f"❌ 入库失败：{exc}")

    st.divider()

    # ---- 文档列表 ----
    st.subheader("📚 知识库内容")
    if st.button("🔄 刷新列表"):
        st.rerun()

    try:
        docs_data = client.list_documents(st.session_state.collection)
        docs = docs_data.get("documents", [])

        m1, m2, m3 = st.columns(3)
        m1.metric("文档数", docs_data.get("total_docs", 0))
        m2.metric("片段总数", docs_data.get("total_chunks", 0))
        m3.metric("集合", docs_data.get("collection", "-"))

        if not docs:
            st.info("知识库还是空的。上传文件或粘贴文本后就能开始提问了。")
        else:
            for d in docs:
                c1, c2, c3, c4 = st.columns([4, 1, 1, 1])
                c1.markdown(f"**{d['filename']}**  \n`{d['doc_id']}`")
                c2.caption(f"{d['chunks']} 段")
                c3.caption(d.get("source_type", "-"))
                if c4.button("🗑️ 删除", key=f"del-{d['doc_id']}"):
                    try:
                        client.delete_document(d["doc_id"], st.session_state.collection)
                        st.toast(f"已删除 {d['filename']}")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"删除失败：{exc}")
    except Exception as exc:
        st.error(f"获取文档列表失败：{exc}")


# ------------------------------------------------------------
#  Tab 3：使用说明
# ------------------------------------------------------------
with tab_help:
    st.markdown(
        """
### 三步开始

1. **入库**：切到「📁 知识库管理」，上传 PDF/Word/TXT，点「开始入库」。
   系统会自动**加载 → 切分 → 向量化 → 存进 ChromaDB**。
2. **提问**：切回「💬 智能问答」，直接用自然语言提问。
3. **看引用**：每条回答下方都能展开引用来源，含**文件名、页码、原文片段、相关度分数**。

---

### 多轮对话

同一个会话 ID 下，系统会记住之前的对话：

> 你：公司的年假有多少天？  
> 你：**那病假呢？** ← 指代词会被自动改写成「公司的病假有多少天？」再去检索

点侧边栏「🆕 新会话」可以开始一段全新对话。

---

### 关于「离线抽取模式」

如果没配置大模型 API Key，系统会自动切换到离线模式：
**检索、引用溯源全都正常工作**，只是答案从原文里抽取拼接，不由语言模型生成。

切换方法：在项目根目录 `.env` 里填 `LLM_API_KEY`，重启后端。

---

### 检索参数怎么调

| 现象 | 调整 |
|---|---|
| 答案缺少关键信息 | 调大「检索片段数 top_k」（5 → 8） |
| 答案里混入无关内容 | 调小 top_k（5 → 3） |
| 专有名词/编号检索不到 | 确认混合检索已开启（默认开） |
| 召回回来的片段总是同一段 | 调小 `.env` 里的 `CHUNK_OVERLAP`，或调大 `CHUNK_SIZE` |

---

### 常见问题

**Q：PDF 入库后检索不到内容？**  
A：多半是**扫描版 PDF**（内容是图片）。本系统只用文本层提取（pypdf），
不内置 OCR。先用 OCR 工具转成文本型 PDF 再上传。

**Q：换过 Embedding 模型后报维度不匹配？**  
A：不同模型的向量维度不同，同一个集合不能混用。
删除 `data/chroma` 目录重建，或换一个集合名。

**Q：回答里的引用编号对不上？**  
A：这是 RAG 最典型的 bug。检查 `app/core/prompts.py` 里的引用规则，
以及 `chain.py` 的 `_build_citations()`。本项目对「模型忘标引用」有兜底，
会在 `retrieval_debug` 里标记。
        """
    )
