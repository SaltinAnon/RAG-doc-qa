---
title: RAG 智能文档问答
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.41.1
app_file: app.py
pinned: false
license: mit
---

> ⚠️ **这是 Hugging Face Space 的 README 头部**。
> 部署时把它**追加到你 Space 仓库的 README.md 最顶部**（包含 `---` 分隔符），
> HF 靠这段 YAML 元数据识别这是 Streamlit 应用。
>
> 详细部署步骤见 `docs/` 或本项目 GitHub 仓库的说明。

# 📚 RAG 智能文档问答

基于 **LangChain + ChromaDB** 的检索增强问答系统，支持**引用溯源**。

上传 PDF / Word / Markdown 文档，系统会自动建立向量索引；
提问时通过混合检索（向量 + BM25 + RRF 融合 + 启发式重排）找出最相关片段，
生成带 `[n]` 引用标记的答案，每条引用都能展开查看**文件名、页码、原文片段与相关度**。

## 功能

- 多格式解析：PDF（按页）/ Word（含表格）/ Markdown / TXT / CSV
- 中文感知切分：三级分隔符降级 + 跨页句子拼接 + 页码映射 + 近似去重
- 混合检索：向量 + BM25 并行召回，RRF 融合，启发式重排序
- 引用溯源：编号化上下文 + 强制引用 Prompt + 正则回查 + 漏标兜底
- 多轮对话：按会话隔离，带查询改写

## 说明

默认运行在**离线抽取模式**（未配置大模型 API Key）：检索与引用溯源完全正常，
答案直接从原文抽取拼接，**未经语言模型生成**。

如果要接入真实大模型，在 Space 的 **Settings → Variables and secrets**
里添加 `LLM_PROVIDER` / `LLM_MODEL` / `LLM_API_KEY`。

> ⚠️ **千万不要在公开 Space 上填自己的付费 API Key** —— 任何人访问都会消耗你的额度。
> 建议保持离线模式演示，或把 Space 设为私有。

## 源码

完整项目（含 FastAPI 后端、Docker 部署、pytest 测试、可复现的评测脚本）见 GitHub 仓库。
