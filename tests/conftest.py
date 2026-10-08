"""pytest 共享 fixture。

## 测试隔离的三个要点

1. **向量库必须隔离**：每个测试用例用独立的临时目录，
   否则测试之间会互相污染（上一个测试入库的文档出现在下一个测试的检索结果里）。
2. **强制离线模式**：测试绝不能真的去调 OpenAI —— 会花钱、会因网络抖动 flaky、
   别人的 CI 没配 Key 就会全红。所以 fixture 里强制
   `EMBEDDING_PROVIDER=hash` + `LLM_PROVIDER=offline`。
3. **单例必须重置**：项目里用了大量 `@lru_cache` 和模块级单例来避免重复加载模型，
   测试里必须显式重置，否则第二个用例会拿到上一个用例的旧状态。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 让 tests/ 下的文件能 import app 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402


def _reset_all_singletons() -> None:
    """重置项目里所有模块级单例 / 缓存。

    新增单例时记得往这里加一行 —— 忘了会导致「测试单独跑能过、一起跑就挂」
    这种最难查的问题。
    """
    from app.core.chain import reset_rag_chain_singleton
    from app.core.embeddings import get_embeddings
    from app.core.ingest import reset_ingestor_singleton
    from app.core.llm import reset_llm_singleton
    from app.core.retriever import reset_retriever_singleton
    from app.core.vectorstore import reset_vector_store_singleton

    reset_rag_chain_singleton()
    reset_ingestor_singleton()
    reset_retriever_singleton()
    reset_vector_store_singleton()
    get_embeddings.cache_clear()
    reset_llm_singleton()


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """每个测试用例自动应用：独立向量库目录 + 强制离线模式。

    这是 `autouse=True` 的，所有测试都会生效，不需要手动声明。
    """
    monkeypatch.setattr(settings, "chroma_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "embedding_provider", "hash")
    monkeypatch.setattr(settings, "embedding_dim", 256)  # 小维度，测试更快
    monkeypatch.setattr(settings, "llm_provider", "offline")
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, "api_key", "")  # 默认关闭鉴权
    monkeypatch.setattr(settings, "collection_name", "test_collection")
    monkeypatch.setattr(settings, "hybrid_enabled", True)
    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr(settings, "chunk_size", 200)
    monkeypatch.setattr(settings, "chunk_overlap", 40)

    _reset_all_singletons()
    yield
    _reset_all_singletons()


@pytest.fixture
def sample_texts() -> list[str]:
    """一组可复用的中文测试文本。"""
    return [
        "员工入职满一年后开始享受带薪年休假。工龄一年以上不满十年的，每年享有 5 天年假；"
        "工龄十年以上不满二十年的，每年享有 10 天年假；工龄二十年以上的，每年享有 15 天年假。",
        "公司为全体正式员工缴纳五险一金，其中住房公积金缴存比例为 12%。"
        "公司提供年度体检，标准为每人每年 1200 元。",
        "员工提出离职应提前 30 日以书面形式通知公司；试用期员工提前 3 日通知即可。"
        "离职证明在完成全部交接手续后 5 个工作日内出具。",
    ]


@pytest.fixture
def seeded_store(sample_texts):
    """已经入库了样例数据的向量库（供检索类测试使用）。

    Returns:
        已初始化并写入数据的 `VectorStore` 实例。
    """
    from app.core.embeddings import get_embeddings
    from app.core.vectorstore import VectorStore

    store = VectorStore()
    emb = get_embeddings()
    vectors = emb.embed_documents(sample_texts)

    from langchain_core.documents import Document

    docs = [
        Document(
            page_content=t,
            metadata={
                "chunk_id": f"test-{i}",
                "doc_id": f"doc{i}",
                "source": f"测试文档{i}.md",
                "source_type": "md",
                "page": i + 1,
                "chunk_index": 0,
            },
        )
        for i, t in enumerate(sample_texts)
    ]
    store.add_documents(docs, vectors)
    return store
