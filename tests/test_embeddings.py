"""Embedding 测试。

重点验证 `LocalHashEmbeddings`（项目的零依赖兜底方案）的语义性质，
因为整条链路在无 Key 时都依赖它。
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.embeddings import (
    LocalHashEmbeddings,
    describe_embeddings,
    get_embeddings,
)


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度（向量库用的是这个度量）。"""
    va, vb = np.asarray(a), np.asarray(b)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(va @ vb / (na * nb))


class TestLocalHashEmbeddings:
    """兜底 Embedding 的行为契约。"""

    def test_dimension(self):
        emb = LocalHashEmbeddings(dimension=128)
        assert len(emb.embed_query("测试")) == 128
        assert len(emb.embed_documents(["a", "b"])[0]) == 128

    def test_batch_size_matches_input(self):
        emb = LocalHashEmbeddings(dimension=64)
        assert len(emb.embed_documents(["一", "二", "三"])) == 3

    def test_normalized(self):
        """必须 L2 归一化 —— 否则「余弦相似度 = 点积」这个前提不成立。"""
        vec = LocalHashEmbeddings(dimension=256).embed_query("年假规定员工手册")
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5

    def test_deterministic(self):
        """跨调用必须稳定。用 Python 内置 hash() 会因进程随机盐而失败。"""
        emb = LocalHashEmbeddings(dimension=256)
        assert emb.embed_query("同样的文本") == emb.embed_query("同样的文本")

    def test_empty_text_gives_zero_vector(self):
        vec = LocalHashEmbeddings(dimension=64).embed_query("")
        assert all(v == 0.0 for v in vec)

    def test_similar_text_scores_higher(self):
        """核心语义性质：字面相近的文本相似度必须显著更高。"""
        emb = LocalHashEmbeddings(dimension=1024)
        q = emb.embed_query("员工的年假有多少天")
        near = emb.embed_query("员工每年享有 5 天带薪年假，入职满一年后开始享受")
        far = emb.embed_query("今天天气不错，我们去公园散步吧")

        assert cosine(q, near) > cosine(q, far)
        assert cosine(q, near) > 0.15  # 相关文本应有明显的正相关

    def test_identical_text_scores_one(self):
        emb = LocalHashEmbeddings(dimension=512)
        assert abs(cosine(emb.embed_query("年假"), emb.embed_query("年假")) - 1.0) < 1e-6

    def test_query_and_document_use_same_algo(self):
        """embed_query 与 embed_documents 必须同算法。

        这是 RAG 最隐蔽的 bug 之一：两边用了不同模型/不同预处理，
        检索结果会完全随机，而且不报错。这里用一个断言把它钉死。
        """
        emb = LocalHashEmbeddings(dimension=256)
        text = "住房公积金缴存比例为 12%"
        assert emb.embed_query(text) == emb.embed_documents([text])[0]

    def test_rejects_invalid_dimension(self):
        with pytest.raises(ValueError, match="dimension"):
            LocalHashEmbeddings(dimension=0)

    def test_bigram_beats_unigram_for_chinese(self):
        """bigram 加权后，中文语义区分度应该比纯 unigram 更好。

        验证方式：「年假」vs「假年」——字完全相同，顺序不同。
        带 bigram 时相似度应该明显低于 1（否则说明 bigram 没起作用）。
        """
        emb = LocalHashEmbeddings(dimension=1024)
        a = emb.embed_query("年假")
        b = emb.embed_query("假年")
        assert cosine(a, b) < 0.95

    def test_repr_contains_dim(self):
        assert "256" in repr(LocalHashEmbeddings(dimension=256))


class TestFactory:
    """工厂与降级逻辑。"""

    def test_hash_provider_explicit(self):
        emb = get_embeddings("hash")
        assert isinstance(emb, LocalHashEmbeddings)

    def test_unknown_provider_falls_back(self):
        # 传一个不存在的名字，也不能抛异常，必须降级
        emb = get_embeddings("不存在的提供商")
        assert isinstance(emb, LocalHashEmbeddings)

    def test_openai_without_key_falls_back(self, monkeypatch):
        """没有 Key 时必须降级，而不是抛异常。

        这条测试保证「别人 clone 下来不带 Key 也能跑」这个卖点不会回归。
        """
        from app.config import settings

        monkeypatch.setattr(settings, "llm_api_key", "")
        monkeypatch.setattr(settings, "embedding_provider", "openai")
        get_embeddings.cache_clear()

        emb = get_embeddings("openai")
        assert isinstance(emb, LocalHashEmbeddings)
        get_embeddings.cache_clear()

    def test_describe(self):
        assert "hash" in describe_embeddings(LocalHashEmbeddings(dimension=64))
