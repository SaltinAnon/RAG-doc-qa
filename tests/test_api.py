"""FastAPI 接口测试（使用 TestClient，全程离线，不依赖任何外部服务）。

这些测试的价值：
1. **回归保护**：改一个字段名能立刻发现前后端契约被破坏；
2. **CI 可跑**：不联网、不花钱，GitHub Actions 上 100% 稳定；
3. **活文档**：比 README 更准确地描述接口实际行为。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

API = "/api/v1"


@pytest.fixture
def client():
    """带 lifespan 的测试客户端（临时向量库目录由 conftest 的 fixture 提供）。"""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def seeded(client):
    """已经入库了两段文本的客户端。"""
    client.post(
        f"{API}/ingest/text",
        json={
            "text": "员工入职满一年后开始享受带薪年休假。工龄一年以上不满十年的，每年享有 5 天年假；"
                    "工龄十年以上不满二十年的，每年享有 10 天年假。年假以自然年为计算周期。",
            "title": "员工手册-年假",
        },
    )
    client.post(
        f"{API}/ingest/text",
        json={
            "text": "公司为全体正式员工缴纳五险一金，其中住房公积金缴存比例为 12%。"
                    "公司提供年度体检，标准为每人每年 1200 元，入职满一年的员工可享受。",
            "title": "员工手册-福利",
        },
    )
    return client


# ============================================================
#  基础接口
# ============================================================
class TestBasics:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert body["code"] == 0
        assert body["data"]["name"]

    def test_healthz(self, client):
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_readyz(self, client):
        r = client.get("/readyz")
        assert r.status_code == 200
        data = r.json()
        assert data["ready"] is True
        assert data["chroma_ok"] is True
        assert data["llm_offline"] is True  # 测试环境强制离线

    def test_openapi_docs_available(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        assert f"{API}/query" in r.json()["paths"]


# ============================================================
#  入库
# ============================================================
class TestIngestApi:
    def test_ingest_text(self, client):
        r = client.post(
            f"{API}/ingest/text",
            json={"text": "这是一段用于测试入库接口的文本内容。" * 20, "title": "测试文本"},
        )
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["chunks"] > 0
        assert data["doc_id"]
        assert data["filename"] == "测试文本"

    def test_ingest_empty_text_rejected(self, client):
        r = client.post(f"{API}/ingest/text", json={"text": "   ", "title": "空"})
        assert r.status_code == 400
        assert "detail" in r.json()

    def test_ingest_text_validation_error_format(self, client):
        """参数校验错误必须走统一响应格式，而不是 FastAPI 的裸 422 结构。"""
        r = client.post(f"{API}/ingest/text", json={"title": "缺少 text"})
        assert r.status_code == 422
        body = r.json()
        assert body["code"] == 422
        assert isinstance(body["data"]["errors"], list)

    def test_ingest_file(self, client):
        content = ("# 报销制度\n\n差旅费用报销标准：一线城市住宿每晚不超过 600 元。" * 15).encode()
        r = client.post(
            f"{API}/ingest/file",
            files={"file": ("报销制度.md", content, "text/markdown")},
            data={"collection": "default"},
        )
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["chunks"] > 0
        assert data["filename"] == "报销制度.md"

    def test_unsupported_file_type(self, client):
        r = client.post(
            f"{API}/ingest/file",
            files={"file": ("恶意.exe", b"MZ\x90\x00", "application/octet-stream")},
        )
        assert r.status_code == 400
        assert "不支持" in r.json()["detail"]

    def test_empty_file_rejected(self, client):
        r = client.post(
            f"{API}/ingest/file",
            files={"file": ("空.txt", b"", "text/plain")},
        )
        assert r.status_code == 400

    def test_path_traversal_filename_sanitized(self, client):
        """文件名里的路径穿越必须被清洗掉。"""
        content = ("安全测试内容。" * 30).encode()
        r = client.post(
            f"{API}/ingest/file",
            files={"file": ("../../../../etc/passwd.md", content, "text/markdown")},
        )
        assert r.status_code == 200
        assert "/" not in r.json()["data"]["filename"]
        assert ".." not in r.json()["data"]["filename"]


# ============================================================
#  文档管理
# ============================================================
class TestDocumentsApi:
    def test_list_empty(self, client):
        r = client.get(f"{API}/documents")
        assert r.status_code == 200
        assert r.json()["data"]["total_docs"] == 0

    def test_list_after_ingest(self, seeded):
        r = seeded.get(f"{API}/documents")
        data = r.json()["data"]
        assert data["total_docs"] == 2
        assert data["total_chunks"] > 0
        assert {d["filename"] for d in data["documents"]} == {"员工手册-年假", "员工手册-福利"}

    def test_delete_document(self, seeded):
        listing = seeded.get(f"{API}/documents").json()["data"]
        doc_id = listing["documents"][0]["doc_id"]

        r = seeded.delete(f"{API}/documents/{doc_id}")
        assert r.status_code == 200
        assert r.json()["data"]["deleted_chunks"] > 0

        after = seeded.get(f"{API}/documents").json()["data"]
        assert after["total_docs"] == 1

    def test_delete_nonexistent(self, seeded):
        r = seeded.delete(f"{API}/documents/不存在的id")
        assert r.status_code == 404
        assert "doc_id" in r.json()["detail"]


# ============================================================
#  问答
# ============================================================
class TestQueryApi:
    def test_query_without_documents_returns_503(self, client):
        """知识库为空时给 503 + 可操作的指引，而不是一个空答案。"""
        r = client.post(f"{API}/query", json={"question": "年假多少天"})
        assert r.status_code == 503
        assert "ingest" in r.json()["detail"].lower() or "上传" in r.json()["detail"]

    def test_query_returns_answer_with_citations(self, seeded):
        r = seeded.post(f"{API}/query", json={"question": "员工入职满一年后有多少天年假？"})
        assert r.status_code == 200

        data = r.json()["data"]
        assert data["answer"]
        assert len(data["citations"]) > 0
        assert data["offline_mode"] is True
        assert data["model"] == "offline-extractive"

        c = data["citations"][0]
        for field in ("index", "doc_id", "source", "page", "score", "snippet"):
            assert field in c

    def test_citation_source_is_real_document(self, seeded):
        r = seeded.post(f"{API}/query", json={"question": "住房公积金缴存比例是多少"})
        sources = {c["source"] for c in r.json()["data"]["citations"]}
        assert sources <= {"员工手册-年假", "员工手册-福利"}

    def test_answerable_question_is_not_refused(self, seeded):
        """接口层要暴露结构化的拒答字段，调用方不该去匹配 answer 文本。"""
        r = seeded.post(f"{API}/query", json={"question": "员工入职满一年后有多少天年假？"})
        data = r.json()["data"]

        assert data["refused"] is False
        assert data["refusal_reason"] is None

    def test_unanswerable_question_is_flagged_as_refused(self, seeded):
        """知识库外的内容：refused=true + 原因 + 空的引用列表。"""
        r = seeded.post(f"{API}/query", json={"question": "公司的股票代码是多少？"})
        assert r.status_code == 200

        data = r.json()["data"]
        assert data["refused"] is True
        assert data["refusal_reason"] in {"no_retrieval", "model_refused"}
        assert data["citations"] == []

    def test_citation_fallback_flag_present(self, seeded):
        """retrieval_debug 里要有 citation_fallback —— 它与模块文档的承诺对应。"""
        r = seeded.post(f"{API}/query", json={"question": "住房公积金比例"})
        assert "citation_fallback" in r.json()["data"]["retrieval_debug"]

    def test_query_latency_and_debug(self, seeded):
        r = seeded.post(f"{API}/query", json={"question": "住房公积金比例"})
        data = r.json()["data"]

        assert data["latency_ms"] >= 0
        debug = data["retrieval_debug"]
        for key in ("vector_hits", "bm25_hits", "fused", "reranked", "used_hybrid"):
            assert key in debug

    def test_query_top_k_honored(self, seeded):
        r = seeded.post(f"{API}/query", json={"question": "年假", "top_k": 1})
        assert r.status_code == 200
        assert len(r.json()["data"]["citations"]) <= 3

    def test_empty_question_rejected(self, seeded):
        r = seeded.post(f"{API}/query", json={"question": "   "})
        assert r.status_code == 400

    def test_session_memory(self, seeded):
        """同一 session 的两次问答应该被记录。"""
        seeded.post(
            f"{API}/query",
            json={"question": "年假多少天", "session_id": "s-test"},
        )
        seeded.post(
            f"{API}/query",
            json={"question": "住房公积金比例", "session_id": "s-test"},
        )

        stats = seeded.get(f"{API}/sessions/stats").json()["data"]
        assert stats["active_sessions"] >= 1
        assert stats["total_turns"] >= 2

    def test_session_reset(self, seeded):
        seeded.post(f"{API}/query", json={"question": "年假", "session_id": "s-reset"})
        before = seeded.get(f"{API}/sessions/stats").json()["data"]["total_turns"]

        r = seeded.post(f"{API}/sessions/s-reset/reset")
        assert r.status_code == 200
        assert r.json()["data"]["existed"] is True

        after = seeded.get(f"{API}/sessions/stats").json()["data"]["total_turns"]
        assert after < before

    def test_config_status(self, seeded):
        r = seeded.get(f"{API}/config/status")
        assert r.status_code == 200
        data = r.json()["data"]
        assert "embedding" in data
        assert data["llm_offline"] is True
        assert "memory" in data


# ============================================================
#  鉴权
# ============================================================
class TestAuth:
    def test_disabled_by_default(self, client):
        """API_KEY 为空时应该关闭鉴权（本地开发体验）。"""
        r = client.get(f"{API}/documents")
        assert r.status_code == 200

    def test_enabled_rejects_without_key(self, client, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "api_key", "super-secret-key")

        r = client.get(f"{API}/documents")
        assert r.status_code == 401
        assert "API Key" in r.json()["detail"]

    def test_enabled_accepts_x_api_key(self, client, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "api_key", "super-secret-key")

        r = client.get(f"{API}/documents", headers={"X-API-Key": "super-secret-key"})
        assert r.status_code == 200

    def test_enabled_accepts_bearer(self, client, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "api_key", "super-secret-key")

        r = client.get(f"{API}/documents", headers={"Authorization": "Bearer super-secret-key"})
        assert r.status_code == 200

    def test_wrong_key_rejected(self, client, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "api_key", "super-secret-key")

        r = client.get(f"{API}/documents", headers={"X-API-Key": "wrong-key"})
        assert r.status_code == 401

    def test_health_endpoints_not_protected(self, client, monkeypatch):
        """健康检查必须免鉴权，否则容器编排探针会一直失败。"""
        from app.config import settings

        monkeypatch.setattr(settings, "api_key", "super-secret-key")

        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 200


# ============================================================
#  可观测性
# ============================================================
class TestObservability:
    def test_request_id_header(self, client):
        r = client.get("/healthz")
        assert "X-Request-ID" in r.headers
        assert len(r.headers["X-Request-ID"]) == 8

    def test_custom_request_id_echoed(self, client):
        r = client.get("/healthz", headers={"X-Request-ID": "my-trace-id"})
        assert r.headers["X-Request-ID"] == "my-trace-id"

    def test_process_time_header(self, client):
        r = client.get("/healthz")
        assert "X-Process-Time-Ms" in r.headers
