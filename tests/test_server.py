"""服务端 API 集成测试 + SQLite 持久化测试。

注意：不依赖真实 key —— embedding 走 hash 降级、LLM 走 mock，
因此可在离线环境验证：索引落盘/加载、会话持久、chat/retrieve/health 全通。
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

from app.config import load_config
from app.server import make_app


@pytest.fixture()
def client():
    cfg = load_config()
    # 用临时库，规避污染真实 data/kefu.db
    cfg["server"]["store_path"] = os.path.join(tempfile.mkdtemp(), "test.db")
    cfg["embedding"]["mode"] = "hash"
    cfg["llm"]["mock"] = True
    app = make_app(cfg, force_rebuild=True)
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["chunks"] > 0
    assert body["embedding"] == "hash"
    assert body["llm"] == "mock"
    # 索引同步状态 & embedding 指纹应暴露（运维据此发现「换了模型/改了 md 索引没重建」）
    assert body["index_synced"] is True
    assert body["vector_dim"] > 0
    assert body["embedding_fingerprint"]


def test_create_and_get_session(client):
    r = client.post("/sessions", json={"user_id": "u1"})
    assert r.status_code == 200
    sid = r.json()["session_id"]
    # 发送提问 → 应持久化到该会话
    cr = client.post("/chat", json={"session_id": sid, "message": "退款多久到账", "user_id": "u1"})
    assert cr.status_code == 200
    body = cr.json()
    assert body["session_id"] == sid
    assert body["evidence"], "应有检索证据"
    # 读回历史（需归属校验：用创建时相同 user_id）
    hr = client.get(f"/sessions/{sid}", params={"user_id": "u1"})
    assert hr.status_code == 200
    msgs = hr.json()["messages"]
    roles = [m["role"] for m in msgs]
    assert "user" in roles and "assistant" in roles
    # 越权访问他人会话应被拒
    assert client.get(f"/sessions/{sid}", params={"user_id": "hacker"}).status_code == 403


def test_chat_epoch(client):
    r = client.post("/chat", json={"message": "满多少免运费", "user_id": "u2"})
    assert r.status_code == 200
    b = r.json()
    assert b["intent"] in ("answer",)
    # 无 session_id 应自动建会话
    assert b["session_id"]


def test_retrieve(client):
    r = client.post("/retrieve?text=退款&top_k=3")
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert hits, "应命中退款相关块"


def test_stream_returns_sse(client):
    with client.stream("POST", "/chat/stream",
                       json={"message": "优惠券可以叠加吗", "user_id": "u3"}) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    assert "data:" in body
    assert "event: done" in body


def test_delete_session(client):
    r = client.post("/sessions", json={"user_id": "u9"})
    sid = r.json()["session_id"]
    # 他人无权删除
    assert client.delete(f"/sessions/{sid}", params={"user_id": "other"}).status_code == 403
    d = client.delete(f"/sessions/{sid}", params={"user_id": "u9"})
    assert d.status_code == 200
    assert client.get(f"/sessions/{sid}", params={"user_id": "u9"}).status_code == 404