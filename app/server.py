"""FastAPI 服务：把客服 Agent 暴露为可被前端/网关调用的 HTTP API。

- GET  /health        健康检查（含 embedding/llm 模式、chunk 数）
- POST /sessions      创建会话（多用户）
- GET  /sessions/{id} 会话历史
- DELETE/sessions/{id} 删除会话
- POST /chat          单轮问答（同步返回 answer + evidence + confidence）
- POST /chat/stream   流式问答（SSE，逐 token 输出）
- POST /retrieve      仅检索（调试索引质量）

阻塞的 LLM/DB 操作以同步 def 端点运行在线程池，不阻塞事件循环。
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import load_config
from .kefu import build_runtime

logger = logging.getLogger("kefu.server")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    user_id: str = "anonymous"
    top_k: int = Field(default=5, ge=1, le=20)   # 有界：负值/畸形值直接 422，而非静默接受


class CreateSession(BaseModel):
    user_id: str = "anonymous"


app = FastAPI(title="智能客服 Agent", version="0.1.0")
# CORS：允许前端跨域调用；白名单按需收紧 origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
rt: "Runtime" | None = None


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    """为每个请求注入 trace_id 并记录耗时，便于跨模块定位。"""
    trace_id = request.headers.get("X-Trace-ID") or uuid.uuid4().hex[:12]
    request.state.trace_id = trace_id
    import time
    start = time.time()
    response = await call_next(request)
    dur = (time.time() - start) * 1000
    if dur > 200:
        logger.info("trace=%s method=%s path=%s status=%d dur=%.0fms",
                    trace_id, request.method, request.url.path,
                    response.status_code, dur)
    return response


def runtime() -> "Runtime":
    if rt is None:
        raise HTTPException(503, "服务尚未初始化")
    return rt


# ------------------------------------------------ 会话
@app.post("/sessions", tags=["session"])
def create_session(body: CreateSession):
    return {"session_id": runtime().store.create(body.user_id)["id"],
            "user_id": body.user_id}


@app.get("/sessions/{sid}", tags=["session"])
def get_session(sid: str, user_id: str = "anonymous"):
    s = runtime().store.get(sid)
    if not s:
        raise HTTPException(404, "会话不存在")
    if s["user_id"] != user_id:
        raise HTTPException(403, "无权访问他人会话")
    hist = runtime().store.history(sid)
    return {"session_id": sid, "messages": [
        {"role": m.role, "content": m.content, "created_at": m.created_at} for m in hist]}


@app.delete("/sessions/{sid}", tags=["session"])
def delete_session(sid: str, user_id: str = "anonymous"):
    r = runtime()
    s = r.store.get(sid)
    if not s:
        raise HTTPException(404, "会话不存在")
    if s["user_id"] != user_id:
        raise HTTPException(403, "无权访问他人会话")
    r.store.delete(sid)
    return {"ok": True}


@app.get("/sessions", tags=["session"])
def list_sessions(user_id: str = "anonymous"):
    return runtime().store.list_for_user(user_id)


# ------------------------------------------------ 问答
@app.post("/chat", tags=["chat"])
def chat(req: ChatRequest):
    r = runtime()
    sid = _session_id(r, req)
    hist = _history(r, sid)
    result = r.agent.chat(req.message, top_k=_capped_top_k(r, req), history=hist)
    _persist(r, sid, req.message, result.reply)
    return {
        "session_id": sid,
        "reply": result.reply,
        "intent": result.intent,
        "confidence": result.confidence,
        "evidence": result.evidence,
        "query": result.query,
        "error_code": result.error_code,
    }


@app.post("/chat/stream", tags=["chat"])
def chat_stream(req: ChatRequest):
    import queue
    import threading
    r = runtime()
    sid = _session_id(r, req)
    hist = _history(r, sid)

    q: "queue.Queue" = queue.Queue(maxsize=200)
    state: dict = {}

    def sink(piece: str):
        try:
            q.put(("delta", piece), timeout=5)
        except queue.Full:
            logger.warning("SSE 下游消费缓慢，丢弃增量 token")

    def run():
        try:
            out = r.agent.stream(req.message, top_k=_capped_top_k(r, req),
                                 history=hist, on_delta=sink)
        except Exception:  # noqa: BLE001
            # agent.stream 自身已兜底，这里再兜一层：保证 close 帧必达，避免客户端悬挂
            logger.exception("stream 线程异常")
            out = {"status": "error", "text": "系统繁忙，请稍后重试", "evidence": [],
                   "intent": "error", "query": req.message, "error_code": "stream_error"}
        finally:
            # 关键：无论生成成功/失败都推送 close，否则 gen() 的阻塞读会悬挂
            q.put(("close", None))
        state.update(out)
        if out["status"] == "ok":
            _persist(r, sid, req.message, out["text"])

    threading.Thread(target=run, daemon=True).start()

    def gen():
        # 首帧元数据
        yield "event: meta\ndata: " + json.dumps(
            {"session_id": sid}, ensure_ascii=False) + "\n\n"
        while True:
            try:
                kind, payload = q.get(timeout=30)   # 超时兜底：上游卡死也不悬挂客户端
            except queue.Empty:
                yield "event: error\ndata: {\"detail\":\"生成超时\"}\n\n"
                break
            if kind == "close":
                break
            yield "data: " + json.dumps({"delta": payload}, ensure_ascii=False) + "\n\n"
        # 尾帧元数据（含证据/置信度）
        yield "event: meta\ndata: " + json.dumps(
            {"session_id": sid, "status": state.get("status"),
             "evidence": state.get("evidence", []), "intent": state.get("intent"),
             "query": state.get("query"), "error_code": state.get("error_code", "")},
            ensure_ascii=False) + "\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ------------------------------------------------ 检索调试
@app.post("/retrieve", tags=["debug"])
def retrieve(text: str = Query(...), top_k: int = Query(5, ge=1, le=20)):
    r = runtime()
    cfg_cap = int(r.cfg["index"]["vector"].get("top_k", 8))
    top_k = max(1, min(top_k, cfg_cap))
    hits = r.retriever.retrieve(text, top_k=top_k)
    return {"query": text, "hits": [
        {"chunk_id": h.chunk_id, "text": h.text, "section_path": h.section_path,
         "category": h.category, "source": h.source} for h in hits]}


# ------------------------------------------------ 健康/系统
@app.get("/health", tags=["system"])
def health():
    r = runtime()
    return {"status": "ok", "embedding": r.embedding_mode, "llm": r.llm_name,
            "chunks": len(r.kb.chunks), "faqs": len(r.kb.faqs),
            "index_synced": r.index_synced, "vector_dim": r.vector_dim,
            "embedding_fingerprint": r.embedding_fingerprint}


# ------------------------------------------------ 工具
def _capped_top_k(r, req) -> int:
    """客户端 top_k 设上/下限，且不越过配置的检索深度，保证线上行为可复现可调优。"""
    cfg_cap = int(r.cfg["index"]["vector"].get("top_k", 8))
    return max(1, min(req.top_k, cfg_cap))


def _session_id(r, req: ChatRequest) -> str:
    if req.session_id and r.store.get(req.session_id):
        return req.session_id
    return r.store.create(req.user_id)["id"]


def _history(r, sid: str) -> list[dict]:
    return r.store.to_history_messages(sid, limit=10)


def _persist(r, sid: str, user: str, reply: str):
    r.store.add_message(sid, "user", user)
    r.store.add_message(sid, "assistant", reply)


# 供 uvicorn 引用
def make_app(cfg: dict | None = None, force_rebuild: bool = False):
    global rt
    _configure_logging()
    cfg = cfg or load_config()
    rt = build_runtime(cfg, force_rebuild=force_rebuild)
    logger.info("服务就绪 embedding=%s llm=%s chunks=%d faqs=%d",
                rt.embedding_mode, rt.llm_name, len(rt.kb.chunks), len(rt.kb.faqs))
    return app


def _configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


if __name__ == "__main__":
    import uvicorn
    cfg = load_config()
    server = cfg["server"]
    app_ = make_app(cfg)
    uvicorn.run(app_, host=server.get("host", "0.0.0.0"),
                port=int(server.get("port", 8000)))