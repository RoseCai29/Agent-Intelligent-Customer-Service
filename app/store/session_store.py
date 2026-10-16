"""会话与消息持久化（多用户、会话隔离、重启不丢）。"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from .db import ConnectionKeeper
from .schema import init_db


@dataclass
class Message:
    role: str
    content: str
    created_at: str = ""


def _sid() -> str:
    return uuid.uuid4().hex


class SessionStore:
    def __init__(self, keeper: ConnectionKeeper):
        self.keeper = keeper
        init_db(keeper)

    def create(self, user_id: str = "anonymous") -> dict:
        sid = _sid()
        self.keeper.execute(
            "INSERT INTO sessions(id, user_id) VALUES(?, ?)", (sid, user_id))
        return self.get(sid)

    def get(self, session_id: str) -> dict | None:
        row = self.keeper.query("SELECT * FROM sessions WHERE id=?", (session_id,))
        return dict(row[0]) if row else None

    def list_for_user(self, user_id: str, limit: int = 50) -> list[dict]:
        rows = self.keeper.query(
            "SELECT * FROM sessions WHERE user_id=? ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit))
        return [dict(r) for r in rows]

    def delete(self, session_id: str) -> bool:
        cur = self.keeper.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        return cur.rowcount > 0

    def touch(self, session_id: str):
        self.keeper.execute(
            "UPDATE sessions SET updated_at=datetime('now') WHERE id=?", (session_id,))

    # ---- 消息 ----
    def add_message(self, session_id: str, role: str, content: str) -> None:
        self.keeper.execute(
            "INSERT INTO messages(session_id, role, content) VALUES(?,?,?)",
            (session_id, role, content))
        self.touch(session_id)

    def history(self, session_id: str, limit: int = 100) -> list[Message]:
        rows = self.keeper.query(
            "SELECT role, content, created_at FROM messages WHERE session_id=? "
            "ORDER BY id DESC LIMIT ?", (session_id, limit))
        msgs = [Message(role=r["role"], content=r["content"], created_at=r["created_at"]) for r in rows]
        msgs.reverse()
        return msgs

    def to_history_messages(self, session_id: str, limit: int = 10) -> list[dict]:
        """转为 LLM 记忆 messages（user/assistant 交替）。"""
        h = self.history(session_id, limit=limit * 2)
        out = []
        for m in h:
            if m.role in ("user", "assistant"):
                out.append({"role": m.role, "content": m.content})
        return out[-limit * 2:]