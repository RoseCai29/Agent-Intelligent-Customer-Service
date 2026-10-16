"""建表、初始化。"""
from __future__ import annotations

from .db import ConnectionKeeper


def init_db(keeper: ConnectionKeeper) -> None:
    with keeper.lock:
        conn = keeper.connect()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role       TEXT NOT NULL,           -- 'user' | 'assistant' | 'system'
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, id);
            CREATE TABLE IF NOT EXISTS chunks (
                id            TEXT PRIMARY KEY,
                text          TEXT NOT NULL,
                doc_id        INTEGER,
                section_path  TEXT,
                title         TEXT,
                category      TEXT,
                source        TEXT,
                faq_question  TEXT,
                vector        BLOB              -- 归一化向量的 float32 序列
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_category ON chunks(category);
            """
        )
        conn.commit()