"""SQLite 连接管理：线程安全（每连接一个串行锁，WAL 模式支持并发读写）。"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None
_conn_lock = threading.Lock()


class ConnectionKeeper:
    """持有一个连接 + 一把锁，供多线程串行访问单个 SQLite 连接。"""
    def __init__(self, db_path: str | Path):
        self.path = str(db_path)
        self.lock = threading.RLock()
        self.conn: sqlite3.Connection | None = None

    def connect(self):
        with self.lock:
            if self.conn is None:
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(self.path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA foreign_keys=ON")
                self.conn = conn
            return self.conn

    def execute(self, sql, params=()):
        conn = self.connect()
        with self.lock:
            cur = conn.execute(sql, params)
            conn.commit()
            return cur

    def query(self, sql, params=()):
        conn = self.connect()
        with self.lock:
            cur = conn.execute(sql, params)
            return cur.fetchall()

    def close(self):
        with self.lock:
            if self.conn is not None:
                self.conn.close()
                self.conn = None