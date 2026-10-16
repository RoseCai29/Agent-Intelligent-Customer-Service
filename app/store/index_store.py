"""索引持久化：把 chunk(+向量) 落到 SQLite，加载时免重建、免重新 embedding。

- save(ctx): 对 wiki 全量重建 embedding，upsert 到 chunks 表
- load():    若库中有数据则读回 (chunks, matrix)，供重建 VectorIndex / InvertedIndex
- 设计取舍：向量以 BLOB(float32) 存库；BM25 postings 不做持久化，
  每次 load 时由 chunk 文本廉价复算（tokenize 即可），避免存两份。
"""
from __future__ import annotations

import hashlib

import numpy as np

from ..wiki.models import WikiIndex, Chunk
from ..embedding.embedder import BaseEmbedder
from .db import ConnectionKeeper
from .schema import init_db


def _decode_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _fingerprint(embedder: BaseEmbedder, chunks: list[Chunk]) -> dict:
    """构建索引时记录的 embedding 指纹 + 内容指纹。

    - embedding 指纹：mode + dim，检测「换模型/换维度后仍在用旧向量」；
    - 内容指纹     ：chunk 文本的 sha256 前缀，检测「改了 md 后内容不更新」。
    """
    h = hashlib.sha256()
    for c in sorted(chunks, key=lambda c: c.id):
        h.update(c.text.encode("utf-8"))
    return {"embedding_mode": embedder.name, "dim": str(embedder.dim),
            "content_hash": h.hexdigest()[:16]}


class IndexStore:
    def __init__(self, keeper: ConnectionKeeper):
        self.keeper = keeper
        init_db(keeper)

    # ---- 保存 ----
    def build_and_save(self, wiki: WikiIndex, embedder: BaseEmbedder) -> int:
        chunks = wiki.chunks
        if not chunks:
            return 0
        rows = []
        # 分批 embedding，避免一次请求 body 过大
        vectors = embedder.encode([c.text for c in chunks])
        for c, vec in zip(chunks, vectors):
            rows.append((c.id, c.text, c.doc_id, c.section_path, c.title,
                         c.category, c.source, c.faq_question,
                         np.asarray(vec, dtype=np.float32).tobytes()))
        # DELETE + INSERT 放同一事务，避免并发读观察到"半空表"
        with self.keeper.lock:
            conn = self.keeper.connect()
            conn.execute("BEGIN")
            conn.execute("DELETE FROM chunks")
            conn.executemany(
                "INSERT INTO chunks(id, text, doc_id, section_path, title, category, "
                "source, faq_question, vector) VALUES(?,?,?,?,?,?,?,?,?);", rows)
            conn.commit()
        self.save_meta(_fingerprint(embedder, chunks))
        return len(rows)

    def save_chunks(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        rows = []
        for c, vec in zip(chunks, vectors):
            rows.append((c.id, c.text, c.doc_id, c.section_path, c.title,
                         c.category, c.source, c.faq_question,
                         np.asarray(vec, dtype=np.float32).tobytes()))
        with self.keeper.lock:
            conn = self.keeper.connect()
            conn.executemany(
                "INSERT OR REPLACE INTO chunks(id, text, doc_id, section_path, title, "
                "category, source, faq_question, vector) VALUES(?,?,?,?,?,?,?,?,?);",
                rows)
            conn.commit()
        return len(rows)

    # ---- 加载 ----
    def count(self) -> int:
        row = self.keeper.query("SELECT COUNT(*) AS c FROM chunks")
        return row[0]["c"]

    def load(self) -> tuple[list[Chunk], np.ndarray]:
        rows = self.keeper.query(
            "SELECT id, text, doc_id, section_path, title, category, source, "
            "faq_question, vector FROM chunks ORDER BY rowid")
        chunks: list[Chunk] = []
        vecs: list[np.ndarray] = []
        for r in rows:
            chunks.append(Chunk(
                id=r["id"], text=r["text"], doc_id=r["doc_id"],
                sec_id=None, section_path=r["section_path"] or "", title=r["title"] or "",
                category=r["category"] or "", source=r["source"] or "",
                faq_question=r["faq_question"],
            ))
            blob = r["vector"]
            vecs.append(_decode_vec(blob) if blob else np.zeros(0, dtype=np.float32))
        matrix = np.stack(vecs).astype(np.float32) if vecs else np.zeros((0, 0), dtype=np.float32)
        return chunks, matrix

    # ---- 索引指纹 ----
    def save_meta(self, meta: dict) -> None:
        with self.keeper.lock:
            conn = self.keeper.connect()
            conn.executemany(
                "INSERT OR REPLACE INTO index_meta(key, value) VALUES(?,?)",
                list(meta.items()))
            conn.commit()

    def load_meta(self) -> dict:
        rows = self.keeper.query("SELECT key, value FROM index_meta")
        return {r["key"]: r["value"] for r in rows}