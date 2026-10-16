"""倒排索引 + BM25 检索。

BM25 评分：对查询词 q，score = IDF(q) * tf * (k1+1) / (tf + k1*(1 - b + b*dl/avgdl))
- 自研：用上述标准公式，无外部库
- 倒排最终存储：term -> [(doc_idx, tf)，...]，doc_idx 对应 wiki.chunks 的扁平下标
- 支持 IDF 平滑（+1）避免未见词除零

返回 (chunk_id, score)，score 为 BM25 原始值（未归一化，供 RRF 融合）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..wiki.models import WikiIndex
from ..wiki.tokenizer import tokenize


@dataclass
class Hit:
    chunk_id: str
    score: float


class InvertedIndex:
    def __init__(self, chunk_ids: list[str], postings: dict[str, dict[int, int]],
                 doc_lens: list[int], avgdl: float, idf: dict[str, float], k1: float, b: float):
        self.chunk_ids = chunk_ids
        self.postings = postings
        self.doc_lens = doc_lens
        self.avgdl = avgdl
        self.idf = idf
        self.k1 = k1
        self.b = b

    @classmethod
    def build(cls, wiki, tokenizer_cfg: dict | None = None,
              k1: float = 1.5, b: float = 0.75) -> "InvertedIndex":
        cfg = tokenizer_cfg or {}
        min_term_len = int(cfg.get("min_term_len", 2))
        drop = bool(cfg.get("drop_stopwords", True))
        chunks = getattr(wiki, "chunks", wiki)  # WikiIndex/KnowledgeBase/list
        n = len(chunks)
        postings: dict[str, dict[int, int]] = {}
        doc_lens = [0] * n
        for i, c in enumerate(chunks):
            toks = tokenize(c.text, min_term_len=min_term_len, drop_stopwords=drop)
            doc_lens[i] = len(toks)
            seen: dict[str, int] = {}
            for t in toks:
                seen[t] = seen.get(t, 0) + 1
            for t, tf in seen.items():
                postings.setdefault(t, {})[i] = tf
        df = {t: len(p) for t, p in postings.items()}
        idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        avgdl = sum(doc_lens) / n if n else 0.0
        return cls(chunks_id_or_ids(chunks), postings, doc_lens, avgdl, idf, k1, b)

    def query(self, text: str, top_k: int = 8) -> list[Hit]:
        toks = tokenize(text, min_term_len=2, drop_stopwords=self.drop())
        n = len(self.chunk_ids)
        if n == 0 or not toks:
            return []
        scores: dict[int, float] = {}
        for t in toks:
            idf = self.idf.get(t, 0.0)
            if idf <= 0:
                continue
            for doc_idx, tf in self.postings.get(t, {}).items():
                dl = self.doc_lens[doc_idx]
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avgdl if self.avgdl else 1.0))
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * tf * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [Hit(self.chunk_ids[i], s) for i, s in ranked]

    def drop(self) -> bool:
        return True

    # 小工具：统计每 chunk 的词数（供调试）
    @staticmethod
    def chunk_token_count(chunk_text: str, min_term_len: int = 2) -> int:
        return len(tokenize(chunk_text, min_term_len=min_term_len))


def chunks_id_or_ids(chunks) -> list[str]:
    return [c.id for c in chunks]