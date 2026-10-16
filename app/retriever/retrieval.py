"""混合检索：向量（语义）+ BM25（关键词）→ RRF 融合 → （可选）重排。

为什么用 RRF：向量与 BM25 分数尺度、分布完全不同，直接相加会偏向某一侧；
RRF 只依赖"排名"，鲁棒且无需调尺度，公式 sum 1/(k+rank)。

管线：retriever.retrieve(query) -> List[RetrievedChunk]
RetrievedChunk 含 chunk + raw 分数 + 融合分 + 来源(哪些通道召回)。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..index.vector_index import VectorIndex
from ..index.inverted_index import InvertedIndex
from ..wiki.models import WikiIndex
from ..embedding.embedder import BaseEmbedder
from .reranker import Reranker


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    section_path: str
    title: str
    category: str
    source: str
    vector_score: float
    bm25_score: float
    fused_score: float      # 重排后总分
    from_faq: bool = False


class RRF_Weights:
    def __init__(self, vector: float = 1.0, inverted: float = 1.0, k: int = 60):
        self.vector = vector
        self.inverted = inverted
        self.k = k


class Retriever:
    def __init__(self, wiki: WikiIndex, vector_index: VectorIndex,
                 inverted_index: InvertedIndex, embedder: BaseEmbedder,
                 cfg: dict | None = None, reranker: "Reranker" | None = None):
        self.wiki = wiki
        self.vector_index = vector_index
        self.inverted_index = inverted_index
        self.embedder = embedder
        self.cfg = cfg or {}
        self.reranker = reranker
        hy = (cfg or {}).get("hybrid", {})
        self.k_const = int(hy.get("k", 60))
        w = hy.get("weights", {"vector": 1.0, "inverted": 1.0})
        self.w_vector = float(w.get("vector", 1.0))
        self.w_inverted = float(w.get("inverted", 1.0))

    # ---- 查询向量化 + 召回 ----
    def _vector_recall(self, query_vec: np.ndarray, top_k: int):
        return self.vector_index.query(query_vec, top_k=top_k)

    def _bm25_recall(self, query: str, top_k: int):
        return self.inverted_index.query(query, top_k=top_k)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        cfg_v = (self.cfg.get("index") if self.cfg else None) or {}
        vec_k = cfg_v.get("vector", {}).get("top_k", 8) if cfg_v else 8
        bm_k = cfg_v.get("inverted", {}).get("top_k", 8) if cfg_v else 8
        vec_hits = self._vector_recall(self.embedder.embed(query), vec_k)
        bm_hits = self._bm25_recall(query, bm_k)

        # RRF 融合
        rrf: dict[str, float] = {}
        for rank, h in enumerate(sorted(vec_hits, key=lambda x: -x.score)):
            rrf[h.chunk_id] = rrf.get(h.chunk_id, 0.0) + self.w_vector / (self.k_const + rank + 1)
        for rank, h in enumerate(sorted(bm_hits, key=lambda x: -x.score)):
            rrf[h.chunk_id] = rrf.get(h.chunk_id, 0.0) + self.w_inverted / (self.k_const + rank + 1)

        vec_sc = {h.chunk_id: h.score for h in vec_hits}
        bm_sc = {h.chunk_id: h.score for h in bm_hits}

        merged: list[RetrievedChunk] = []
        for cid, fused in sorted(rrf.items(), key=lambda kv: kv[1], reverse=True):
            chunk = self.wiki.get_chunk(cid)
            if chunk is None:
                continue
            merged.append(RetrievedChunk(
                chunk_id=cid, text=chunk.text, section_path=chunk.section_path,
                title=chunk.title, category=chunk.category, source=chunk.source,
                vector_score=vec_sc.get(cid, 0.0), bm25_score=bm_sc.get(cid, 0.0),
                fused_score=fused,
            ))

        # FAQ 精确命中提升（结构化知识库的杀手锏）
        self._faq_exact_boost(query, merged)

        # 重排
        if self.reranker is not None and self.reranker.enabled:
            merged = self.reranker.rerank(query, merged)
        return merged[:top_k]

    def _faq_exact_boost(self, query: str, merged: list[RetrievedChunk]):
        """如果用户问题与某条 FAQ 的问题文本高度重合，把对应 chunk/来源顶到最前。"""
        best = None
        best_q = 0.0
        for f in self.wiki.faqs:
            q_norm = f.question.strip().strip('?？')
            qq = query.strip().strip('?？')
            if not q_norm:
                continue
            # 简单重合度：问题句间包含 or 共现单词比例
            ratio = _overlap_ratio(qq, q_norm)
            if ratio > best_q:
                best_q = ratio
                best = f
        if best is not None and best_q >= 0.6:
            # 找到对应 Markdown FAQ 块对应的 chunk（text 含答案）
            target = [c for c in merged if best.answer and best.answer.strip()[:30] in c.text]
            # 找不到就构造一个临时直达项
            fake = RetrievedChunk(
                chunk_id=f"faq:{best.id}", text=best.answer, section_path="FAQ",
                title=best.title, category=best.category, source=best.source,
                vector_score=0.0, bm25_score=0.0, fused_score=99.0, from_faq=True,
            )
            merged.insert(0, fake)


def _overlap_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 0.9
    a = set(a)
    b = set(b)
    return len(a & b) / max(len(a | b), 1)