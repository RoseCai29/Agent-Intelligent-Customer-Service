"""重排器：混合通道后对候选做精细打分。

两层：
1. 无模型启发式加分（结合结构化知识库元信息）：
   - FAQ 精确命中已在 retrieval 阶段置顶
   - 标题/keyword 重合加分、category 加分
2. 可选交叉编码重排（cross-encoder/ms-marco-MiniLM-L-6-v2），对 query 与块做联合编码，
   逼近语义细节；未安装 sentence-transformers 时自动跳过，仅用启发式 + RRF 排序。

说明：本项目知识库规模小，交叉编码可用；生产上可替换为更准的中文 ce 模型。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..wiki.tokenizer import tokenize

if TYPE_CHECKING:
    from .retrieval import RetrievedChunk


class Reranker:
    def __init__(self, cfg: dict | None = None, use_cross_encoder: bool = True):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.boost = cfg.get("boost", {})
        self.cross_encoder = None
        if use_cross_encoder and self.enabled:
            try:
                from sentence_transformers import CrossEncoder
                model = cfg.get("cross_encoder_model",
                                "cross-encoder/ms-marco-MiniLM-L-6-v2")
                self.cross_encoder = CrossEncoder(model)
            except Exception:
                self.cross_encoder = None

    def rerank(self, query: str, items: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not items:
            return items
        # 1) 交叉编码（真实语义）得分
        ce_scores: dict[str, float] = {}
        if self.cross_encoder is not None:
            pairs = [(query, it.text) for it in items]
            scores = self.cross_encoder.predict(pairs)
            if getattr(scores, "ndim", 0) == 2 and scores.shape[1] == 2:
                scores = scores[:, 1] - scores[:, 0]  # 取 【相关】 类的logit
            for it, s in zip(items, scores):
                ce_scores[it.chunk_id] = float(s)
        else:
            # 降级启发式语义分：token 重合度
            qset = set(tokenize(query, min_term_len=2))
            for it in items:
                tset = set(tokenize(it.text, min_term_len=2))
                ce_scores[it.chunk_id] = _f1(qset, tset)

        for it in items:
            score = it.fused_score
            if ce_scores:
                score += 0.5 * ce_scores.get(it.chunk_id, 0.0)
            # 启发式加分
            score += it.vector_score * 0.2 + it.bm25_score * 0.1
            score += self._heuristic_boost(query, it)
            it.fused_score = score
        items.sort(key=lambda x: x.fused_score, reverse=True)
        return items

    def _heuristic_boost(self, query: str, it: RetrievedChunk) -> float:
        boost = 0.0
        q = query.strip().strip('?？')
        # 标题命中
        if it.title and q in it.title:
            boost += float(self.boost.get("title_keyword_boost", 0.5))
        # 分类命中
        if it.category and q in it.category:
            boost += float(self.boost.get("category_boost", 0.3))
        return boost


def _f1(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    p = inter / len(b)
    r = inter / len(a)
    return 2 * p * r / (p + r)