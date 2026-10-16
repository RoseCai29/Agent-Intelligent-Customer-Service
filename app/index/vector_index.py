"""向量索引：存储全部 chunk 的 embedding（L2 归一化），做余弦相似度检索。

因为 embedding 已归一化，余弦相似度 = 点积。采用暴力遍历 + numpy 向量化的方式，
对小规模（本项目的知识库）是正确且简单可讲的选择；规模化时上可换 ANN（HNSW/IVF）。
支持 top_k 检索，返回 (chunk_id, score) 列表，score≈相似度。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..wiki.models import WikiIndex
from ..embedding.embedder import BaseEmbedder


@dataclass
class Hit:
    chunk_id: str
    score: float


class VectorIndex:
    def __init__(self, chunk_ids: list[str], matrix: np.ndarray):
        self.chunk_ids = chunk_ids                 # 与 matrix 行一一对应
        self.matrix = matrix                        # [N, dim] 已归一化
        self._pos = {cid: i for i, cid in enumerate(chunk_ids)}

    @classmethod
    def build(cls, wiki, embedder: BaseEmbedder) -> "VectorIndex":
        chunks = getattr(wiki, "chunks", wiki)  # WikiIndex/KnowledgeBase/list
        ids = [c.id for c in chunks]
        if not ids:
            return cls([], np.zeros((0, embedder.dim), dtype=np.float32))
        matrix = embedder.encode([c.text for c in chunks])
        return cls(ids, matrix)

    @classmethod
    def from_vectors(cls, chunk_ids: list[str], matrix: np.ndarray) -> "VectorIndex":
        return cls(chunk_ids, matrix)

    def query(self, query_vec: np.ndarray, top_k: int = 8) -> list[Hit]:
        if self.matrix.shape[0] == 0:
            return []
        sims = self.matrix @ query_vec  # [N,]
        order = np.argsort(-sims)[:top_k]
        return [Hit(self.chunk_ids[i], float(sims[i])) for i in order]

    def get_matrix(self) -> np.ndarray:
        return self.matrix