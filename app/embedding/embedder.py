"""可插拔 Embedding 层。

支持三种模式（由 config.embedding.mode 控制）：
- local :  本地 sentence-transformers + bge-small-zh（推荐，效果最好，离线可跑一次下载）
- hash  :  纯 Python 哈希词袋向量（零依赖，离线立刻可用；效果一般但链路完整）
- api   :  远程 embedding API（如 siliconflow BAAI/bge-m3）

统一 expose：encode(texts: list[str]) -> (np.ndarray[N, dim])，已 L2 归一化。
"""
from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

import numpy as np

from ..config import resolve_secret
from ..wiki.tokenizer import tokenize


class BaseEmbedder(ABC):
    dim = 0
    name = "base"

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def build_embedder(cfg: dict):
    emb = cfg["embedding"]
    mode = emb.get("mode", "api")
    if mode == "local":
        # 明确要求本地模型：加载失败则直接报错
        return LocalEmbedder(emb)
    if mode == "auto":
        # 仅当本地已缓存模型时使用 local（避免触发联网下载），否则快速降级 hash
        try:
            return LocalEmbedder(emb, cached_only=True)
        except Exception:
            return HashEmbedder(emb)
    if mode == "hash":
        return HashEmbedder(emb)
    if mode == "api":
        key = resolve_secret(cfg)["embedding_key"]
        if not key and emb.get("fallback_hash", True):
            return HashEmbedder(emb)
        return APIEmbedder(emb, key)
    raise ValueError(f"未知 embedding.mode: {mode}")


# ---------------------------------------------------------------- local
class LocalEmbedder(BaseEmbedder):
    name = "local"
    dim = 512

    def __init__(self, emb: dict, cached_only: bool = False):
        from sentence_transformers import SentenceTransformer
        name = emb.get("model_name", "BAAI/bge-small-zh-v1.5")
        # cached_only=True 时 local_files_only 不触发联网下载，未缓存直接抛错回落 hash
        self.model = SentenceTransformer(name, local_files_only=cached_only)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


# ---------------------------------------------------------------- hash 回退
class HashEmbedder(BaseEmbedder):
    """零依赖词袋哈希向量：把 token 哈希映射到 dim 维超平面，再 L2 归一化。
    只保证同义词近似匹配弱于 local，但能体现完整 RAG 链路。"""
    name = "hash"

    def __init__(self, emb: dict):
        self.dim = int(emb.get("dim", 512))
        self._plane = self._build_plane()

    def _build_plane(self) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.normal(0.0, 1.0, size=(self.dim,)).astype(np.float32)

    @staticmethod
    def _hash(token: str) -> int:
        return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            vec = np.zeros(self.dim, dtype=np.float32)
            for tok in tokenize(text, min_term_len=2, drop_stopwords=False):
                idx = self._hash(tok) % self.dim
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            out[i] = vec
        return out


# ---------------------------------------------------------------- api
class APIEmbedder(BaseEmbedder):
    name = "api"

    def __init__(self, emb: dict, api_key: str | None):
        import requests
        self._requests = requests
        self.api_key = api_key
        self.base_url = emb.get("api_base_url", "https://api.siliconflow.cn/v1/embeddings")
        self.model = emb.get("api_model", "BAAI/bge-m3")
        self.dim = int(emb.get("dim", 1024))
        self.batch = int(emb.get("batch", 16))

    def _headers(self) -> dict:
        if not self.api_key:
            raise RuntimeError(
                "embedding 使用 API 模式，但未配置密钥。请设置环境变量 "
                "SILICONFLOW_API_KEY。")
        return {"Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"}

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch):
            batch = list(texts[i:i + self.batch])
            out.append(self._one_batch(batch))
        return np.vstack(out) / (np.linalg.norm(np.vstack(out), axis=1, keepdims=True) + 1e-9)

    def _one_batch(self, batch: list[str]) -> np.ndarray:
        import time
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                resp = self._requests.post(
                    self.base_url, headers=self._headers(),
                    json={"model": self.model, "input": batch}, timeout=30)
                if resp.status_code == 429:
                    time.sleep(1 * (attempt + 1))
                    last_err = RuntimeError("限流 429")
                    continue
                if resp.status_code >= 400:
                    raise RuntimeError(f"Embedding API 错误 [{resp.status_code}]: {resp.text[:200]}")
                rows = sorted(resp.json()["data"], key=lambda x: x["index"])
                return np.asarray([r["embedding"] for r in rows], dtype=np.float32)
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Embedding API 重试失败: {last_err}")