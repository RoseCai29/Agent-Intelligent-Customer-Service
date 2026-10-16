"""组装完整 Agent：加载配置 → wiki → embedding → 索引 → 检索 → LLM → Agent。"""
from __future__ import annotations

from ..config import load_config
from ..wiki.loader import load_directory
from ..embedding import build_embedder
from ..index.vector_index import VectorIndex
from ..index.inverted_index import InvertedIndex
from ..retriever.retrieval import Retriever
from ..retriever.reranker import Reranker
from ..llm.base import build_llm
from .agent import CustomerServiceAgent


def build_pipeline(cfg: dict | None = None, use_cross_encoder: bool = True):
    cfg = cfg or load_config()
    wiki_cfg = cfg["wiki"]
    embedder = build_embedder(cfg)

    wiki = load_directory(wiki_cfg["data_dirs"], chunk_cfg=wiki_cfg.get("chunker"))

    vector_index = VectorIndex.build(wiki, embedder)
    inverted_cfg = cfg["index"]["inverted"]
    inverted_index = InvertedIndex.build(
        wiki, tokenizer_cfg=cfg.get("tokenizer"),
        k1=float(inverted_cfg.get("k1", 1.5)), b=float(inverted_cfg.get("b", 0.75)),
    )
    reranker = Reranker(cfg["rerank"], use_cross_encoder=use_cross_encoder)
    retriever = Retriever(wiki, vector_index, inverted_index, embedder,
                          cfg=cfg, reranker=reranker)
    llm = build_llm(cfg)
    agent_cfg = cfg["agent"]
    agent = CustomerServiceAgent(
        llm=llm, retriever=retriever,
        system_prompt=agent_cfg["system_prompt"],
        fallback=agent_cfg.get("no_answer_fallback", ""),
        max_history=int(agent_cfg.get("max_history", 6)),
    )
    return agent, {"wiki": wiki, "embedder": embedder, "retriever": retriever,
                   "vector_index": vector_index, "inverted_index": inverted_index}