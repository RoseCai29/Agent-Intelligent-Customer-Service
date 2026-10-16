"""服务端应用工厂：连接 SQLite 持久化 + 索引 + 检索 + Agent。

启动策略：
- 库里已有 chunk 向量 → 直接从库加载（免重新 embedding、免重建），启动即用
- 库为空 / 显式 force_rebuild → 从 markdown 重建并落盘（首次或知识库升级）
- wiki .md 仅用于低频解析获取 FAQ 与结构；实际的 chunk/向量以 SQLite 为权威
"""
from __future__ import annotations

from .config import load_config
from .store.db import ConnectionKeeper
from .store.session_store import SessionStore
from .store.index_store import IndexStore
from .wiki.loader import load_directory
from .wiki.models import KnowledgeBase
from .embedding import build_embedder
from .index.vector_index import VectorIndex
from .index.inverted_index import InvertedIndex
from .retriever.retrieval import Retriever
from .retriever.reranker import Reranker
from .llm.base import build_llm
from .agent.agent import CustomerServiceAgent


class Runtime:
    """持有一切运行依赖的容器。"""
    def __init__(self, cfg: dict, embedder, wiki, kb, vector, inverted,
                 retriever, llm, agent, store: SessionStore, index_store: IndexStore,
                 keeper: ConnectionKeeper):
        self.cfg = cfg
        self.embedder = embedder
        self.wiki = wiki
        self.kb = kb
        self.vector_idx = vector
        self.inverted_idx = inverted
        self.retriever = retriever
        self.llm = llm
        self.agent = agent
        self.store = store
        self.index_store = index_store
        self.keeper = keeper
        # 报告用
        self.embedding_mode = embedder.name
        self.llm_name = llm.name


def build_runtime(cfg: dict | None = None, force_rebuild: bool = False,
                  use_cross_encoder: bool = False) -> Runtime:
    cfg = cfg or load_config()
    server_cfg = cfg["server"]
    db_path = server_cfg.get("store_path", "data/kefu.db")

    keeper = ConnectionKeeper(db_path)
    session_store = SessionStore(keeper)
    index_store = IndexStore(keeper)
    embedder = build_embedder(cfg)

    # 1) markdown 解析 → 取 FAQ 与结构（轻量，始终执行）
    wiki = load_directory(cfg["wiki"]["data_dirs"], chunk_cfg=cfg["wiki"].get("chunker"))

    # 2) 索引来源：优先从库加载，否则重建并落盘
    if not force_rebuild and index_store.count() > 0:
        chunks, matrix = index_store.load()
        kb = KnowledgeBase(chunks, wiki.faqs)
        vector_idx = VectorIndex.from_vectors([c.id for c in chunks], matrix)
    else:
        index_store.build_and_save(wiki, embedder)   # 重新 embedding 并落盘
        kb = KnowledgeBase.from_wiki(wiki)
        vector_idx = VectorIndex.build(kb, embedder)

    inverted_idx = InvertedIndex.build(kb, tokenizer_cfg=cfg.get("tokenizer"),
                                       k1=float(cfg["index"]["inverted"].get("k1", 1.5)),
                                       b=float(cfg["index"]["inverted"].get("b", 0.75)))
    reranker = Reranker(cfg["rerank"], use_cross_encoder=use_cross_encoder)
    retriever = Retriever(kb, vector_idx, inverted_idx, embedder, cfg=cfg, reranker=reranker)
    llm = build_llm(cfg)
    agent_cfg = cfg["agent"]
    agent = CustomerServiceAgent(
        llm=llm, retriever=retriever, system_prompt=agent_cfg["system_prompt"],
        fallback=agent_cfg.get("no_answer_fallback", ""),
        max_history=int(agent_cfg.get("max_history", 6)),
    )
    return Runtime(cfg, embedder, wiki, kb, vector_idx, inverted_idx, retriever,
                   llm, agent, session_store, index_store, keeper)