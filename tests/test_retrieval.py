"""RAG 检索质量测试：验证混合检索能召回正确知识块/FAQ。"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import load_config
from app.embedding import build_embedder
from app.wiki.loader import load_directory
from app.index.vector_index import VectorIndex
from app.index.inverted_index import InvertedIndex
from app.retriever.retrieval import Retriever
from app.retriever.reranker import Reranker


def _build_retriever(cfg):
    wiki = load_directory(cfg["wiki"]["data_dirs"], chunk_cfg=cfg["wiki"].get("chunker"))
    embedder = build_embedder(cfg)
    vindex = VectorIndex.build(wiki, embedder)
    iindex = InvertedIndex.build(wiki, tokenizer_cfg=cfg.get("tokenizer"))
    reranker = Reranker(cfg["rerank"], use_cross_encoder=False)
    return Retriever(wiki, vindex, iindex, embedder, cfg=cfg, reranker=reranker)


def test_tokenizer_bigram():
    from app.wiki.tokenizer import tokenize
    toks = tokenize("退款多久到账")
    assert "退款" in toks
    assert "到账" in toks


def test_retrieve_refund_hits_correct():
    cfg = load_config()
    r = _build_retriever(cfg)
    res = r.retrieve("退款多久到账", top_k=3)
    assert res, "应有检索结果"
    assert any("退款" in x.text or "到账" in x.text or x.category == "售后" for x in res), \
        f"应召回退款相关信息: {[x.section_path for x in res]}"


def test_retrieve_shipping_free():
    cfg = load_config()
    r = _build_retriever(cfg)
    res = r.retrieve("满多少免运费", top_k=3)
    assert any("免邮" in x.section_path or "运费" in x.section_path for x in res)


def test_faq_exact_match_boosted():
    cfg = load_config()
    r = _build_retriever(cfg)
    res = r.retrieve("退款多久到账", top_k=3)
    assert res[0].from_faq, "FAQ 精确命中应置顶"


def test_lexical_query_hits_shipping():
    """关键词查询：验证 BM25 能命中发货/物流内容。"""
    cfg = load_config()
    r = _build_retriever(cfg)
    res = r.retrieve("商品什么时候发货", top_k=3)
    assert any("发货" in x.text or "物流" in x.text or "顺延" in x.text for x in res), \
        f"应命中物流块: {[x.section_path for x in res]}"


def test_vector_and_inverted_both_return():
    cfg = load_config()
    r = _build_retriever(cfg)
    vec = r._vector_recall(r.embedder.embed("退款"), 5)
    bm = r._bm25_recall("退款", 5)
    assert vec and bm, "向量与BM25都应返回结果"


if __name__ == "__main__":
    # 简单断言式运行，无需 pytest
    for fn in [test_tokenizer_bigram, test_retrieve_refund_hits_correct,
               test_retrieve_shipping_free, test_faq_exact_match_boosted,
               test_lexical_query_hits_shipping, test_vector_and_inverted_both_return]:
        fn()
        print(f"PASS {fn.__name__}")
    print("全部通过")