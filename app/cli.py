"""智能客服 Agent CLI。

用法：
  python -m app.cli                           # 进入交互式对话
  python -m app.cli --query "退款多久到账"     # 单次询问
  python -m app.cli --retrieve "退款"          # 只看检索结果（离线演示检索质量）
"""
from __future__ import annotations

import argparse

from .config import load_config
from .agent.pipeline import build_pipeline


def main():
    args = _parse()
    cfg = load_config()
    agent, parts = build_pipeline(cfg, use_cross_encoder=not args.no_ce)

    if args.retrieve:
        _show_retrieval(parts["retriever"], args.retrieve)
        return

    if args.query:
        result = agent.chat(args.query, top_k=args.top_k)
        print(f"Q: {args.query}")
        print(f"A: {result.reply}")
        _show_confidence(result)
        return

    print("客服 Agent 已就绪（输入 exit/退出 结束）。")
    if agent.llm.name == "mock":
        print("提示：未配置 DEEPSEEK_API_KEY，当前用 mock 生成（可演示 RAG 链路）。")
    while True:
        try:
            u = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return
        if not u:
            continue
        result = agent.chat(u, top_k=args.top_k)
        print(f"客服> {result.reply}")
        _show_confidence(result)
        if result.intent == "exit":
            return


def _show_retrieval(retriever, query: str):
    res = retriever.retrieve(query, top_k=5)
    print(f"查询: {query}")
    if not res:
        print("  无结果")
        return
    for i, r in enumerate(res, 1):
        print(f"  #{i} [{r.section_path}] score={r.fused_score:.3f} v={r.vector_score:.3f} bm25={r.bm25_score:.3f}")
        print(f"      {r.text[:80]}")


def _show_confidence(result):
    if result.used_retrieval:
        print(f"[置信度 {result.confidence:.2f} | 来源 {len(result.evidence)} 条]")
        for e in result.evidence:
            print(f"   ↳ {e['category']}/{e['section_path']}  {e['source']}")


def _parse():
    p = argparse.ArgumentParser(description="智能客服 Agent")
    p.add_argument("--query", default=None, help="单次询问")
    p.add_argument("--retrieve", default=None, help="只做检索(离线验证检索质量)")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--no-ce", action="store_true", help="禁用交叉编码重排")
    return p.parse_args()


if __name__ == "__main__":
    main()