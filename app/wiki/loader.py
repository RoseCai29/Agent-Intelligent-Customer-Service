"""从磁盘加载 wiki 文档并解析为结构化 WikiIndex。"""
from __future__ import annotations

import re
from pathlib import Path

from .models import Document, WikiIndex, Faq
from .parser import parse_markdown


def load_directory(dirs: list[str] | str, chunk_cfg: dict | None = None) -> WikiIndex:
    index = WikiIndex()
    if isinstance(dirs, str):
        dirs = [dirs]
    seen_sources: set[str] = set()
    for d in dirs:
        base = Path(d)
        for md in sorted(base.rglob("*.md")):
            if md.name.startswith("_"):
                continue
            text = md.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                seen_sources.add(str(md))
                doc = parse_markdown(text, source=str(md), chunk_cfg=chunk_cfg)
                _index_flatten(index, doc)
    return index


def _index_flatten(index: WikiIndex, doc: Document):
    index.register(doc)


def index_faqs(index: WikiIndex) -> list[Faq]:
    faqs = []
    for doc in index.documents:
        faqs.extend(doc.faqs)
    return faqs