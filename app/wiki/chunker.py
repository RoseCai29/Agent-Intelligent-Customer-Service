"""分层感知分块器。

策略（面试可讲）：
- 以段落（\n\n）为基本单元，优先用段落边界切分，保持语义完整；
- 段落过长时回退到句子边界（按中英文句号/问号/叹号/换行），仍超限则硬切；
- 支持块间重叠（overlap）以缓解"关键信息恰好跨块"的问题；
- 层级感知在 parser 层已经保证按 Section 独立分块，块并不过界标题。

参数（config.wiki.chunker）：
- max_chars    : 目标块字符数
- overlap_chars: 相邻块重叠字符数
- respect_headings : 是否优先在段落/句边界切（总是为真，此处保留开关位）
"""
from __future__ import annotations

import re


def split_text(text: str, cfg: dict | None = None) -> list[str]:
    cfg = cfg or {}
    max_chars = int(cfg.get("max_chars", 200))
    overlap = int(cfg.get("overlap_chars", 40))
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""

    def flush():
        nonlocal buf
        for piece in _cut(buf, max_chars, overlap):
            chunks.append(piece)
        if overlap and len(buf) > overlap:
            keep = buf[-overlap:]
            buf = keep
        else:
            buf = ""

    for para in paragraphs:
        if not buf:
            buf = para
        elif len(buf) + 1 + len(para) <= max_chars:
            buf = buf + "\n" + para
        else:
            # 若本段单独就超长，直接单独切割，不与累计的 buf 混切
            if len(para) > max_chars:
                flush()
                if buf.strip():
                    chunks.extend(_cut(buf, max_chars, overlap))
                    buf = ""
                for piece in _cut(para, max_chars, 0):
                    chunks.append(piece)
                buf = ""
            else:
                flush()
                buf = para

    if buf.strip():
        chunks.extend(_cut(buf, max_chars, overlap))
    return _dedupe_pieces(chunks)


def _cut(segment: str, max_chars: int, overlap: int) -> list[str]:
    """对单一连续段做句子优先、硬切兜底的切割。"""
    if len(segment) <= max_chars:
        return [segment]

    sentences = [s for s in _SENT_SPLIT.split(segment) if s.strip()] or [segment]

    parts: list[str] = []
    cur = ""
    for sent in sentences:
        if len(sent) > max_chars:  # 单个超长句子：硬切
            if cur:
                parts.append(cur)
                cur = ""
            for j in range(0, len(sent), max_chars):
                parts.append(sent[j:j + max_chars])
            continue
        if cur and len(cur) + len(sent) > max_chars:
            parts.append(cur)
            cur = ""
        cur += sent
    if cur:
        parts.append(cur)

    return _apply_overlap(parts, max_chars, overlap)


_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*|\n+")


def _apply_overlap(parts: list[str], max_chars: int, overlap: int) -> list[str]:
    out = []
    prev = ""
    for part in parts:
        if prev:
            merged = prev + part
            # 尽量保留上一块尾部的 overlap 与当前块
            out.append(merged[:max_chars])
        else:
            out.append(part[:max_chars])
        prev = part
    return [p for p in out if p.strip()]


def _dedupe_pieces(pieces: list[str]) -> list[str]:
    seen, out = set(), []
    for p in pieces:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out