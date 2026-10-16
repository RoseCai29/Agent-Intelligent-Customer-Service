"""分层感知分块器。

策略（面试可讲）：
- 以段落（\n\n）为基本单元，优先用段落边界切分，保持语义完整；
- 段落过长时回退到句子边界（按中英文句号/问号/叹号/换行），仍超限则硬切；
- 支持块间重叠（overlap）以缓解"关键信息恰好跨块"的问题；
- 层级感知在 parser 层已经保证按 Section 独立分块，块并不过界标题。

正确性不变量：任意一段正文切块后，各块拼接必须覆盖原文
（len("".join(chunks)) >= len(src) - 段落分隔空白），不允许丢内容。

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
    for block in _group_blocks(paragraphs, max_chars):
        # 每个块独立切割并携带块内重叠；块与块之间互不重复，避免内容被二次切进索引
        chunks.extend(_cut(block, max_chars, overlap))
    return _dedupe_pieces(chunks)


def _group_blocks(paragraphs: list[str], max_chars: int) -> list[str]:
    """把段落贪心聚成 <= max_chars 的块；超长段落单独成块（交由 _cut 内部再切）。

    保证每个段落只落在唯一一个块里 → 切块后不丢内容（覆盖不变量成立）。
    """
    blocks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(para) > max_chars:
            if buf:
                blocks.append(buf)
                buf = ""
            blocks.append(para)
            continue
        if buf and len(buf) + 1 + len(para) > max_chars:
            blocks.append(buf)
            buf = ""
        buf = (buf + "\n" if buf else "") + para
    if buf:
        blocks.append(buf)
    return blocks


def _cut(segment: str, max_chars: int, overlap: int) -> list[str]:
    """对单一连续段做句子优先、硬切兜底的切割。返回块拼接后覆盖整个 segment。"""
    if len(segment) <= max_chars:
        return [segment]

    sentences = [s for s in _SENT_SPLIT.split(segment) if s.strip()] or [segment]

    # 组句预算扣掉 overlap，给前一/下一块的尾部重叠留出塞入空间，
    # 保证 overlap 真实生效（prev_tail + part <= max_chars 恒成立）。
    budget = max(1, max_chars - overlap)

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
        if cur and len(cur) + len(sent) > budget:
            parts.append(cur)
            cur = ""
        cur += sent
    if cur:
        parts.append(cur)

    return _apply_overlap(parts, max_chars, overlap)


_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*|\n+")


def _apply_overlap(parts: list[str], max_chars: int, overlap: int) -> list[str]:
    """让后一块拼上前一块的尾部 overlap 字符，实现真正的前后块重叠。

    兜底：若拼接后超过 max_chars，则放弃拼接直接用当前块（前块尾字仍在前块里，不丢内容）。
    覆盖不变量：每个 part 原样出现在某一 chunk 中 → 拼接结果覆盖整个输入段。
    """
    out: list[str] = []
    prev_tail = ""
    for part in parts:
        if prev_tail and len(prev_tail) + len(part) <= max_chars:
            chunk = prev_tail + part
        else:
            chunk = part
        out.append(chunk)
        prev_tail = chunk[-overlap:] if overlap else ""
    return [p for p in out if p.strip()]


def _dedupe_pieces(pieces: list[str]) -> list[str]:
    seen, out = set(), []
    for p in pieces:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out