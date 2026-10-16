"""Markdown 文档解析 → 结构化 Document（含层级感知分块）。

流程：
1. 读取 front-matter（---category: xxx---）得到分类，取第一个 # 标题作为文档 title
2. 按标题(#/##/###…)把正文切分为 Section，维护完整标题路径 path="售后 > 退款 > 处理时效"
3. 识别 "## FAQ / ## 常见问题" 下 "### 问题？" 作为 Faq（问题和紧随正文）。
   但 ## FAQ 本身不会生成可检索的普通 Section 正文（避免重复）。
4. 其余正文逐 Section 经 [chunker] 分层感知分块。

返回 Document（chunks 带 sec_id、section_path、title、category、source，可直接溯源）。
"""
from __future__ import annotations

import re

from .chunker import split_text
from .models import Chunk, Document, Faq, Section

_HEADING = re.compile(r"^(#{1,6})\s+(.+)\s*$")
_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.S)


class _Counter:
    _n = 0
    def next(self) -> int:
        self._n += 1
        return self._n


# 全局自增计数器：保证跨文档的 doc.id / chunk.id 全局唯一，避免注册到索引时被覆盖
_GLOBAL = _Counter()


def parse_markdown(text: str, source: str = "", doc_title: str | None = None,
                   chunk_cfg: dict | None = None) -> Document:
    counter = _Counter()
    text = text.lstrip("\ufeff")

    category = "默认"
    m = _FRONT.match(text)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                if k.strip() == "category":
                    category = v.strip()
        text = text[m.end():]

    title = (doc_title or source).strip()
    # 若正文第一个非空标题是单个 # 的 H1，将其作为文档标题
    lines = [ln.rstrip() for ln in text.splitlines()]

    h1_matches = [ln for ln in lines if _HEADING.match(ln) and _HEADING.match(ln).group(1) == "#"]
    if h1_matches:
        title = _HEADING.match(h1_matches[0]).group(2).strip()

    sections: list[Section] = []
    faqs: list[Faq] = []
    section_bodies: dict[int, list[str]] = {}   # sec_id -> 原始行

    stack: list[tuple[int, str]] = []           # (level, title)
    cur_sec: int | None = None                 # 当前普通 Section id
    in_faq = False
    pending_faq_question: str | None = None
    faq_body: list[str] = []

    def flush_faq():
        nonlocal pending_faq_question, faq_body
        if pending_faq_question and faq_body and _strip_lines(faq_body):
            # FAQ id 用全局自增计数器保证跨文档唯一，避免同 id 互相覆盖
            faqs.append(Faq(id=_GLOBAL.next(), question=pending_faq_question,
                            answer=_strip_lines(faq_body), category=category,
                            source=source, title=title))
        pending_faq_question = None
        faq_body = []

    def flush_section():
        nonlocal cur_sec
        cur_sec = None

    for raw in lines:
        ln = raw.strip()
        if not ln:
            continue
        h = _HEADING.match(ln)
        if not h:
            if pending_faq_question is not None:
                faq_body.append(ln)
            elif cur_sec is not None:
                section_bodies.setdefault(cur_sec, []).append(ln)
            continue

        level = len(h.group(1))
        heading = h.group(2).strip()
        if heading.lower() in {"faq", "常见问题"}:
            flush_section()
            flush_faq()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
            in_faq = True
            continue

        if in_faq and level > 2:
            # FAQ 条目："### 问题？"
            flush_section()
            flush_faq()
            pending_faq_question = heading
            continue

        # 普通标题：先收尾 FAQ，再进入常规 section 逻辑
        in_faq = False
        flush_faq()
        flush_section()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
        parent_path = " > ".join(t for _, t in stack[:-1])
        path = parent_path + (" > " if parent_path else "") + heading
        if level == 1:
            continue  # H1 为文档标题，不单独成 section
        sec_id = counter.next()
        sections.append(Section(id=sec_id, level=level, title=heading, path=path))
        cur_sec = sec_id
        in_faq = False

    flush_faq()
    flush_section()

    doc = Document(id=_GLOBAL.next(), title=title, category=category,
                   source=source, sections=sections, faqs=faqs)

    # ---- 分层感知分块 ----
    chunks: list[Chunk] = []
    for sec in sections:
        body = _strip_lines(section_bodies.get(sec.id, []))
        if not body:
            continue
        for i, piece in enumerate(split_text(body, chunk_cfg or {})):
            chunks.append(Chunk(
                id=f"doc:{doc.id}/sec:{sec.id}:{i}",
                text=piece,
                doc_id=doc.id,
                sec_id=sec.id,
                section_path=sec.path,
                title=title,
                category=category,
                source=source,
            ))
    for sec in sections:
        sec.chunks = [c for c in chunks if c.sec_id == sec.id]

    return doc


def _strip_lines(lines: list[str]) -> str:
    out = [ln for ln in lines]
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out).strip()