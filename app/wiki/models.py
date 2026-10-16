"""Wiki 结构化知识库数据模型。

结构化设计（区别于纯文本语料的卖点）：
- Document ：一个 markdown 文件（一个知识点大主题），带 category 分类、title、来源路径
- Section  ：文档内的层级章节（# / ## / ### …），可嵌套，含小节标题
- Chunk    ：真正被检索的最小单元，绑定上游 Document/Section/Faq，携带固定 id
- Faq      ：FAQ 问题-答案对，问题文本参与精确匹配（启发式重排加分）

关系：Document 1—N Section；Chunk 归属于一个最具体的 Section；FAQ 独立且保序。
Chunk.id 作为检索与生成引用（证据溯源）的统一主键。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    id: str                     # 唯一主键，形如 "doc:1/sec:2:0"
    text: str                   # 块文本
    doc_id: int
    sec_id: Optional[int] = None
    section_path: str = ""      # 层级标题路径，如 "售后 > 退款 > 处理时效"
    title: str = ""             # 命中的文档标题，用于溯源与加分
    category: str = ""
    source: str = ""            # markdown 文件路径
    faq_question: Optional[str] = None   # 若来自 FAQ 则记录问题原文
    metadata: dict = field(default_factory=dict)


@dataclass
class Section:
    id: int
    level: int                 # 1 = '#', 2 = '##' ...
    title: str
    path: str                  # 完整标题路径
    chunks: list[Chunk] = field(default_factory=list)


@dataclass
class Document:
    id: int
    title: str
    category: str
    source: str
    sections: list[Section] = field(default_factory=list)
    faqs: list[Faq] = field(default_factory=list)

    @property
    def chunks(self) -> list[Chunk]:
        out = []
        for s in self.sections:
            out.extend(s.chunks)
        return out


@dataclass
class Faq:
    id: int
    question: str
    answer: str
    category: str
    source: str
    title: str = ""


@dataclass
class WikiIndex:
    """从 markdown 解析得到的检索快照（documents + faqs + 扁平 chunk 映射）。"""
    documents: list[Document] = field(default_factory=list)
    faqs: list[Faq] = field(default_factory=list)
    _chunks_by_id: dict[str, Chunk] = field(default_factory=dict)
    _faqs_by_id: dict[int, Faq] = field(default_factory=dict)

    def register(self, doc: Document):
        self.documents.append(doc)
        for c in doc.chunks:
            if c.id in self._chunks_by_id:
                raise ValueError(f"重复 chunk id: {c.id}（doc.id 生成不稳定，需全局唯一）")
            self._chunks_by_id[c.id] = c
        for f in doc.faqs:
            if f.id in self._faqs_by_id:
                raise ValueError(f"重复 FAQ id: {f.id}（id 必须是全局唯一，否则按 id 溯源会串文档）")
            self.faqs.append(f)
            self._faqs_by_id[f.id] = f

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks_by_id.values())

    def get_chunk(self, cid: str) -> Chunk | None:
        return self._chunks_by_id.get(cid)

    def get_faq(self, fid: int) -> Faq | None:
        return self._faqs_by_id.get(fid)


class KnowledgeBase:
    """轻量检索用知识容器：chunks + faqs，兼容从 SQLite 加载或从 markdown 构造。"""
    def __init__(self, chunks: list[Chunk] | None = None, faqs: list[Faq] | None = None):
        self.chunks = chunks or []
        self.faqs = faqs or []
        self._by_id = {c.id: c for c in self.chunks}

    def get_chunk(self, cid: str) -> Chunk | None:
        return self._by_id.get(cid)

    def all_chunks(self) -> list[Chunk]:
        return self.chunks

    @classmethod
    def from_wiki(cls, wiki: WikiIndex) -> "KnowledgeBase":
        return cls(wiki.chunks, wiki.faqs)