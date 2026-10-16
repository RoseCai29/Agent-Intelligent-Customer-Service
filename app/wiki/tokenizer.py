"""自研中文分词器。

设计动机（面试可讲）：中文没有空格分词，直接按 char 切分会让倒排索引丢失词义；
引入分词器过大（jieba 等），因此采用 **2-gram（bigram）切分**：既能捕捉常用双字词
（"退款""到账"），又天然对未登录词、错别字健壮，时间复杂度 O(n)，无需词典。

策略：
- CJK 汉字 → 连续汉字区段内滑窗取 bigram
- 连续字母/数字英文整体保留（含大小写归一），保留"IDS"、"3天"等
- 中英混合用连接 token 衔接，避免跨语言拼接噪音
- 可选丢弃停用字（仅影响倒排索引，不影响向量）

此输出供 BM25 倒排索引、查询解析与启发式重排复用。
"""
from __future__ import annotations

import re

_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_LATIN = re.compile(r"[a-zA-Z0-9]+")
_SPLIT = re.compile(r"[^\u4e00-\u9fff\u3400-\u4dbfa-zA-Z0-9]")


def tokenize(text: str, min_term_len: int = 2, drop_stopwords: bool = True) -> list[str]:
    """返回去重后的小写词组列表（保持出现顺序，供 TF 统计无需保序）。
    min_term_len=2：bigram 与英文单词均取其长度 ≥2。"""
    text = text.lower()
    tokens: list[str] = []
    segments = _SPLIT.split(text)
    for seg in segments:
        if not seg:
            continue
        lats = _LATIN.findall(seg)
        if lats:
            # 段内只含英数
            if len(seg) >= min_term_len:
                tokens.append(seg)
            continue
        # 纯中文段 → bigram
        chars = seg
        for i in range(len(chars) - 1):
            tokens.append(chars[i:i + 2])
        if len(chars) < min_term_len and len(chars) == 1:
            tokens.append(chars)  # 单字保留，供精确匹配
    if drop_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens


# 中文常见停用字（用于倒排索引，减小"的/了"等无意义 bigram 噪音）
_STOPWORDS: frozenset[str] = frozenset(
    {"的了", "了个", "是吗", "的呢", "吗呢", "吧啊", "啊把", "的个", "个问",
     "问题", "答案", "请问", "您好", "我们", "你们", "可以", "需要", "按照",
     "那么", "但是", "所以", "以及", "或者", "如何", "什么", "怎么"}
)


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def term_frequencies(tokens: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in tokens:
        out[t] = out.get(t, 0) + 1
    return out