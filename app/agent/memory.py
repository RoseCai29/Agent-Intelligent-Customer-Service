"""会话记忆 + query 改写（多轮指代消解）。

- 保留最近 N 轮（user/assistant，成对修剪）
- 轻量启发式改写：以指代词/省略片段开头的问题，拼接上一轮用户问题，使其可独立检索
  （"那多久到账" → "退款 那多久到账"）。生产上可用 LLM 改写更稳，这里用规则以便演示与测试。
"""
from __future__ import annotations

_REFERRING = ("那", "这个", "它", "然后", "那多久", "那是不是", "怎么办理", "能否", "还能")


class SessionMemory:
    def __init__(self, max_history: int = 6):
        self.max_history = max_history
        self.history: list[dict] = []   # [{"role","content"}]

    def add_user(self, text: str):
        if self.history and self.history[-1]["role"] == "user":
            self.history[-1]["content"] = text
        else:
            self.history.append({"role": "user", "content": text})
        auto_prune(self.history, self.max_history)

    def add_assistant(self, text: str):
        self.history.append({"role": "assistant", "content": _truncate(text, 300)})
        auto_prune(self.history, self.max_history)

    def as_messages(self) -> list[dict]:
        return list(self.history)


def auto_prune(hist: list[dict], max_history: int):
    while len(hist) > max_history * 2:
        hist.pop(0)


def rewrite_query(query: str, memory: SessionMemory) -> str:
    """把含指代的追问改写成独立、可检索的完整问题。"""
    q = query.strip()
    last_user = _last_user(memory)
    if not last_user:
        return q
    # 短片段或典型指代开头 → 拼接上一轮用户问题
    if len(q) < 10 or any(q.startswith(r) for r in _REFERRING):
        return f"{last_user} {q}"
    return q


def _last_user(memory: SessionMemory) -> str | None:
    for m in reversed(memory.history):
        if m["role"] == "user":
            return m["content"]
    return None


def _truncate(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + "…"