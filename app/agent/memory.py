"""会话记忆（多轮）。

- 保留最近 N 轮（user/assistant，成对修剪）
- query 改写（指代消解）见 agent._rewrite，此处不再重复维护死代码
"""
from __future__ import annotations


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


def _truncate(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + "…"