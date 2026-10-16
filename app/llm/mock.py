"""Deterministic mock LLM：无 API key 时保证链路可跑、可测、可演示。

从 user 消息中提取"参考资料"，做简单的关键词抽取式摘要，避免编造，便于离线验证 RAG 效果。
"""
from __future__ import annotations

import re

from .base import BaseLLM


class MockLLM(BaseLLM):
    name = "mock"

    def chat(self, messages: list[dict], **kwargs) -> str:
        user_text = ""
        for m in reversed(messages):
            if m["role"] == "user":
                user_text = m["content"]
                break
        context = self._extract_context(user_text)
        if context:
            # 抽取式：直接从参考块中摘关键句，保证"没有编造"
            lines = _dedupe(_destyle(context))
            body = "\n".join("   · " + ln for ln in lines)
            return (
                "[mock 离线答复] 依据参考资料：\n"
                f"{body}\n\n"
                "（提示：配置 DEEPSEEK_API_KEY 后即调用真实 DeepSeek 生成。）"
            )
        return "[mock 离线答复] 参考文档中暂未覆盖该问题，建议转人工客服。"

    @staticmethod
    def _extract_context(text: str) -> str:
        m = re.search(r"参考资料[:：]\s*(.*?)(?:\n\n[^\n]|$)", text, re.S)
        return m.group(1) if m else ""


def _destyle(text: str) -> list[str]:
    lines = []
    for ln in text.splitlines():
        ln = re.sub(r"^\s*[#>*`|\-•]+\s*", "", ln).strip()
        if ln:
            lines.append(ln[:120])
    return lines[:5]


def _dedupe(lines: list[str]) -> list[str]:
    seen, out = set(), []
    for ln in lines:
        if ln not in seen:
            seen.add(ln)
            out.append(ln)
    return out