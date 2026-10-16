"""LLM 抽象层。"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import resolve_secret


def build_llm(cfg: dict):
    """根据配置构建 LLM：DeepSeek；无 key 且 mock=True 时降级为 MockLLM。"""
    secrets = resolve_secret(cfg)
    llm_cfg = cfg["llm"]
    if secrets["llm_key"]:
        from .deepseek import DeepSeekLLM
        return DeepSeekLLM(cfg=llm_cfg, api_key=secrets["llm_key"])
    if llm_cfg.get("mock", True):
        from .mock import MockLLM
        return MockLLM(cfg=llm_cfg)
    raise RuntimeError("未配置 LLM API Key（环境变量 DEEPSEEK_API_KEY），且 mock 已关闭。")


class BaseLLM(ABC):
    """统一的 LLM chat 接口。"""
    name = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.temperature = float(cfg.get("temperature", 0.3))
        self.max_tokens = int(cfg.get("max_tokens", 800))

    @abstractmethod
    def chat(self, messages: list[dict], **kwargs) -> str:
        """messages: [{"role","content"}, ...]（system/user/assistant）。返回回答文本。"""
        raise NotImplementedError

    def complete(
        self,
        system: str | None = None,
        user: str | None = None,
        **kwargs,
    ) -> str:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        if user:
            messages.append({"role": "user", "content": user})
        return self.chat(messages, **kwargs)

    async def achat(self, messages: list[dict], **kwargs) -> str:
        """默认同步；需要异步的 provider 可覆盖。"""
        return self.chat(messages, **kwargs)

    def stream_chat(self, messages: list[dict], **kwargs) -> "Iterator[str]":
        """默认将完整回答作为单个 token 输出；支持流式的 provider 覆盖。"""
        from typing import Iterator
        ans = self.chat(messages, **kwargs)
        if ans:
            yield ans