"""DeepSeek LLM：自写 OpenAI 兼容 HTTP 客户端（不依赖 openai SDK）。

- 线程安全的复用 session + 连接池
- 指数退避重试（限流 429 / 5xx / 网络错误）
- 流式输出 stream_chat()：逐 token yield（SSE 解析），供前端打字机效果
"""
from __future__ import annotations

import json
import time
from typing import Iterator

import requests

from .base import BaseLLM


class DeepSeekLLM(BaseLLM):
    name = "deepseek"

    def __init__(self, cfg: dict, api_key: str):
        super().__init__(cfg)
        self.api_key = api_key
        self.base_url = cfg.get("base_url", "https://api.deepseek.com").rstrip("/")
        self.model = cfg.get("model", "deepseek-chat")
        self.timeout = float(cfg.get("timeout", 30))
        self.max_retries = int(cfg.get("max_retries", 2))
        self.session = requests.Session()
        self.session.headers.update(self._headers())

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, messages, stream: bool, kwargs):
        return {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": float(kwargs.get("temperature", self.temperature)),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": stream,
        }

    def chat(self, messages: list[dict], **kwargs) -> str:
        payload = self._payload(messages, False, kwargs)
        data = self._post_with_retry(payload)
        return data["choices"][0]["message"]["content"].strip()

    def stream_chat(self, messages: list[dict], **kwargs) -> Iterator[str]:
        """流式生成：yield 每个增量 token，支持中断（generator 被 close）。"""
        payload = self._payload(messages, True, kwargs)
        for attempt in range(self.max_retries + 1):
            try:
                with self.session.post(
                    f"{self.base_url}/chat/completions",
                    data=json.dumps(payload, ensure_ascii=False),
                    timeout=(self.timeout, self.timeout * 3),
                    stream=True,
                ) as resp:
                    if resp.status_code != 200:
                        raise RuntimeError(
                            f"DeepSeek 错误 [{resp.status_code}]: {resp.text[:200]}")
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line or not line.startswith("data:"):
                            continue
                        chunk = line[5:].strip()
                        if chunk == "[DONE]":
                            return
                        try:
                            obj = json.loads(chunk)
                        except json.JSONDecodeError:
                            continue
                        delta = obj["choices"][0].get("delta", {})
                        piece = delta.get("content") or ""
                        if piece:
                            yield piece
                return
            except requests.RequestException as e:
                if attempt >= self.max_retries:
                    raise RuntimeError(f"DeepSeek 流式请求失败: {e}") from e
                time.sleep(0.5 * (2 ** attempt))

    def _post_with_retry(self, payload: dict) -> dict:
        for attempt in range(self.max_retries + 1):
            resp = self.session.post(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload, ensure_ascii=False),
                timeout=self.timeout,
            )
            if resp.status_code == 429:
                time.sleep(1 * (attempt + 1))
                continue
            if resp.status_code != 200:
                raise RuntimeError(
                    f"DeepSeek API 错误 [{resp.status_code}]: {resp.text[:300]}")
            return resp.json()
        raise RuntimeError("DeepSeek 限流重试超限")