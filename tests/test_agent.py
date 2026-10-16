"""Agent 规则回归测试：改写、preflight、异常收敛（统一文案 + 密钥脱敏 + error_code）。

评审实测：真实问题被当问候吞掉、短问题被误拼上一话题、LLM 401/密钥明文回传。
本文件锁住修复后的行为。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent.agent import (  # noqa: E402
    CustomerServiceAgent, _is_greeting, _is_thanks, _rewrite, redact,
    _LLM_ERROR_TEXT,
)


# ---------------------------------------------------------------- preflight
def test_pure_greeting_is_greeting():
    assert _is_greeting("你好")
    assert _is_greeting("你好呀")
    assert _is_greeting("hi")


def test_question_with_greeting_prefix_is_not_greeting():
    # 评审实测：'你好，退款多久到账' 被当问候跳过检索，这是 bug
    assert not _is_greeting("你好，退款多久到账")
    assert not _is_greeting("hi 怎么改地址")


def test_pure_thanks_is_thanks():
    assert _is_thanks("谢谢")
    assert _is_thanks("明白了，谢谢")


def test_thanks_with_content_is_not_swallowed():
    # 评审实测：'谢谢，但是退款还没到账' 被整句吞掉、完全不检索
    assert not _is_thanks("谢谢，但是退款还没到账")
    assert not _is_thanks("谢谢，那运费怎么算")


# ---------------------------------------------------------------- rewrite
def test_referring_question_prepends_last_topic():
    hist = [{"role": "user", "content": "我买的鞋什么时候发货"},
            {"role": "assistant", "content": "已为您的订单发货"}]
    assert _rewrite("那优惠券呢", hist) == "我买的鞋什么时候发货 那优惠券呢"


def test_plain_short_question_not_polluted():
    # 评审实测：'退款要多久呀'(9字) 被误拼上一话题造成漂移
    hist = [{"role": "user", "content": "退款多久到账"}, {"role": "assistant", "content": "1-3 天"}]
    assert _rewrite("退款要多久呀", hist) == "退款要多久呀"


def test_rewrite_no_history_no_change():
    assert _rewrite("那多久到账", []) == "那多久到账"


# ---------------------------------------------------------------- error convergence
def test_redact_secret():
    assert redact("DeepSeek 错误 [401]: invalid api key sk-abcdef123") == \
        "DeepSeek 错误 [401]: invalid api key sk-***"


def _failing_agent():
    class BoomLLM:
        name = "boom"
        def chat(self, messages):
            raise RuntimeError("DeepSeek 错误 [401]: sk-abcdef123 secret")

        def stream_chat(self, messages):
            raise RuntimeError("DeepSeek 错误 [401]: sk-abcdef123 secret")

    class EmptyRetriever:
        def retrieve(self, query, top_k=5):
            return []

    return CustomerServiceAgent(llm=BoomLLM(), retriever=EmptyRetriever(),
                                system_prompt="你是一个客服", fallback="抱歉")


def test_llm_error_is_uniform_no_secret():
    """对外回复不含异常原文 / 密钥；error_code=llm_error；详情已脱敏。"""
    res = _failing_agent().chat("退款多久到账")
    assert res.reply == _LLM_ERROR_TEXT
    assert res.intent == "error"
    assert res.error_code == "llm_error"
    assert "sk-abcdef123" not in res.reply
    assert "sk-***" in res.error_detail
    assert "sk-abcdef123" not in res.error_detail


def test_llm_error_stream_is_uniform_no_secret():
    out = _failing_agent().stream("退款多久到账")
    assert out["status"] == "error"
    assert out["text"] == _LLM_ERROR_TEXT
    assert out["error_code"] == "llm_error"
    assert "sk-abcdef123" not in out["text"]