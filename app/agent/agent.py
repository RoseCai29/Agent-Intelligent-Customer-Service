"""客服 Agent 主流程。

流程：
1. 握手词/提示意图判断（规则）→ 返回问候/提示，不检索
2. query 改写（多轮指代消解）
3. 检索（混合 RRF + 重排）
4. 拼装上下文（参考资料 + 会话历史）→ LLM 生成
5. 输出：answer + evidence(来源引用) + confidence + fallback

整条链路自研，无 LangChain/LlamaIndex。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .memory import SessionMemory
from ..llm.base import BaseLLM
from ..retriever.retrieval import RetrievedChunk

_GREETINGS = ("你好", "您好", "hello", "hi", "在吗", "在不在", "hey")
_THANKS = ("谢谢", "感谢", "懂了", "清楚了", "明白了")
_QUIT = ("退出", "quit", "exit", "再见", "拜拜")


@dataclass
class AgentResult:
    reply: str
    intent: str = "answer"
    evidence: list[dict] = field(default_factory=list)   # [{"title","section_path","source","text"}]
    confidence: float = 0.5
    used_retrieval: bool = True
    query: str = ""


class CustomerServiceAgent:
    def __init__(self, llm: BaseLLM, retriever, system_prompt: str,
                 fallback: str, max_history: int = 6):
        self.llm = llm
        self.retriever = retriever
        self.system_prompt = system_prompt
        self.fallback = fallback
        self.session = SessionMemory(max_history=max_history)

    def chat(self, user_input: str, top_k: int = 5, history: list[dict] | None = None,
             memory: SessionMemory | None = None) -> AgentResult:
        """一轮问答。

        - history：调用方提供的先前对话（{role,content}，来自持久化会话），用于多轮改写与上下文
        - memory ：可用的内存记忆；缺省用 self.session（CLI 场景）
        """
        mem = memory or self.session
        hist = history if history is not None else mem.as_messages()
        meta = self.preflight(user_input, mem)
        if meta is not None:
            mem.add_user(user_input)
            mem.add_assistant(meta.reply)
            return meta

        # ---- 改写 + 检索 ----
        query = _rewrite(user_input, hist)
        retrieved = self.retriever.retrieve(query, top_k=top_k)
        confidence = self._confidence(retrieved)

        # ---- 生成 ----
        context = self._build_context(retrieved)
        messages = self._build_messages(hist, query, context, user_input)
        try:
            answer = self.llm.chat(messages)
        except Exception as e:  # noqa: BLE001
            return AgentResult(reply=f"抱歉，生成回答时出错：{e}", intent="error",
                               evidence=self._evidence(retrieved), used_retrieval=True,
                               query=query, confidence=0.0)
        mem.add_user(user_input)
        mem.add_assistant(answer)
        if not retrieved:
            return AgentResult(reply=f"{self.fallback}\n（未能从知识库检索到相关信息。）",
                               intent="answer", evidence=[], used_retrieval=True,
                               query=query, confidence=0.0)
        return AgentResult(reply=answer, intent="answer", evidence=self._evidence(retrieved),
                           confidence=confidence, used_retrieval=True, query=query)

    def preflight(self, user_input: str, mem: SessionMemory) -> AgentResult | None:
        """非检索意图（问候/致谢/退出）快速返回；否则 None。"""
        msg = user_input.strip().lower()
        if msg in _QUIT:
            return AgentResult(reply="再见，谢谢咨询！", intent="exit", used_retrieval=False)
        if _is_greeting(user_input.strip()):
            return AgentResult(reply="您好呀，我是智能客服小助手，有什么可以帮您？（如：退款多久到账？）",
                               intent="greeting", used_retrieval=False)
        if _is_thanks(user_input.strip()):
            return AgentResult(reply="不客气～还有其它问题随时问我哦。", intent="thanks",
                               used_retrieval=False)
        return None

    def stream(self, user_input: str, top_k: int = 5, history: list[dict] | None = None,
               memory: SessionMemory | None = None, on_delta=None) -> dict:
        """生成的同时逐 token 回调 on_delta(piece)，供 SSE 即时推流。

        返回 {"status","text","tokens","evidence","intent","query"}。
        """
        mem = memory or self.session
        hist = history if history is not None else mem.as_messages()
        meta = self.preflight(user_input, mem)
        if meta is not None:
            mem.add_user(user_input)
            mem.add_assistant(meta.reply)
            if on_delta:
                on_delta(meta.reply)
            return {"status": "preflight", "text": meta.reply, "tokens": [meta.reply],
                    "evidence": [], "intent": meta.intent, "query": user_input}

        query = _rewrite(user_input, hist)
        retrieved = self.retriever.retrieve(query, top_k=top_k)
        context = self._build_context(retrieved)
        messages = self._build_messages(hist, query, context, user_input)
        tokens = []
        try:
            for piece in self.llm.stream_chat(messages):
                tokens.append(piece)
                if on_delta:
                    on_delta(piece)
            answer = "".join(tokens)
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "text": str(e), "tokens": [], "evidence": [],
                    "intent": "error", "query": query}
        mem.add_user(user_input)
        mem.add_assistant(answer)
        if not retrieved:
            answer = f"{self.fallback}\n（未能从知识库检索到相关信息。）"
            tokens = [answer]
            if on_delta:
                on_delta(answer)
        return {"status": "ok", "text": answer, "tokens": tokens,
                "evidence": self._evidence(retrieved), "intent": "answer", "query": query}

    # ---------------- 内部 ----------------
    def _build_context(self, retrieved: list[RetrievedChunk]) -> str:
        blocks = []
        for it in retrieved:
            blocks.append(
                f"[{it.section_path or it.title or it.category}]\n{it.text}"
            )
        return "\n\n".join(blocks) if blocks else "(空)"

    def _build_messages(self, history, query: str, context: str, user_input: str):
        content = (
            f"用户问题：{query}\n\n"
            f"参考资料：\n{context}"
        )
        messages = [{"role": "system", "content": self.system_prompt}]
        # 注入持久化历史（多轮）；用 query(改写后) 而非原句作最后 user
        messages.extend(history[-8:])
        messages.append({"role": "user", "content": content})
        return messages

    def _evidence(self, retrieved: list[RetrievedChunk]) -> list[dict]:
        out = []
        for it in retrieved[:3]:
            out.append({
                "title": it.title,
                "section_path": it.section_path,
                "category": it.category,
                "source": it.source,
                "text": it.text[:120],
            })
        return out

    def _confidence(self, retrieved: list[RetrievedChunk]) -> float:
        if not retrieved:
            return 0.0
        if retrieved[0].from_faq:
            return 0.95
        top = retrieved[0] if retrieved else None
        if top is None:
            return 0.5
        # 融合分越高越可信；结合是否有 FAQ/高重合
        base = min(top.fused_score / 6.0, 1.0)
        return min(0.55 + 0.4 * base, 0.9)


def _is_greeting(text: str) -> bool:
    t = text.strip()
    return any(g in t.lower() for g in _GREETINGS) and len(t) <= 12


def _is_thanks(text: str) -> bool:
    return any(x in text for x in _THANKS)


_REFERRING = ("那", "这个", "它", "然后", "那多久", "那是不是", "怎么办理", "能否", "还能")


def _rewrite(query: str, history: list[dict]) -> str:
    """把含指代的追问改写成独立、可检索的完整问题（基于历史消息列表）。"""
    q = query.strip()
    last_user = None
    for m in reversed(history):
        if m["role"] == "user":
            last_user = m["content"]
            break
    if not last_user:
        return q
    if len(q) < 10 or any(q.startswith(r) for r in _REFERRING):
        return f"{last_user} {q}"
    return q