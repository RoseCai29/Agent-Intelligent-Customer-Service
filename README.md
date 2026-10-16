# kefu-agent · 智能客服 Agent / Intelligent Customer Service Agent

[中文版](#中文) · [English](#english)

---

## 中文

**自研智能客服 Agent：RAG 全链路 + 结构化 Wiki 知识库**（不用 LangChain / LlamaIndex）

从零实现、无框架黑盒的客服问答系统。分词、向量索引、BM25 倒排、RRF 混合检索、
重排、SQLite 持久化、FastAPI 服务全部自研，便于理解底层原理，也方便深入讲解。

### 特性

- **自研 RAG 全链路**：中文 bigram 分词 → 向量检索（余弦）+ BM25 倒排 → RRF 混合融合 → 重排
- **结构化 Wiki 层**：Markdown 层级解析、FAQ 问答对、分层感知分块、证据溯源 + 置信度
- **可插拔 Embedding**：SiliconFlow `bge-m3`（真实语义）/ 本地 bge / 离线 hash 降级
- **LLM**：DeepSeek 流式生成（SSE），未配 Key 自动降级为确定性 mock
- **SQLite 落盘**：索引/向量持久化（进程重启不重建）+ 多用户会话持久化
- **API 服务**：`/chat`、`/chat/stream`、`/retrieve`、`/sessions`、`/health`，含 trace_id 日志
- **一键部署**：Docker / docker-compose，pytest 测试 12 项

### 架构

```
用户问题
   │
   ▼
┌────────────────────  Agent 层  ────────────────────┐
│ 意图判断 → 多轮记忆(query改写) → 检索 → 上下文拼装   │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┼───────────────┐
        ▼              ▼               ▼
   向量检索(余弦)   倒排检索(BM25)    [重排(交叉编码+启发式)]
        └───── RRF 混合融合 ──────┘
                       │
                       ▼
              Wiki 结构化知识库(分块/条级索引)
                       │
                       ▼
              ┌─ 生成层（DeepSeek / mock）
              └─ 输出: 答案 + 证据引用 + 置信度
```

### 快速开始

```bash
pip install -r requirements.txt

# 0) 配置密钥（复制并填写，不配置会自动降级）
cp .env.example .env        # 填入 DEEPSEEK_API_KEY / SILICONFLOW_API_KEY

# 1) 以 API 服务方式运行（推荐）
python -m app.server         # http://localhost:8000，见下方 API 用例

# 或 CLI 对话：
python -m app.cli --query "退款多久到账"
```

API 用例：

```bash
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"message":"退款多久到账","user_id":"u1"}'

curl -X POST localhost:8000/chat/stream -H "Content-Type: application/json" \
  -d '{"message":"优惠券能叠加吗?"}'   # SSE 逐 token 流式

curl "localhost:8000/health"            # 健康检查(含 embedding/llm/chunk 数)
```

### 结构化 Wiki 示意（`data/wiki/`）

```markdown
---
category: 售后
---
# 退款

## 处理时效
用户申请退款后，1~3 个工作日内原路退回...

## FAQ
### 退款多久到账？
一般 1-3 个工作日，节假日顺延...
```

### 测试

```bash
pip install pytest && pytest
```

---

## English

**A self-built intelligent customer-service Agent: full RAG pipeline + structured Wiki layer** (no LangChain / LlamaIndex).

Built from scratch with no framework black boxes — tokenization, vector index, BM25 inverted index,
RRF hybrid fusion, reranking, SQLite persistence, and the FastAPI service are all implemented by hand,
so the internals are easy to understand and deep-dive into.

### Features

- **Self-built full RAG pipeline**: Chinese bigram tokenizer → vector search (cosine) + BM25 → RRF hybrid fusion → reranker
- **Structured Wiki layer**: Markdown hierarchy parsing, FAQ pairs, heading-aware chunking, evidence citation + confidence
- **Pluggable embedding**: SiliconFlow `bge-m3` (real semantics) / local bge / offline hash fallback
- **LLM**: DeepSeek streaming generation (SSE), automatic graceful fallback to a deterministic mock without a key
- **SQLite persistence**: index/vector persisted (no rebuild across restarts) + multi-user session persistence
- **API service**: `/chat`, `/chat/stream`, `/retrieve`, `/sessions`, `/health` with per-request `trace_id` logging
- **One-command deploy**: Docker / docker-compose, 12 pytest tests

### Architecture

```
User Question
   │
   ▼
┌────────────────────  Agent Layer  ────────────────────┐
│ intent → multi-turn memory(query rewrite) → retrieve  │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┼───────────────┐
        ▼              ▼               ▼
   vector(cosine)   inverted(BM25)  [rerank(cross-encoder+heuristic)]
        └───── RRF hybrid fusion ─────┘
                       │
                       ▼
            Structured Wiki KB (chunk/item index)
                       │
                       ▼
        ┌─ Generation (DeepSeek / mock)
        └─ Output: answer + evidence + confidence
```

### Quick Start

```bash
pip install -r requirements.txt

# setup keys (optional — auto-fallback if unset)
copy .env.example .env       # fill in DEEPSEEK_API_KEY / SILICONFLOW_API_KEY

# run as an API service (recommended)
python -m app.server         # http://localhost:8000, see examples below

# or chat via CLI:
python -m app.cli --query "How long for a refund?"
```

API examples:

```bash
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"message":"How long for a refund?","user_id":"u1"}'

curl -X POST localhost:8000/chat/stream -H "Content-Type: application/json" \
  -d '{"message":"Can coupons stack?"}'   # SSE token streaming

curl "localhost:8000/health"            # health check (embedding / llm / chunk count)
```

### Structured Wiki Sample (`data/wiki/`)

```markdown
---
category: after-sales
---
# Refund

## Processing Time
After a refund is requested, it is returned within 1-3 business days...

## FAQ
### How long does a refund take?
Generally 1-3 business days, longer around holidays...
```

### Tests

```bash
pip install pytest && pytest
```