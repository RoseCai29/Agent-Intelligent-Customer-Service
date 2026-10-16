# kefu-agent：自研智能客服 Agent（RAG 全链路 + 结构化 Wiki 知识库）

从零实现的一个客服问答 Agent，包含 **RAG 全链路** 与 **结构化 Wiki 层**。
不依赖 LangChain/LlamaIndex，检索、索引、分词、生成均为自研，便于理解与面试深入讲解。

## 架构

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

## 快速开始

```bash
pip install -r requirements.txt
pip install sentence-transformers      # 可选：本地向量化，未装则用 hash 回退

# 1) 导入/更新 wiki 文档并建索引
python -m app.ingest

# 2) 对话
python -m app.cli            # 无 DEEPSEEK_API_KEY 时自动用 mock

# 3) 离线直接测检索
python -m app.cli --query "退款多久到账"
```

## 结构化 Wiki 示意（data/wiki/）

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

## 测试
```bash
pip install pytest && pytest
```