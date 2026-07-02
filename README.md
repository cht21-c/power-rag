# Power RAG Agent

电厂智能运维知识库 — 基于 RAG 的工业文档检索问答系统。

## 架构

```
用户 → FastAPI/SSE 前端 → LangGraph Agent → 意图理解(LLM) → 路由分发
                                                    ├── RAG 检索(Hybrid: 向量+BM25) → Qdrant
                                                    └── 图纸匹配(LLM语义) → MySQL
```

## 核心能力

- **语义路由**：纯 LLM 判断用户意图（知识问答 / 图纸查找 / 歧义澄清），无关键词硬编码
- **混合检索**：BGE-M3 向量 + BM25 → RRF 融合排序
- **智能图纸匹配**：口语化表达（"那个泵的图"）→ LLM 语义匹配图纸库
- **双路径分块**：散文走 SemanticChunker，结构化表格走规则分块
- **多轮对话**：上下文记忆 + 指代消解 + 歧义澄清
- **审计追踪**：全链路 trace_id 日志
- **流式输出**：SSE token-by-token，前端 Markdown 渲染

## 快速开始

```bash
# 1. 安装依赖
pip install -r camera_sdk_agent/requirements.txt

# 2. 配置 .env
cp .env.example .env
# 编辑 .env 填写 DEEPSEEK_API_KEY 等

# 3. 启动 Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# 4. 摄入文档
python camera_sdk_agent/ingest_pipeline.py

# 5. 启动服务
uvicorn backend.server:app --host 0.0.0.0 --port 8765
```

访问 http://localhost:8765

## 配置

关键环境变量（详见 `camera_sdk_agent/config.py`）：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | - | DeepSeek API Key |
| `QDRANT_HOST` | `localhost` | Qdrant 服务地址 |
| `RAG_MIN_SCORE_STRICT` | `0.02` | 检索拒答阈值 |
| `INTENT_CONFIDENCE_THRESHOLD` | `0.55` | 意图置信度阈值 |
| `STRUCTURED_BLOCK_THRESHOLD` | `0.6` | 结构化检测阈值 |
| `LLM_MAX_CONTEXT_TOKENS` | `8000` | LLM 上下文预算 |

## 项目结构

```
├── backend/                  # FastAPI + SSE 前端
│   ├── server.py             # API 服务
│   └── index.html            # 前端页面
├── camera_sdk_agent/
│   ├── agent/                # LangGraph Agent
│   │   ├── graph.py          # 核心状态图
│   │   ├── prompts.py        # LLM 提示词
│   │   ├── tools.py          # RAG 工具定义
│   │   └── drawing_match.py  # LLM 图纸语义匹配
│   ├── ingestion/            # 文档摄取
│   │   ├── parsers.py        # PDF/MD/HTML 解析
│   │   ├── chunker.py        # 双路径分块
│   │   ├── embedder.py       # BGE-M3 嵌入
│   │   └── structured_detector.py  # 结构化检测
│   ├── retrieval/            # 检索模块
│   │   ├── retriever.py      # 混合检索器
│   │   └── reranker.py       # RRF 融合
│   ├── store/                # 向量存储
│   │   └── qdrant_store.py   # Qdrant 操作
│   ├── framework/            # 基础设施
│   │   ├── audit/            # 审计追踪
│   │   ├── auth/             # 认证鉴权
│   │   ├── llm_utils/        # LLM 工具
│   │   ├── memory/           # 对话压缩
│   │   └── tool_registry/    # 工具注册
│   └── infra-ai/             # 工程健壮性
│       └── model_router/     # 模型路由+熔断器
├── tests/                    # 77 个测试用例
├── docs/                     # 文档
└── scripts/                  # 运维脚本
```

## 技术栈

- **Agent**: LangGraph + LangChain
- **LLM**: DeepSeek-Chat
- **Embedding**: BGE-M3 (sentence-transformers)
- **向量库**: Qdrant
- **分块**: chonkie SemanticChunker + 规则分块
- **OCR**: RapidOCR (ONNX Runtime)
- **前端**: 原生 HTML/CSS/JS + marked.js + SSE
- **后端**: FastAPI + uvicorn

## License

MIT
