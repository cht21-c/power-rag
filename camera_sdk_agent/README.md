# Camera SDK RAG Agent v1.0

基于 RAG 的相机 SDK 智能问答系统，支持多品牌文档混合检索。

## 架构

```
用户提问 → 意图理解 → 混合检索(向量+BM25) → RRF融合 → DeepSeek生成 → 回答
```

| 组件 | 技术 |
|------|------|
| 向量数据库 | Qdrant (Docker) |
| Embedding | BAAI/bge-m3 (FP32/FP16) |
| RAG 框架 | LlamaIndex |
| 检索策略 | 向量 + BM25 + RRF 融合 |
| Agent | LangGraph 状态图 |
| LLM | DeepSeek (ChatOpenAI 兼容) |
| 文档解析 | PyMuPDF (PDF) + 原生 (MD/HTML) |
| Chat UI | Chainlit |

## 快速开始

### 1. 环境配置

```bash
conda create -n camera_sdk_agent python=3.11 -y
conda activate camera_sdk_agent
pip install -r requirements.txt
pip install chainlit
```

### 2. 启动 Qdrant

```bash
docker rm -f qdrant-camera-sdk 2>/dev/null
docker run -d --name qdrant-camera-sdk -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

### 3. 配置 API Key

创建 `.env` 文件：

```
DEEPSEEK_API_KEY=sk-your-key-here
EMBED_DEVICE=cuda          # 有 GPU 用 cuda，否则 cpu
EMBED_BATCH_SIZE=4         # 4GB 显存建议 4
```

### 4. 放入文档

按品牌分类放入 `sdk_docs/`：

```
sdk_docs/
├── dahua/
│   ├── 产品手册1.pdf
│   └── 产品手册2.pdf
└── haikang/
    └── 用户手册.pdf
```

### 5. 一键摄取

```bash
# 预览（不写入 Qdrant）
python ingest_pipeline.py --dry-run --verbose

# 正式摄取
python ingest_pipeline.py --force-recreate
```

### 6. 启动问答

**Chainlit Web UI（推荐）：**

```bash
chainlit run chat_ui.py
```

浏览器打开 `http://localhost:8000`，底部下拉框切换品牌过滤。

**命令行交互：**

```bash
python main.py
```

支持命令：
```
/brand dahua     # 切换到大华品牌
/brand clear     # 清除品牌过滤
/config          # 查看当前配置
/help            # 帮助
```

**单次查询：**

```bash
python main.py --query "如何初始化大华相机？" --brand dahua
```

## 项目结构

```
camera_sdk_agent/
├── ingestion/          # 文档解析 & 分块 & Embedding
│   ├── parsers.py      # PDF/MD/HTML 解析
│   ├── chunker.py      # 函数粒度分块 (512 tokens)
│   └── embedder.py     # BGE-M3 封装 (FP16/FP32)
├── store/
│   └── qdrant_store.py # Qdrant CRUD 操作
├── retrieval/
│   ├── retriever.py    # 向量 + BM25 混合检索
│   └── reranker.py     # RRF 融合
├── agent/
│   ├── graph.py        # LangGraph 状态图
│   ├── tools.py        # RAG 检索工具
│   └── prompts.py      # System prompt 模板
├── chat_ui.py          # Chainlit Web UI
├── main.py             # CLI 入口
├── ingest_pipeline.py  # 文档摄取流水线
├── config.py           # 全部配置
├── sdk_docs/           # 原始文档存放目录
└── requirements.txt    # Python 依赖
```

## GPU 配置

| 显存 | 建议配置 |
|------|---------|
| ≥8GB | `EMBED_DEVICE=cuda` `EMBED_BATCH_SIZE=16` |
| 4GB | `EMBED_DEVICE=cuda` `EMBED_BATCH_SIZE=4` (FP16 自动启用) |
| 无 GPU | `EMBED_DEVICE=cpu` `EMBED_BATCH_SIZE=32` |

## 离线模式

模型已缓存到 `~/.cache/huggingface/hub/`，无网络时设置环境变量跳过 HuggingFace 检查：

```powershell
$env:HF_HUB_OFFLINE=1
python ingest_pipeline.py --force-recreate
```
