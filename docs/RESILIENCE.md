# 工程健壮性调优指南

## 模型路由

### 候选模型配置

通过 `MODEL_CANDIDATES` 环境变量配置（逗号分隔）：

```bash
# 单一模型（当前默认）
MODEL_CANDIDATES=deepseek-chat

# 多模型降级
MODEL_CANDIDATES=deepseek-chat,deepseek-reasoner
```

### 熔断器参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `CIRCUIT_BREAKER_FAIL_THRESHOLD` | 5 | 连续失败 N 次触发熔断 |
| `CIRCUIT_BREAKER_COOLDOWN_SEC` | 60 | 熔断后冷却时间（秒），到期进入半开试探 |
| `STREAM_FIRST_TOKEN_TIMEOUT_SEC` | 10 | 流式首包超时（秒） |

### 三态行为

```
CLOSED ──(连续失败≥阈值)──> OPEN ──(冷却到期)──> HALF_OPEN
   ^                                                  |   |
   └──────────(试探成功)────────────────────────────────┘   │
                                                            │
                    (试探失败，重新熔断)─────────────────────┘
```

### 调优建议

| 场景 | fail_threshold | cooldown_sec |
|------|---------------|-------------|
| 开发/测试 | 3 | 10 |
| 生产保守 | 5 | 60 |
| 生产激进（快速切换） | 3 | 30 |

---

## 记忆压缩

### 参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `MEMORY_WINDOW_TURNS` | 3 | 保留最近 N 轮原文不压缩 |
| `MEMORY_COMPRESS_TRIGGER_TOKENS` | 3500 | 总消息 token 估算超此值触发压缩 |

### 行为

1. 对话历史总 token > `TRIGGER_TOKENS` 且轮数 > `WINDOW_TURNS` → 触发压缩
2. 保留最近 `WINDOW_TURNS` 轮原文
3. 更早的消息交由 LLM 生成结构化摘要（保留设备型号、参数、规程名称等实体）
4. 压缩 LLM 调用失败 → 降级为硬截断（保留最近消息至不超限）
5. 已存在摘要 + 新超窗消息 → 合并压缩

---

## 工具注册

### 注册新工具

```python
from framework.tool_registry.registry import register_tool

@register_tool(name="weather_query", description="查询指定地点的天气")
def weather_query(location: str) -> str:
    # ...
    return result
```

无需修改 `agent/graph.py`——工具会自动被 LLM 发现和调用。

### 工具名单开关

```bash
# 只启用指定工具（逗号分隔）
ENABLED_TOOLS=ocr_image,rag_search

# 空 = 全部启用（默认）
ENABLED_TOOLS=
```

---

## 入库报告

每次执行 `python ingest_pipeline.py` 后在 `logs/ingest/` 生成 `ingest_YYYYMMDD_HHMMSS.json`：

```json
{
  "total_docs": 50,
  "success_docs": 47,
  "failed_docs": 3,
  "total_chunks": 1234,
  "steps": [
    {"step": "discover", "total": 50, "success": 50, "failed": 0},
    {"step": "parse",    "total": 50, "success": 48, "failed": 2, "failed_items": [...]},
    {"step": "chunk",    "total": 48, "success": 48, "failed": 0},
    ...
  ]
}
```
