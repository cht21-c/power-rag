# 审计日志配置指南

## 概述

等保三级要求"审计追踪"。本模块对 LangGraph Agent 的所有节点（query_understand → route_decision → mysql_query/rag_retrieve → llm_generate）进行全链路审计。

## 审计记录字段

每条审计记录包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| trace_id | string | 一次完整请求的唯一标识（16字符 hex） |
| user_id | string | 操作者身份（关联认证模块） |
| node_name | string | 当前节点名 |
| input_summary | string | 入参摘要（截断至 500 字符） |
| output_summary | string | 出参摘要（截断至 500 字符） |
| route_decision | string | 路由决策（rag / mysql / mysql_miss_fallback） |
| latency_ms | float | 节点耗时（毫秒） |
| status | string | success / error |
| timestamp | string | ISO 8601 时间戳 |
| extra | object | 扩展字段（可选） |

## 日志存储

- **格式**：JSON Lines（`.jsonl`），每行一条 JSON 记录
- **路径**：`logs/audit/YYYY-MM-DD.jsonl`（按天切割）
- **编码**：UTF-8

## 查询日志

### 按 trace_id 查询完整链路

```bash
python scripts/query_audit_log.py --trace-id abc123def456
```

输出示例：

```
======================================================================
  Trace ID: abc123def456 | User: admin01
======================================================================
  [14:23:01.234] OK  query_understand      12.3ms -> [rag]
  [14:23:01.567] OK  route_decision          0.5ms
  [14:23:02.123] OK  rag_retrieve          556.0ms
  [14:23:03.890] OK  llm_generate         1767.0ms
----------------------------------------------------------------------
  Total latency: 2335.8ms  |  Nodes: 4
======================================================================
```

### 按时间范围查询

```bash
python scripts/query_audit_log.py --from 2026-07-01 --to 2026-07-05
```

### 按用户查询

```bash
python scripts/query_audit_log.py --user-id admin01 --from 2026-07-01
```

### 查看错误记录

```bash
python scripts/query_audit_log.py --status error --from 2026-07-01
```

### JSON 格式输出

```bash
python scripts/query_audit_log.py --trace-id abc123 --json
```

## 架构说明

```
framework/audit/
├── __init__.py   # 模块入口
├── sink.py       # AuditSink 抽象 + FileAuditSink + DBAuditSink(预留)
└── tracer.py     # AuditTracer + audit_node 装饰器 + 节点包装
```

### AuditSink 接口

```python
class AuditSink(ABC):
    @abstractmethod
    def write(self, record: Dict[str, Any]) -> None: ...
```

- `FileAuditSink`：当前实现，写入 JSON Lines 文件
- `DBAuditSink`：预留接口，后续可接入 SQLite/MySQL

## Fail-Open 设计

审计日志写入失败**不会阻断主流程**（等保要求系统可用性优先）：

```python
try:
    self._sink.write(record)
except Exception as e:
    logger.error("Audit log write failed (fail-open): %s", e)
```

写入失败时会在系统日志（logging）中记录 error，不影响问答功能。

## 集成方式

### agent/graph.py 自动注入

节点函数在 graph 构建时自动包裹审计装饰器（零侵入）：

```python
# 在 create_agent_graph() 中
tracer = get_tracer()
_make_audit = lambda name: audit_node(name) if tracer else lambda f: f

query_understand_fn = _make_audit("query_understand")(query_understand_fn)
mysql_query_fn = _make_audit("mysql_query")(mysql_query_fn)
# ...
```

### 手动打点（如需）

```python
from framework.audit.tracer import get_tracer

tracer = get_tracer()
trace_id = tracer.generate_trace_id()

tracer.record(
    trace_id=trace_id,
    user_id="admin01",
    node_name="custom_action",
    input_summary="...",
    output_summary="...",
    latency_ms=123.4,
    status="success",
)
```

## 等保合规对照

| 等保要求 | 实现方式 |
|---------|---------|
| 审计记录生成 | 每个 Agent 节点自动记录 |
| 审计记录内容 | trace_id + user_id + 节点名 + 出入参摘要 + 耗时 + 状态 |
| 审计记录保护 | 本地文件存储，后续可切数据库 |
| 审计记录查询 | query_audit_log.py 按 trace_id/时间/用户/状态查询 |
| 审计进程保护 | fail-open 设计，审计失败不影响主流程 |
