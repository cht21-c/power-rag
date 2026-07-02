"""
审计追踪器 - 全链路审计日志的核心模块

提供：
- AuditTracer: trace_id 生成、审计记录工厂、节点包装
- audit_node / audit_node_with_route: LangGraph 节点包装装饰器
- 各节点出入参自动截断（前500字符）

用法:
    from framework.audit.tracer import AuditTracer, audit_node, audit_node_with_route

    tracer = AuditTracer(sink=FileAuditSink())

    @audit_node(tracer, "rag_retrieve")
    def rag_retrieve_node(state):
        ...
"""

import json
import time
import uuid
import functools
import logging
from typing import Any, Callable, Dict, Optional

from .sink import AuditSink, FileAuditSink

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 摘要截断
# ---------------------------------------------------------------------------

MAX_SUMMARY_LENGTH = 500


def _summarize(obj: Any) -> str:
    """将任意对象截断为 500 字符摘要。"""
    if obj is None:
        return "None"
    if isinstance(obj, str):
        return obj[:MAX_SUMMARY_LENGTH]
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
        return s[:MAX_SUMMARY_LENGTH]
    except Exception:
        return str(obj)[:MAX_SUMMARY_LENGTH]


# ---------------------------------------------------------------------------
# AuditTracer
# ---------------------------------------------------------------------------


class AuditTracer:
    """审计追踪器。

    管理 trace_id 生成和审计记录写入。
    """

    def __init__(self, sink: Optional[AuditSink] = None):
        self._sink = sink or FileAuditSink()

    def generate_trace_id(self) -> str:
        return uuid.uuid4().hex[:16]

    def record(self, trace_id: str, user_id: str, node_name: str,
               input_summary: str = "", output_summary: str = "",
               route_decision: str = "", latency_ms: float = 0,
               status: str = "success", extra: Optional[Dict[str, Any]] = None) -> None:
        """写入一条审计记录。写入失败不影响主流程。"""
        record = {
            "trace_id": trace_id,
            "user_id": user_id,
            "node_name": node_name,
            "input_summary": input_summary,
            "output_summary": output_summary,
            "route_decision": route_decision,
            "latency_ms": round(latency_ms, 2),
            "status": status,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.", time.gmtime())
                         + f"{int((time.time() % 1) * 1000):03d}Z",
        }
        if extra:
            record["extra"] = extra
        try:
            self._sink.write(record)
        except Exception as e:
            logger.error("AuditTracer.record() failed (fail-open): %s", e)


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

_tracer: Optional[AuditTracer] = None


def get_tracer(sink: Optional[AuditSink] = None) -> AuditTracer:
    global _tracer
    if _tracer is None:
        _tracer = AuditTracer(sink)
    return _tracer


# ---------------------------------------------------------------------------
# LangGraph 节点包装装饰器
# ---------------------------------------------------------------------------


def audit_node(tracer_or_node_name, node_name: Optional[str] = None):
    """统一审计包装器 — 支持两种调用方式：

    @audit_node("rag_retrieve")
    def my_node(state): ...

    或：

    @audit_node(tracer, "rag_retrieve")
    def my_node(state): ...

    自动记录：节点名、入参/出参摘要、耗时、状态
    """
    if isinstance(tracer_or_node_name, str):
        # 短格式：@audit_node("name")
        _t = None  # 延迟从全局拿
        _name = tracer_or_node_name
    else:
        # 带 tracer 格式：@audit_node(tracer, "name")
        _t = tracer_or_node_name
        _name = node_name

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(state: dict):
            tracer = _t or get_tracer()
            if tracer is None:
                return func(state)

            user = {"user_id": "unknown", "role": "operator"}
            try:
                from framework.auth.middleware import get_current_user
                u = get_current_user()
                if u:
                    user = u
            except ImportError:
                pass

            trace_id = state.get("_trace_id", "") or tracer.generate_trace_id()
            state["_trace_id"] = trace_id

            input_summary = _summarize(state)
            t0 = time.time()

            try:
                result = func(state)
                latency = (time.time() - t0) * 1000

                route_decision = ""
                if _name == "route_decision":
                    route_decision = result.get("route", "")
                elif _name == "mysql_query":
                    has_results = bool(result.get("formatted_context", ""))
                    route_decision = "mysql_hit" if has_results else "mysql_miss_fallback"

                status = result.get("error") or "success"
                if status and status != "success":
                    status = f"error:{status}" if status != "no_context" else "success"

                tracer.record(
                    trace_id=trace_id,
                    user_id=user.get("user_id", "unknown"),
                    node_name=_name,
                    input_summary=input_summary,
                    output_summary=_summarize(result),
                    route_decision=route_decision,
                    latency_ms=latency,
                    status=status,
                )
                return result
            except Exception as e:
                latency = (time.time() - t0) * 1000
                tracer.record(
                    trace_id=trace_id,
                    user_id=user.get("user_id", "unknown"),
                    node_name=_name,
                    input_summary=input_summary,
                    output_summary=f"{type(e).__name__}: {e}",
                    route_decision="",
                    latency_ms=latency,
                    status="error",
                )
                raise

        return wrapper
    return decorator


def audit_node_with_route(tracer, node_name: str, route_field: str = "route"):
    """带路由信息记录的审计包装器（用于 route_decision 节点）。

    与 audit_node 的区别：额外记录路由决策。
    """
    return audit_node(tracer, node_name)
