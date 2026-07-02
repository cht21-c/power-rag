"""
审计日志模块单元测试

覆盖：
- AuditTracer 记录写入
- trace_id 唯一性
- 写入失败不影响主流程（fail-open）
- audit_node 装饰器
- FileAuditSink 按天切割
"""

import json
import os
import tempfile
import time
import pytest

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.audit.sink import FileAuditSink
from framework.audit.tracer import AuditTracer, get_tracer, audit_node, _summarize


class TestAuditTracer:
    """AuditTracer 核心功能测试。"""

    def test_record_write(self, tmp_path):
        sink = FileAuditSink(str(tmp_path))
        tracer = AuditTracer(sink)

        trace_id = tracer.generate_trace_id()
        assert len(trace_id) == 16

        tracer.record(
            trace_id=trace_id,
            user_id="test_user",
            node_name="query_understand",
            input_summary="test input",
            output_summary="test output",
            route_decision="rag",
            latency_ms=42.5,
            status="success",
        )

        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        logfile = tmp_path / f"{today}.jsonl"
        assert logfile.exists()

        lines = logfile.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["trace_id"] == trace_id
        assert record["user_id"] == "test_user"
        assert record["node_name"] == "query_understand"
        assert record["latency_ms"] == 42.5
        assert record["status"] == "success"

    def test_trace_id_unique(self):
        tracer = AuditTracer()
        ids = {tracer.generate_trace_id() for _ in range(100)}
        assert len(ids) == 100

    def test_fail_open(self, tmp_path):
        """写入失败不应抛异常（fail-open）。"""
        # 创建一个目录作为"文件"，导致写入失败
        sink = FileAuditSink(str(tmp_path))
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        logfile = tmp_path / f"{today}.jsonl"
        logfile.mkdir(exist_ok=True)  # 把文件路径变成目录

        tracer = AuditTracer(sink)
        # 不应该抛异常
        tracer.record(
            trace_id=tracer.generate_trace_id(),
            user_id="test",
            node_name="test_node",
            status="success",
        )

    def test_record_all_fields(self, tmp_path):
        sink = FileAuditSink(str(tmp_path))
        tracer = AuditTracer(sink)

        tracer.record(
            trace_id="abc123",
            user_id="admin",
            node_name="llm_generate",
            input_summary="query: how to...",
            output_summary="answer: you should...",
            route_decision="mysql_hit",
            latency_ms=1234.56,
            status="success",
            extra={"model": "deepseek-chat", "tokens": 500},
        )

        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        logfile = tmp_path / f"{today}.jsonl"
        record = json.loads(logfile.read_text(encoding="utf-8").strip())
        assert record["route_decision"] == "mysql_hit"
        assert record["extra"]["model"] == "deepseek-chat"


class TestSummarize:
    """截断函数测试。"""

    def test_short_string(self):
        assert _summarize("hello") == "hello"

    def test_long_string_truncated(self):
        long_text = "x" * 1000
        result = _summarize(long_text)
        assert len(result) == 500

    def test_dict_summarize(self):
        d = {"key": "value"}
        result = _summarize(d)
        assert "key" in result

    def test_none(self):
        assert _summarize(None) == "None"


class TestAuditNodeDecorator:
    """audit_node 装饰器测试。"""

    def test_basic_wrapping(self, tmp_path):
        sink = FileAuditSink(str(tmp_path))
        tracer = AuditTracer(sink)

        @audit_node(tracer, "test_node")
        def my_node(state):
            state["result"] = "done"
            return state

        state = {"query": "test"}
        result = my_node(state)
        assert result["result"] == "done"
        assert "_trace_id" in result

    def test_error_recording(self, tmp_path):
        sink = FileAuditSink(str(tmp_path))
        tracer = AuditTracer(sink)

        @audit_node(tracer, "error_node")
        def bad_node(state):
            raise ValueError("test error")

        with pytest.raises(ValueError):
            bad_node({"query": "test"})

        # 确认错误被记录
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        logfile = tmp_path / f"{today}.jsonl"
        records = logfile.read_text(encoding="utf-8").strip().split("\n")
        last = json.loads(records[-1])
        assert last["status"] == "error"
        assert last["status"] == "error"

    def test_multiple_nodes_same_trace(self, tmp_path):
        sink = FileAuditSink(str(tmp_path))
        tracer = AuditTracer(sink)

        @audit_node(tracer, "step1")
        def step1(state):
            state["step1_done"] = True
            return state

        @audit_node(tracer, "step2")
        def step2(state):
            state["step2_done"] = True
            return state

        state = {"query": "test pipeline"}
        state = step1(state)
        state = step2(state)

        tid = state["_trace_id"]
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        logfile = tmp_path / f"{today}.jsonl"
        records = [json.loads(l) for l in logfile.read_text(encoding="utf-8").strip().split("\n")]
        assert len(records) == 2
        assert records[0]["trace_id"] == tid
        assert records[1]["trace_id"] == tid
        assert records[0]["node_name"] == "step1"
        assert records[1]["node_name"] == "step2"
