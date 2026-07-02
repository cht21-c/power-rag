"""
P2 工程健壮性单元测试

覆盖: 熔断器三态、模型路由降级、记忆压缩、工具注册
"""
import os, sys, time, pytest

_project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project)
sys.path.insert(0, os.path.join(_project, "camera_sdk_agent"))
sys.path.insert(0, os.path.join(_project, "infra-ai"))


# ============================================================================
# 熔断器测试
# ============================================================================

class TestCircuitBreaker:
    from model_router.circuit_breaker import CircuitBreaker, CircuitState

    def test_initial_state(self):
        cb = self.CircuitBreaker("test", fail_threshold=3, cooldown_sec=10)
        assert cb.state == self.CircuitState.CLOSED
        assert cb.fail_count == 0
        assert cb.allow_request() is True

    def test_transition_to_open(self):
        cb = self.CircuitBreaker("test", fail_threshold=3, cooldown_sec=10)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == self.CircuitState.OPEN
        assert not cb.allow_request()

    def test_half_open_after_cooldown(self):
        cb = self.CircuitBreaker("test", fail_threshold=3, cooldown_sec=0)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == self.CircuitState.OPEN
        assert cb.allow_request() is True
        assert cb.state == self.CircuitState.HALF_OPEN

    def test_half_open_success_recovery(self):
        cb = self.CircuitBreaker("test", fail_threshold=3, cooldown_sec=0)
        for _ in range(3):
            cb.record_failure()
        cb.allow_request()  # -> HALF_OPEN
        cb.record_success()
        assert cb.state == self.CircuitState.CLOSED
        assert cb.fail_count == 0

    def test_half_open_failure_back_to_open(self):
        cb = self.CircuitBreaker("test", fail_threshold=3, cooldown_sec=0)
        for _ in range(3):
            cb.record_failure()
        cb.allow_request()
        cb.record_failure()
        assert cb.state == self.CircuitState.OPEN


# ============================================================================
# 模型路由测试
# ============================================================================

class TestModelRouter:
    """需要 mock LangChain 调用。"""

    def test_router_requires_clients(self):
        from model_router.model_router import ModelRouter
        with pytest.raises(ValueError):
            ModelRouter([])

    def test_router_basic(self):
        from model_router.model_client import DeepSeekClient
        from model_router.model_router import ModelRouter
        c = DeepSeekClient(model="test", api_key="sk-test")
        router = ModelRouter([c], fail_threshold=10)
        assert router.active_model_name == "test"
        assert len(router.get_status()["breakers"]) == 1

    def test_router_status(self):
        from model_router.model_client import DeepSeekClient
        from model_router.model_router import ModelRouter
        c = DeepSeekClient(model="m1", api_key="k1")
        c2 = DeepSeekClient(model="m2", api_key="k2")
        router = ModelRouter([c, c2])
        status = router.get_status()
        assert status["active_model"] == "m1"
        assert len(status["breakers"]) == 2


# ============================================================================
# 记忆压缩测试
# ============================================================================

class TestConversationCompressor:
    def test_no_compression_needed(self):
        from framework.memory.compressor import ConversationCompressor
        comp = ConversationCompressor(None, window_turns=3)
        msgs = [{"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"}]
        assert not comp.needs_compression(msgs)

    def test_compression_needed_long(self):
        from framework.memory.compressor import ConversationCompressor
        comp = ConversationCompressor(None, window_turns=0, trigger_tokens=20)
        long_msgs = [{"role": "user", "content": "x" * 500},
                     {"role": "assistant", "content": "y" * 500}]
        assert comp.needs_compression(long_msgs)

    def test_hard_truncate(self):
        from framework.memory.compressor import ConversationCompressor
        comp = ConversationCompressor(None)
        msgs = [{"role": "user", "content": "a" * 100},
                {"role": "assistant", "content": "b" * 100}]
        result = comp.hard_truncate(msgs, max_tokens=5)
        assert len(result) <= 2


# ============================================================================
# 工具注册测试
# ============================================================================

class TestToolRegistry:
    def test_register_and_list(self):
        from framework.tool_registry.registry import register_tool, list_tools

        @register_tool(name="test_tool", description="A test tool")
        def test_tool(x: str) -> str:
            return x

        tools = list_tools()
        names = [t["name"] for t in tools]
        assert "test_tool" in names

    def test_get_registered_tools(self):
        pytest.importorskip("langchain_core")
        from framework.tool_registry.registry import get_registered_tools, register_tool

        @register_tool(name="echo2", description="Echo back")
        def echo2(x: str) -> str:
            return x

        tools = get_registered_tools()
        assert len(tools) >= 1
