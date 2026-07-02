"""模型路由器：管理候选模型列表 + 熔断 + 降级"""
import logging, sys, time
from typing import Any, Callable, Dict, Iterator, List, Optional

from .model_client import ModelClient, DeepSeekClient
from .circuit_breaker import CircuitBreaker, CircuitState

logger = logging.getLogger(__name__)


class ModelRouter:
    """持有多个候选 ModelClient，按优先级路由，失败自动降级。

    用法:
        router = ModelRouter([deepseek_client])
        router = ModelRouter([deepseek_client, fallback_client])
        answer = router.generate(messages)
    """

    def __init__(self, clients: List[ModelClient], fail_threshold: int = 5,
                 cooldown_sec: int = 60, first_token_timeout: float = 10.0):
        if not clients:
            raise ValueError("At least one ModelClient required")
        self._clients = clients
        self._active_idx = 0
        self._breakers = [
            CircuitBreaker(f"model-{i}", fail_threshold, cooldown_sec)
            for i in range(len(clients))
        ]
        self.first_token_timeout = first_token_timeout

    @property
    def active_client(self) -> ModelClient:
        return self._clients[self._active_idx]

    @property
    def active_model_name(self) -> str:
        c = self.active_client
        return getattr(c, 'model', str(c.__class__.__name__))

    def _try_next(self) -> bool:
        """切换到下一个候选模型，返回是否还有可用模型。"""
        for i in range(len(self._clients)):
            self._active_idx = (self._active_idx + 1) % len(self._clients)
            if self._breakers[self._active_idx].allow_request():
                logger.warning("[ModelRouter] Switched to model-%d: %s",
                               self._active_idx, self.active_model_name)
                return True
        return False

    def _reset_all_breakers(self):
        """所有熔断器重置（在尝试了所有模型都失败后）。"""
        for cb in self._breakers:
            cb.state = CircuitState.CLOSED
            cb.fail_count = 0
        self._active_idx = 0

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """生成回答，自动容错降级。"""
        errors = []
        start_time = time.time()

        for attempt in range(len(self._clients)):
            idx = self._active_idx
            breaker = self._breakers[idx]

            if not breaker.allow_request():
                logger.warning("[ModelRouter] Model-%d circuit OPEN, skipping", idx)
                errors.append(f"model-{idx}: circuit_open")
                if not self._try_next():
                    break
                continue

            try:
                client = self._clients[idx]
                result = client.generate(messages, **kwargs)
                breaker.record_success()
                logger.debug("[ModelRouter] Model-%d success (%.1fs)", idx, time.time() - start_time)
                return result
            except Exception as e:
                breaker.record_failure()
                errors.append(f"model-{idx}: {e}")
                logger.error("[ModelRouter] Model-%d failed: %s", idx, e)
                if not self._try_next():
                    break

        # 所有模型都失败 → 重置熔断器并报错
        self._reset_all_breakers()
        error_msg = "所有候选模型均不可用"
        if errors:
            error_msg += ":\n" + "\n".join(f"  - {e}" for e in errors)
        raise RuntimeError(error_msg)

    def generate_stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """流式生成，带首包超时。"""
        errors = []
        for attempt in range(len(self._clients)):
            idx = self._active_idx
            breaker = self._breakers[idx]
            if not breaker.allow_request():
                errors.append(f"model-{idx}: circuit_open")
                if not self._try_next():
                    break
                continue
            try:
                client = self._clients[idx]
                gen = client.generate_stream(messages, **kwargs)
                t0 = time.time()
                first = next(gen, None)
                if first is None:
                    raise TimeoutError("No token received")
                if time.time() - t0 > self.first_token_timeout:
                    raise TimeoutError(f"First token timeout ({self.first_token_timeout}s)")
                yield first
                for chunk in gen:
                    yield chunk
                breaker.record_success()
                return
            except Exception as e:
                breaker.record_failure()
                errors.append(f"model-{idx}: {e}")
                if not self._try_next():
                    break
        self._reset_all_breakers()
        error_msg = "所有候选模型均不可用"
        if errors:
            error_msg += ":\n" + "\n".join(f"  - {e}" for e in errors)
        yield f"\n\n❌ {error_msg}"

    def get_status(self) -> dict:
        return {
            "active_index": self._active_idx,
            "active_model": self.active_model_name,
            "breakers": [
                {"name": cb.name, "state": cb.state.value, "fail_count": cb.fail_count}
                for cb in self._breakers
            ],
        }


# 工厂函数
def create_default_router(api_key: str, base_url: str = "https://api.deepseek.com",
                          model: str = "deepseek-chat", **kwargs) -> ModelRouter:
    """从配置创建默认路由器。"""
    from config import (
        MODEL_CANDIDATES, CIRCUIT_BREAKER_FAIL_THRESHOLD,
        CIRCUIT_BREAKER_COOLDOWN_SEC, STREAM_FIRST_TOKEN_TIMEOUT_SEC,
    )
    candidates = MODEL_CANDIDATES if MODEL_CANDIDATES else [model]
    clients = []
    for m in candidates:
        m = m.strip()
        clients.append(DeepSeekClient(model=m, api_key=api_key, base_url=base_url))
    return ModelRouter(
        clients,
        fail_threshold=CIRCUIT_BREAKER_FAIL_THRESHOLD,
        cooldown_sec=CIRCUIT_BREAKER_COOLDOWN_SEC,
        first_token_timeout=STREAM_FIRST_TOKEN_TIMEOUT_SEC,
    )
