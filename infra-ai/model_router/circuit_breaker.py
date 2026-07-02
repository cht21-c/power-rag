"""轻量级三态熔断器（CLOSED → OPEN → HALF_OPEN → CLOSED）"""
import time, logging
from enum import Enum

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, name: str, fail_threshold: int = 5, cooldown_sec: int = 60):
        self.name = name
        self.fail_threshold = fail_threshold
        self.cooldown_sec = cooldown_sec
        self.state = CircuitState.CLOSED
        self.fail_count = 0
        self.last_fail_time = 0.0
        self.last_state_change = time.time()

    def record_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self._transition(CircuitState.CLOSED)
        self.fail_count = 0

    def record_failure(self):
        self.fail_count += 1
        self.last_fail_time = time.time()
        if self.state == CircuitState.CLOSED and self.fail_count >= self.fail_threshold:
            self._transition(CircuitState.OPEN)
        elif self.state == CircuitState.HALF_OPEN:
            self._transition(CircuitState.OPEN)

    def allow_request(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_fail_time >= self.cooldown_sec:
                self._transition(CircuitState.HALF_OPEN)
                return True
            return False
        return True  # HALF_OPEN

    def _transition(self, new_state: CircuitState):
        old = self.state
        self.state = new_state
        self.last_state_change = time.time()
        msg = f"[CircuitBreaker:{self.name}] {old.value} -> {new_state.value}"
        if new_state == CircuitState.OPEN:
            logger.warning(msg)
        else:
            logger.info(msg)
