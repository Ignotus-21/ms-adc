import os
import time
from enum import Enum
from typing import Callable, Any, Tuple, Optional

class CircuitState(Enum):
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Failing, fast fallback
    HALF_OPEN = "HALF_OPEN"# Testing recovery

class CircuitBreaker:
    """
    Protects downstream metrology systems from cascading timeouts, 429 quota exhaustion, and cloud outages (Comp 21).
    Enforces a strict SLA execution budget (default 15.0s) before tripping to local edge fallback.
    """
    def __init__(self, failure_threshold: int = 3, recovery_time_s: float = 5.0, timeout_s: Optional[float] = None):
        self.failure_threshold = failure_threshold
        self.recovery_time_s = recovery_time_s
        self.timeout_s = timeout_s if timeout_s is not None else float(os.getenv("CIRCUIT_BREAKER_TIMEOUT_S", "15.0"))
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = CircuitState.CLOSED

    def execute(self, primary_fn: Callable[[], Any], fallback_fn: Callable[[], Any]) -> Tuple[Any, str]:
        current_time = time.time()

        if self.state == CircuitState.OPEN:
            if current_time - self.last_failure_time > self.recovery_time_s:
                self.state = CircuitState.HALF_OPEN
            else:
                return fallback_fn(), "CIRCUIT_OPEN_FALLBACK"

        start_call = time.time()
        try:
            result = primary_fn()
            duration = time.time() - start_call
            
            # Enforce execution timeout SLA
            if duration > self.timeout_s:
                raise TimeoutError(f"Execution SLA breached: took {duration:.2f}s > {self.timeout_s}s limit")

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
            return result, "PRIMARY_SUCCESS"
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
            return fallback_fn(), f"FALLBACK_TRIGGERED: {str(e)}"
