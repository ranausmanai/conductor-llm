"""
Circuit breaker pattern for Conductor.

Prevents cascading failures by temporarily blocking requests to failing providers.
"""

import time
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Optional


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout_seconds: float = 60.0  # Time before trying again
    half_open_max_requests: int = 3  # Max requests in half-open state


class CircuitBreakerError(Exception):
    """Raised when circuit is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    Tracks failure rates and temporarily blocks requests to failing services.
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Circuit breaker name (e.g., provider name)
            config: Configuration. Uses defaults if not provided.
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._lock = Lock()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_requests = 0

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state

    def call(self, func, *args, **kwargs):
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        with self._lock:
            self._check_and_update_state()

            if self._state == CircuitState.OPEN:
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Service temporarily unavailable."
                )

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_requests >= self.config.half_open_max_requests:
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is HALF-OPEN. "
                        f"Max test requests reached."
                    )
                self._half_open_requests += 1

        # Execute outside lock
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _check_and_update_state(self) -> None:
        """Check if state should transition."""
        if self._state == CircuitState.OPEN:
            # Check if timeout expired
            if (
                self._last_failure_time
                and time.time() - self._last_failure_time >= self.config.timeout_seconds
            ):
                self._state = CircuitState.HALF_OPEN
                self._half_open_requests = 0

    def _on_success(self) -> None:
        """Record successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self._half_open_requests = 0
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def _on_failure(self) -> None:
        """Record failed call."""
        with self._lock:
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Failure in half-open -> back to open
                self._state = CircuitState.OPEN
                self._success_count = 0
                self._half_open_requests = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Manually reset circuit to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._half_open_requests = 0

    def get_stats(self) -> dict:
        """
        Get circuit breaker statistics.

        Returns:
            Dictionary with stats
        """
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time,
                "time_until_retry": (
                    max(
                        0,
                        self.config.timeout_seconds
                        - (time.time() - self._last_failure_time),
                    )
                    if self._last_failure_time and self._state == CircuitState.OPEN
                    else 0
                ),
            }
