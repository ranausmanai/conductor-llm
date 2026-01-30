"""
Rate limiting for Conductor.

Prevents excessive API usage and protects against runaway costs.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from typing import Optional


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    max_cost_per_hour_usd: float = 10.0


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""
    pass


class RateLimiter:
    """
    Token bucket rate limiter with cost-based limits.

    Tracks requests and costs per client to enforce limits.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Initialize rate limiter.

        Args:
            config: Rate limit configuration. Uses defaults if not provided.
        """
        self.config = config or RateLimitConfig()
        self._lock = Lock()

        # Track request timestamps per client
        self._requests_minute: dict[str, deque] = defaultdict(lambda: deque())
        self._requests_hour: dict[str, deque] = defaultdict(lambda: deque())

        # Track costs per client
        self._costs_hour: dict[str, list[tuple[float, float]]] = defaultdict(list)

    def check_and_record(
        self,
        client_id: str,
        estimated_cost_usd: float = 0.0,
    ) -> None:
        """
        Check if request is allowed and record it.

        Args:
            client_id: Client identifier
            estimated_cost_usd: Estimated cost of this request

        Raises:
            RateLimitError: If rate limit is exceeded
        """
        with self._lock:
            now = time.time()

            # Clean old entries
            self._clean_old_entries(client_id, now)

            # Check per-minute limit
            if len(self._requests_minute[client_id]) >= self.config.requests_per_minute:
                raise RateLimitError(
                    f"Rate limit exceeded: {self.config.requests_per_minute} "
                    f"requests per minute for client '{client_id}'"
                )

            # Check per-hour limit
            if len(self._requests_hour[client_id]) >= self.config.requests_per_hour:
                raise RateLimitError(
                    f"Rate limit exceeded: {self.config.requests_per_hour} "
                    f"requests per hour for client '{client_id}'"
                )

            # Check cost limit
            total_cost = sum(cost for _, cost in self._costs_hour[client_id])
            if total_cost + estimated_cost_usd > self.config.max_cost_per_hour_usd:
                raise RateLimitError(
                    f"Cost limit exceeded: ${total_cost + estimated_cost_usd:.2f} "
                    f"(max: ${self.config.max_cost_per_hour_usd:.2f} per hour "
                    f"for client '{client_id}')"
                )

            # Record request
            self._requests_minute[client_id].append(now)
            self._requests_hour[client_id].append(now)
            self._costs_hour[client_id].append((now, estimated_cost_usd))

    def _clean_old_entries(self, client_id: str, now: float) -> None:
        """Remove entries older than the time windows."""
        # Clean minute window (60 seconds)
        while (
            self._requests_minute[client_id]
            and now - self._requests_minute[client_id][0] > 60
        ):
            self._requests_minute[client_id].popleft()

        # Clean hour window (3600 seconds)
        while (
            self._requests_hour[client_id]
            and now - self._requests_hour[client_id][0] > 3600
        ):
            self._requests_hour[client_id].popleft()

        # Clean cost tracking
        self._costs_hour[client_id] = [
            (ts, cost)
            for ts, cost in self._costs_hour[client_id]
            if now - ts <= 3600
        ]

    def get_stats(self, client_id: str) -> dict:
        """
        Get current usage statistics for a client.

        Args:
            client_id: Client identifier

        Returns:
            Dictionary with usage stats
        """
        with self._lock:
            now = time.time()
            self._clean_old_entries(client_id, now)

            total_cost = sum(cost for _, cost in self._costs_hour[client_id])

            return {
                "client_id": client_id,
                "requests_last_minute": len(self._requests_minute[client_id]),
                "requests_last_hour": len(self._requests_hour[client_id]),
                "cost_last_hour_usd": total_cost,
                "requests_per_minute_limit": self.config.requests_per_minute,
                "requests_per_hour_limit": self.config.requests_per_hour,
                "cost_per_hour_limit_usd": self.config.max_cost_per_hour_usd,
                "minute_remaining": (
                    self.config.requests_per_minute
                    - len(self._requests_minute[client_id])
                ),
                "hour_remaining": (
                    self.config.requests_per_hour
                    - len(self._requests_hour[client_id])
                ),
                "cost_remaining_usd": (
                    self.config.max_cost_per_hour_usd - total_cost
                ),
            }

    def reset(self, client_id: Optional[str] = None) -> None:
        """
        Reset rate limits for a client or all clients.

        Args:
            client_id: Client to reset, or None for all clients
        """
        with self._lock:
            if client_id:
                self._requests_minute.pop(client_id, None)
                self._requests_hour.pop(client_id, None)
                self._costs_hour.pop(client_id, None)
            else:
                self._requests_minute.clear()
                self._requests_hour.clear()
                self._costs_hour.clear()
