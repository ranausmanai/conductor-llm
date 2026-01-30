"""
Metrics and observability for Conductor.

Provides structured logging and metrics collection for monitoring.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, UTC
from typing import Optional, Any
from pathlib import Path


@dataclass
class MetricEvent:
    """A single metric event."""
    timestamp: str
    event_type: str  # request, decision, outcome, error
    request_id: str
    client_id: str
    data: dict[str, Any]


class MetricsCollector:
    """
    Collects and aggregates metrics from Conductor operations.

    Provides both real-time stats and historical tracking.
    """

    def __init__(
        self,
        metrics_file: Optional[Path] = None,
        enable_logging: bool = True,
    ):
        """
        Initialize metrics collector.

        Args:
            metrics_file: Optional file to write metrics to (JSONL format)
            enable_logging: Whether to enable structured logging
        """
        self.metrics_file = metrics_file
        self.enable_logging = enable_logging

        # Configure logger
        self.logger = logging.getLogger("conductor.metrics")
        if enable_logging and not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                )
            )
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # In-memory metrics
        self._events: list[MetricEvent] = []
        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)

    def record_request(
        self,
        request_id: str,
        client_id: str,
        task_type: str,
        prompt_length: int,
        **extra: Any,
    ) -> None:
        """
        Record incoming request.

        Args:
            request_id: Request identifier
            client_id: Client identifier
            task_type: Type of task
            prompt_length: Length of prompt in characters
            **extra: Additional fields
        """
        self._record_event(
            event_type="request",
            request_id=request_id,
            client_id=client_id,
            data={
                "task_type": task_type,
                "prompt_length": prompt_length,
                **extra,
            },
        )
        self._counters["requests_total"] += 1
        self._counters[f"requests_by_task_{task_type}"] += 1
        self._counters[f"requests_by_client_{client_id}"] += 1

    def record_decision(
        self,
        request_id: str,
        client_id: str,
        strategy_id: str,
        model_id: str,
        predicted_cost: float,
        predicted_latency_p95: int,
        fallback_used: bool,
        **extra: Any,
    ) -> None:
        """
        Record routing decision.

        Args:
            request_id: Request identifier
            client_id: Client identifier
            strategy_id: Chosen strategy
            model_id: Chosen model
            predicted_cost: Predicted cost in USD
            predicted_latency_p95: Predicted P95 latency
            fallback_used: Whether fallback was used
            **extra: Additional fields
        """
        self._record_event(
            event_type="decision",
            request_id=request_id,
            client_id=client_id,
            data={
                "strategy_id": strategy_id,
                "model_id": model_id,
                "predicted_cost_usd": predicted_cost,
                "predicted_latency_p95_ms": predicted_latency_p95,
                "fallback_used": fallback_used,
                **extra,
            },
        )
        self._counters[f"decisions_by_strategy_{strategy_id}"] += 1
        self._counters[f"decisions_by_model_{model_id}"] += 1
        if fallback_used:
            self._counters["decisions_fallback"] += 1

    def record_outcome(
        self,
        request_id: str,
        client_id: str,
        actual_cost: float,
        actual_latency: int,
        finish_reason: str,
        sla_met: bool,
        **extra: Any,
    ) -> None:
        """
        Record execution outcome.

        Args:
            request_id: Request identifier
            client_id: Client identifier
            actual_cost: Actual cost in USD
            actual_latency: Actual latency in milliseconds
            finish_reason: How the LLM finished
            sla_met: Whether SLA was met
            **extra: Additional fields
        """
        self._record_event(
            event_type="outcome",
            request_id=request_id,
            client_id=client_id,
            data={
                "actual_cost_usd": actual_cost,
                "actual_latency_ms": actual_latency,
                "finish_reason": finish_reason,
                "sla_met": sla_met,
                **extra,
            },
        )
        self._counters["outcomes_total"] += 1
        self._counters[f"outcomes_{finish_reason}"] += 1
        if sla_met:
            self._counters["outcomes_sla_met"] += 1
        else:
            self._counters["outcomes_sla_violated"] += 1

        # Record in histograms
        self._histograms["cost_usd"].append(actual_cost)
        self._histograms["latency_ms"].append(actual_latency)

    def record_error(
        self,
        request_id: str,
        client_id: str,
        error_type: str,
        error_message: str,
        **extra: Any,
    ) -> None:
        """
        Record error.

        Args:
            request_id: Request identifier
            client_id: Client identifier
            error_type: Type of error
            error_message: Error message
            **extra: Additional fields
        """
        self._record_event(
            event_type="error",
            request_id=request_id,
            client_id=client_id,
            data={
                "error_type": error_type,
                "error_message": error_message,
                **extra,
            },
        )
        self._counters["errors_total"] += 1
        self._counters[f"errors_{error_type}"] += 1

        if self.enable_logging:
            self.logger.error(
                f"Error in request {request_id}: {error_type} - {error_message}"
            )

    def _record_event(
        self,
        event_type: str,
        request_id: str,
        client_id: str,
        data: dict,
    ) -> None:
        """Record a metric event."""
        event = MetricEvent(
            timestamp=datetime.now(UTC).isoformat(),
            event_type=event_type,
            request_id=request_id,
            client_id=client_id,
            data=data,
        )

        self._events.append(event)

        # Write to file if configured
        if self.metrics_file:
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(asdict(event)) + "\n")

        # Log if enabled
        if self.enable_logging:
            self.logger.info(
                f"{event_type.upper()}: request_id={request_id}, "
                f"client_id={client_id}, data={data}"
            )

    def get_stats(self) -> dict:
        """
        Get aggregated statistics.

        Returns:
            Dictionary with metrics summary
        """
        import statistics as stats

        cost_values = self._histograms.get("cost_usd", [])
        latency_values = self._histograms.get("latency_ms", [])

        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "cost": {
                "total_usd": sum(cost_values),
                "avg_usd": stats.mean(cost_values) if cost_values else 0,
                "p50_usd": stats.median(cost_values) if cost_values else 0,
                "p95_usd": (
                    stats.quantiles(cost_values, n=20)[18]
                    if len(cost_values) >= 20
                    else (max(cost_values) if cost_values else 0)
                ),
            },
            "latency": {
                "avg_ms": stats.mean(latency_values) if latency_values else 0,
                "p50_ms": stats.median(latency_values) if latency_values else 0,
                "p95_ms": (
                    stats.quantiles(latency_values, n=20)[18]
                    if len(latency_values) >= 20
                    else (max(latency_values) if latency_values else 0)
                ),
                "p99_ms": (
                    stats.quantiles(latency_values, n=100)[98]
                    if len(latency_values) >= 100
                    else (max(latency_values) if latency_values else 0)
                ),
            },
            "total_events": len(self._events),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._events.clear()
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
