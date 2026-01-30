"""
Logger for Conductor.

Persists decisions and outcomes for calibration and analysis.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator
import csv

from conductor.schemas import (
    Request,
    RequestFeatures,
    Decision,
    Outcome,
    LogEntry,
    TaskType,
    UrgencyTier,
    LatencyPercentile,
    FinishReason,
)


class LogStore(ABC):
    """Abstract base class for log storage backends."""
    
    @abstractmethod
    def write(self, entry: LogEntry) -> None:
        """Write a log entry."""
        pass
    
    @abstractmethod
    def read_all(self) -> Iterator[LogEntry]:
        """Read all log entries."""
        pass
    
    @abstractmethod
    def read_range(
        self,
        start: datetime,
        end: datetime,
    ) -> Iterator[LogEntry]:
        """Read log entries in a time range."""
        pass


class InMemoryLogStore(LogStore):
    """
    In-memory log store for testing.
    """
    
    def __init__(self):
        self.entries: list[LogEntry] = []
    
    def write(self, entry: LogEntry) -> None:
        self.entries.append(entry)
    
    def read_all(self) -> Iterator[LogEntry]:
        return iter(self.entries)
    
    def read_range(
        self,
        start: datetime,
        end: datetime,
    ) -> Iterator[LogEntry]:
        for entry in self.entries:
            if start <= entry.decision.timestamp_utc <= end:
                yield entry
    
    def clear(self) -> None:
        self.entries.clear()
    
    def __len__(self) -> int:
        return len(self.entries)


class JSONLLogStore(LogStore):
    """
    JSON Lines file-based log store.
    
    Each line is a complete JSON object representing a LogEntry.
    """
    
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
    
    def write(self, entry: LogEntry) -> None:
        """Append entry to JSONL file."""
        record = self._entry_to_dict(entry)
        
        with open(self.path, 'a') as f:
            f.write(json.dumps(record, default=str) + '\n')
    
    def read_all(self) -> Iterator[LogEntry]:
        """Read all entries from file."""
        if not self.path.exists():
            return
        
        with open(self.path, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    yield self._dict_to_entry(record)
    
    def read_range(
        self,
        start: datetime,
        end: datetime,
    ) -> Iterator[LogEntry]:
        """Read entries in time range."""
        for entry in self.read_all():
            if start <= entry.decision.timestamp_utc <= end:
                yield entry
    
    def _entry_to_dict(self, entry: LogEntry) -> dict:
        """Convert LogEntry to dictionary."""
        return {
            "request": {
                "request_id": entry.request.request_id,
                "client_id": entry.request.client_id,
                "trace_id": entry.request.trace_id,
                "timestamp_utc": entry.request.timestamp_utc.isoformat(),
                "prompt": entry.request.prompt,
                "system_prompt": entry.request.system_prompt,
                "task_type": entry.request.task_type.value,
                "max_latency_ms": entry.request.max_latency_ms,
                "latency_percentile": entry.request.latency_percentile.value,
                "min_quality_score": entry.request.min_quality_score,
                "max_cost_usd": entry.request.max_cost_usd,
                "expected_output_tokens": entry.request.expected_output_tokens,
                "allow_batching": entry.request.allow_batching,
                "require_tools": entry.request.require_tools,
                "require_verifier": entry.request.require_verifier,
            },
            "features": {
                "request_id": entry.features.request_id,
                "task_type": entry.features.task_type.value,
                "input_token_count": entry.features.input_token_count,
                "estimated_output_tokens": entry.features.estimated_output_tokens,
                "complexity_score": entry.features.complexity_score,
                "urgency_tier": entry.features.urgency_tier.value,
                "has_structured_output": entry.features.has_structured_output,
                "max_latency_ms": entry.features.max_latency_ms,
                "latency_percentile": entry.features.latency_percentile.value,
                "min_quality_score": entry.features.min_quality_score,
                "max_cost_usd": entry.features.max_cost_usd,
                "allow_batching": entry.features.allow_batching,
                "require_tools": entry.features.require_tools,
                "require_verifier": entry.features.require_verifier,
            },
            "decision": {
                "decision_id": entry.decision.decision_id,
                "request_id": entry.decision.request_id,
                "timestamp_utc": entry.decision.timestamp_utc.isoformat(),
                "strategy_id": entry.decision.strategy_id,
                "model_id": entry.decision.model_id,
                "temperature": entry.decision.temperature,
                "max_tokens": entry.decision.max_tokens,
                "batching_delay_ms": entry.decision.batching_delay_ms,
                "verifier_enabled": entry.decision.verifier_enabled,
                "tools_enabled": entry.decision.tools_enabled,
                "predicted_cost_usd": entry.decision.predicted_cost_usd,
                "predicted_latency_p50_ms": entry.decision.predicted_latency_p50_ms,
                "predicted_latency_p95_ms": entry.decision.predicted_latency_p95_ms,
                "predicted_latency_p99_ms": entry.decision.predicted_latency_p99_ms,
                "predicted_quality_risk": entry.decision.predicted_quality_risk,
                "why": entry.decision.why,
                "strategies_considered": entry.decision.strategies_considered,
                "strategies_filtered_out": entry.decision.strategies_filtered_out,
                "fallback_used": entry.decision.fallback_used,
                "filter_reasons": entry.decision.filter_reasons,
                "policy_version": entry.decision.policy_version,
                "coefficient_version": entry.decision.coefficient_version,
            },
            "outcome": {
                "outcome_id": entry.outcome.outcome_id,
                "decision_id": entry.outcome.decision_id,
                "request_id": entry.outcome.request_id,
                "timestamp_utc": entry.outcome.timestamp_utc.isoformat(),
                "actual_cost_usd": entry.outcome.actual_cost_usd,
                "actual_latency_ms": entry.outcome.actual_latency_ms,
                "actual_input_tokens": entry.outcome.actual_input_tokens,
                "actual_output_tokens": entry.outcome.actual_output_tokens,
                "finish_reason": entry.outcome.finish_reason.value,
                "http_status": entry.outcome.http_status,
                "retry_count": entry.outcome.retry_count,
                "verifier_passed": entry.outcome.verifier_passed,
                "format_valid": entry.outcome.format_valid,
                "quality_proxy_score": entry.outcome.quality_proxy_score,
                "sla_met": entry.outcome.sla_met,
                "cost_error_pct": entry.outcome.cost_error_pct,
                "latency_error_pct": entry.outcome.latency_error_pct,
            },
        }
    
    def _dict_to_entry(self, record: dict) -> LogEntry:
        """Convert dictionary to LogEntry."""
        req_data = record["request"]
        request = Request(
            request_id=req_data["request_id"],
            client_id=req_data["client_id"],
            trace_id=req_data.get("trace_id"),
            timestamp_utc=datetime.fromisoformat(req_data["timestamp_utc"]),
            prompt=req_data["prompt"],
            system_prompt=req_data.get("system_prompt"),
            task_type=TaskType(req_data["task_type"]),
            max_latency_ms=req_data["max_latency_ms"],
            latency_percentile=LatencyPercentile(req_data["latency_percentile"]),
            min_quality_score=req_data["min_quality_score"],
            max_cost_usd=req_data.get("max_cost_usd"),
            expected_output_tokens=req_data.get("expected_output_tokens"),
            allow_batching=req_data.get("allow_batching", True),
            require_tools=req_data.get("require_tools", False),
            require_verifier=req_data.get("require_verifier", False),
        )
        
        feat_data = record["features"]
        features = RequestFeatures(
            request_id=feat_data["request_id"],
            task_type=TaskType(feat_data["task_type"]),
            input_token_count=feat_data["input_token_count"],
            estimated_output_tokens=feat_data["estimated_output_tokens"],
            complexity_score=feat_data["complexity_score"],
            urgency_tier=UrgencyTier(feat_data["urgency_tier"]),
            has_structured_output=feat_data["has_structured_output"],
            max_latency_ms=feat_data["max_latency_ms"],
            latency_percentile=LatencyPercentile(feat_data["latency_percentile"]),
            min_quality_score=feat_data["min_quality_score"],
            max_cost_usd=feat_data.get("max_cost_usd"),
            allow_batching=feat_data.get("allow_batching", True),
            require_tools=feat_data.get("require_tools", False),
            require_verifier=feat_data.get("require_verifier", False),
        )
        
        dec_data = record["decision"]
        decision = Decision(
            decision_id=dec_data["decision_id"],
            request_id=dec_data["request_id"],
            timestamp_utc=datetime.fromisoformat(dec_data["timestamp_utc"]),
            strategy_id=dec_data["strategy_id"],
            model_id=dec_data["model_id"],
            temperature=dec_data["temperature"],
            max_tokens=dec_data["max_tokens"],
            batching_delay_ms=dec_data["batching_delay_ms"],
            verifier_enabled=dec_data["verifier_enabled"],
            tools_enabled=dec_data["tools_enabled"],
            predicted_cost_usd=dec_data["predicted_cost_usd"],
            predicted_latency_p50_ms=dec_data["predicted_latency_p50_ms"],
            predicted_latency_p95_ms=dec_data["predicted_latency_p95_ms"],
            predicted_latency_p99_ms=dec_data["predicted_latency_p99_ms"],
            predicted_quality_risk=dec_data["predicted_quality_risk"],
            why=dec_data["why"],
            strategies_considered=dec_data["strategies_considered"],
            strategies_filtered_out=dec_data["strategies_filtered_out"],
            fallback_used=dec_data["fallback_used"],
            filter_reasons=dec_data.get("filter_reasons", {}),
            policy_version=dec_data.get("policy_version", "v1.0.0"),
            coefficient_version=dec_data.get("coefficient_version", "default"),
        )
        
        out_data = record["outcome"]
        outcome = Outcome(
            outcome_id=out_data["outcome_id"],
            decision_id=out_data["decision_id"],
            request_id=out_data["request_id"],
            timestamp_utc=datetime.fromisoformat(out_data["timestamp_utc"]),
            actual_cost_usd=out_data["actual_cost_usd"],
            actual_latency_ms=out_data["actual_latency_ms"],
            actual_input_tokens=out_data["actual_input_tokens"],
            actual_output_tokens=out_data["actual_output_tokens"],
            finish_reason=FinishReason(out_data["finish_reason"]),
            http_status=out_data["http_status"],
            retry_count=out_data["retry_count"],
            verifier_passed=out_data.get("verifier_passed"),
            format_valid=out_data["format_valid"],
            quality_proxy_score=out_data["quality_proxy_score"],
            sla_met=out_data["sla_met"],
            cost_error_pct=out_data["cost_error_pct"],
            latency_error_pct=out_data["latency_error_pct"],
        )
        
        return LogEntry(
            request=request,
            features=features,
            decision=decision,
            outcome=outcome,
        )


class Logger:
    """
    Main logger interface for Conductor.
    """
    
    def __init__(self, store: Optional[LogStore] = None):
        self.store = store or InMemoryLogStore()
    
    def log(
        self,
        request: Request,
        features: RequestFeatures,
        decision: Decision,
        outcome: Outcome,
    ) -> LogEntry:
        """
        Log a complete request/decision/outcome.
        
        Args:
            request: Original request.
            features: Extracted features.
            decision: Routing decision.
            outcome: Execution outcome.
        
        Returns:
            The logged entry.
        """
        entry = LogEntry(
            request=request,
            features=features,
            decision=decision,
            outcome=outcome,
        )
        
        self.store.write(entry)
        return entry
    
    def get_entries(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> list[LogEntry]:
        """
        Get log entries, optionally filtered by time range.
        """
        if start and end:
            return list(self.store.read_range(start, end))
        return list(self.store.read_all())
    
    def export_csv(self, path: Path) -> int:
        """
        Export log entries to CSV.
        
        Returns:
            Number of entries exported.
        """
        entries = list(self.store.read_all())
        
        if not entries:
            return 0
        
        fieldnames = [
            "request_id",
            "timestamp",
            "task_type",
            "input_tokens",
            "output_tokens",
            "strategy_id",
            "model_id",
            "predicted_cost",
            "actual_cost",
            "cost_error_pct",
            "predicted_latency_p50",
            "actual_latency",
            "latency_error_pct",
            "predicted_quality_risk",
            "quality_proxy_score",
            "sla_met",
            "fallback_used",
            "why",
        ]
        
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for entry in entries:
                writer.writerow({
                    "request_id": entry.request.request_id,
                    "timestamp": entry.decision.timestamp_utc.isoformat(),
                    "task_type": entry.features.task_type.value,
                    "input_tokens": entry.features.input_token_count,
                    "output_tokens": entry.outcome.actual_output_tokens,
                    "strategy_id": entry.decision.strategy_id,
                    "model_id": entry.decision.model_id,
                    "predicted_cost": entry.decision.predicted_cost_usd,
                    "actual_cost": entry.outcome.actual_cost_usd,
                    "cost_error_pct": entry.outcome.cost_error_pct,
                    "predicted_latency_p50": entry.decision.predicted_latency_p50_ms,
                    "actual_latency": entry.outcome.actual_latency_ms,
                    "latency_error_pct": entry.outcome.latency_error_pct,
                    "predicted_quality_risk": entry.decision.predicted_quality_risk,
                    "quality_proxy_score": entry.outcome.quality_proxy_score,
                    "sla_met": entry.outcome.sla_met,
                    "fallback_used": entry.decision.fallback_used,
                    "why": entry.decision.why,
                })
        
        return len(entries)
