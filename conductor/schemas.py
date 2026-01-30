"""
Data schemas for Conductor.

All request, decision, and outcome data structures.
"""

from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from typing import Optional
import uuid


class TaskType(str, Enum):
    """Types of LLM tasks."""
    CLASSIFY = "classify"
    EXTRACT_JSON = "extract_json"
    SUMMARIZE = "summarize"
    REWRITE = "rewrite"
    GENERATE_LONG = "generate_long"
    CHAT = "chat"


class UrgencyTier(str, Enum):
    """Urgency levels for requests."""
    REALTIME = "realtime"      # <500ms
    INTERACTIVE = "interactive" # <2s
    BATCH = "batch"            # >2s OK


class LatencyPercentile(str, Enum):
    """Which percentile the latency SLA applies to."""
    P50 = "p50"
    P95 = "p95"
    P99 = "p99"


class Provider(str, Enum):
    """LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    SELF_HOSTED = "self_hosted"


class FinishReason(str, Enum):
    """LLM finish reasons."""
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALL = "tool_call"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class Request:
    """
    Incoming request from client.
    
    This is what the user sends to Conductor.
    """
    # Task specification
    prompt: str
    task_type: TaskType
    system_prompt: Optional[str] = None
    
    # Constraints (client-specified SLAs)
    max_latency_ms: int = 5000
    latency_percentile: LatencyPercentile = LatencyPercentile.P95
    min_quality_score: float = 0.8
    max_cost_usd: Optional[float] = None
    
    # Hints
    expected_output_tokens: Optional[int] = None
    allow_batching: bool = True
    require_tools: bool = False
    require_verifier: bool = False
    
    # Identity (auto-generated if not provided)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    client_id: str = "default"
    trace_id: Optional[str] = None
    timestamp_utc: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class RequestFeatures:
    """
    Extracted features from a request.
    
    This is computed by the Classifier from the raw Request.
    """
    request_id: str
    task_type: TaskType
    input_token_count: int
    estimated_output_tokens: int
    complexity_score: float  # 0.0-1.0
    urgency_tier: UrgencyTier
    has_structured_output: bool
    
    # Pass-through from request
    max_latency_ms: int
    latency_percentile: LatencyPercentile
    min_quality_score: float
    max_cost_usd: Optional[float]
    allow_batching: bool
    require_tools: bool
    require_verifier: bool


@dataclass
class Strategy:
    """
    A routing strategy configuration.
    
    Defines which model to use and with what parameters.
    """
    strategy_id: str
    model_id: str
    temperature: float
    max_tokens: int
    batching_delay_ms: int
    verifier_enabled: bool
    tools_enabled: bool
    
    # Applicability rules
    min_input_tokens: int = 0
    max_input_tokens: int = 128000
    supported_task_types: list[TaskType] = field(default_factory=list)
    
    # Provider config
    provider: Provider = Provider.OPENAI
    endpoint: Optional[str] = None
    timeout_ms: int = 30000


@dataclass
class Prediction:
    """
    Predicted outcomes for a strategy.
    
    This is what the Predictor computes for each strategy.
    """
    strategy_id: str
    cost_usd: float
    latency_p50_ms: int
    latency_p95_ms: int
    latency_p99_ms: int
    quality_risk: float  # 0.0-1.0, lower is better (probability of failure)
    
    @property
    def expected_quality(self) -> float:
        """Expected quality score (1 - risk)."""
        return 1.0 - self.quality_risk


@dataclass
class Decision:
    """
    The routing decision made by Conductor.
    
    Includes the chosen strategy, predictions, and explanation.
    """
    # Identity
    decision_id: str
    request_id: str
    timestamp_utc: datetime
    
    # Chosen strategy
    strategy_id: str
    model_id: str
    temperature: float
    max_tokens: int
    batching_delay_ms: int
    verifier_enabled: bool
    tools_enabled: bool
    
    # Predictions at decision time
    predicted_cost_usd: float
    predicted_latency_p50_ms: int
    predicted_latency_p95_ms: int
    predicted_latency_p99_ms: int
    predicted_quality_risk: float
    
    # Explanation
    why: str
    strategies_considered: int
    strategies_filtered_out: int
    fallback_used: bool
    filter_reasons: dict[str, str] = field(default_factory=dict)
    
    # Metadata
    policy_version: str = "v1.0.0"
    coefficient_version: str = "default"


@dataclass
class Outcome:
    """
    Actual execution outcome.
    
    Logged after the LLM call completes.
    """
    # Identity
    outcome_id: str
    decision_id: str
    request_id: str
    timestamp_utc: datetime
    
    # Actual measurements
    actual_cost_usd: float
    actual_latency_ms: int
    actual_input_tokens: int
    actual_output_tokens: int
    
    # Quality signals
    finish_reason: FinishReason
    http_status: int
    retry_count: int
    verifier_passed: Optional[bool]
    format_valid: bool
    
    # Computed
    quality_proxy_score: float
    sla_met: bool
    cost_error_pct: float
    latency_error_pct: float
    
    # Response content
    response_text: Optional[str] = None


@dataclass 
class LogEntry:
    """
    Complete log entry combining request, decision, and outcome.
    
    Used for calibration and replay evaluation.
    """
    request: Request
    features: RequestFeatures
    decision: Decision
    outcome: Outcome
