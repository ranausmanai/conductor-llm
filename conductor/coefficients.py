"""
Coefficients for prediction formulas.

These are the tunable parameters that control cost, latency, and quality predictions.
They can be calibrated from logged outcomes.
"""

from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Optional
import json
from pathlib import Path


@dataclass
class ModelCoefficients:
    """Coefficients for a specific model."""
    
    # Cost (per 1K tokens, USD)
    cost_per_input_1k: float
    cost_per_output_1k: float
    
    # Latency
    base_latency_ms: int
    latency_per_input_token_ms: float
    latency_per_output_token_ms: float
    p95_multiplier: float
    p99_multiplier: float
    
    # Quality risk (base probability of failure)
    base_quality_risk: float


@dataclass
class TaskCoefficients:
    """Coefficients for task types."""
    
    # Base output tokens by task type
    base_output_tokens: int
    
    # Quality risk adjustment by task type
    task_quality_risk: float


@dataclass
class Coefficients:
    """
    All coefficients for the prediction engine.
    
    This is the central configuration for how Conductor makes predictions.
    """
    
    # Version tracking
    version: str = "default"
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    # ==========================================================================
    # MODEL COEFFICIENTS
    # ==========================================================================
    models: dict[str, ModelCoefficients] = field(default_factory=lambda: {
        
        # OpenAI models
        "gpt-4o-mini": ModelCoefficients(
            cost_per_input_1k=0.00015,
            cost_per_output_1k=0.0006,
            base_latency_ms=80,  # Reduced from 150ms
            latency_per_input_token_ms=0.01,  # Reduced from 0.02
            latency_per_output_token_ms=4.0,  # Reduced from 8.0
            p95_multiplier=1.4,  # Reduced from 1.8
            p99_multiplier=1.8,  # Reduced from 2.5
            base_quality_risk=0.08,  # Reduced from 0.15
        ),

        "gpt-4o": ModelCoefficients(
            cost_per_input_1k=0.0025,
            cost_per_output_1k=0.01,
            base_latency_ms=150,  # Reduced from 300ms
            latency_per_input_token_ms=0.015,  # Reduced from 0.03
            latency_per_output_token_ms=8.0,  # Reduced from 15.0
            p95_multiplier=1.5,  # Reduced from 2.0
            p99_multiplier=2.0,  # Reduced from 3.0
            base_quality_risk=0.03,  # Reduced from 0.05
        ),
        
        # Anthropic models
        "claude-3-haiku-20240307": ModelCoefficients(
            cost_per_input_1k=0.00025,
            cost_per_output_1k=0.00125,
            base_latency_ms=100,  # Reduced from 200ms
            latency_per_input_token_ms=0.01,  # Reduced from 0.02
            latency_per_output_token_ms=5.0,  # Reduced from 10.0
            p95_multiplier=1.4,  # Reduced from 1.7
            p99_multiplier=1.8,  # Reduced from 2.3
            base_quality_risk=0.07,  # Reduced from 0.12
        ),

        "claude-3-5-sonnet-20241022": ModelCoefficients(
            cost_per_input_1k=0.003,
            cost_per_output_1k=0.015,
            base_latency_ms=200,  # Reduced from 400ms
            latency_per_input_token_ms=0.02,  # Reduced from 0.04
            latency_per_output_token_ms=10.0,  # Reduced from 20.0
            p95_multiplier=1.5,  # Reduced from 2.2
            p99_multiplier=2.0,  # Reduced from 3.5
            base_quality_risk=0.02,  # Reduced from 0.04
        ),
        
        # Self-hosted (example)
        "llama-3-70b": ModelCoefficients(
            cost_per_input_1k=0.0005,
            cost_per_output_1k=0.0005,
            base_latency_ms=100,
            latency_per_input_token_ms=0.01,
            latency_per_output_token_ms=25.0,
            p95_multiplier=1.5,
            p99_multiplier=2.0,
            base_quality_risk=0.20,
        ),
    })
    
    # ==========================================================================
    # TASK COEFFICIENTS
    # ==========================================================================
    tasks: dict[str, TaskCoefficients] = field(default_factory=lambda: {
        "classify": TaskCoefficients(
            base_output_tokens=10,
            task_quality_risk=0.01,  # Reduced from 0.02
        ),
        "extract_json": TaskCoefficients(
            base_output_tokens=150,
            task_quality_risk=0.04,  # Reduced from 0.08
        ),
        "summarize": TaskCoefficients(
            base_output_tokens=200,  # Will be adjusted by input length
            task_quality_risk=0.02,  # Reduced from 0.05
        ),
        "rewrite": TaskCoefficients(
            base_output_tokens=0,  # Uses input length * 1.1
            task_quality_risk=0.015,  # Reduced from 0.03
        ),
        "generate_long": TaskCoefficients(
            base_output_tokens=0,  # Uses max_tokens * 0.8
            task_quality_risk=0.04,  # Reduced from 0.10
        ),
        "chat": TaskCoefficients(
            base_output_tokens=200,
            task_quality_risk=0.02,  # Reduced from 0.05
        ),
    })
    
    # ==========================================================================
    # FIXED OVERHEADS
    # ==========================================================================
    verifier_fixed_cost_usd: float = 0.002
    verifier_latency_ms: int = 800
    verifier_risk_reduction: float = 0.10  # Reduces quality_risk by this amount
    
    tool_overhead_cost_usd: float = 0.0005
    tool_latency_ms: int = 200
    
    # ==========================================================================
    # COMPLEXITY COEFFICIENTS
    # ==========================================================================
    complexity_risk_coeff: float = 0.08  # Reduced from 0.15
    temperature_risk_penalty: float = 0.02  # Reduced from 0.05
    long_input_risk: float = 0.02  # Reduced from 0.05
    long_input_threshold: int = 4000
    
    def get_model_coeffs(self, model_id: str) -> ModelCoefficients:
        """Get coefficients for a model, with fallback to gpt-4o-mini."""
        return self.models.get(model_id, self.models["gpt-4o-mini"])
    
    def get_task_coeffs(self, task_type: str) -> TaskCoefficients:
        """Get coefficients for a task type, with fallback to chat."""
        return self.tasks.get(task_type, self.tasks["chat"])
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "updated_at": self.updated_at.isoformat(),
            "models": {
                model_id: {
                    "cost_per_input_1k": mc.cost_per_input_1k,
                    "cost_per_output_1k": mc.cost_per_output_1k,
                    "base_latency_ms": mc.base_latency_ms,
                    "latency_per_input_token_ms": mc.latency_per_input_token_ms,
                    "latency_per_output_token_ms": mc.latency_per_output_token_ms,
                    "p95_multiplier": mc.p95_multiplier,
                    "p99_multiplier": mc.p99_multiplier,
                    "base_quality_risk": mc.base_quality_risk,
                }
                for model_id, mc in self.models.items()
            },
            "tasks": {
                task_type: {
                    "base_output_tokens": tc.base_output_tokens,
                    "task_quality_risk": tc.task_quality_risk,
                }
                for task_type, tc in self.tasks.items()
            },
            "verifier_fixed_cost_usd": self.verifier_fixed_cost_usd,
            "verifier_latency_ms": self.verifier_latency_ms,
            "verifier_risk_reduction": self.verifier_risk_reduction,
            "tool_overhead_cost_usd": self.tool_overhead_cost_usd,
            "tool_latency_ms": self.tool_latency_ms,
            "complexity_risk_coeff": self.complexity_risk_coeff,
            "temperature_risk_penalty": self.temperature_risk_penalty,
            "long_input_risk": self.long_input_risk,
            "long_input_threshold": self.long_input_threshold,
        }
    
    def save(self, path: Path) -> None:
        """Save coefficients to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Coefficients":
        """Create from dictionary."""
        coeffs = cls()
        coeffs.version = data.get("version", "default")
        
        if "updated_at" in data:
            coeffs.updated_at = datetime.fromisoformat(data["updated_at"])
        
        if "models" in data:
            coeffs.models = {
                model_id: ModelCoefficients(**mc_data)
                for model_id, mc_data in data["models"].items()
            }
        
        if "tasks" in data:
            coeffs.tasks = {
                task_type: TaskCoefficients(**tc_data)
                for task_type, tc_data in data["tasks"].items()
            }
        
        # Load fixed overheads
        for key in ["verifier_fixed_cost_usd", "verifier_latency_ms", 
                    "verifier_risk_reduction", "tool_overhead_cost_usd",
                    "tool_latency_ms", "complexity_risk_coeff",
                    "temperature_risk_penalty", "long_input_risk",
                    "long_input_threshold"]:
            if key in data:
                setattr(coeffs, key, data[key])
        
        return coeffs


def load_coefficients(path: Optional[Path] = None) -> Coefficients:
    """
    Load coefficients from file or return defaults.
    
    Args:
        path: Path to JSON coefficients file. If None, uses defaults.
    
    Returns:
        Coefficients instance.
    """
    if path is None:
        return Coefficients()
    
    if not path.exists():
        return Coefficients()
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    return Coefficients.from_dict(data)


# Default coefficients instance
DEFAULT_COEFFICIENTS = Coefficients()
