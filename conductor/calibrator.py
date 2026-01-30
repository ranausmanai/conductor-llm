"""
Calibrator for Conductor.

Updates prediction coefficients based on logged outcomes.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional
import statistics

from conductor.schemas import LogEntry
from conductor.coefficients import Coefficients, ModelCoefficients
from conductor.logger import LogStore


@dataclass
class CalibrationResult:
    """Result of a calibration run."""
    model_id: str
    entries_analyzed: int
    
    # Cost calibration
    cost_mae: float  # Mean absolute error
    cost_mape: float  # Mean absolute percentage error
    
    # Latency calibration
    latency_mae: float
    latency_mape: float
    actual_p50: int
    actual_p95: int
    actual_p99: int
    new_p95_multiplier: float
    new_p99_multiplier: float
    
    # Quality calibration
    actual_failure_rate: float
    predicted_avg_risk: float
    new_base_risk: float
    
    # Changes made
    base_latency_adjustment: int
    base_risk_adjustment: float


class Calibrator:
    """
    Calibrates prediction coefficients from logged outcomes.
    
    The calibration algorithm:
    1. Group log entries by model_id
    2. For each model:
       a. Compute prediction errors (predicted vs actual)
       b. Adjust base_latency to reduce systematic bias
       c. Recompute p95/p99 multipliers from actual distribution
       d. Adjust base_quality_risk from actual failure rate
    3. Save updated coefficients
    """
    
    def __init__(
        self,
        coefficients: Coefficients,
        min_entries: int = 100,
        dampening_factor: float = 0.5,
    ):
        """
        Args:
            coefficients: Current coefficients to update.
            min_entries: Minimum entries required to calibrate a model.
            dampening_factor: How much to apply adjustments (0-1).
                             Lower = more conservative updates.
        """
        self.coefficients = coefficients
        self.min_entries = min_entries
        self.dampening_factor = dampening_factor
    
    def calibrate(
        self,
        entries: list[LogEntry],
    ) -> list[CalibrationResult]:
        """
        Calibrate coefficients from log entries.
        
        Args:
            entries: Log entries to learn from.
        
        Returns:
            List of calibration results per model.
        """
        # Group by model
        by_model: dict[str, list[LogEntry]] = defaultdict(list)
        for entry in entries:
            model_id = entry.decision.model_id
            by_model[model_id].append(entry)
        
        results = []
        
        for model_id, model_entries in by_model.items():
            if len(model_entries) < self.min_entries:
                continue
            
            result = self._calibrate_model(model_id, model_entries)
            results.append(result)
        
        # Update coefficients version
        self.coefficients.version = datetime.now(UTC).isoformat()
        self.coefficients.updated_at = datetime.now(UTC)
        
        return results
    
    def _calibrate_model(
        self,
        model_id: str,
        entries: list[LogEntry],
    ) -> CalibrationResult:
        """Calibrate coefficients for a single model."""
        
        # Get current coefficients
        current = self.coefficients.get_model_coeffs(model_id)
        
        # =======================================================================
        # COST CALIBRATION
        # =======================================================================
        cost_errors = []
        cost_pct_errors = []
        
        for entry in entries:
            pred = entry.decision.predicted_cost_usd
            actual = entry.outcome.actual_cost_usd
            
            if pred > 0:
                error = actual - pred
                pct_error = abs(error / pred * 100)
                cost_errors.append(abs(error))
                cost_pct_errors.append(pct_error)
        
        cost_mae = statistics.mean(cost_errors) if cost_errors else 0
        cost_mape = statistics.mean(cost_pct_errors) if cost_pct_errors else 0
        
        # Cost is deterministic from tokens, so we don't adjust coefficients
        # But we could flag if pricing changed significantly
        
        # =======================================================================
        # LATENCY CALIBRATION
        # =======================================================================
        latencies = [e.outcome.actual_latency_ms for e in entries]
        latencies.sort()
        
        n = len(latencies)
        actual_p50 = latencies[int(n * 0.5)]
        actual_p95 = latencies[int(n * 0.95)]
        actual_p99 = latencies[min(int(n * 0.99), n - 1)]
        
        # Compute prediction errors
        latency_errors = []
        latency_pct_errors = []
        
        for entry in entries:
            pred = entry.decision.predicted_latency_p50_ms
            actual = entry.outcome.actual_latency_ms
            
            if pred > 0:
                error = actual - pred
                pct_error = abs(error / pred * 100)
                latency_errors.append(abs(error))
                latency_pct_errors.append(pct_error)
        
        latency_mae = statistics.mean(latency_errors) if latency_errors else 0
        latency_mape = statistics.mean(latency_pct_errors) if latency_pct_errors else 0
        
        # Compute systematic bias in predictions
        pred_p50_values = [e.decision.predicted_latency_p50_ms for e in entries]
        avg_pred_p50 = statistics.mean(pred_p50_values)
        
        # Adjust base_latency to reduce bias
        bias = actual_p50 - avg_pred_p50
        base_latency_adjustment = 0
        
        if abs(bias) > 50:  # Only adjust if bias > 50ms
            adjustment = int(bias * self.dampening_factor)
            new_base_latency = current.base_latency_ms + adjustment
            
            # Ensure non-negative
            if new_base_latency > 0:
                base_latency_adjustment = adjustment
                
                # Update coefficient
                if model_id in self.coefficients.models:
                    self.coefficients.models[model_id] = ModelCoefficients(
                        cost_per_input_1k=current.cost_per_input_1k,
                        cost_per_output_1k=current.cost_per_output_1k,
                        base_latency_ms=new_base_latency,
                        latency_per_input_token_ms=current.latency_per_input_token_ms,
                        latency_per_output_token_ms=current.latency_per_output_token_ms,
                        p95_multiplier=current.p95_multiplier,
                        p99_multiplier=current.p99_multiplier,
                        base_quality_risk=current.base_quality_risk,
                    )
        
        # Recompute multipliers from actual distribution
        new_p95_mult = actual_p95 / actual_p50 if actual_p50 > 0 else current.p95_multiplier
        new_p99_mult = actual_p99 / actual_p50 if actual_p50 > 0 else current.p99_multiplier
        
        # Apply dampening
        new_p95_mult = current.p95_multiplier + (new_p95_mult - current.p95_multiplier) * self.dampening_factor
        new_p99_mult = current.p99_multiplier + (new_p99_mult - current.p99_multiplier) * self.dampening_factor
        
        # Update multipliers
        if model_id in self.coefficients.models:
            mc = self.coefficients.models[model_id]
            self.coefficients.models[model_id] = ModelCoefficients(
                cost_per_input_1k=mc.cost_per_input_1k,
                cost_per_output_1k=mc.cost_per_output_1k,
                base_latency_ms=mc.base_latency_ms,
                latency_per_input_token_ms=mc.latency_per_input_token_ms,
                latency_per_output_token_ms=mc.latency_per_output_token_ms,
                p95_multiplier=new_p95_mult,
                p99_multiplier=new_p99_mult,
                base_quality_risk=mc.base_quality_risk,
            )
        
        # =======================================================================
        # QUALITY CALIBRATION
        # =======================================================================
        # Count actual failures (quality_proxy_score < 0.8)
        failures = sum(1 for e in entries if e.outcome.quality_proxy_score < 0.8)
        actual_failure_rate = failures / len(entries)
        
        # Average predicted risk
        predicted_risks = [e.decision.predicted_quality_risk for e in entries]
        predicted_avg_risk = statistics.mean(predicted_risks)
        
        # Adjust base risk
        risk_diff = actual_failure_rate - predicted_avg_risk
        base_risk_adjustment = 0.0
        
        if abs(risk_diff) > 0.03:  # Only adjust if difference > 3%
            adjustment = risk_diff * self.dampening_factor
            new_base_risk = current.base_quality_risk + adjustment
            
            # Clamp to valid range
            new_base_risk = max(0.01, min(0.5, new_base_risk))
            base_risk_adjustment = adjustment
            
            # Update coefficient
            if model_id in self.coefficients.models:
                mc = self.coefficients.models[model_id]
                self.coefficients.models[model_id] = ModelCoefficients(
                    cost_per_input_1k=mc.cost_per_input_1k,
                    cost_per_output_1k=mc.cost_per_output_1k,
                    base_latency_ms=mc.base_latency_ms,
                    latency_per_input_token_ms=mc.latency_per_input_token_ms,
                    latency_per_output_token_ms=mc.latency_per_output_token_ms,
                    p95_multiplier=mc.p95_multiplier,
                    p99_multiplier=mc.p99_multiplier,
                    base_quality_risk=new_base_risk,
                )
        
        return CalibrationResult(
            model_id=model_id,
            entries_analyzed=len(entries),
            cost_mae=cost_mae,
            cost_mape=cost_mape,
            latency_mae=latency_mae,
            latency_mape=latency_mape,
            actual_p50=actual_p50,
            actual_p95=actual_p95,
            actual_p99=actual_p99,
            new_p95_multiplier=new_p95_mult,
            new_p99_multiplier=new_p99_mult,
            actual_failure_rate=actual_failure_rate,
            predicted_avg_risk=predicted_avg_risk,
            new_base_risk=self.coefficients.models.get(
                model_id, current
            ).base_quality_risk,
            base_latency_adjustment=base_latency_adjustment,
            base_risk_adjustment=base_risk_adjustment,
        )
    
    def save_coefficients(self, path: Path) -> None:
        """Save calibrated coefficients to file."""
        self.coefficients.save(path)
    
    def print_report(self, results: list[CalibrationResult]) -> str:
        """Generate a human-readable calibration report."""
        lines = []
        lines.append("=" * 70)
        lines.append("CALIBRATION REPORT")
        lines.append(f"Timestamp: {datetime.now(UTC).isoformat()}")
        lines.append("=" * 70)
        
        for result in results:
            lines.append("")
            lines.append(f"Model: {result.model_id}")
            lines.append(f"  Entries analyzed: {result.entries_analyzed}")
            lines.append("")
            lines.append("  COST:")
            lines.append(f"    MAE: ${result.cost_mae:.6f}")
            lines.append(f"    MAPE: {result.cost_mape:.1f}%")
            lines.append("")
            lines.append("  LATENCY:")
            lines.append(f"    MAE: {result.latency_mae:.0f}ms")
            lines.append(f"    MAPE: {result.latency_mape:.1f}%")
            lines.append(f"    Actual P50/P95/P99: {result.actual_p50}/{result.actual_p95}/{result.actual_p99}ms")
            lines.append(f"    New P95 multiplier: {result.new_p95_multiplier:.2f}")
            lines.append(f"    New P99 multiplier: {result.new_p99_multiplier:.2f}")
            lines.append(f"    Base latency adjustment: {result.base_latency_adjustment:+d}ms")
            lines.append("")
            lines.append("  QUALITY:")
            lines.append(f"    Actual failure rate: {result.actual_failure_rate:.1%}")
            lines.append(f"    Predicted avg risk: {result.predicted_avg_risk:.1%}")
            lines.append(f"    New base risk: {result.new_base_risk:.2f}")
            lines.append(f"    Base risk adjustment: {result.base_risk_adjustment:+.3f}")
            lines.append("-" * 70)
        
        return "\n".join(lines)
