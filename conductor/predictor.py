"""
Predictor for Conductor.

Computes cost, latency, and quality predictions for each strategy.
Uses deterministic formulas with tunable coefficients.
"""

from conductor.schemas import (
    RequestFeatures,
    Strategy,
    Prediction,
    TaskType,
)
from conductor.coefficients import Coefficients, DEFAULT_COEFFICIENTS


class Predictor:
    """
    Predicts cost, latency, and quality risk for request/strategy combinations.
    
    All predictions use deterministic formulas (no ML) with coefficients
    that can be calibrated from logged outcomes.
    """
    
    def __init__(self, coefficients: Coefficients = None):
        self.coefficients = coefficients or DEFAULT_COEFFICIENTS
    
    def predict(
        self,
        features: RequestFeatures,
        strategy: Strategy,
    ) -> Prediction:
        """
        Predict outcomes for a request/strategy combination.
        
        Args:
            features: Extracted request features.
            strategy: The strategy to evaluate.
        
        Returns:
            Prediction with cost, latency, and quality estimates.
        """
        # Get model coefficients
        model_coeffs = self.coefficients.get_model_coeffs(strategy.model_id)
        
        # Adjust output tokens for strategy's max_tokens limit
        estimated_output = min(
            features.estimated_output_tokens,
            strategy.max_tokens
        )
        
        # Predict cost
        cost_usd = self._predict_cost(
            features.input_token_count,
            estimated_output,
            strategy,
            model_coeffs,
        )
        
        # Predict latency
        latency_p50, latency_p95, latency_p99 = self._predict_latency(
            features.input_token_count,
            estimated_output,
            strategy,
            model_coeffs,
        )
        
        # Predict quality risk
        quality_risk = self._predict_quality_risk(
            features,
            strategy,
            model_coeffs,
        )
        
        return Prediction(
            strategy_id=strategy.strategy_id,
            cost_usd=cost_usd,
            latency_p50_ms=latency_p50,
            latency_p95_ms=latency_p95,
            latency_p99_ms=latency_p99,
            quality_risk=quality_risk,
        )
    
    def _predict_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        strategy: Strategy,
        model_coeffs,
    ) -> float:
        """
        Predict cost in USD.
        
        Formula:
            cost = (input_tokens * cost_per_input) + 
                   (output_tokens * cost_per_output) +
                   verifier_cost + tool_cost
        """
        # Base cost from tokens
        input_cost = (input_tokens / 1000) * model_coeffs.cost_per_input_1k
        output_cost = (output_tokens / 1000) * model_coeffs.cost_per_output_1k
        
        cost = input_cost + output_cost
        
        # Add verifier cost if enabled
        if strategy.verifier_enabled:
            cost += self.coefficients.verifier_fixed_cost_usd
        
        # Add tool overhead if enabled
        if strategy.tools_enabled:
            cost += self.coefficients.tool_overhead_cost_usd
        
        return round(cost, 6)
    
    def _predict_latency(
        self,
        input_tokens: int,
        output_tokens: int,
        strategy: Strategy,
        model_coeffs,
    ) -> tuple[int, int, int]:
        """
        Predict latency at P50, P95, and P99.
        
        Formula:
            p50 = base_latency + 
                  (input_tokens * latency_per_input) +
                  (output_tokens * latency_per_output) +
                  batching_delay + verifier_latency + tool_latency
            p95 = p50 * p95_multiplier
            p99 = p50 * p99_multiplier
        """
        # Base latency
        p50 = model_coeffs.base_latency_ms
        
        # Add token processing time
        p50 += input_tokens * model_coeffs.latency_per_input_token_ms
        p50 += output_tokens * model_coeffs.latency_per_output_token_ms
        
        # Add batching delay
        p50 += strategy.batching_delay_ms
        
        # Add verifier latency if enabled
        if strategy.verifier_enabled:
            p50 += self.coefficients.verifier_latency_ms
        
        # Add tool latency if enabled
        if strategy.tools_enabled:
            p50 += self.coefficients.tool_latency_ms
        
        # Compute percentiles
        p95 = int(p50 * model_coeffs.p95_multiplier)
        p99 = int(p50 * model_coeffs.p99_multiplier)
        p50 = int(p50)
        
        return p50, p95, p99
    
    def _predict_quality_risk(
        self,
        features: RequestFeatures,
        strategy: Strategy,
        model_coeffs,
    ) -> float:
        """
        Predict quality risk (probability of failure, 0.0 to 1.0).
        
        Lower is better.
        
        Formula:
            risk = base_risk + task_risk + 
                   (complexity * complexity_coeff) +
                   temp_penalty + long_input_penalty -
                   verifier_reduction
        """
        # Start with model base risk
        risk = model_coeffs.base_quality_risk
        
        # Add task-specific risk
        task_coeffs = self.coefficients.get_task_coeffs(features.task_type.value)
        risk += task_coeffs.task_quality_risk
        
        # Add complexity risk
        risk += features.complexity_score * self.coefficients.complexity_risk_coeff
        
        # Add temperature penalty (if temp > 0.5)
        if strategy.temperature > 0.5:
            risk += self.coefficients.temperature_risk_penalty
        
        # Add long input penalty
        if features.input_token_count > self.coefficients.long_input_threshold:
            risk += self.coefficients.long_input_risk
        
        # Subtract verifier reduction
        if strategy.verifier_enabled:
            risk -= self.coefficients.verifier_risk_reduction
        
        # Task-strategy match bonus
        # Tools are better for JSON extraction
        if features.task_type == TaskType.EXTRACT_JSON and strategy.tools_enabled:
            risk -= 0.05
        
        # Clamp to valid range
        return max(0.0, min(1.0, risk))
    
    def predict_all(
        self,
        features: RequestFeatures,
        strategies: list[Strategy],
    ) -> list[tuple[Strategy, Prediction]]:
        """
        Predict outcomes for all strategies.
        
        Args:
            features: Extracted request features.
            strategies: List of strategies to evaluate.
        
        Returns:
            List of (strategy, prediction) tuples.
        """
        return [
            (strategy, self.predict(features, strategy))
            for strategy in strategies
        ]
