"""
Router for Conductor.

Makes the final routing decision by filtering strategies and selecting
the cheapest one that meets constraints.
"""

from datetime import datetime, UTC
from typing import Optional
import uuid

from conductor.schemas import (
    RequestFeatures,
    Strategy,
    Prediction,
    Decision,
    LatencyPercentile,
)
from conductor.strategies import get_all_strategies
from conductor.predictor import Predictor
from conductor.coefficients import Coefficients, DEFAULT_COEFFICIENTS


class Router:
    """
    Routes requests to the optimal strategy.
    
    The routing algorithm:
    1. Filter strategies by capability (task type, token limits)
    2. Filter by SLA constraints (latency, quality, cost)
    3. Select cheapest surviving strategy
    4. If nothing survives, fallback to highest quality
    """
    
    def __init__(
        self,
        coefficients: Coefficients = None,
        strategies: dict[str, Strategy] = None,
        policy_version: str = "v1.0.0",
    ):
        self.coefficients = coefficients or DEFAULT_COEFFICIENTS
        self.predictor = Predictor(self.coefficients)
        self.strategies = strategies or get_all_strategies()
        self.policy_version = policy_version
    
    def route(self, features: RequestFeatures) -> Decision:
        """
        Route a request to the optimal strategy.
        
        Args:
            features: Extracted request features.
        
        Returns:
            Decision with chosen strategy and explanation.
        """
        all_strategies = list(self.strategies.values())
        filter_reasons: dict[str, str] = {}
        
        # =======================================================================
        # STEP 1: Filter by capability
        # =======================================================================
        applicable = []
        for strategy in all_strategies:
            reason = self._check_capability(features, strategy)
            if reason:
                filter_reasons[strategy.strategy_id] = reason
            else:
                applicable.append(strategy)
        
        # =======================================================================
        # STEP 2: Compute predictions for applicable strategies
        # =======================================================================
        predictions = self.predictor.predict_all(features, applicable)
        
        # =======================================================================
        # STEP 3: Filter by SLA constraints
        # =======================================================================
        meeting_sla = []
        for strategy, pred in predictions:
            reason = self._check_sla(features, strategy, pred)
            if reason:
                filter_reasons[strategy.strategy_id] = reason
            else:
                meeting_sla.append((strategy, pred))
        
        # =======================================================================
        # STEP 4: Select best strategy
        # =======================================================================
        fallback_used = False
        
        if meeting_sla:
            # Sort by cost ascending
            meeting_sla.sort(key=lambda x: x[1].cost_usd)
            
            # Tiebreak by quality risk, then latency
            min_cost = meeting_sla[0][1].cost_usd
            same_cost = [x for x in meeting_sla if x[1].cost_usd == min_cost]
            
            if len(same_cost) > 1:
                same_cost.sort(key=lambda x: (x[1].quality_risk, x[1].latency_p50_ms))
            
            chosen_strategy, chosen_pred = same_cost[0]
            
            why = (
                f"Lowest cost (${chosen_pred.cost_usd:.4f}) among "
                f"{len(meeting_sla)} strategies meeting "
                f"{features.latency_percentile.value.upper()}<{features.max_latency_ms}ms "
                f"and quality>{features.min_quality_score:.0%}"
            )
        else:
            # FALLBACK: No strategy meets all constraints
            fallback_used = True
            
            if predictions:
                # Choose strategy with lowest quality risk
                predictions.sort(key=lambda x: x[1].quality_risk)
                chosen_strategy, chosen_pred = predictions[0]
                
                why = (
                    f"FALLBACK: No strategy met all constraints. "
                    f"Using '{chosen_strategy.strategy_id}' for best quality "
                    f"(risk={chosen_pred.quality_risk:.2f}). "
                    f"Violated: check filter_reasons for details."
                )
            else:
                # No applicable strategies at all - this shouldn't happen
                # but handle it gracefully
                raise ValueError(
                    f"No applicable strategies for request. "
                    f"Task type: {features.task_type}, "
                    f"Input tokens: {features.input_token_count}"
                )
        
        # =======================================================================
        # STEP 5: Build decision
        # =======================================================================
        return Decision(
            decision_id=str(uuid.uuid4()),
            request_id=features.request_id,
            timestamp_utc=datetime.now(UTC),
            
            strategy_id=chosen_strategy.strategy_id,
            model_id=chosen_strategy.model_id,
            temperature=chosen_strategy.temperature,
            max_tokens=chosen_strategy.max_tokens,
            batching_delay_ms=chosen_strategy.batching_delay_ms,
            verifier_enabled=chosen_strategy.verifier_enabled,
            tools_enabled=chosen_strategy.tools_enabled,
            
            predicted_cost_usd=chosen_pred.cost_usd,
            predicted_latency_p50_ms=chosen_pred.latency_p50_ms,
            predicted_latency_p95_ms=chosen_pred.latency_p95_ms,
            predicted_latency_p99_ms=chosen_pred.latency_p99_ms,
            predicted_quality_risk=chosen_pred.quality_risk,
            
            why=why,
            strategies_considered=len(all_strategies),
            strategies_filtered_out=len(all_strategies) - len(meeting_sla),
            fallback_used=fallback_used,
            filter_reasons=filter_reasons,
            
            policy_version=self.policy_version,
            coefficient_version=self.coefficients.version,
        )
    
    def _check_capability(
        self,
        features: RequestFeatures,
        strategy: Strategy,
    ) -> Optional[str]:
        """
        Check if strategy can handle the request.
        
        Returns:
            None if capable, or reason string if not.
        """
        # Check task type support
        if features.task_type not in strategy.supported_task_types:
            return f"task_type '{features.task_type.value}' not supported"
        
        # Check input token limits
        if features.input_token_count < strategy.min_input_tokens:
            return f"input too short ({features.input_token_count} < {strategy.min_input_tokens})"
        
        if features.input_token_count > strategy.max_input_tokens:
            return f"input too long ({features.input_token_count} > {strategy.max_input_tokens})"
        
        # Check client requirements
        if features.require_tools and not strategy.tools_enabled:
            return "tools required but not enabled"
        
        if features.require_verifier and not strategy.verifier_enabled:
            return "verifier required but not enabled"
        
        if not features.allow_batching and strategy.batching_delay_ms > 0:
            return "batching not allowed but strategy uses batching"
        
        return None
    
    def _check_sla(
        self,
        features: RequestFeatures,
        strategy: Strategy,
        pred: Prediction,
    ) -> Optional[str]:
        """
        Check if prediction meets SLA constraints.
        
        Returns:
            None if meets SLA, or reason string if not.
        """
        # Check latency SLA
        sla_latency = features.max_latency_ms
        
        if features.latency_percentile == LatencyPercentile.P50:
            if pred.latency_p50_ms > sla_latency:
                return f"P50 {pred.latency_p50_ms}ms > SLA {sla_latency}ms"
        
        elif features.latency_percentile == LatencyPercentile.P95:
            if pred.latency_p95_ms > sla_latency:
                return f"P95 {pred.latency_p95_ms}ms > SLA {sla_latency}ms"
        
        elif features.latency_percentile == LatencyPercentile.P99:
            if pred.latency_p99_ms > sla_latency:
                return f"P99 {pred.latency_p99_ms}ms > SLA {sla_latency}ms"
        
        # Check quality threshold
        expected_quality = 1.0 - pred.quality_risk
        if expected_quality < features.min_quality_score:
            return f"quality {expected_quality:.2f} < threshold {features.min_quality_score}"
        
        # Check cost budget
        if features.max_cost_usd is not None:
            if pred.cost_usd > features.max_cost_usd:
                return f"cost ${pred.cost_usd:.4f} > budget ${features.max_cost_usd:.4f}"
        
        return None


def route_request(features: RequestFeatures, **kwargs) -> Decision:
    """
    Convenience function to route a request.
    
    Args:
        features: Extracted request features.
        **kwargs: Passed to Router constructor.
    
    Returns:
        Decision.
    """
    router = Router(**kwargs)
    return router.route(features)
