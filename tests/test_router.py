"""Tests for the Router component."""

import pytest
from conductor.router import Router
from conductor.classifier import Classifier
from conductor.schemas import Request, TaskType, LatencyPercentile
from conductor.strategies import STRATEGIES


class TestRouter:
    """Test suite for Router."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.router = Router()
        self.classifier = Classifier()
    
    def _classify_and_route(self, request: Request):
        """Helper to classify and route a request."""
        features = self.classifier.classify(request)
        return self.router.route(features)
    
    def test_basic_routing(self):
        """Test basic routing returns a decision."""
        request = Request(
            prompt="What is 2 + 2?",
            task_type=TaskType.CLASSIFY,
            max_latency_ms=2000,
            min_quality_score=0.8,
        )
        
        decision = self._classify_and_route(request)
        
        assert decision is not None
        assert decision.strategy_id in STRATEGIES
        assert decision.why is not None
        assert len(decision.why) > 0
    
    def test_routing_prefers_cheaper(self):
        """Test that routing prefers cheaper strategies when constraints allow."""
        # Relaxed constraints should allow cheap strategy
        request = Request(
            prompt="Simple question?",
            task_type=TaskType.CLASSIFY,
            max_latency_ms=5000,
            min_quality_score=0.7,
        )
        
        decision = self._classify_and_route(request)
        
        # Should pick one of the cheaper strategies
        cheap_strategies = ["turbo-cheap", "batch-cheap"]
        assert decision.strategy_id in cheap_strategies or decision.predicted_cost_usd < 0.01
    
    def test_routing_respects_latency_sla(self):
        """Test that routing respects latency SLA."""
        # Tight latency should exclude slow strategies
        request = Request(
            prompt="Quick!",
            task_type=TaskType.CLASSIFY,
            max_latency_ms=300,
            latency_percentile=LatencyPercentile.P95,
            min_quality_score=0.7,
        )
        
        decision = self._classify_and_route(request)
        
        # Either meets SLA or uses fallback
        if not decision.fallback_used:
            assert decision.predicted_latency_p95_ms <= 300
    
    def test_routing_respects_quality_threshold(self):
        """Test that routing respects quality threshold."""
        # High quality requirement
        request = Request(
            prompt="Important task",
            task_type=TaskType.SUMMARIZE,
            max_latency_ms=10000,
            min_quality_score=0.95,
        )
        
        decision = self._classify_and_route(request)
        
        # Either meets quality or uses fallback
        if not decision.fallback_used:
            expected_quality = 1 - decision.predicted_quality_risk
            assert expected_quality >= 0.95
    
    def test_routing_filters_by_task_type(self):
        """Test that routing only considers strategies supporting the task type."""
        # GENERATE_LONG is only supported by some strategies
        request = Request(
            prompt="Write a long article about...",
            task_type=TaskType.GENERATE_LONG,
            max_latency_ms=30000,
            min_quality_score=0.8,
        )
        
        decision = self._classify_and_route(request)
        
        # Check that chosen strategy supports this task type
        strategy = STRATEGIES[decision.strategy_id]
        assert TaskType.GENERATE_LONG in strategy.supported_task_types
    
    def test_routing_respects_require_tools(self):
        """Test that require_tools constraint is respected."""
        request = Request(
            prompt="Extract JSON",
            task_type=TaskType.EXTRACT_JSON,
            max_latency_ms=5000,
            min_quality_score=0.8,
            require_tools=True,
        )
        
        decision = self._classify_and_route(request)
        
        # Either uses tools or fallback
        if not decision.fallback_used:
            assert decision.tools_enabled is True
    
    def test_routing_respects_require_verifier(self):
        """Test that require_verifier constraint is respected."""
        request = Request(
            prompt="Critical task",
            task_type=TaskType.SUMMARIZE,
            max_latency_ms=10000,
            min_quality_score=0.8,
            require_verifier=True,
        )
        
        decision = self._classify_and_route(request)
        
        # Either uses verifier or fallback
        if not decision.fallback_used:
            assert decision.verifier_enabled is True
    
    def test_routing_respects_no_batching(self):
        """Test that allow_batching=False is respected."""
        request = Request(
            prompt="No batching please",
            task_type=TaskType.CLASSIFY,
            max_latency_ms=5000,
            min_quality_score=0.8,
            allow_batching=False,
        )
        
        decision = self._classify_and_route(request)
        
        # Should not use batching strategy
        assert decision.batching_delay_ms == 0
    
    def test_routing_respects_cost_budget(self):
        """Test that max_cost_usd is respected."""
        request = Request(
            prompt="Budget constrained",
            task_type=TaskType.CHAT,
            max_latency_ms=5000,
            min_quality_score=0.7,
            max_cost_usd=0.0001,  # Very tight budget
        )
        
        decision = self._classify_and_route(request)
        
        # Either under budget or fallback
        if not decision.fallback_used:
            assert decision.predicted_cost_usd <= 0.0001
    
    def test_fallback_when_no_strategy_meets_constraints(self):
        """Test fallback behavior when constraints can't be met."""
        # Impossible constraints
        request = Request(
            prompt="Impossible",
            task_type=TaskType.GENERATE_LONG,
            max_latency_ms=10,  # Way too fast
            min_quality_score=0.99,
        )
        
        decision = self._classify_and_route(request)
        
        # Should use fallback
        assert decision.fallback_used is True
        assert "FALLBACK" in decision.why
    
    def test_filter_reasons_populated(self):
        """Test that filter_reasons explains why strategies were rejected."""
        request = Request(
            prompt="Test",
            task_type=TaskType.GENERATE_LONG,  # Limited strategy support
            max_latency_ms=500,
            min_quality_score=0.95,
        )
        
        decision = self._classify_and_route(request)
        
        # Should have filter reasons for rejected strategies
        assert len(decision.filter_reasons) > 0
        
        # Check that reasons are descriptive
        for strategy_id, reason in decision.filter_reasons.items():
            assert len(reason) > 0
    
    def test_why_explanation_is_informative(self):
        """Test that the 'why' explanation is informative."""
        request = Request(
            prompt="Test",
            task_type=TaskType.CLASSIFY,
            max_latency_ms=2000,
            min_quality_score=0.8,
        )
        
        decision = self._classify_and_route(request)
        
        # Why should mention cost, strategies, and constraints
        assert "cost" in decision.why.lower() or "FALLBACK" in decision.why
        
        # Should not be empty or generic
        assert len(decision.why) > 20
    
    def test_strategies_considered_count(self):
        """Test that strategies_considered is accurate."""
        request = Request(
            prompt="Test",
            task_type=TaskType.CLASSIFY,
            max_latency_ms=2000,
            min_quality_score=0.8,
        )
        
        decision = self._classify_and_route(request)
        
        # Should consider all strategies
        assert decision.strategies_considered == len(STRATEGIES)
        
        # Filtered out should be less than or equal to considered
        assert decision.strategies_filtered_out <= decision.strategies_considered
    
    def test_long_input_filtered(self):
        """Test that strategies with max_input_tokens are respected."""
        # Very long prompt
        long_prompt = "This is a test. " * 2000  # ~8000 tokens
        
        request = Request(
            prompt=long_prompt,
            task_type=TaskType.SUMMARIZE,
            max_latency_ms=30000,
            min_quality_score=0.8,
        )
        
        decision = self._classify_and_route(request)
        
        # Should not pick turbo-cheap (max 4000 tokens)
        if decision.strategy_id == "turbo-cheap":
            # If it did pick turbo-cheap, check that input was under limit
            features = self.classifier.classify(request)
            assert features.input_token_count <= 4000
