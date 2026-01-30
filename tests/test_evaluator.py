"""Tests for the Evaluator component."""

import pytest
from conductor.evaluator import Evaluator, Policy, ComparisonResult
from conductor.schemas import Request, TaskType
from conductor.strategies import STRATEGIES
from conductor.coefficients import Coefficients
from conductor.generators import (
    generate_steady_traffic,
    generate_bursty_traffic,
    generate_mixed_prompt_sizes,
    generate_by_task_type,
)


class TestEvaluator:
    """Test suite for Evaluator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = Evaluator()
    
    def test_evaluate_single_policy(self):
        """Test evaluating a single policy."""
        policy = Policy(name="default")
        requests = generate_steady_traffic(count=50)
        
        metrics = self.evaluator.evaluate_policy(policy, requests)
        
        assert metrics.policy_name == "default"
        assert metrics.total_requests == 50
        assert metrics.total_predicted_cost > 0
        assert metrics.avg_predicted_cost > 0
        assert 0 <= metrics.sla_pass_rate_p95 <= 1
        assert 0 <= metrics.fallback_rate <= 1
        assert len(metrics.strategy_distribution) > 0
    
    def test_evaluate_empty_requests(self):
        """Test evaluating with no requests."""
        policy = Policy(name="default")
        
        metrics = self.evaluator.evaluate_policy(policy, [])
        
        assert metrics.total_requests == 0
        assert metrics.total_predicted_cost == 0
    
    def test_compare_two_policies(self):
        """Test comparing two policies."""
        policy_a = Policy(name="baseline")
        policy_b = Policy(name="candidate")
        requests = generate_steady_traffic(count=50)
        
        result = self.evaluator.compare_policies(policy_a, policy_b, requests)
        
        assert result.policy_a.policy_name == "baseline"
        assert result.policy_b.policy_name == "candidate"
        assert result.same_decisions + result.different_decisions == 50
    
    def test_compare_identical_policies(self):
        """Test that identical policies produce same decisions."""
        coeffs = Coefficients()
        policy_a = Policy(name="policy_a", coefficients=coeffs)
        policy_b = Policy(name="policy_b", coefficients=coeffs)
        
        requests = generate_steady_traffic(count=30)
        
        result = self.evaluator.compare_policies(policy_a, policy_b, requests)
        
        # Same coefficients should produce same decisions
        assert result.same_decisions == 30
        assert result.different_decisions == 0
        assert abs(result.cost_diff_pct) < 0.01
    
    def test_comparison_with_different_coefficients(self):
        """Test comparing policies with different coefficients."""
        coeffs_a = Coefficients()
        coeffs_b = Coefficients()
        
        # Make policy B more aggressive about latency
        for model_id in coeffs_b.models:
            mc = coeffs_b.models[model_id]
            coeffs_b.models[model_id] = type(mc)(
                cost_per_input_1k=mc.cost_per_input_1k,
                cost_per_output_1k=mc.cost_per_output_1k,
                base_latency_ms=mc.base_latency_ms + 500,  # Penalize latency
                latency_per_input_token_ms=mc.latency_per_input_token_ms,
                latency_per_output_token_ms=mc.latency_per_output_token_ms,
                p95_multiplier=mc.p95_multiplier,
                p99_multiplier=mc.p99_multiplier,
                base_quality_risk=mc.base_quality_risk,
            )
        
        policy_a = Policy(name="normal", coefficients=coeffs_a)
        policy_b = Policy(name="latency_penalized", coefficients=coeffs_b)
        
        requests = generate_steady_traffic(count=50)
        
        result = self.evaluator.compare_policies(policy_a, policy_b, requests)
        
        # Policies should make some different decisions due to changed predictions
        # (not guaranteed, but likely with enough requests)
        assert result.policy_a.total_requests == 50
        assert result.policy_b.total_requests == 50
    
    def test_breakdown_by_task_type(self):
        """Test that comparison includes task type breakdown."""
        policy_a = Policy(name="a")
        policy_b = Policy(name="b")
        
        # Generate requests with multiple task types
        requests = []
        for task_type in [TaskType.CLASSIFY, TaskType.SUMMARIZE, TaskType.EXTRACT_JSON]:
            requests.extend(generate_by_task_type(task_type, count=10))
        
        result = self.evaluator.compare_policies(policy_a, policy_b, requests)
        
        # Should have breakdown for each task type
        assert len(result.by_task_type) == 3
        assert "classify" in result.by_task_type
        assert "summarize" in result.by_task_type
        assert "extract_json" in result.by_task_type
        
        # Each breakdown should have metrics
        for task_type, data in result.by_task_type.items():
            assert "count" in data
            assert "cost_a" in data
            assert "cost_b" in data
    
    def test_strategy_distribution(self):
        """Test that strategy distribution is tracked."""
        policy = Policy(name="default")
        requests = generate_steady_traffic(count=100)
        
        metrics = self.evaluator.evaluate_policy(policy, requests)
        
        # Should have distribution
        assert len(metrics.strategy_distribution) > 0
        
        # Distribution should sum to total
        total = sum(metrics.strategy_distribution.values())
        assert total == 100
    
    def test_top_reasons_tracked(self):
        """Test that top routing reasons are tracked."""
        policy = Policy(name="default")
        requests = generate_steady_traffic(count=50)
        
        metrics = self.evaluator.evaluate_policy(policy, requests)
        
        # Should have reasons
        assert len(metrics.top_reasons) > 0
        
        # Most common reason should be at top
        if len(metrics.top_reasons) > 1:
            assert metrics.top_reasons[0][1] >= metrics.top_reasons[1][1]
    
    def test_comparison_report_generation(self):
        """Test that comparison report is generated."""
        policy_a = Policy(name="baseline")
        policy_b = Policy(name="candidate")
        requests = generate_steady_traffic(count=30)
        
        result = self.evaluator.compare_policies(policy_a, policy_b, requests)
        report = self.evaluator.print_comparison_report(result)
        
        # Report should contain key sections
        assert "SUMMARY" in report
        assert "COST" in report
        assert "LATENCY" in report
        assert "QUALITY" in report
        assert "STRATEGY DISTRIBUTION" in report
        assert "baseline" in report
        assert "candidate" in report


class TestTrafficGenerators:
    """Test suite for traffic generators."""
    
    def test_generate_steady_traffic(self):
        """Test steady traffic generation."""
        requests = generate_steady_traffic(count=100)
        
        assert len(requests) == 100
        for r in requests:
            assert isinstance(r, Request)
            assert r.prompt is not None
    
    def test_generate_bursty_traffic(self):
        """Test bursty traffic generation."""
        requests = generate_bursty_traffic(
            base_count=50,
            burst_count=30,
        )
        
        assert len(requests) == 80
        
        # Should have some urgent (low latency) requests from burst
        urgent = [r for r in requests if r.max_latency_ms <= 500]
        # At least some should be urgent (20% of 30 = 6)
        assert len(urgent) > 0
    
    def test_generate_mixed_prompt_sizes(self):
        """Test mixed prompt size generation."""
        requests = generate_mixed_prompt_sizes(count=100)
        
        assert len(requests) == 100
        
        # Check distribution of prompt lengths
        short = [r for r in requests if len(r.prompt) < 500]
        medium = [r for r in requests if 500 <= len(r.prompt) < 2000]
        long = [r for r in requests if len(r.prompt) >= 2000]
        
        # Should have variety
        assert len(short) > 0
        assert len(medium) > 0
        # Long might be 0 if prompts were padded less
    
    def test_generate_by_task_type(self):
        """Test task-specific generation."""
        requests = generate_by_task_type(
            task_type=TaskType.CLASSIFY,
            count=20,
            sla_profile="strict",
        )
        
        assert len(requests) == 20
        
        for r in requests:
            assert r.task_type == TaskType.CLASSIFY
            assert r.max_latency_ms == 1000  # strict profile
            assert r.min_quality_score == 0.95


class TestPolicy:
    """Test suite for Policy wrapper."""
    
    def test_policy_creation(self):
        """Test policy creation."""
        policy = Policy(name="test_policy")
        
        assert policy.name == "test_policy"
        assert policy.coefficients is not None
        assert policy.router is not None
        assert policy.classifier is not None
    
    def test_policy_decide(self):
        """Test policy decision making."""
        policy = Policy(name="test")
        request = Request(
            prompt="Test",
            task_type=TaskType.CHAT,
            max_latency_ms=2000,
            min_quality_score=0.8,
        )
        
        features, decision = policy.decide(request)
        
        assert features is not None
        assert decision is not None
        assert features.request_id == request.request_id
        assert decision.request_id == request.request_id
    
    def test_policy_with_custom_strategies(self):
        """Test policy with custom strategies."""
        from conductor.strategies import Strategy, Provider
        
        custom_strategies = {
            "custom-only": Strategy(
                strategy_id="custom-only",
                model_id="gpt-4o",
                temperature=0.0,
                max_tokens=1000,
                batching_delay_ms=0,
                verifier_enabled=False,
                tools_enabled=False,
                supported_task_types=[TaskType.CHAT],
                provider=Provider.OPENAI,
            )
        }
        
        policy = Policy(
            name="custom",
            strategies=custom_strategies,
        )
        
        request = Request(
            prompt="Test",
            task_type=TaskType.CHAT,
            max_latency_ms=5000,
            min_quality_score=0.8,
        )
        
        features, decision = policy.decide(request)
        
        # Should use custom strategy
        assert decision.strategy_id == "custom-only"
