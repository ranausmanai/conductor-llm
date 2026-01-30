"""Tests for the Predictor component."""

import pytest
from conductor.predictor import Predictor
from conductor.classifier import Classifier
from conductor.schemas import Request, RequestFeatures, TaskType, LatencyPercentile, UrgencyTier
from conductor.strategies import get_strategy, STRATEGIES
from conductor.coefficients import Coefficients


class TestPredictor:
    """Test suite for Predictor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.predictor = Predictor()
        self.classifier = Classifier()
    
    def _make_features(
        self,
        input_tokens: int = 100,
        output_tokens: int = 50,
        task_type: TaskType = TaskType.CHAT,
        complexity: float = 0.3,
    ) -> RequestFeatures:
        """Create test features."""
        return RequestFeatures(
            request_id="test-123",
            task_type=task_type,
            input_token_count=input_tokens,
            estimated_output_tokens=output_tokens,
            complexity_score=complexity,
            urgency_tier=UrgencyTier.INTERACTIVE,
            has_structured_output=False,
            max_latency_ms=2000,
            latency_percentile=LatencyPercentile.P95,
            min_quality_score=0.8,
            max_cost_usd=None,
            allow_batching=True,
            require_tools=False,
            require_verifier=False,
        )
    
    def test_cost_prediction_basic(self):
        """Test basic cost prediction."""
        features = self._make_features(input_tokens=1000, output_tokens=500)
        strategy = get_strategy("turbo-cheap")
        
        pred = self.predictor.predict(features, strategy)
        
        # Cost should be positive
        assert pred.cost_usd > 0
        
        # GPT-4o-mini: $0.00015/1K input, $0.0006/1K output
        # Note: turbo-cheap has max_tokens=256, so output is capped
        # Expected: (1000/1000 * 0.00015) + (256/1000 * 0.0006) = 0.000304
        assert 0.0002 < pred.cost_usd < 0.0005
    
    def test_cost_prediction_with_verifier(self):
        """Test that verifier adds to cost."""
        features = self._make_features()
        
        strategy_no_verifier = get_strategy("turbo-quality")
        strategy_with_verifier = get_strategy("quality-verified")
        
        pred_no = self.predictor.predict(features, strategy_no_verifier)
        pred_with = self.predictor.predict(features, strategy_with_verifier)
        
        # Verifier should add cost
        assert pred_with.cost_usd > pred_no.cost_usd
    
    def test_cost_scales_with_tokens(self):
        """Test that cost scales with token count."""
        features_small = self._make_features(input_tokens=100, output_tokens=50)
        features_large = self._make_features(input_tokens=1000, output_tokens=500)
        strategy = get_strategy("turbo-quality")
        
        pred_small = self.predictor.predict(features_small, strategy)
        pred_large = self.predictor.predict(features_large, strategy)
        
        # Larger request should cost more
        assert pred_large.cost_usd > pred_small.cost_usd
        # Should scale roughly linearly (10x tokens ≈ 10x cost)
        ratio = pred_large.cost_usd / pred_small.cost_usd
        assert 5 < ratio < 15
    
    def test_latency_prediction_basic(self):
        """Test basic latency prediction."""
        features = self._make_features()
        strategy = get_strategy("turbo-cheap")
        
        pred = self.predictor.predict(features, strategy)
        
        # Latencies should be positive and ordered
        assert pred.latency_p50_ms > 0
        assert pred.latency_p95_ms > pred.latency_p50_ms
        assert pred.latency_p99_ms > pred.latency_p95_ms
    
    def test_latency_includes_batching_delay(self):
        """Test that batching delay is included in latency."""
        features = self._make_features()
        
        strategy_no_batch = get_strategy("turbo-cheap")  # batching_delay_ms=0
        strategy_batch = get_strategy("batch-cheap")      # batching_delay_ms=500
        
        pred_no_batch = self.predictor.predict(features, strategy_no_batch)
        pred_batch = self.predictor.predict(features, strategy_batch)
        
        # Batching should add latency
        assert pred_batch.latency_p50_ms > pred_no_batch.latency_p50_ms
        # Difference should be at least the batching delay
        assert pred_batch.latency_p50_ms - pred_no_batch.latency_p50_ms >= 400
    
    def test_latency_includes_verifier(self):
        """Test that verifier adds to latency."""
        features = self._make_features()
        
        strategy_no_verifier = get_strategy("turbo-quality")
        strategy_with_verifier = get_strategy("quality-verified")
        
        pred_no = self.predictor.predict(features, strategy_no_verifier)
        pred_with = self.predictor.predict(features, strategy_with_verifier)
        
        # Verifier should add latency
        assert pred_with.latency_p50_ms > pred_no.latency_p50_ms
    
    def test_quality_risk_basic(self):
        """Test basic quality risk prediction."""
        features = self._make_features()
        strategy = get_strategy("turbo-quality")
        
        pred = self.predictor.predict(features, strategy)
        
        # Quality risk should be in valid range
        assert 0 <= pred.quality_risk <= 1
        assert 0 <= pred.expected_quality <= 1
        assert pred.expected_quality == 1 - pred.quality_risk
    
    def test_quality_risk_model_comparison(self):
        """Test that better models have lower quality risk."""
        features = self._make_features()
        
        strategy_mini = get_strategy("turbo-cheap")   # gpt-4o-mini
        strategy_full = get_strategy("turbo-quality") # gpt-4o
        
        pred_mini = self.predictor.predict(features, strategy_mini)
        pred_full = self.predictor.predict(features, strategy_full)
        
        # GPT-4o should have lower risk than GPT-4o-mini
        assert pred_full.quality_risk < pred_mini.quality_risk
    
    def test_quality_risk_verifier_reduction(self):
        """Test that verifier reduces quality risk."""
        features = self._make_features()
        
        strategy_no_verifier = get_strategy("turbo-quality")
        strategy_with_verifier = get_strategy("quality-verified")
        
        pred_no = self.predictor.predict(features, strategy_no_verifier)
        pred_with = self.predictor.predict(features, strategy_with_verifier)
        
        # Verifier should reduce risk
        assert pred_with.quality_risk < pred_no.quality_risk
    
    def test_quality_risk_complexity_impact(self):
        """Test that higher complexity increases quality risk."""
        features_simple = self._make_features(complexity=0.1)
        features_complex = self._make_features(complexity=0.9)
        strategy = get_strategy("turbo-cheap")
        
        pred_simple = self.predictor.predict(features_simple, strategy)
        pred_complex = self.predictor.predict(features_complex, strategy)
        
        # Complex requests should have higher risk
        assert pred_complex.quality_risk > pred_simple.quality_risk
    
    def test_quality_risk_temperature_impact(self):
        """Test that high temperature increases quality risk."""
        features = self._make_features()
        
        strategy_low_temp = get_strategy("turbo-quality")  # temp=0.0
        strategy_high_temp = get_strategy("long-form")     # temp=0.7
        
        pred_low = self.predictor.predict(features, strategy_low_temp)
        pred_high = self.predictor.predict(features, strategy_high_temp)
        
        # High temperature should have higher risk (all else equal)
        # Note: models are the same (gpt-4o), so difference is temperature
        assert pred_high.quality_risk > pred_low.quality_risk
    
    def test_predict_all_strategies(self):
        """Test predicting for all strategies."""
        features = self._make_features(task_type=TaskType.SUMMARIZE)
        strategies = list(STRATEGIES.values())
        
        predictions = self.predictor.predict_all(features, strategies)
        
        assert len(predictions) == len(strategies)
        
        for strategy, pred in predictions:
            assert pred.strategy_id == strategy.strategy_id
            assert pred.cost_usd > 0
            assert pred.latency_p50_ms > 0
    
    def test_tools_bonus_for_json_extraction(self):
        """Test that tools get quality bonus for JSON extraction."""
        features = self._make_features(task_type=TaskType.EXTRACT_JSON)
        
        strategy_no_tools = get_strategy("turbo-quality")
        strategy_tools = get_strategy("tool-augmented")
        
        pred_no = self.predictor.predict(features, strategy_no_tools)
        pred_tools = self.predictor.predict(features, strategy_tools)
        
        # Tools should have lower risk for JSON extraction
        # (assuming same model quality otherwise)
        # Note: Both use gpt-4o, so tools should win
        assert pred_tools.quality_risk <= pred_no.quality_risk
