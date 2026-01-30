"""Tests for the Classifier component."""

import pytest
from conductor.classifier import Classifier
from conductor.schemas import Request, TaskType, UrgencyTier, LatencyPercentile


class TestClassifier:
    """Test suite for Classifier."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = Classifier()
    
    def test_basic_classification(self):
        """Test basic request classification."""
        request = Request(
            prompt="What is 2 + 2?",
            task_type=TaskType.CLASSIFY,
            max_latency_ms=2000,
        )
        
        features = self.classifier.classify(request)
        
        assert features.request_id == request.request_id
        assert features.task_type == TaskType.CLASSIFY
        assert features.input_token_count > 0
        assert 0 <= features.complexity_score <= 1
    
    def test_token_estimation(self):
        """Test token count estimation."""
        # Short prompt
        short_request = Request(
            prompt="Hello",
            task_type=TaskType.CHAT,
        )
        short_features = self.classifier.classify(short_request)
        
        # Long prompt
        long_prompt = "This is a much longer prompt " * 100
        long_request = Request(
            prompt=long_prompt,
            task_type=TaskType.CHAT,
        )
        long_features = self.classifier.classify(long_request)
        
        assert long_features.input_token_count > short_features.input_token_count
    
    def test_complexity_score_simple(self):
        """Test complexity score for simple prompts."""
        request = Request(
            prompt="What time is it?",
            task_type=TaskType.CHAT,
        )
        
        features = self.classifier.classify(request)
        
        # Simple question should have low complexity
        assert features.complexity_score < 0.3
    
    def test_complexity_score_complex(self):
        """Test complexity score for complex prompts."""
        complex_prompt = """
        Please explain step by step why the following code doesn't work:
        
        def calculate(x, y):
            if x > y:
                return x - y
            elif x < y:
                return y - x
            else:
                return 0
        
        1. What is the issue?
        2. How can it be fixed?
        3. Are there any edge cases we should consider?
        """
        
        request = Request(
            prompt=complex_prompt,
            task_type=TaskType.CHAT,
        )
        
        features = self.classifier.classify(request)
        
        # Complex prompt should have higher complexity
        assert features.complexity_score > 0.3
    
    def test_urgency_tier_realtime(self):
        """Test urgency tier for realtime latency."""
        request = Request(
            prompt="Quick question",
            task_type=TaskType.CHAT,
            max_latency_ms=300,
        )
        
        features = self.classifier.classify(request)
        
        assert features.urgency_tier == UrgencyTier.REALTIME
    
    def test_urgency_tier_interactive(self):
        """Test urgency tier for interactive latency."""
        request = Request(
            prompt="Question",
            task_type=TaskType.CHAT,
            max_latency_ms=1500,
        )
        
        features = self.classifier.classify(request)
        
        assert features.urgency_tier == UrgencyTier.INTERACTIVE
    
    def test_urgency_tier_batch(self):
        """Test urgency tier for batch latency."""
        request = Request(
            prompt="No rush",
            task_type=TaskType.CHAT,
            max_latency_ms=10000,
        )
        
        features = self.classifier.classify(request)
        
        assert features.urgency_tier == UrgencyTier.BATCH
    
    def test_structured_output_detection(self):
        """Test detection of structured output requests."""
        json_request = Request(
            prompt="Extract the data as JSON: ...",
            task_type=TaskType.EXTRACT_JSON,
        )
        
        features = self.classifier.classify(json_request)
        
        assert features.has_structured_output is True
    
    def test_output_token_estimation_by_task(self):
        """Test output token estimation varies by task type."""
        # Classification should estimate few tokens
        classify_request = Request(
            prompt="Is this spam?",
            task_type=TaskType.CLASSIFY,
        )
        classify_features = self.classifier.classify(classify_request)
        
        # Long-form should estimate more tokens
        generate_request = Request(
            prompt="Write a blog post",
            task_type=TaskType.GENERATE_LONG,
        )
        generate_features = self.classifier.classify(generate_request)
        
        assert generate_features.estimated_output_tokens > classify_features.estimated_output_tokens
    
    def test_client_provided_output_tokens(self):
        """Test that client-provided output token estimate is used."""
        request = Request(
            prompt="Some prompt",
            task_type=TaskType.CHAT,
            expected_output_tokens=500,
        )
        
        features = self.classifier.classify(request)
        
        assert features.estimated_output_tokens == 500
    
    def test_passthrough_constraints(self):
        """Test that constraints are passed through."""
        request = Request(
            prompt="Test",
            task_type=TaskType.CHAT,
            max_latency_ms=1234,
            latency_percentile=LatencyPercentile.P99,
            min_quality_score=0.92,
            max_cost_usd=0.05,
            allow_batching=False,
            require_tools=True,
            require_verifier=True,
        )
        
        features = self.classifier.classify(request)
        
        assert features.max_latency_ms == 1234
        assert features.latency_percentile == LatencyPercentile.P99
        assert features.min_quality_score == 0.92
        assert features.max_cost_usd == 0.05
        assert features.allow_batching is False
        assert features.require_tools is True
        assert features.require_verifier is True
