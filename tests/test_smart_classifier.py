"""Tests for smart classifier."""

import pytest
from conductor.smart_classifier import (
    SmartClassifier,
    SmartClassification,
    smart_classify,
    CLASSIFICATION_PROMPT,
)


class TestSmartClassifier:
    """Test smart classifier functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = SmartClassifier(cache_enabled=True)

    def test_fallback_simple_prompt(self):
        """Test fallback heuristic identifies simple prompts."""
        # Simple prompts should get low scores
        simple_prompts = [
            "What is 2+2?",
            "Who is the president of USA?",
            "Define photosynthesis",
            "List 5 colors",
            "Yes or no: Is the sky blue?",
        ]
        
        for prompt in simple_prompts:
            score = self.classifier._fallback_score(prompt)
            assert score <= 5, f"Simple prompt '{prompt}' got score {score}"

    def test_fallback_complex_prompt(self):
        """Test fallback heuristic identifies complex prompts."""
        # Complex prompts should get higher scores
        complex_prompts = [
            "Explain why the sky is blue, step by step with physics",
            "Analyze and compare the economic policies of X and Y",
            "Debug this code and explain the fix: def foo(): pass",
            "Design an architecture for a distributed system",
            "Implement a binary search tree with balancing",
        ]
        
        for prompt in complex_prompts:
            score = self.classifier._fallback_score(prompt)
            assert score >= 5, f"Complex prompt '{prompt}' got score {score}"

    def test_parse_score_valid(self):
        """Test score parsing with valid inputs."""
        assert self.classifier._parse_score("5") == 5
        assert self.classifier._parse_score("  7  ") == 7
        assert self.classifier._parse_score("Score: 8") == 8
        assert self.classifier._parse_score("The complexity is 3.") == 3

    def test_parse_score_clamps(self):
        """Test score parsing clamps to 1-10."""
        assert self.classifier._parse_score("0") == 1
        assert self.classifier._parse_score("15") == 10
        assert self.classifier._parse_score("-5") == 5  # Finds 5, not -5

    def test_parse_score_invalid(self):
        """Test score parsing with invalid inputs returns default."""
        assert self.classifier._parse_score("") == 5
        assert self.classifier._parse_score(None) == 5
        assert self.classifier._parse_score("no number here") == 5

    def test_complexity_score_conversion(self):
        """Test that raw scores convert correctly to 0-1 range."""
        # Score 1 -> 0.0, Score 10 -> 1.0
        assert (1 - 1) / 9.0 == 0.0
        assert (10 - 1) / 9.0 == 1.0
        assert (5 - 1) / 9.0 == pytest.approx(0.444, rel=0.01)

    def test_caching(self):
        """Test that caching works."""
        prompt = "Test prompt for caching"
        
        # First call - not cached
        result1 = self.classifier.classify(prompt)
        assert result1.from_cache is False
        
        # Second call - should be cached
        result2 = self.classifier.classify(prompt)
        assert result2.from_cache is True
        assert result2.complexity_score == result1.complexity_score

    def test_cache_disabled(self):
        """Test that caching can be disabled."""
        classifier = SmartClassifier(cache_enabled=False)
        prompt = "Test prompt"
        
        result1 = classifier.classify(prompt)
        result2 = classifier.classify(prompt)
        
        # Both should not be from cache
        assert result1.from_cache is False
        assert result2.from_cache is False

    def test_get_reasoning(self):
        """Test reasoning generation."""
        assert "Trivial" in self.classifier._get_reasoning(1)
        assert "Simple" in self.classifier._get_reasoning(3)
        assert "Moderate" in self.classifier._get_reasoning(5)
        assert "Complex" in self.classifier._get_reasoning(7)
        assert "Very complex" in self.classifier._get_reasoning(9)

    def test_cache_key_generation(self):
        """Test cache key handles long prompts."""
        short = "short"
        long = "x" * 1000
        
        key1 = self.classifier._get_cache_key(short)
        key2 = self.classifier._get_cache_key(long)
        
        assert len(key1) < 600  # Should be bounded
        assert len(key2) < 600  # Should be bounded
        assert key1 != key2

    def test_cache_eviction(self):
        """Test cache eviction when full."""
        classifier = SmartClassifier(cache_enabled=True, max_cache_size=3)
        
        # Fill cache
        classifier.classify("prompt1")
        classifier.classify("prompt2")
        classifier.classify("prompt3")
        
        stats = classifier.get_cache_stats()
        assert stats["size"] == 3
        
        # Add one more - should evict oldest
        classifier.classify("prompt4")
        
        stats = classifier.get_cache_stats()
        assert stats["size"] == 3

    def test_clear_cache(self):
        """Test cache clearing."""
        self.classifier.classify("test")
        assert self.classifier.get_cache_stats()["size"] > 0
        
        self.classifier.clear_cache()
        assert self.classifier.get_cache_stats()["size"] == 0

    def test_classification_prompt_format(self):
        """Test that classification prompt is properly formatted."""
        assert "{prompt}" in CLASSIFICATION_PROMPT
        formatted = CLASSIFICATION_PROMPT.format(prompt="test")
        assert "test" in formatted
        assert "{prompt}" not in formatted


class TestSmartClassifyConvenience:
    """Test the convenience function."""

    def test_smart_classify_returns_classification(self):
        """Test smart_classify returns proper type."""
        result = smart_classify("What is 2+2?")
        assert isinstance(result, SmartClassification)
        assert 0 <= result.complexity_score <= 1
        assert result.reasoning is not None
        assert 1 <= result.raw_score <= 10
