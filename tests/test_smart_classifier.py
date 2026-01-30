"""Tests for smart classifier."""

import pytest
from conductor.smart_classifier import SmartClassifier, SmartClassification


class TestSmartClassifier:
    """Test smart classifier."""

    def test_fallback_simple_prompt(self):
        """Simple prompts should get low scores in fallback."""
        classifier = SmartClassifier()
        # Force fallback by not having API key
        score = classifier._fallback_score("What is 2+2?")
        assert score <= 4

    def test_fallback_complex_prompt(self):
        """Complex prompts should get high scores in fallback."""
        classifier = SmartClassifier()
        score = classifier._fallback_score("Analyze the economic implications of this policy")
        assert score >= 6

    def test_parse_score_valid(self):
        """Valid scores should parse correctly."""
        classifier = SmartClassifier()
        assert classifier._parse_score("5") == 5
        assert classifier._parse_score("10") == 10
        assert classifier._parse_score("1") == 1

    def test_parse_score_clamps(self):
        """Scores outside 1-10 should be clamped."""
        classifier = SmartClassifier()
        assert classifier._parse_score("0") == 1
        assert classifier._parse_score("15") == 10

    def test_parse_score_invalid(self):
        """Invalid scores should default to 5."""
        classifier = SmartClassifier()
        assert classifier._parse_score("abc") == 5
        assert classifier._parse_score("") == 5

    def test_complexity_score_conversion(self):
        """Raw scores should convert to 0-1 correctly."""
        classifier = SmartClassifier()
        # Score 1 -> 0.0, Score 10 -> 1.0
        assert (1 - 1) / 9.0 == 0.0
        assert (10 - 1) / 9.0 == 1.0
        assert (5 - 1) / 9.0 == pytest.approx(0.444, rel=0.01)

    def test_caching(self):
        """Identical prompts should be cached."""
        classifier = SmartClassifier(cache_enabled=True)

        # First call - not cached
        result1 = classifier.classify("Test prompt")
        assert result1.from_cache is False

        # Second call - should be cached
        result2 = classifier.classify("Test prompt")
        assert result2.from_cache is True

    def test_cache_disabled(self):
        """Cache can be disabled."""
        classifier = SmartClassifier(cache_enabled=False)

        result1 = classifier.classify("Test prompt")
        result2 = classifier.classify("Test prompt")

        assert result1.from_cache is False
        assert result2.from_cache is False

    def test_get_reasoning(self):
        """Reasoning should be appropriate for score."""
        classifier = SmartClassifier()

        assert "trivial" in classifier._get_reasoning(1).lower()
        assert "simple" in classifier._get_reasoning(3).lower()
        assert "moderate" in classifier._get_reasoning(5).lower()
        assert "complex" in classifier._get_reasoning(7).lower()

    def test_cache_key_generation(self):
        """Cache keys should be deterministic."""
        classifier = SmartClassifier()

        key1 = classifier._cache_key("test")
        key2 = classifier._cache_key("test")
        key3 = classifier._cache_key("different")

        assert key1 == key2
        assert key1 != key3

    def test_clear_cache(self):
        """Cache should be clearable."""
        classifier = SmartClassifier(cache_enabled=True)

        classifier.classify("Test")
        assert len(classifier._cache) > 0

        classifier.clear_cache()
        assert len(classifier._cache) == 0
