"""Tests for the simple API."""

import pytest
from conductor import llm_dry_run, Response
from conductor.simple import _is_complex, _calculate_cost, CHEAP_MODEL, QUALITY_MODEL


class TestComplexityDetection:
    """Test prompt complexity detection."""

    def test_simple_question(self):
        """Simple questions should be detected as simple."""
        assert _is_complex("What is 2+2?") is False
        assert _is_complex("What is the capital of France?") is False
        assert _is_complex("Who is the president?") is False

    def test_simple_short_prompt(self):
        """Short prompts should default to simple."""
        assert _is_complex("Hello") is False
        assert _is_complex("Hi there") is False
        assert _is_complex("Thanks") is False

    def test_complex_keywords(self):
        """Prompts with complex keywords should be detected."""
        assert _is_complex("Analyze this document") is True
        assert _is_complex("Debug this code") is True
        assert _is_complex("Write a legal contract") is True
        assert _is_complex("Create a business plan") is True

    def test_complex_long_prompt(self):
        """Long prompts should be detected as complex."""
        long_prompt = "Please help me with " + "x " * 500
        assert _is_complex(long_prompt) is True

    def test_summarize_is_simple(self):
        """Summarization tasks should be simple."""
        assert _is_complex("Summarize this article") is False

    def test_translate_is_simple(self):
        """Translation tasks should be simple."""
        assert _is_complex("Translate this to Spanish") is False


class TestDryRun:
    """Test dry run functionality."""

    def test_dry_run_simple(self):
        """Dry run should predict cheap model for simple prompts."""
        result = llm_dry_run("What is 2+2?")
        assert result["model"] == CHEAP_MODEL
        assert "Simple" in result["reason"] or "simple" in result["reason"]

    def test_dry_run_complex(self):
        """Dry run should predict quality model for complex prompts."""
        result = llm_dry_run("Analyze this legal contract and identify risks")
        assert result["model"] == QUALITY_MODEL
        assert "Complex" in result["reason"] or "complex" in result["reason"]

    def test_dry_run_force_high(self):
        """Dry run with quality=high should always use quality model."""
        result = llm_dry_run("What is 2+2?", quality="high")
        assert result["model"] == QUALITY_MODEL

    def test_dry_run_force_low(self):
        """Dry run with quality=low should always use cheap model."""
        result = llm_dry_run("Analyze complex document", quality="low")
        assert result["model"] == CHEAP_MODEL


class TestCostCalculation:
    """Test cost calculation."""

    def test_cost_calculation_mini(self):
        """Cost calculation for gpt-4o-mini."""
        # 1000 input + 1000 output tokens
        cost = _calculate_cost("gpt-4o-mini", 1000, 1000)
        # $0.15/1M input + $0.60/1M output
        expected = (1000 / 1_000_000) * 0.15 + (1000 / 1_000_000) * 0.60
        assert cost == pytest.approx(expected, rel=0.01)

    def test_cost_calculation_4o(self):
        """Cost calculation for gpt-4o."""
        cost = _calculate_cost("gpt-4o", 1000, 1000)
        # $2.50/1M input + $10.00/1M output
        expected = (1000 / 1_000_000) * 2.50 + (1000 / 1_000_000) * 10.00
        assert cost == pytest.approx(expected, rel=0.01)

    def test_mini_cheaper_than_4o(self):
        """gpt-4o-mini should always be cheaper than gpt-4o."""
        mini_cost = _calculate_cost("gpt-4o-mini", 1000, 1000)
        full_cost = _calculate_cost("gpt-4o", 1000, 1000)
        assert mini_cost < full_cost


class TestResponse:
    """Test Response dataclass."""

    def test_response_str(self):
        """Response should stringify to text."""
        r = Response(
            text="Hello world",
            model="gpt-4o-mini",
            cost=0.001,
            saved=0.009,
            baseline_cost=0.01,
            prompt_tokens=10,
            completion_tokens=20,
        )
        assert str(r) == "Hello world"

    def test_response_repr(self):
        """Response repr should show key info."""
        r = Response(
            text="Hello world",
            model="gpt-4o-mini",
            cost=0.001,
            saved=0.009,
            baseline_cost=0.01,
            prompt_tokens=10,
            completion_tokens=20,
        )
        repr_str = repr(r)
        assert "gpt-4o-mini" in repr_str
        assert "$0.001" in repr_str or "0.001" in repr_str


class TestImports:
    """Test that imports work."""

    def test_import_llm(self):
        """llm function should be importable."""
        from conductor import llm
        assert callable(llm)

    def test_import_dry_run(self):
        """llm_dry_run function should be importable."""
        from conductor import llm_dry_run
        assert callable(llm_dry_run)

    def test_import_response(self):
        """Response class should be importable."""
        from conductor import Response
        assert Response is not None
