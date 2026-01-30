"""Tests for session functionality."""

import pytest
from conductor.session import Session, StepResult, BudgetExceededError
from conductor.simple import Response


# Mock llm function for testing
def mock_llm(prompt: str, quality: str = "auto", **kwargs) -> Response:
    """Mock LLM function that returns predictable results."""
    return Response(
        text=f"Response to: {prompt[:20]}",
        model="gpt-4o-mini" if quality != "high" else "gpt-4o",
        cost=0.001,
        saved=0.009,
        baseline_cost=0.01,
        prompt_tokens=10,
        completion_tokens=20,
    )


class TestSession:
    """Test session functionality."""

    def test_session_context_manager(self):
        """Session should work as context manager."""
        session = Session(llm_func=mock_llm)

        with session as s:
            assert s._started is True

        assert session._ended is True

    def test_session_tracks_steps(self):
        """Session should track steps."""
        with Session(llm_func=mock_llm) as s:
            s.llm("First prompt")
            s.llm("Second prompt")

        assert s.step_count == 2
        assert len(s.steps) == 2

    def test_session_tracks_cost(self):
        """Session should track total cost."""
        with Session(llm_func=mock_llm) as s:
            s.llm("First")
            s.llm("Second")

        assert s.total_cost == 0.002  # 2 * 0.001

    def test_session_tracks_savings(self):
        """Session should track savings."""
        with Session(llm_func=mock_llm) as s:
            s.llm("First")
            s.llm("Second")

        assert s.total_saved == 0.018  # 2 * 0.009
        assert s.baseline_cost == 0.02  # 2 * 0.01

    def test_session_savings_pct(self):
        """Session should calculate savings percentage."""
        with Session(llm_func=mock_llm) as s:
            s.llm("Test")

        assert s.savings_pct == pytest.approx(90.0, rel=0.01)

    def test_session_budget_remaining(self):
        """Session should track remaining budget."""
        with Session(llm_func=mock_llm, budget_usd=1.00) as s:
            s.llm("Test")

        assert s.budget_remaining == pytest.approx(0.999, rel=0.01)

    def test_session_no_budget(self):
        """Session without budget should have None remaining."""
        with Session(llm_func=mock_llm) as s:
            s.llm("Test")

        assert s.budget_remaining is None

    def test_session_strict_budget_raises(self):
        """Strict budget should raise when exceeded."""
        with pytest.raises(BudgetExceededError):
            with Session(llm_func=mock_llm, budget_usd=0.0001, budget_strategy="strict") as s:
                s.llm("First")
                s.llm("Second")  # Should raise

    def test_session_requires_context(self):
        """Calling llm outside context should raise."""
        session = Session(llm_func=mock_llm)

        with pytest.raises(RuntimeError):
            session.llm("Test")

    def test_session_get_stats(self):
        """Session should return stats dict."""
        with Session(llm_func=mock_llm, session_id="test-session") as s:
            s.llm("Test")

        stats = s.get_stats()

        assert stats["session_id"] == "test-session"
        assert stats["step_count"] == 1
        assert stats["total_cost"] == 0.001

    def test_session_get_report(self):
        """Session should generate readable report."""
        with Session(llm_func=mock_llm) as s:
            s.llm("Test")

        report = s.get_report()

        assert "Actual cost" in report
        assert "Baseline cost" in report
        assert "saved" in report.lower()

    def test_session_repr(self):
        """Session repr should show key info."""
        with Session(llm_func=mock_llm, session_id="my-session") as s:
            s.llm("Test")

        repr_str = repr(s)
        assert "my-session" in repr_str
        assert "steps=1" in repr_str


class TestStepResult:
    """Test StepResult dataclass."""

    def test_step_result_creation(self):
        """StepResult should store all fields."""
        from datetime import datetime, timezone

        step = StepResult(
            step_id="step-1",
            step_number=1,
            prompt="Test prompt",
            text="Test response",
            model="gpt-4o-mini",
            cost=0.001,
            latency_ms=100,
            timestamp=datetime.now(timezone.utc),
            baseline_cost=0.01,
            saved=0.009,
        )

        assert step.step_id == "step-1"
        assert step.cost == 0.001
        assert step.saved == 0.009
