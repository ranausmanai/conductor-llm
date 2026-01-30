"""Tests for session and workflow functionality."""

import pytest
from unittest.mock import MagicMock, patch
from conductor.session import (
    Session,
    StepResult,
    SessionStats,
    WorkflowBuilder,
    Workflow,
    BudgetExceededError,
)
from conductor.schemas import TaskType


class MockResponse:
    """Mock conductor response for testing."""
    def __init__(self, text="response", model="gpt-4o-mini", cost=0.001, latency=100):
        self.text = text
        self.model_used = model
        self.cost = cost
        self.latency_ms = latency


class MockConductor:
    """Mock conductor for testing sessions."""
    def __init__(self):
        self.call_count = 0
        self.calls = []

    def complete(self, prompt, task_type, **kwargs):
        self.call_count += 1
        self.calls.append({"prompt": prompt, "task_type": task_type, **kwargs})
        return MockResponse(
            text=f"response-{self.call_count}",
            cost=kwargs.get("_mock_cost", 0.001),
            latency=kwargs.get("_mock_latency", 100),
        )

    def session(self, session_id=None, budget_usd=None, **kwargs):
        """Return a session for this mock conductor."""
        return Session(self, session_id, budget_usd, **kwargs)


class TestSession:
    """Test session functionality."""

    def test_session_context_manager(self):
        """Test session works as context manager."""
        conductor = MockConductor()
        session = Session(conductor, session_id="test-session")

        assert not session._is_active

        with session:
            assert session._is_active
            session.complete("test", TaskType.CLASSIFY)

        assert not session._is_active

    def test_session_tracks_steps(self):
        """Test session tracks all steps."""
        conductor = MockConductor()

        with Session(conductor, "test") as session:
            session.complete("prompt 1", TaskType.CLASSIFY)
            session.complete("prompt 2", TaskType.SUMMARIZE)
            session.complete("prompt 3", TaskType.EXTRACT_JSON)

        assert len(session.steps) == 3
        assert session.step_count == 3
        assert session.steps[0].step_number == 1
        assert session.steps[1].step_number == 2
        assert session.steps[2].step_number == 3

    def test_session_tracks_total_cost(self):
        """Test session calculates total cost."""
        conductor = MockConductor()

        with Session(conductor, "test") as session:
            session.complete("p1", TaskType.CLASSIFY, _mock_cost=0.001)
            session.complete("p2", TaskType.CLASSIFY, _mock_cost=0.002)
            session.complete("p3", TaskType.CLASSIFY, _mock_cost=0.003)

        assert session.total_cost == pytest.approx(0.006, rel=0.01)

    def test_session_tracks_total_latency(self):
        """Test session calculates total latency."""
        conductor = MockConductor()

        with Session(conductor, "test") as session:
            session.complete("p1", TaskType.CLASSIFY, _mock_latency=100)
            session.complete("p2", TaskType.CLASSIFY, _mock_latency=200)
            session.complete("p3", TaskType.CLASSIFY, _mock_latency=150)

        assert session.total_latency_ms == 450

    def test_session_budget_remaining(self):
        """Test budget remaining calculation."""
        conductor = MockConductor()

        with Session(conductor, "test", budget_usd=0.01) as session:
            session.complete("p1", TaskType.CLASSIFY, _mock_cost=0.003)
            assert session.budget_remaining == pytest.approx(0.007, rel=0.01)

            session.complete("p2", TaskType.CLASSIFY, _mock_cost=0.004)
            assert session.budget_remaining == pytest.approx(0.003, rel=0.01)

    def test_session_budget_utilization(self):
        """Test budget utilization percentage."""
        conductor = MockConductor()

        with Session(conductor, "test", budget_usd=0.01) as session:
            session.complete("p1", TaskType.CLASSIFY, _mock_cost=0.005)

        assert session.budget_utilization == pytest.approx(50.0, rel=1)

    def test_session_no_budget(self):
        """Test session without budget."""
        conductor = MockConductor()

        with Session(conductor, "test") as session:
            session.complete("p1", TaskType.CLASSIFY)

        assert session.budget_remaining is None
        assert session.budget_utilization is None

    def test_session_strict_budget_raises(self):
        """Test strict budget mode raises on exceed."""
        conductor = MockConductor()

        with pytest.raises(BudgetExceededError):
            with Session(conductor, "test", budget_usd=0.001, budget_strategy="strict") as session:
                session.complete("p1", TaskType.CLASSIFY, _mock_cost=0.002)
                session.complete("p2", TaskType.CLASSIFY, _mock_cost=0.001)  # Should raise

    def test_session_adaptive_budget_adjusts(self):
        """Test adaptive budget mode adjusts kwargs."""
        conductor = MockConductor()

        with Session(conductor, "test", budget_usd=0.01, budget_strategy="adaptive") as session:
            # First call - plenty of budget
            session.complete("p1", TaskType.CLASSIFY, _mock_cost=0.008)

            # Second call - budget low (<20%), should adjust
            session.complete("p2", TaskType.CLASSIFY, _mock_cost=0.001)

        # Check that max_cost_usd was added to the second call
        assert "max_cost_usd" in conductor.calls[1]

    def test_session_get_stats(self):
        """Test session statistics generation."""
        conductor = MockConductor()

        with Session(conductor, "test-stats", budget_usd=0.05) as session:
            session.complete("p1", TaskType.CLASSIFY, _mock_cost=0.01)
            session.complete("p2", TaskType.CLASSIFY, _mock_cost=0.02)

        stats = session.get_stats()

        assert isinstance(stats, SessionStats)
        assert stats.session_id == "test-stats"
        assert stats.total_steps == 2
        assert stats.total_cost_usd == pytest.approx(0.03, rel=0.01)
        assert stats.budget_usd == 0.05
        assert stats.budget_remaining_usd == pytest.approx(0.02, rel=0.01)
        assert "gpt-4o-mini" in stats.models_used

    def test_session_requires_context_manager(self):
        """Test session raises if not in context."""
        conductor = MockConductor()
        session = Session(conductor, "test")

        with pytest.raises(RuntimeError, match="not active"):
            session.complete("test", TaskType.CLASSIFY)

    def test_session_repr(self):
        """Test session string representation."""
        conductor = MockConductor()
        session = Session(conductor, "my-session", budget_usd=0.10)

        repr_str = repr(session)
        assert "my-session" in repr_str
        assert "budget=" in repr_str


class TestSessionParallel:
    """Test parallel execution."""

    def test_parallel_execution(self):
        """Test parallel executes all requests."""
        conductor = MockConductor()

        with Session(conductor, "test") as session:
            results = session.parallel([
                {"prompt": "p1", "task_type": TaskType.CLASSIFY},
                {"prompt": "p2", "task_type": TaskType.CLASSIFY},
                {"prompt": "p3", "task_type": TaskType.CLASSIFY},
            ])

        assert len(results) == 3
        assert conductor.call_count == 3

    def test_parallel_preserves_order(self):
        """Test parallel results are in original order."""
        conductor = MockConductor()

        with Session(conductor, "test") as session:
            results = session.parallel([
                {"prompt": "first", "task_type": TaskType.CLASSIFY},
                {"prompt": "second", "task_type": TaskType.CLASSIFY},
                {"prompt": "third", "task_type": TaskType.CLASSIFY},
            ])

        # Results should be in order despite parallel execution
        assert results[0].text == "response-1"
        assert results[1].text == "response-2"
        assert results[2].text == "response-3"

    def test_map_over_items(self):
        """Test map applies template to each item."""
        conductor = MockConductor()

        with Session(conductor, "test") as session:
            results = session.map(
                items=["doc1", "doc2", "doc3"],
                prompt_template="Summarize: {item}",
                task_type=TaskType.SUMMARIZE,
            )

        assert len(results) == 3
        assert "Summarize: doc1" in conductor.calls[0]["prompt"]
        assert "Summarize: doc2" in conductor.calls[1]["prompt"]
        assert "Summarize: doc3" in conductor.calls[2]["prompt"]


class TestWorkflowBuilder:
    """Test workflow builder and execution."""

    def test_workflow_builder_chain(self):
        """Test workflow builder chaining."""
        conductor = MockConductor()

        workflow = (
            conductor.workflow_builder("test-workflow")
            if hasattr(conductor, 'workflow_builder')
            else WorkflowBuilder(conductor, "test-workflow")
        )

        workflow = (
            WorkflowBuilder(conductor, "test-workflow")
            .add_step("step1", TaskType.SUMMARIZE, "Summarize: {input}")
            .add_step("step2", TaskType.EXTRACT_JSON, "Extract: {step1}")
            .build()
        )

        assert isinstance(workflow, Workflow)

    def test_workflow_execution(self):
        """Test workflow runs all steps."""
        conductor = MockConductor()

        workflow = (
            WorkflowBuilder(conductor, "test-workflow")
            .add_step("summarize", TaskType.SUMMARIZE, "Summarize: {input}")
            .add_step("extract", TaskType.EXTRACT_JSON, "Extract: {summarize}")
            .build()
        )

        result = workflow.run(input="test document")

        assert "result" in result
        assert "steps" in result
        assert "stats" in result
        assert "summarize" in result["steps"]
        assert "extract" in result["steps"]

    def test_workflow_passes_outputs(self):
        """Test workflow passes step outputs to next steps."""
        conductor = MockConductor()

        workflow = (
            WorkflowBuilder(conductor, "chain")
            .add_step("step1", TaskType.CLASSIFY, "Process: {input}")
            .add_step("step2", TaskType.CLASSIFY, "Continue: {step1}")
            .build()
        )

        workflow.run(input="start")

        # Second call should include output from first
        assert "response-1" in conductor.calls[1]["prompt"]


class TestSessionSavings:
    """Test savings tracking functionality."""

    def test_session_tracks_savings(self):
        """Test session calculates savings vs baseline."""
        conductor = MockConductor()

        with Session(conductor, "test") as session:
            session.complete("Short prompt", TaskType.CLASSIFY, _mock_cost=0.0001)

        # Should have calculated some savings
        assert session.baseline_cost > 0
        assert session.total_savings >= 0
        assert session.savings_pct >= 0

    def test_session_savings_report(self):
        """Test human-readable savings report."""
        conductor = MockConductor()

        with Session(conductor, "test-report") as session:
            session.complete("p1", TaskType.CLASSIFY, _mock_cost=0.001)
            session.complete("p2", TaskType.CLASSIFY, _mock_cost=0.002)

        report = session.get_savings_report()

        assert "Savings Report" in report
        assert "test-report" in report
        assert "Actual cost" in report
        assert "Baseline cost" in report
        assert "You saved" in report
        assert "gpt-4o-mini" in report

    def test_stats_include_savings(self):
        """Test that stats include savings info."""
        conductor = MockConductor()

        with Session(conductor, "test") as session:
            session.complete("p1", TaskType.CLASSIFY, _mock_cost=0.001)

        stats = session.get_stats()

        assert hasattr(stats, 'baseline_cost_usd')
        assert hasattr(stats, 'total_savings_usd')
        assert hasattr(stats, 'savings_pct')

    def test_repr_includes_savings(self):
        """Test session repr shows savings percentage."""
        conductor = MockConductor()

        with Session(conductor, "test") as session:
            session.complete("p1", TaskType.CLASSIFY, _mock_cost=0.001)

        repr_str = repr(session)
        assert "saved=" in repr_str


class TestStepResult:
    """Test StepResult dataclass."""

    def test_step_result_creation(self):
        """Test creating a step result."""
        from datetime import datetime, UTC

        step = StepResult(
            step_id="test-1",
            step_number=1,
            prompt="test prompt",
            task_type="classify",
            text="response",
            model_used="gpt-4o-mini",
            cost_usd=0.001,
            latency_ms=100,
            timestamp=datetime.now(UTC),
            baseline_cost_usd=0.005,
            savings_usd=0.004,
        )

        assert step.step_id == "test-1"
        assert step.cost_usd == 0.001
        assert step.latency_ms == 100
        assert step.baseline_cost_usd == 0.005
        assert step.savings_usd == 0.004
