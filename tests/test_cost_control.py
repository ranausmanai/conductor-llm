"""Tests for cost control."""

import pytest
from conductor.cost_control import (
    CostController,
    UserBudget,
    Period,
    BudgetExceededError,
)


class TestUserBudgets:
    """Test user budget management."""

    def test_set_user_budget(self):
        """Test setting a user budget."""
        controller = CostController()

        budget = controller.set_user_budget(
            "user_123",
            daily_usd=5.00,
            monthly_usd=50.00,
            hard_limit=True,
        )

        assert budget.user_id == "user_123"
        assert budget.daily_usd == 5.00
        assert budget.monthly_usd == 50.00

    def test_get_user_budget(self):
        """Test retrieving a user budget."""
        controller = CostController()
        controller.set_user_budget("user_123", daily_usd=10.00)

        budget = controller.get_user_budget("user_123")
        assert budget is not None
        assert budget.daily_usd == 10.00

        # Non-existent user
        assert controller.get_user_budget("unknown") is None

    def test_remove_user_budget(self):
        """Test removing a user budget."""
        controller = CostController()
        controller.set_user_budget("user_123", daily_usd=10.00)

        assert controller.remove_user_budget("user_123") is True
        assert controller.get_user_budget("user_123") is None

    def test_check_budget_no_budget_set(self):
        """Test check_budget when no budget is set."""
        controller = CostController()

        result = controller.check_budget("user_123", estimated_cost=100.00)

        assert result["allowed"] is True
        assert result["warnings"] == []

    def test_check_budget_hard_limit_exceeded(self):
        """Test check_budget raises when hard limit exceeded."""
        controller = CostController()
        controller.set_user_budget("user_123", daily_usd=0.01, hard_limit=True)

        # Record costs up to limit
        controller.record_cost("user_123", 0.02, "gpt-4o", "chat")

        # Should raise
        with pytest.raises(BudgetExceededError):
            controller.check_budget("user_123", estimated_cost=0.01)

    def test_check_budget_soft_limit_warning(self):
        """Test check_budget warns when soft limit exceeded."""
        controller = CostController()
        controller.set_user_budget("user_123", daily_usd=0.01, hard_limit=False)

        # Record costs over limit
        controller.record_cost("user_123", 0.02, "gpt-4o", "chat")

        result = controller.check_budget("user_123", estimated_cost=0.01)

        assert result["allowed"] is True
        assert len(result["warnings"]) > 0


class TestCostRecording:
    """Test cost recording."""

    def test_record_cost(self):
        """Test recording a cost."""
        controller = CostController()

        record = controller.record_cost(
            user_id="user_123",
            cost_usd=0.005,
            model="gpt-4o-mini",
            task_type="chat",
        )

        assert record.user_id == "user_123"
        assert record.cost_usd == 0.005
        assert record.model == "gpt-4o-mini"

    def test_records_accumulate(self):
        """Test that records accumulate."""
        controller = CostController()

        controller.record_cost("user_1", 0.01, "gpt-4o", "chat")
        controller.record_cost("user_1", 0.02, "gpt-4o", "chat")
        controller.record_cost("user_2", 0.03, "gpt-4o-mini", "chat")

        summary = controller.get_summary()
        assert summary["total_records"] == 3


class TestCostReports:
    """Test cost reporting."""

    def test_get_cost_report_by_user(self):
        """Test cost report grouped by user."""
        controller = CostController()

        controller.record_cost("user_1", 0.10, "gpt-4o", "chat")
        controller.record_cost("user_1", 0.20, "gpt-4o", "chat")
        controller.record_cost("user_2", 0.05, "gpt-4o-mini", "chat")

        report = controller.get_cost_report(group_by="user", period=Period.DAILY)

        assert report.total_cost_usd == pytest.approx(0.35, rel=0.01)
        assert report.total_requests == 3
        assert "user_1" in report.breakdown

    def test_get_cost_report_by_model(self):
        """Test cost report grouped by model."""
        controller = CostController()

        controller.record_cost("user_1", 0.10, "gpt-4o", "chat")
        controller.record_cost("user_1", 0.02, "gpt-4o-mini", "chat")

        report = controller.get_cost_report(group_by="model", period=Period.DAILY)

        assert "gpt-4o" in report.breakdown
        assert "gpt-4o-mini" in report.breakdown

    def test_get_user_report(self):
        """Test user-specific report."""
        controller = CostController()
        controller.set_user_budget("user_123", daily_usd=10.00)

        controller.record_cost("user_123", 0.10, "gpt-4o", "chat")
        controller.record_cost("user_123", 0.20, "gpt-4o-mini", "chat")

        report = controller.get_user_report("user_123")

        assert report["user_id"] == "user_123"
        assert report["total_requests"] == 2
        assert report["total_cost_usd"] == pytest.approx(0.30, rel=0.01)


class TestUtilities:
    """Test utility functions."""

    def test_get_summary(self):
        """Test getting summary."""
        controller = CostController()
        controller.set_user_budget("user_1", daily_usd=10.00)
        controller.record_cost("user_1", 0.50, "gpt-4o", "chat")

        summary = controller.get_summary()

        assert summary["total_users_with_budgets"] == 1
        assert summary["total_records"] == 1

    def test_clear_records(self):
        """Test clearing records."""
        controller = CostController()

        controller.record_cost("user_1", 0.01, "gpt-4o", "chat")
        controller.clear_records()

        assert controller.get_summary()["total_records"] == 0

    def test_export_records_dict(self):
        """Test exporting records as dict."""
        controller = CostController()
        controller.record_cost("user_1", 0.01, "gpt-4o", "chat")

        records = controller.export_records(format="dict")

        assert len(records) == 1
        assert records[0]["user_id"] == "user_1"

    def test_export_records_csv(self):
        """Test exporting records as CSV."""
        controller = CostController()
        controller.record_cost("user_1", 0.01, "gpt-4o", "chat")

        csv = controller.export_records(format="csv")

        assert "user_id" in csv
        assert "user_1" in csv


class TestBudgetExceededError:
    """Test BudgetExceededError."""

    def test_error_message(self):
        """Test error contains useful info."""
        error = BudgetExceededError("user_123", "daily", 10.50, 10.00)

        assert error.user_id == "user_123"
        assert error.period == "daily"
        assert "user_123" in str(error)
        assert "daily" in str(error)
