"""Tests for cost control functionality."""

import pytest
from datetime import datetime, UTC, timedelta
from conductor.cost_control import (
    CostController,
    UserBudget,
    CostRecord,
    CostReport,
    Period,
    BudgetExceededError,
    AnomalyAlert,
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
        assert budget.hard_limit is True

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
        assert controller.remove_user_budget("user_123") is False  # Already removed

    def test_check_budget_no_budget_set(self):
        """Test check_budget when no budget is set."""
        controller = CostController()

        result = controller.check_budget("user_123", estimated_cost=100.00)

        assert result["allowed"] is True
        assert result["warnings"] == []

    def test_check_budget_within_limits(self):
        """Test check_budget when within limits."""
        controller = CostController()
        controller.set_user_budget("user_123", daily_usd=10.00)

        # Record some costs
        controller.record_cost("user_123", 2.00, "gpt-4o-mini", "classify")

        result = controller.check_budget("user_123", estimated_cost=1.00)

        assert result["allowed"] is True
        assert result["current_spend"]["daily"] == 2.00

    def test_check_budget_hard_limit_exceeded(self):
        """Test check_budget raises when hard limit exceeded."""
        controller = CostController()
        controller.set_user_budget("user_123", daily_usd=5.00, hard_limit=True)

        # Record costs up to limit
        controller.record_cost("user_123", 5.00, "gpt-4o", "generate")

        # Should raise
        with pytest.raises(BudgetExceededError) as exc_info:
            controller.check_budget("user_123", estimated_cost=0.01)

        assert exc_info.value.user_id == "user_123"
        assert exc_info.value.period == "daily"

    def test_check_budget_soft_limit_warning(self):
        """Test check_budget warns when soft limit exceeded."""
        controller = CostController()
        controller.set_user_budget("user_123", daily_usd=5.00, hard_limit=False)

        # Record costs over limit
        controller.record_cost("user_123", 6.00, "gpt-4o", "generate")

        result = controller.check_budget("user_123", estimated_cost=0.01)

        assert result["allowed"] is True
        assert len(result["warnings"]) > 0
        assert "exceeded" in result["warnings"][0].lower()

    def test_check_budget_approaching_limit_warning(self):
        """Test check_budget warns when approaching limit (>80%)."""
        controller = CostController()
        controller.set_user_budget("user_123", daily_usd=10.00)

        # Record 85% of budget
        controller.record_cost("user_123", 8.50, "gpt-4o", "generate")

        result = controller.check_budget("user_123", estimated_cost=0.01)

        assert result["allowed"] is True
        assert len(result["warnings"]) > 0


class TestCostRecording:
    """Test cost recording functionality."""

    def test_record_cost(self):
        """Test recording a cost."""
        controller = CostController()

        record = controller.record_cost(
            user_id="user_123",
            cost_usd=0.005,
            model="gpt-4o-mini",
            task_type="classify",
            feature="chat",
        )

        assert record.user_id == "user_123"
        assert record.cost_usd == 0.005
        assert record.model == "gpt-4o-mini"
        assert record.task_type == "classify"
        assert record.feature == "chat"

    def test_records_accumulate(self):
        """Test that records accumulate."""
        controller = CostController()

        controller.record_cost("user_1", 0.01, "gpt-4o", "classify")
        controller.record_cost("user_1", 0.02, "gpt-4o", "summarize")
        controller.record_cost("user_2", 0.03, "gpt-4o-mini", "classify")

        summary = controller.get_summary()
        assert summary["total_records"] == 3


class TestAlerts:
    """Test alert functionality."""

    def test_add_alert(self):
        """Test adding an alert."""
        controller = CostController()

        alert_id = controller.add_alert(
            threshold_usd=100.00,
            period=Period.DAILY,
            callback=lambda x: None,
        )

        assert alert_id.startswith("alert-")

    def test_remove_alert(self):
        """Test removing an alert."""
        controller = CostController()

        alert_id = controller.add_alert(threshold_usd=100.00)

        assert controller.remove_alert(alert_id) is True
        assert controller.remove_alert(alert_id) is False

    def test_alert_triggers_callback(self):
        """Test that alert triggers callback when threshold reached."""
        controller = CostController()

        triggered = []

        def callback(info):
            triggered.append(info)

        controller.add_alert(
            threshold_usd=0.05,
            period=Period.DAILY,
            callback=callback,
        )

        # Record costs to exceed threshold
        controller.record_cost("user_1", 0.06, "gpt-4o", "classify")

        assert len(triggered) == 1
        assert triggered[0]["total"] >= 0.05

    def test_alert_cooldown(self):
        """Test alert cooldown prevents re-triggering."""
        controller = CostController()

        triggered_count = [0]

        def callback(info):
            triggered_count[0] += 1

        controller.add_alert(
            threshold_usd=0.01,
            period=Period.DAILY,
            callback=callback,
            cooldown_minutes=60,
        )

        # Record multiple costs
        controller.record_cost("user_1", 0.02, "gpt-4o", "classify")
        controller.record_cost("user_1", 0.02, "gpt-4o", "classify")
        controller.record_cost("user_1", 0.02, "gpt-4o", "classify")

        # Should only trigger once due to cooldown
        assert triggered_count[0] == 1


class TestCostReports:
    """Test cost reporting functionality."""

    def test_get_cost_report_by_user(self):
        """Test cost report grouped by user."""
        controller = CostController()

        controller.record_cost("user_1", 0.10, "gpt-4o", "classify")
        controller.record_cost("user_1", 0.20, "gpt-4o", "summarize")
        controller.record_cost("user_2", 0.05, "gpt-4o-mini", "classify")

        report = controller.get_cost_report(group_by="user", period=Period.DAILY)

        assert report.total_cost_usd == pytest.approx(0.35, rel=0.01)
        assert report.total_requests == 3
        assert "user_1" in report.breakdown
        assert "user_2" in report.breakdown
        assert report.breakdown["user_1"] == pytest.approx(0.30, rel=0.01)

    def test_get_cost_report_by_model(self):
        """Test cost report grouped by model."""
        controller = CostController()

        controller.record_cost("user_1", 0.10, "gpt-4o", "classify")
        controller.record_cost("user_1", 0.02, "gpt-4o-mini", "classify")
        controller.record_cost("user_2", 0.01, "gpt-4o-mini", "classify")

        report = controller.get_cost_report(group_by="model", period=Period.DAILY)

        assert "gpt-4o" in report.breakdown
        assert "gpt-4o-mini" in report.breakdown
        assert report.breakdown["gpt-4o"] == pytest.approx(0.10, rel=0.01)
        assert report.breakdown["gpt-4o-mini"] == pytest.approx(0.03, rel=0.01)

    def test_get_cost_report_top_spenders(self):
        """Test top spenders in report."""
        controller = CostController()

        controller.record_cost("user_1", 0.50, "gpt-4o", "classify")
        controller.record_cost("user_2", 0.30, "gpt-4o", "classify")
        controller.record_cost("user_3", 0.20, "gpt-4o", "classify")

        report = controller.get_cost_report(group_by="user", top_n=2)

        assert len(report.top_spenders) == 2
        assert report.top_spenders[0][0] == "user_1"  # Highest spender
        assert report.top_spenders[1][0] == "user_2"

    def test_get_user_report(self):
        """Test user-specific report."""
        controller = CostController()
        controller.set_user_budget("user_123", daily_usd=10.00)

        controller.record_cost("user_123", 0.10, "gpt-4o", "classify")
        controller.record_cost("user_123", 0.20, "gpt-4o-mini", "summarize")

        report = controller.get_user_report("user_123")

        assert report["user_id"] == "user_123"
        assert report["total_requests"] == 2
        assert report["total_cost_usd"] == pytest.approx(0.30, rel=0.01)
        assert report["budget_status"]["daily"]["limit"] == 10.00


class TestAnomalyDetection:
    """Test anomaly detection functionality."""

    def test_anomaly_detection_disabled_by_default_needs_history(self):
        """Test that anomaly detection needs history."""
        controller = CostController(enable_anomaly_detection=True)

        alerts = []

        def callback(alert):
            alerts.append(alert)

        controller.enable_anomaly_detection("high", callback)

        # Not enough history - shouldn't trigger
        controller.record_cost("user_1", 0.01, "gpt-4o", "classify")
        controller.record_cost("user_1", 0.01, "gpt-4o", "classify")

        assert len(alerts) == 0  # Not enough history

    def test_anomaly_detection_triggers_on_spike(self):
        """Test anomaly detection triggers on spending spike."""
        controller = CostController(enable_anomaly_detection=True)

        alerts = []

        def callback(alert):
            alerts.append(alert)

        controller.enable_anomaly_detection("high", callback)

        # Build history with some variance (required for stdev calculation)
        costs = [0.01, 0.012, 0.009, 0.011, 0.01, 0.013, 0.008, 0.01, 0.011, 0.009]
        for cost in costs:
            controller.record_cost("user_1", cost, "gpt-4o-mini", "classify")

        # Sudden massive spike (100x normal)
        controller.record_cost("user_1", 1.00, "gpt-4o", "generate")

        # Should detect anomaly
        assert len(alerts) >= 1
        assert alerts[0].anomaly_type == "spike"
        assert alerts[0].user_id == "user_1"

    def test_disable_anomaly_detection(self):
        """Test disabling anomaly detection."""
        controller = CostController(enable_anomaly_detection=True)

        controller.disable_anomaly_detection()
        assert controller._anomaly_enabled is False


class TestUtilities:
    """Test utility functions."""

    def test_get_summary(self):
        """Test getting summary."""
        controller = CostController()
        controller.set_user_budget("user_1", daily_usd=10.00)
        controller.record_cost("user_1", 0.50, "gpt-4o", "classify")

        summary = controller.get_summary()

        assert summary["total_users_with_budgets"] == 1
        assert summary["total_records"] == 1
        assert "total_spend" in summary

    def test_clear_records(self):
        """Test clearing records."""
        controller = CostController()

        controller.record_cost("user_1", 0.01, "gpt-4o", "classify")
        controller.record_cost("user_1", 0.02, "gpt-4o", "classify")

        controller.clear_records()

        assert controller.get_summary()["total_records"] == 0

    def test_export_records_dict(self):
        """Test exporting records as dict."""
        controller = CostController()

        controller.record_cost("user_1", 0.01, "gpt-4o", "classify")

        records = controller.export_records(format="dict")

        assert len(records) == 1
        assert records[0]["user_id"] == "user_1"
        assert records[0]["cost_usd"] == 0.01

    def test_export_records_csv(self):
        """Test exporting records as CSV."""
        controller = CostController()

        controller.record_cost("user_1", 0.01, "gpt-4o", "classify")

        csv = controller.export_records(format="csv")

        assert "user_id" in csv
        assert "cost_usd" in csv
        assert "user_1" in csv


class TestBudgetExceededError:
    """Test BudgetExceededError exception."""

    def test_error_message(self):
        """Test error message format."""
        error = BudgetExceededError("user_123", "daily", 10.50, 10.00)

        assert error.user_id == "user_123"
        assert error.period == "daily"
        assert error.spent == 10.50
        assert error.limit == 10.00
        assert "user_123" in str(error)
        assert "daily" in str(error)
