"""
Cost Control for Conductor.

"Never get surprised by an LLM bill again."

Features:
- Per-user cost tracking and budget limits
- Real-time alerts when thresholds are hit
- Cost attribution reports (by user, feature, model)
- Anomaly detection for unusual spending
"""

from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from typing import Optional, Callable, Any
from collections import defaultdict
from enum import Enum
import threading
import statistics


class Period(str, Enum):
    """Time periods for budgets and reports."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ALL_TIME = "all_time"


@dataclass
class UserBudget:
    """Budget configuration for a user."""
    user_id: str
    hourly_usd: Optional[float] = None
    daily_usd: Optional[float] = None
    weekly_usd: Optional[float] = None
    monthly_usd: Optional[float] = None
    hard_limit: bool = True  # If True, block requests over budget. If False, just warn.
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class CostRecord:
    """A single cost record."""
    user_id: str
    cost_usd: float
    model: str
    task_type: str
    feature: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict = field(default_factory=dict)


@dataclass
class Alert:
    """Alert configuration."""
    alert_id: str
    threshold_usd: float
    period: Period
    callback: Callable[[dict], None]
    triggered: bool = False
    last_triggered: Optional[datetime] = None
    cooldown_minutes: int = 60  # Don't re-trigger within this period


@dataclass
class CostReport:
    """Cost report data."""
    period: Period
    start_time: datetime
    end_time: datetime
    total_cost_usd: float
    total_requests: int
    breakdown: dict[str, float]  # key -> cost
    group_by: str
    top_spenders: list[tuple[str, float]]  # [(key, cost), ...]
    average_cost_per_request: float


@dataclass
class AnomalyAlert:
    """Alert for detected anomaly."""
    user_id: str
    anomaly_type: str  # "spike", "unusual_pattern", "new_high"
    current_value: float
    expected_value: float
    deviation_pct: float
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class BudgetExceededError(Exception):
    """Raised when a user's budget has been exceeded."""
    def __init__(self, user_id: str, period: str, spent: float, limit: float):
        self.user_id = user_id
        self.period = period
        self.spent = spent
        self.limit = limit
        super().__init__(
            f"User '{user_id}' exceeded {period} budget: "
            f"${spent:.4f} spent of ${limit:.4f} limit"
        )


class CostController:
    """
    Central controller for LLM cost management.

    Features:
    - Per-user budgets with automatic enforcement
    - Real-time alerts when thresholds are hit
    - Cost attribution reports
    - Anomaly detection

    Example:
        ```python
        controller = CostController()

        # Set user budgets
        controller.set_user_budget("user_123", daily_usd=5.00, monthly_usd=50.00)

        # Set alerts
        controller.add_alert(
            threshold_usd=100.00,
            period=Period.DAILY,
            callback=lambda info: send_slack(f"Daily spend hit ${info['total']}")
        )

        # Check before making a request
        controller.check_budget("user_123", estimated_cost=0.01)

        # Record after request completes
        controller.record_cost(
            user_id="user_123",
            cost_usd=0.008,
            model="gpt-4o-mini",
            task_type="classify"
        )

        # Get reports
        report = controller.get_cost_report(group_by="user", period=Period.DAILY)
        ```
    """

    def __init__(
        self,
        enable_anomaly_detection: bool = True,
        anomaly_sensitivity: str = "medium",  # "low", "medium", "high"
        anomaly_callback: Optional[Callable[[AnomalyAlert], None]] = None,
    ):
        self._budgets: dict[str, UserBudget] = {}
        self._records: list[CostRecord] = []
        self._alerts: dict[str, Alert] = {}
        self._lock = threading.Lock()

        # Anomaly detection
        self._anomaly_enabled = enable_anomaly_detection
        self._anomaly_sensitivity = anomaly_sensitivity
        self._anomaly_callback = anomaly_callback
        self._user_history: dict[str, list[float]] = defaultdict(list)  # user -> daily costs

        # Alert counter for generating IDs
        self._alert_counter = 0

    # =========================================================================
    # Budget Management
    # =========================================================================

    def set_user_budget(
        self,
        user_id: str,
        hourly_usd: Optional[float] = None,
        daily_usd: Optional[float] = None,
        weekly_usd: Optional[float] = None,
        monthly_usd: Optional[float] = None,
        hard_limit: bool = True,
    ) -> UserBudget:
        """
        Set budget limits for a user.

        Args:
            user_id: Unique identifier for the user.
            hourly_usd: Maximum spend per hour.
            daily_usd: Maximum spend per day.
            weekly_usd: Maximum spend per week.
            monthly_usd: Maximum spend per month.
            hard_limit: If True, block requests over budget. If False, just warn.

        Returns:
            The created UserBudget.

        Example:
            ```python
            controller.set_user_budget(
                "user_123",
                daily_usd=5.00,
                monthly_usd=50.00,
                hard_limit=True
            )
            ```
        """
        budget = UserBudget(
            user_id=user_id,
            hourly_usd=hourly_usd,
            daily_usd=daily_usd,
            weekly_usd=weekly_usd,
            monthly_usd=monthly_usd,
            hard_limit=hard_limit,
        )

        with self._lock:
            self._budgets[user_id] = budget

        return budget

    def get_user_budget(self, user_id: str) -> Optional[UserBudget]:
        """Get budget configuration for a user."""
        return self._budgets.get(user_id)

    def remove_user_budget(self, user_id: str) -> bool:
        """Remove budget for a user. Returns True if existed."""
        with self._lock:
            if user_id in self._budgets:
                del self._budgets[user_id]
                return True
            return False

    def check_budget(
        self,
        user_id: str,
        estimated_cost: float = 0.0,
    ) -> dict:
        """
        Check if a user is within budget.

        Args:
            user_id: The user to check.
            estimated_cost: Estimated cost of the upcoming request.

        Returns:
            Dict with budget status: {
                "allowed": bool,
                "user_id": str,
                "current_spend": {"hourly": float, "daily": float, ...},
                "limits": {"hourly": float, "daily": float, ...},
                "warnings": [str, ...]
            }

        Raises:
            BudgetExceededError: If hard_limit is True and budget is exceeded.
        """
        budget = self._budgets.get(user_id)

        if budget is None:
            return {
                "allowed": True,
                "user_id": user_id,
                "current_spend": {},
                "limits": {},
                "warnings": [],
            }

        now = datetime.now(UTC)
        current_spend = self._get_user_spend(user_id, now)
        warnings = []

        # Check each period
        checks = [
            ("hourly", budget.hourly_usd, current_spend.get("hourly", 0)),
            ("daily", budget.daily_usd, current_spend.get("daily", 0)),
            ("weekly", budget.weekly_usd, current_spend.get("weekly", 0)),
            ("monthly", budget.monthly_usd, current_spend.get("monthly", 0)),
        ]

        for period, limit, spent in checks:
            if limit is None:
                continue

            projected = spent + estimated_cost

            if projected > limit:
                if budget.hard_limit:
                    raise BudgetExceededError(user_id, period, spent, limit)
                else:
                    warnings.append(
                        f"{period} budget exceeded: ${spent:.4f} of ${limit:.4f}"
                    )
            elif projected > limit * 0.8:
                warnings.append(
                    f"{period} budget at {(spent/limit)*100:.0f}%: ${spent:.4f} of ${limit:.4f}"
                )

        return {
            "allowed": True,
            "user_id": user_id,
            "current_spend": current_spend,
            "limits": {
                "hourly": budget.hourly_usd,
                "daily": budget.daily_usd,
                "weekly": budget.weekly_usd,
                "monthly": budget.monthly_usd,
            },
            "warnings": warnings,
        }

    def _get_user_spend(self, user_id: str, now: datetime) -> dict[str, float]:
        """Get user's spend for each period."""
        hourly_start = now - timedelta(hours=1)
        daily_start = now - timedelta(days=1)
        weekly_start = now - timedelta(weeks=1)
        monthly_start = now - timedelta(days=30)

        spend = {"hourly": 0.0, "daily": 0.0, "weekly": 0.0, "monthly": 0.0}

        for record in self._records:
            if record.user_id != user_id:
                continue

            if record.timestamp >= hourly_start:
                spend["hourly"] += record.cost_usd
            if record.timestamp >= daily_start:
                spend["daily"] += record.cost_usd
            if record.timestamp >= weekly_start:
                spend["weekly"] += record.cost_usd
            if record.timestamp >= monthly_start:
                spend["monthly"] += record.cost_usd

        return spend

    # =========================================================================
    # Cost Recording
    # =========================================================================

    def record_cost(
        self,
        user_id: str,
        cost_usd: float,
        model: str,
        task_type: str,
        feature: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> CostRecord:
        """
        Record a cost after a request completes.

        Args:
            user_id: The user who made the request.
            cost_usd: Actual cost of the request.
            model: Model used (e.g., "gpt-4o-mini").
            task_type: Type of task (e.g., "classify").
            feature: Optional feature/module name for attribution.
            session_id: Optional session ID for grouping.
            metadata: Optional additional metadata.

        Returns:
            The created CostRecord.
        """
        record = CostRecord(
            user_id=user_id,
            cost_usd=cost_usd,
            model=model,
            task_type=task_type,
            feature=feature,
            session_id=session_id,
            metadata=metadata or {},
        )

        with self._lock:
            self._records.append(record)

            # Update user history for anomaly detection
            self._user_history[user_id].append(cost_usd)

        # Check alerts
        self._check_alerts()

        # Check for anomalies
        if self._anomaly_enabled:
            self._check_anomalies(user_id, cost_usd)

        return record

    # =========================================================================
    # Alerts
    # =========================================================================

    def add_alert(
        self,
        threshold_usd: float,
        period: Period = Period.DAILY,
        callback: Callable[[dict], None] = None,
        cooldown_minutes: int = 60,
    ) -> str:
        """
        Add an alert that triggers when spend exceeds threshold.

        Args:
            threshold_usd: Trigger when total spend exceeds this amount.
            period: Time period for the threshold.
            callback: Function to call when triggered. Receives dict with alert info.
            cooldown_minutes: Don't re-trigger within this period.

        Returns:
            Alert ID for later management.

        Example:
            ```python
            def send_slack(info):
                requests.post(webhook_url, json={"text": f"Spend alert: ${info['total']}"})

            controller.add_alert(
                threshold_usd=100.00,
                period=Period.DAILY,
                callback=send_slack
            )
            ```
        """
        with self._lock:
            self._alert_counter += 1
            alert_id = f"alert-{self._alert_counter}"

            self._alerts[alert_id] = Alert(
                alert_id=alert_id,
                threshold_usd=threshold_usd,
                period=period,
                callback=callback or (lambda x: None),
                cooldown_minutes=cooldown_minutes,
            )

        return alert_id

    def remove_alert(self, alert_id: str) -> bool:
        """Remove an alert. Returns True if existed."""
        with self._lock:
            if alert_id in self._alerts:
                del self._alerts[alert_id]
                return True
            return False

    def _check_alerts(self):
        """Check all alerts and trigger if needed."""
        now = datetime.now(UTC)

        for alert in self._alerts.values():
            # Check cooldown
            if alert.last_triggered:
                cooldown_end = alert.last_triggered + timedelta(minutes=alert.cooldown_minutes)
                if now < cooldown_end:
                    continue

            # Get total spend for period
            total = self._get_total_spend(alert.period)

            if total >= alert.threshold_usd:
                alert.triggered = True
                alert.last_triggered = now

                # Call callback in a thread to not block
                info = {
                    "alert_id": alert.alert_id,
                    "threshold": alert.threshold_usd,
                    "total": total,
                    "period": alert.period.value,
                    "timestamp": now.isoformat(),
                }

                try:
                    alert.callback(info)
                except Exception:
                    pass  # Don't let callback errors break the system

    def _get_total_spend(self, period: Period) -> float:
        """Get total spend for a period."""
        now = datetime.now(UTC)

        if period == Period.HOURLY:
            start = now - timedelta(hours=1)
        elif period == Period.DAILY:
            start = now - timedelta(days=1)
        elif period == Period.WEEKLY:
            start = now - timedelta(weeks=1)
        elif period == Period.MONTHLY:
            start = now - timedelta(days=30)
        else:
            start = datetime.min.replace(tzinfo=UTC)

        return sum(
            r.cost_usd for r in self._records
            if r.timestamp >= start
        )

    # =========================================================================
    # Reports
    # =========================================================================

    def get_cost_report(
        self,
        group_by: str = "user",  # "user", "model", "task_type", "feature"
        period: Period = Period.DAILY,
        top_n: int = 10,
    ) -> CostReport:
        """
        Generate a cost attribution report.

        Args:
            group_by: How to group costs ("user", "model", "task_type", "feature").
            period: Time period for the report.
            top_n: Number of top spenders to include.

        Returns:
            CostReport with breakdown and statistics.

        Example:
            ```python
            report = controller.get_cost_report(group_by="user", period=Period.DAILY)
            print(f"Total: ${report.total_cost_usd:.2f}")
            for user, cost in report.top_spenders:
                print(f"  {user}: ${cost:.4f}")
            ```
        """
        now = datetime.now(UTC)

        if period == Period.HOURLY:
            start = now - timedelta(hours=1)
        elif period == Period.DAILY:
            start = now - timedelta(days=1)
        elif period == Period.WEEKLY:
            start = now - timedelta(weeks=1)
        elif period == Period.MONTHLY:
            start = now - timedelta(days=30)
        else:
            start = datetime.min.replace(tzinfo=UTC)

        # Filter records
        records = [r for r in self._records if r.timestamp >= start]

        # Group costs
        breakdown: dict[str, float] = defaultdict(float)

        for record in records:
            if group_by == "user":
                key = record.user_id
            elif group_by == "model":
                key = record.model
            elif group_by == "task_type":
                key = record.task_type
            elif group_by == "feature":
                key = record.feature or "unknown"
            else:
                key = "all"

            breakdown[key] += record.cost_usd

        # Calculate statistics
        total_cost = sum(breakdown.values())
        total_requests = len(records)
        avg_cost = total_cost / total_requests if total_requests > 0 else 0

        # Top spenders
        top_spenders = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)[:top_n]

        return CostReport(
            period=period,
            start_time=start,
            end_time=now,
            total_cost_usd=total_cost,
            total_requests=total_requests,
            breakdown=dict(breakdown),
            group_by=group_by,
            top_spenders=top_spenders,
            average_cost_per_request=avg_cost,
        )

    def get_user_report(self, user_id: str) -> dict:
        """
        Get detailed cost report for a specific user.

        Returns:
            Dict with user's spending details.
        """
        now = datetime.now(UTC)

        user_records = [r for r in self._records if r.user_id == user_id]

        # Spend by period
        spend = self._get_user_spend(user_id, now)

        # Spend by model
        by_model: dict[str, float] = defaultdict(float)
        for r in user_records:
            by_model[r.model] += r.cost_usd

        # Spend by task type
        by_task: dict[str, float] = defaultdict(float)
        for r in user_records:
            by_task[r.task_type] += r.cost_usd

        # Budget status
        budget = self._budgets.get(user_id)
        budget_status = None
        if budget:
            budget_status = {
                "hourly": {"limit": budget.hourly_usd, "spent": spend.get("hourly", 0)},
                "daily": {"limit": budget.daily_usd, "spent": spend.get("daily", 0)},
                "weekly": {"limit": budget.weekly_usd, "spent": spend.get("weekly", 0)},
                "monthly": {"limit": budget.monthly_usd, "spent": spend.get("monthly", 0)},
            }

        return {
            "user_id": user_id,
            "total_requests": len(user_records),
            "total_cost_usd": sum(r.cost_usd for r in user_records),
            "spend_by_period": spend,
            "spend_by_model": dict(by_model),
            "spend_by_task_type": dict(by_task),
            "budget_status": budget_status,
        }

    # =========================================================================
    # Anomaly Detection
    # =========================================================================

    def enable_anomaly_detection(
        self,
        sensitivity: str = "medium",
        callback: Optional[Callable[[AnomalyAlert], None]] = None,
    ):
        """
        Enable anomaly detection for unusual spending patterns.

        Args:
            sensitivity: "low", "medium", or "high"
            callback: Function to call when anomaly detected.
        """
        self._anomaly_enabled = True
        self._anomaly_sensitivity = sensitivity
        self._anomaly_callback = callback

    def disable_anomaly_detection(self):
        """Disable anomaly detection."""
        self._anomaly_enabled = False

    def _check_anomalies(self, user_id: str, cost: float):
        """Check for spending anomalies for a user."""
        history = self._user_history[user_id]

        if len(history) < 5:
            return  # Need enough history

        # Get threshold based on sensitivity
        sensitivity_thresholds = {
            "low": 3.0,    # 3 standard deviations
            "medium": 2.0, # 2 standard deviations
            "high": 1.5,   # 1.5 standard deviations
        }
        threshold = sensitivity_thresholds.get(self._anomaly_sensitivity, 2.0)

        # Calculate statistics (excluding current cost)
        historical = history[:-1]
        mean = statistics.mean(historical)
        stdev = statistics.stdev(historical) if len(historical) > 1 else 0

        if stdev == 0:
            return  # Can't detect anomalies with no variance

        # Check if current cost is anomalous
        z_score = (cost - mean) / stdev

        if z_score > threshold:
            deviation_pct = ((cost - mean) / mean) * 100 if mean > 0 else 0

            alert = AnomalyAlert(
                user_id=user_id,
                anomaly_type="spike",
                current_value=cost,
                expected_value=mean,
                deviation_pct=deviation_pct,
                message=f"User '{user_id}' spending spike: ${cost:.4f} is {deviation_pct:.0f}% above average (${mean:.4f})",
            )

            if self._anomaly_callback:
                try:
                    self._anomaly_callback(alert)
                except Exception:
                    pass

    # =========================================================================
    # Utility
    # =========================================================================

    def get_summary(self) -> dict:
        """Get overall summary of cost control status."""
        now = datetime.now(UTC)

        return {
            "total_users_with_budgets": len(self._budgets),
            "total_records": len(self._records),
            "active_alerts": len(self._alerts),
            "anomaly_detection_enabled": self._anomaly_enabled,
            "total_spend": {
                "hourly": self._get_total_spend(Period.HOURLY),
                "daily": self._get_total_spend(Period.DAILY),
                "weekly": self._get_total_spend(Period.WEEKLY),
                "monthly": self._get_total_spend(Period.MONTHLY),
                "all_time": self._get_total_spend(Period.ALL_TIME),
            },
            "timestamp": now.isoformat(),
        }

    def clear_records(self, before: Optional[datetime] = None):
        """
        Clear cost records.

        Args:
            before: If provided, only clear records before this time.
                   If None, clear all records.
        """
        with self._lock:
            if before is None:
                self._records.clear()
            else:
                self._records = [r for r in self._records if r.timestamp >= before]

    def export_records(self, format: str = "dict") -> list:
        """
        Export all cost records.

        Args:
            format: "dict" for list of dicts, "csv" for CSV string.

        Returns:
            List of record dicts or CSV string.
        """
        records = [
            {
                "user_id": r.user_id,
                "cost_usd": r.cost_usd,
                "model": r.model,
                "task_type": r.task_type,
                "feature": r.feature,
                "session_id": r.session_id,
                "timestamp": r.timestamp.isoformat(),
            }
            for r in self._records
        ]

        if format == "dict":
            return records
        elif format == "csv":
            if not records:
                return "user_id,cost_usd,model,task_type,feature,session_id,timestamp\n"

            headers = list(records[0].keys())
            lines = [",".join(headers)]
            for r in records:
                lines.append(",".join(str(r.get(h, "")) for h in headers))
            return "\n".join(lines)

        return records


# Global instance for convenience
_default_controller: Optional[CostController] = None


def get_cost_controller() -> CostController:
    """Get or create the default cost controller."""
    global _default_controller
    if _default_controller is None:
        _default_controller = CostController()
    return _default_controller
