"""
Cost Control - Per-user budgets and cost tracking.

Track spending by user and enforce budget limits.
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Any, Callable
from enum import Enum
from conductor.models import UserBudget, CostRecord
from conductor.storage import InMemoryStorage, StorageBackend


class Period(Enum):
    """Time period for budgets and reports."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class BudgetExceededError(Exception):
    """Raised when a user's budget is exceeded."""

    def __init__(self, user_id: str, period: str, spent: float, limit: float):
        self.user_id = user_id
        self.period = period
        self.spent = spent
        self.limit = limit
        super().__init__(
            f"User '{user_id}' exceeded {period} budget: ${spent:.4f} spent of ${limit:.2f} limit"
        )


@dataclass
class CostReport:
    """Cost report for a time period."""
    period: Period
    total_cost_usd: float
    total_requests: int
    breakdown: Dict[str, float]
    top_spenders: List[tuple]


class CostController:
    """Track and control costs per user.

    Example:
        controller = CostController()

        # Set user budget
        controller.set_user_budget("user_123", daily_usd=5.00)

        # Check before each request
        controller.check_budget("user_123", estimated_cost=0.01)

        # Record after each request
        controller.record_cost("user_123", 0.008, "gpt-4o-mini", "chat")

        # Get reports
        report = controller.get_cost_report(group_by="user", period=Period.DAILY)
    """

    def __init__(self, storage: Optional[StorageBackend] = None):
        """Initialize cost controller.

        Args:
            storage: Optional storage backend (defaults to in-memory).
        """
        self._storage: StorageBackend = storage or InMemoryStorage()

    def set_user_budget(
        self,
        user_id: str,
        hourly_usd: Optional[float] = None,
        daily_usd: Optional[float] = None,
        weekly_usd: Optional[float] = None,
        monthly_usd: Optional[float] = None,
        hard_limit: bool = True,
    ) -> UserBudget:
        """Set budget for a user.

        Args:
            user_id: User identifier
            hourly_usd: Hourly budget limit
            daily_usd: Daily budget limit
            weekly_usd: Weekly budget limit
            monthly_usd: Monthly budget limit
            hard_limit: If True, block requests when exceeded

        Returns:
            The created UserBudget
        """
        budget = UserBudget(
            user_id=user_id,
            hourly_usd=hourly_usd,
            daily_usd=daily_usd,
            weekly_usd=weekly_usd,
            monthly_usd=monthly_usd,
            hard_limit=hard_limit,
        )
        return self._storage.set_budget(budget)

    def get_user_budget(self, user_id: str) -> Optional[UserBudget]:
        """Get budget for a user."""
        return self._storage.get_budget(user_id)

    def remove_user_budget(self, user_id: str) -> bool:
        """Remove budget for a user."""
        return self._storage.remove_budget(user_id)

    def record_cost(
        self,
        user_id: str,
        cost_usd: float,
        model: str,
        task_type: str,
        feature: Optional[str] = None,
    ) -> CostRecord:
        """Record a cost event.

        Args:
            user_id: User who made the request
            cost_usd: Cost of the request
            model: Model used
            task_type: Type of task
            feature: Optional feature tag

        Returns:
            The created CostRecord
        """
        record = CostRecord(
            user_id=user_id,
            cost_usd=cost_usd,
            model=model,
            task_type=task_type,
            feature=feature,
        )
        return self._storage.add_record(record)

    def check_budget(
        self,
        user_id: str,
        estimated_cost: float = 0.0,
    ) -> Dict[str, Any]:
        """Check if a user is within budget.

        Args:
            user_id: User to check
            estimated_cost: Estimated cost of next request

        Returns:
            Dict with 'allowed', 'warnings', and 'current_spend'

        Raises:
            BudgetExceededError: If hard limit exceeded
        """
        budget = self._storage.get_budget(user_id)
        if not budget:
            return {"allowed": True, "warnings": [], "current_spend": {}}

        now = datetime.now(timezone.utc)
        warnings = []
        current_spend = {}

        # Check each period
        periods = [
            ("hourly", budget.hourly_usd, timedelta(hours=1)),
            ("daily", budget.daily_usd, timedelta(days=1)),
            ("weekly", budget.weekly_usd, timedelta(weeks=1)),
            ("monthly", budget.monthly_usd, timedelta(days=30)),
        ]

        for period_name, limit, delta in periods:
            if limit is None:
                continue

            cutoff = now - delta
            records = self._storage.list_records_since(cutoff)
            spent = sum(r.cost_usd for r in records if r.user_id == user_id)
            current_spend[period_name] = spent

            # Check if would exceed
            if spent + estimated_cost > limit:
                if budget.hard_limit:
                    raise BudgetExceededError(user_id, period_name, spent, limit)
                else:
                    warnings.append(f"{period_name} budget exceeded: ${spent:.4f} of ${limit:.2f}")
            elif spent > limit * 0.8:
                warnings.append(f"Approaching {period_name} limit: ${spent:.4f} of ${limit:.2f}")

        return {
            "allowed": True,
            "warnings": warnings,
            "current_spend": current_spend,
        }

    def get_cost_report(
        self,
        group_by: str = "user",
        period: Period = Period.DAILY,
        top_n: int = 10,
    ) -> CostReport:
        """Get cost report.

        Args:
            group_by: "user", "model", or "task_type"
            period: Time period to report on
            top_n: Number of top spenders to include

        Returns:
            CostReport with breakdown and top spenders
        """
        now = datetime.now(timezone.utc)

        # Get period cutoff
        if period == Period.HOURLY:
            cutoff = now - timedelta(hours=1)
        elif period == Period.DAILY:
            cutoff = now - timedelta(days=1)
        elif period == Period.WEEKLY:
            cutoff = now - timedelta(weeks=1)
        else:
            cutoff = now - timedelta(days=30)

        # Filter records
        records = self._storage.list_records_since(cutoff)

        # Group
        breakdown: Dict[str, float] = {}
        for r in records:
            if group_by == "user":
                key = r.user_id
            elif group_by == "model":
                key = r.model
            else:
                key = r.task_type

            breakdown[key] = breakdown.get(key, 0) + r.cost_usd

        # Sort for top spenders
        sorted_items = sorted(breakdown.items(), key=lambda x: -x[1])
        top_spenders = sorted_items[:top_n]

        return CostReport(
            period=period,
            total_cost_usd=sum(r.cost_usd for r in records),
            total_requests=len(records),
            breakdown=breakdown,
            top_spenders=top_spenders,
        )

    def get_user_report(self, user_id: str) -> Dict[str, Any]:
        """Get report for a specific user."""
        records = [r for r in self._storage.list_records() if r.user_id == user_id]

        budget = self._storage.get_budget(user_id)
        budget_status = {}
        if budget:
            now = datetime.now(timezone.utc)
            for period_name, limit, delta in [
                ("hourly", budget.hourly_usd, timedelta(hours=1)),
                ("daily", budget.daily_usd, timedelta(days=1)),
                ("weekly", budget.weekly_usd, timedelta(weeks=1)),
                ("monthly", budget.monthly_usd, timedelta(days=30)),
            ]:
                if limit:
                    cutoff = now - delta
                    spent = sum(r.cost_usd for r in records if r.timestamp >= cutoff)
                    budget_status[period_name] = {"limit": limit, "spent": spent}

        return {
            "user_id": user_id,
            "total_requests": len(records),
            "total_cost_usd": sum(r.cost_usd for r in records),
            "budget_status": budget_status,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get overall summary."""
        return {
            "total_users_with_budgets": len(self._storage.list_budgets()),
            "total_records": len(self._storage.list_records()),
            "total_spend": sum(r.cost_usd for r in self._storage.list_records()),
        }

    def clear_records(self):
        """Clear all cost records."""
        self._storage.clear_records()

    def export_records(self, format: str = "dict") -> Any:
        """Export records.

        Args:
            format: "dict" or "csv"

        Returns:
            List of dicts or CSV string
        """
        records = [
            {
                "user_id": r.user_id,
                "cost_usd": r.cost_usd,
                "model": r.model,
                "task_type": r.task_type,
                "timestamp": r.timestamp.isoformat(),
                "feature": r.feature,
                "record_id": r.record_id,
            }
            for r in self._storage.list_records()
        ]

        if format == "csv":
            if not records:
                return "user_id,cost_usd,model,task_type,timestamp,feature\n"
            headers = records[0].keys()
            lines = [",".join(headers)]
            for r in records:
                lines.append(",".join(str(r.get(h, "")) for h in headers))
            return "\n".join(lines)

        return records
