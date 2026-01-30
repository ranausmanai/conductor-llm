"""Shared data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import uuid


@dataclass
class UserBudget:
    """Budget configuration for a user."""
    user_id: str
    hourly_usd: Optional[float] = None
    daily_usd: Optional[float] = None
    weekly_usd: Optional[float] = None
    monthly_usd: Optional[float] = None
    hard_limit: bool = True  # If True, block requests. If False, just warn.


@dataclass
class CostRecord:
    """Record of a single cost event."""
    user_id: str
    cost_usd: float
    model: str
    task_type: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    feature: Optional[str] = None
    record_id: str = field(default_factory=lambda: uuid.uuid4().hex)
