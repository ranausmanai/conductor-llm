"""Tests for storage backends."""

from datetime import datetime, timezone
import tempfile

from conductor.cost_control import CostController
from conductor.models import UserBudget
from conductor.storage import SQLiteStorage


def test_sqlite_storage_persists_records():
    """SQLite storage should persist records across instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = f"{tmpdir}/conductor.db"

        storage = SQLiteStorage(db_path=db_path)
        controller = CostController(storage=storage)
        controller.record_cost("user_1", 0.01, "gpt-4o", "chat")
        storage.close()

        storage2 = SQLiteStorage(db_path=db_path)
        controller2 = CostController(storage=storage2)
        records = controller2.export_records(format="dict")
        assert len(records) == 1
        assert records[0]["user_id"] == "user_1"
        storage2.close()


def test_sqlite_storage_budgets_roundtrip():
    """SQLite storage should store and load budgets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = f"{tmpdir}/conductor.db"
        storage = SQLiteStorage(db_path=db_path)
        controller = CostController(storage=storage)

        controller.set_user_budget("user_1", daily_usd=5.0, hard_limit=False)

        budget = controller.get_user_budget("user_1")
        assert isinstance(budget, UserBudget)
        assert budget.daily_usd == 5.0
        assert budget.hard_limit is False
        storage.close()


def test_sqlite_storage_date_filtering():
    """SQLite storage should support cutoff filtering."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = f"{tmpdir}/conductor.db"
        storage = SQLiteStorage(db_path=db_path)
        controller = CostController(storage=storage)

        controller.record_cost("user_1", 0.01, "gpt-4o", "chat")
        now = datetime.now(timezone.utc)
        recent = storage.list_records_since(now.replace(year=2000))
        assert len(recent) == 1
        storage.close()
