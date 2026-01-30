"""Storage backends for budgets and cost records."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Protocol
import sqlite3

from conductor.models import UserBudget, CostRecord


class StorageBackend(Protocol):
    """Storage backend interface."""

    def set_budget(self, budget: UserBudget) -> UserBudget:
        ...

    def get_budget(self, user_id: str) -> Optional[UserBudget]:
        ...

    def remove_budget(self, user_id: str) -> bool:
        ...

    def list_budgets(self) -> List[UserBudget]:
        ...

    def add_record(self, record: CostRecord) -> CostRecord:
        ...

    def list_records(self) -> List[CostRecord]:
        ...

    def list_records_since(self, cutoff: datetime) -> List[CostRecord]:
        ...

    def clear_records(self) -> None:
        ...


class InMemoryStorage:
    """In-memory storage backend (default)."""

    def __init__(self):
        self._budgets: Dict[str, UserBudget] = {}
        self._records: List[CostRecord] = []

    def set_budget(self, budget: UserBudget) -> UserBudget:
        self._budgets[budget.user_id] = budget
        return budget

    def get_budget(self, user_id: str) -> Optional[UserBudget]:
        return self._budgets.get(user_id)

    def remove_budget(self, user_id: str) -> bool:
        if user_id in self._budgets:
            del self._budgets[user_id]
            return True
        return False

    def list_budgets(self) -> List[UserBudget]:
        return list(self._budgets.values())

    def add_record(self, record: CostRecord) -> CostRecord:
        self._records.append(record)
        return record

    def list_records(self) -> List[CostRecord]:
        return list(self._records)

    def list_records_since(self, cutoff: datetime) -> List[CostRecord]:
        return [r for r in self._records if r.timestamp >= cutoff]

    def clear_records(self) -> None:
        self._records.clear()


class SQLiteStorage:
    """SQLite-backed storage backend."""

    def __init__(self, db_path: str = "conductor.db"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS budgets (
                user_id TEXT PRIMARY KEY,
                hourly_usd REAL,
                daily_usd REAL,
                weekly_usd REAL,
                monthly_usd REAL,
                hard_limit INTEGER
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                record_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                cost_usd REAL NOT NULL,
                model TEXT NOT NULL,
                task_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                feature TEXT
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_records_user ON records(user_id)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_records_time ON records(timestamp)")
        self._conn.commit()

    def set_budget(self, budget: UserBudget) -> UserBudget:
        self._conn.execute(
            """
            INSERT INTO budgets (user_id, hourly_usd, daily_usd, weekly_usd, monthly_usd, hard_limit)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                hourly_usd=excluded.hourly_usd,
                daily_usd=excluded.daily_usd,
                weekly_usd=excluded.weekly_usd,
                monthly_usd=excluded.monthly_usd,
                hard_limit=excluded.hard_limit
            """,
            (
                budget.user_id,
                budget.hourly_usd,
                budget.daily_usd,
                budget.weekly_usd,
                budget.monthly_usd,
                1 if budget.hard_limit else 0,
            ),
        )
        self._conn.commit()
        return budget

    def get_budget(self, user_id: str) -> Optional[UserBudget]:
        row = self._conn.execute(
            "SELECT * FROM budgets WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if not row:
            return None
        return UserBudget(
            user_id=row["user_id"],
            hourly_usd=row["hourly_usd"],
            daily_usd=row["daily_usd"],
            weekly_usd=row["weekly_usd"],
            monthly_usd=row["monthly_usd"],
            hard_limit=bool(row["hard_limit"]),
        )

    def remove_budget(self, user_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM budgets WHERE user_id = ?", (user_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def list_budgets(self) -> List[UserBudget]:
        rows = self._conn.execute("SELECT * FROM budgets").fetchall()
        return [
            UserBudget(
                user_id=row["user_id"],
                hourly_usd=row["hourly_usd"],
                daily_usd=row["daily_usd"],
                weekly_usd=row["weekly_usd"],
                monthly_usd=row["monthly_usd"],
                hard_limit=bool(row["hard_limit"]),
            )
            for row in rows
        ]

    def add_record(self, record: CostRecord) -> CostRecord:
        self._conn.execute(
            """
            INSERT INTO records (record_id, user_id, cost_usd, model, task_type, timestamp, feature)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.record_id,
                record.user_id,
                record.cost_usd,
                record.model,
                record.task_type,
                record.timestamp.isoformat(),
                record.feature,
            ),
        )
        self._conn.commit()
        return record

    def _row_to_record(self, row: sqlite3.Row) -> CostRecord:
        timestamp = datetime.fromisoformat(row["timestamp"])
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return CostRecord(
            user_id=row["user_id"],
            cost_usd=row["cost_usd"],
            model=row["model"],
            task_type=row["task_type"],
            timestamp=timestamp,
            feature=row["feature"],
            record_id=row["record_id"],
        )

    def list_records(self) -> List[CostRecord]:
        rows = self._conn.execute("SELECT * FROM records ORDER BY timestamp ASC").fetchall()
        return [self._row_to_record(row) for row in rows]

    def list_records_since(self, cutoff: datetime) -> List[CostRecord]:
        rows = self._conn.execute(
            "SELECT * FROM records WHERE timestamp >= ? ORDER BY timestamp ASC",
            (cutoff.isoformat(),),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def clear_records(self) -> None:
        self._conn.execute("DELETE FROM records")
        self._conn.commit()

    def export_records(self) -> List[Dict[str, str]]:
        return [asdict(r) for r in self.list_records()]

    def close(self) -> None:
        self._conn.close()
