"""
Conductor - Save money on LLM calls automatically.

Simple usage:
    from conductor import llm

    response = llm("What is 2+2?")
    print(response.text)   # "4"
    print(response.model)  # "gpt-4o-mini" (auto-selected)
    print(response.cost)   # $0.0001
    print(response.saved)  # $0.0009

Sessions (multi-step tracking):
    from conductor import llm, session

    with session(budget_usd=1.00) as s:
        r1 = s.llm("Summarize this")
        r2 = s.llm("Extract from: " + r1.text)

    print(s.total_cost)    # $0.0023
    print(s.total_saved)   # $0.0089

Smart classification (LLM-based):
    from conductor import smart_classify

    result = smart_classify("Analyze this complex document...")
    print(result.complexity_score)  # 0.8 (complex)

Cost control (per-user budgets):
    from conductor import CostController

    controller = CostController()
    controller.set_user_budget("user_123", daily_usd=5.00)
"""

from conductor.simple import llm, llm_dry_run, Response
from conductor.config import get_pricing, set_pricing, get_models, set_models
from conductor.smart_classifier import SmartClassifier, smart_classify, SmartClassification
from conductor.session import Session, StepResult, BudgetExceededError as SessionBudgetError
from conductor.cost_control import (
    CostController,
    CostReport,
    UserBudget,
    Period,
    BudgetExceededError,
    CostRecord,
)
from conductor.storage import InMemoryStorage, SQLiteStorage


def session(
    session_id: str = None,
    budget_usd: float = None,
    budget_strategy: str = "warn",
) -> Session:
    """Create a session to track multiple LLM calls.

    Args:
        session_id: Optional identifier
        budget_usd: Optional budget limit
        budget_strategy: "strict", "warn", or "adaptive"

    Returns:
        Session context manager

    Example:
        with session(budget_usd=1.00) as s:
            r1 = s.llm("Hello")
            r2 = s.llm("World")
        print(s.total_cost)
    """
    return Session(
        llm_func=llm,
        session_id=session_id,
        budget_usd=budget_usd,
        budget_strategy=budget_strategy,
    )


__version__ = "2.1.0"
__all__ = [
    # Simple API
    "llm",
    "llm_dry_run",
    "Response",
    "get_pricing",
    "set_pricing",
    "get_models",
    "set_models",
    # Sessions
    "session",
    "Session",
    "StepResult",
    "SessionBudgetError",
    # Smart classification
    "SmartClassifier",
    "smart_classify",
    "SmartClassification",
    # Cost control
    "CostController",
    "CostReport",
    "UserBudget",
    "Period",
    "BudgetExceededError",
    "CostRecord",
    "InMemoryStorage",
    "SQLiteStorage",
]
