"""
Session - Track costs across multiple LLM calls.

Groups related calls together for cost tracking and budget management.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Callable
from contextlib import contextmanager


class BudgetExceededError(Exception):
    """Raised when a session budget is exceeded."""

    def __init__(self, session_id: str, spent: float, budget: float):
        self.session_id = session_id
        self.spent = spent
        self.budget = budget
        super().__init__(
            f"Session '{session_id}' budget exceeded: ${spent:.4f} spent of ${budget:.2f} budget"
        )


@dataclass
class StepResult:
    """Result of a single step in a session."""
    step_id: str
    step_number: int
    prompt: str
    text: str
    model: str
    cost: float
    latency_ms: int
    timestamp: datetime
    baseline_cost: float = 0.0
    saved: float = 0.0


class Session:
    """Track costs across multiple LLM calls.

    Example:
        with conductor.session("my-workflow", budget_usd=1.00) as s:
            r1 = s.llm("Summarize this")
            r2 = s.llm("Extract entities from: " + r1.text)

        print(s.total_cost)    # $0.0023
        print(s.total_saved)   # $0.0089
        print(s.savings_pct)   # 79%
    """

    def __init__(
        self,
        llm_func: Callable,
        session_id: Optional[str] = None,
        budget_usd: Optional[float] = None,
        budget_strategy: str = "warn",
    ):
        """Initialize a session.

        Args:
            llm_func: The llm function to use for calls
            session_id: Optional identifier for this session
            budget_usd: Optional budget limit
            budget_strategy: "strict" (raise error), "warn" (log warning), "adaptive" (use cheaper models)
        """
        self.llm_func = llm_func
        self.session_id = session_id or f"session-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        self.budget_usd = budget_usd
        self.budget_strategy = budget_strategy

        self._steps: List[StepResult] = []
        self._started = False
        self._ended = False
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None

    def __enter__(self):
        """Start the session."""
        self._started = True
        self._start_time = datetime.now(timezone.utc)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the session."""
        self._ended = True
        self._end_time = datetime.now(timezone.utc)
        return False

    def llm(self, prompt: str, quality: str = "auto", **kwargs) -> Any:
        """Make an LLM call within this session.

        Args:
            prompt: The prompt
            quality: "auto", "high", or "low"
            **kwargs: Additional arguments to pass to llm()

        Returns:
            Response from the LLM
        """
        if not self._started:
            raise RuntimeError("Session not started. Use 'with' statement.")

        # Check budget before call
        if self.budget_usd is not None:
            remaining = self.budget_remaining
            if remaining <= 0:
                if self.budget_strategy == "strict":
                    raise BudgetExceededError(self.session_id, self.total_cost, self.budget_usd)
                elif self.budget_strategy == "adaptive":
                    quality = "low"  # Force cheap model

        # Make the call
        import time
        start = time.time()
        response = self.llm_func(prompt, quality=quality, **kwargs)
        latency_ms = int((time.time() - start) * 1000)

        # Record step
        step = StepResult(
            step_id=f"step-{len(self._steps) + 1}",
            step_number=len(self._steps) + 1,
            prompt=prompt,
            text=response.text,
            model=response.model,
            cost=response.cost,
            latency_ms=latency_ms,
            timestamp=datetime.now(timezone.utc),
            baseline_cost=response.baseline_cost,
            saved=response.saved,
        )
        self._steps.append(step)

        return response

    @property
    def total_cost(self) -> float:
        """Total cost of all steps."""
        return sum(s.cost for s in self._steps)

    @property
    def total_saved(self) -> float:
        """Total amount saved vs baseline."""
        return sum(s.saved for s in self._steps)

    @property
    def baseline_cost(self) -> float:
        """What it would have cost with expensive model for everything."""
        return sum(s.baseline_cost for s in self._steps)

    @property
    def savings_pct(self) -> float:
        """Percentage saved vs baseline."""
        if self.baseline_cost == 0:
            return 0.0
        return (self.total_saved / self.baseline_cost) * 100

    @property
    def budget_remaining(self) -> Optional[float]:
        """Remaining budget, if set."""
        if self.budget_usd is None:
            return None
        return max(0, self.budget_usd - self.total_cost)

    @property
    def budget_utilization(self) -> Optional[float]:
        """Budget utilization percentage, if budget set."""
        if self.budget_usd is None or self.budget_usd == 0:
            return None
        return (self.total_cost / self.budget_usd) * 100

    @property
    def total_latency_ms(self) -> int:
        """Total latency across all steps."""
        return sum(s.latency_ms for s in self._steps)

    @property
    def steps(self) -> List[StepResult]:
        """All steps in this session."""
        return self._steps.copy()

    @property
    def step_count(self) -> int:
        """Number of steps completed."""
        return len(self._steps)

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        models = {}
        for s in self._steps:
            if s.model not in models:
                models[s.model] = {"count": 0, "cost": 0}
            models[s.model]["count"] += 1
            models[s.model]["cost"] += s.cost

        return {
            "session_id": self.session_id,
            "step_count": self.step_count,
            "total_cost": self.total_cost,
            "total_saved": self.total_saved,
            "baseline_cost": self.baseline_cost,
            "savings_pct": self.savings_pct,
            "total_latency_ms": self.total_latency_ms,
            "budget_usd": self.budget_usd,
            "budget_remaining": self.budget_remaining,
            "models": models,
        }

    def get_report(self) -> str:
        """Get a human-readable report."""
        lines = [
            "=" * 50,
            "   Conductor Session Report",
            "=" * 50,
            f"Session: {self.session_id}",
            f"Steps: {self.step_count}",
            "",
            "Cost Breakdown:",
            f"  Actual cost:    ${self.total_cost:.4f}",
            f"  Baseline cost:  ${self.baseline_cost:.4f}",
            f"  You saved:      ${self.total_saved:.4f} ({self.savings_pct:.1f}%)",
        ]

        if self.budget_usd:
            lines.extend([
                "",
                f"Budget: ${self.budget_usd:.2f}",
                f"Remaining: ${self.budget_remaining:.4f}",
            ])

        lines.extend([
            "",
            "Models Used:",
        ])

        models = {}
        for s in self._steps:
            if s.model not in models:
                models[s.model] = 0
            models[s.model] += 1

        for model, count in models.items():
            lines.append(f"  {model}: {count} calls")

        lines.append("=" * 50)
        return "\n".join(lines)

    def __repr__(self):
        return f"Session(id='{self.session_id}', steps={self.step_count}, cost=${self.total_cost:.4f}, saved=${self.total_saved:.4f})"
