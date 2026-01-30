"""
Conductor: LLM Control Plane for intelligent model routing.

Automatically routes LLM requests to the optimal model/strategy 
to minimize cost while meeting latency SLAs and quality thresholds.
"""

from conductor.control_plane import Conductor
from conductor.schemas import (
    Request,
    RequestFeatures,
    Decision,
    Outcome,
    Strategy,
    TaskType,
    UrgencyTier,
    LatencyPercentile,
)
from conductor.strategies import STRATEGIES, get_strategy
from conductor.coefficients import Coefficients, load_coefficients
from conductor.validation import ValidationError
from conductor.smart_classifier import SmartClassifier, smart_classify
from conductor.session import Session, Workflow, WorkflowBuilder
from conductor.session import BudgetExceededError as SessionBudgetExceededError
from conductor.cost_control import (
    CostController,
    CostReport,
    UserBudget,
    Period,
    BudgetExceededError,
    AnomalyAlert,
)

__version__ = "1.3.0"
__all__ = [
    # Core
    "Conductor",
    "Request",
    "RequestFeatures",
    "Decision",
    "Outcome",
    "Strategy",
    "TaskType",
    "UrgencyTier",
    "LatencyPercentile",
    "STRATEGIES",
    "get_strategy",
    "Coefficients",
    "load_coefficients",
    # Validation
    "ValidationError",
    # Smart classification
    "SmartClassifier",
    "smart_classify",
    # Sessions & Workflows
    "Session",
    "Workflow",
    "WorkflowBuilder",
    # Cost Control
    "CostController",
    "CostReport",
    "UserBudget",
    "Period",
    "BudgetExceededError",
    "AnomalyAlert",
]
