"""Cost-aware agent router: token budgets, complexity-based model routing,
confidence-based early exit and cost-per-decision analytics."""

from .analytics import Ledger, compare, summarize
from .budget import BudgetPolicy, BudgetTracker, TokenBudget
from .complexity import Complexity, ComplexityEstimator
from .models import CATALOG, HAIKU, OPUS, SONNET, Ladder, ModelSpec, Route, default_ladder
from .router import CostAwareRouter, Decision, SingleModelRouter, Task

__version__ = "0.1.0"

__all__ = [
    "BudgetPolicy", "BudgetTracker", "CATALOG", "Complexity", "ComplexityEstimator",
    "CostAwareRouter", "Decision", "HAIKU", "Ladder", "Ledger", "ModelSpec", "OPUS",
    "Route", "SONNET", "SingleModelRouter", "Task", "TokenBudget", "compare",
    "default_ladder", "summarize",
]
