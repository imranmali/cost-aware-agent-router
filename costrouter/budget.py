"""Per-task token and dollar budgets."""

from __future__ import annotations

from dataclasses import dataclass, field

from .complexity import Complexity
from .models import Route


@dataclass(frozen=True)
class TokenBudget:
    """Hard limits for one task across all of its attempts."""

    max_output_tokens: int  # per attempt; sent as max_tokens
    max_total_tokens: int  # input + output, summed over attempts
    max_cost_usd: float  # summed over attempts


@dataclass
class BudgetPolicy:
    """Maps a task's complexity tier to a budget.

    Defaults are sized so an easy task can never burn hard-task money, while a
    hard task always has room for at least one top-model attempt.
    """

    output_tokens: tuple[int, int, int] = (1_024, 4_096, 16_000)
    total_tokens: tuple[int, int, int] = (8_000, 40_000, 160_000)
    cost_usd: tuple[float, float, float] = (0.01, 0.12, 0.60)

    def for_tier(self, tier: int) -> TokenBudget:
        return TokenBudget(self.output_tokens[tier], self.total_tokens[tier], self.cost_usd[tier])

    def for_task(self, c: Complexity) -> TokenBudget:
        return self.for_tier(c.tier)


@dataclass
class BudgetTracker:
    """Tracks spend for one task and answers 'can we afford another attempt?'."""

    budget: TokenBudget
    tokens_used: int = 0
    cost_used: float = 0.0
    attempts: int = 0
    log: list = field(default_factory=list)

    @property
    def tokens_left(self) -> int:
        return self.budget.max_total_tokens - self.tokens_used

    @property
    def cost_left(self) -> float:
        return self.budget.max_cost_usd - self.cost_used

    def max_tokens_for(self, input_tokens: int) -> int:
        """Output cap for the next attempt: the per-attempt cap, shrunk to fit what is left."""
        return max(0, min(self.budget.max_output_tokens, self.tokens_left - input_tokens))

    def can_afford(self, route: Route, input_tokens: int, expected_output: int) -> bool:
        if self.max_tokens_for(input_tokens) < min(expected_output, 256):
            return False
        return route.cost(input_tokens, expected_output) <= self.cost_left

    def raise_to(self, budget: TokenBudget) -> None:
        """Widen the limits (never shrink them); spend so far still counts."""
        self.budget = TokenBudget(
            max(self.budget.max_output_tokens, budget.max_output_tokens),
            max(self.budget.max_total_tokens, budget.max_total_tokens),
            max(self.budget.max_cost_usd, budget.max_cost_usd),
        )

    def charge(self, route: Route, input_tokens: int, output_tokens: int) -> float:
        cost = route.cost(input_tokens, output_tokens)
        self.tokens_used += input_tokens + output_tokens
        self.cost_used += cost
        self.attempts += 1
        self.log.append((route.key, input_tokens, output_tokens, cost))
        return cost
