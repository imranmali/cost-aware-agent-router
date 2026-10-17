"""The cost-aware router: complexity -> budget -> cheapest viable rung -> escalate until confident."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from .budget import BudgetPolicy, BudgetTracker
from .complexity import ComplexityEstimator
from .confidence import score_confidence
from .models import Ladder, Route, baseline_route, default_ladder
from .providers.base import Provider, Request


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Attempt:
    route: str
    model_id: str
    effort: str
    input_tokens: int
    output_tokens: int
    max_tokens: int
    cost_usd: float
    confidence: float
    stop_reason: str
    accepted: bool
    latency_ms: float
    correct: Optional[bool] = None


@dataclass
class Decision:
    """Everything that happened while answering one task. One row in the ledger."""

    task_id: str
    strategy: str
    complexity: float
    tier: str
    budget_usd: float  # initial budget for the task's estimated tier
    attempts: list[Attempt]
    final_route: str
    answer: str
    confidence: float
    total_cost_usd: float
    total_tokens: int
    latency_ms: float
    early_exit: bool  # accepted below the top rung
    budget_exhausted: bool  # stopped escalating because the next rung did not fit
    correct: Optional[bool] = None

    @property
    def escalations(self) -> int:
        return max(0, len(self.attempts) - 1)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["escalations"] = self.escalations
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Decision":
        d = {k: v for k, v in d.items() if k != "escalations"}
        d["attempts"] = [Attempt(**a) for a in d["attempts"]]
        return cls(**d)


class CostAwareRouter:
    """Routes each task to the cheapest rung likely to answer it, then escalates on low confidence.

    1. Estimate complexity locally (free).
    2. Give the task a token + dollar budget sized to its complexity.
    3. Start at the rung mapped to that complexity tier (hard tasks skip the cheap rungs).
    4. Accept the first answer whose confidence clears the rung's threshold (early exit);
       otherwise escalate, as long as the next attempt fits in the remaining budget.

    A low-confidence answer is evidence the task was harder than it looked, so with
    `escalation_raises_budget` an escalation lifts the budget to the tier that normally
    starts at the new rung. Without it, a hard task that reads as easy is stuck with an
    easy task's budget and can never reach the model it needs.
    """

    strategy = "cost-aware"

    def __init__(
        self,
        provider: Provider,
        ladder: Optional[Ladder] = None,
        estimator: Optional[ComplexityEstimator] = None,
        budget_policy: Optional[BudgetPolicy] = None,
        start_rung_by_tier: tuple[int, int, int] = (0, 1, 2),  # easy, medium, hard
        escalation_raises_budget: bool = True,
        ledger=None,
    ) -> None:
        self.provider = provider
        self.ladder = ladder or default_ladder()
        self.estimator = estimator or ComplexityEstimator()
        self.budget_policy = budget_policy or BudgetPolicy()
        top = len(self.ladder) - 1
        self.start_rung_by_tier = tuple(min(r, top) for r in start_rung_by_tier)
        self.escalation_raises_budget = escalation_raises_budget
        self.ledger = ledger
        # Running average of output tokens per (route, tier), used to price the next attempt.
        self._expected_out: dict[tuple[str, int], float] = {}

    def expected_output(self, route: Route, tier: int) -> int:
        default = 400 * route.output_multiplier * (1 + 0.5 * tier)
        return int(self._expected_out.get((route.key, tier), default))

    def _tier_for_rung(self, rung: int) -> int:
        return max(t for t, r in enumerate(self.start_rung_by_tier) if r <= rung) if rung >= min(
            self.start_rung_by_tier
        ) else 0

    def _observe_output(self, route: Route, tier: int, tokens: int) -> None:
        key = (route.key, tier)
        prev = self._expected_out.get(key)
        self._expected_out[key] = tokens if prev is None else 0.8 * prev + 0.2 * tokens

    def route(self, task: Task) -> Decision:
        c = self.estimator.estimate(task.prompt)
        tracker = BudgetTracker(self.budget_policy.for_task(c))
        initial_budget = tracker.budget.max_cost_usd
        attempts: list[Attempt] = []
        best: Optional[tuple[float, Attempt, str]] = None
        accepted_at: Optional[int] = None
        budget_exhausted = False

        for i in range(self.start_rung_by_tier[c.tier], len(self.ladder)):
            route = self.ladder[i]
            input_tokens = self.provider.count_tokens(task.prompt, route)
            input_tokens += int(task.metadata.get("context_tokens", 0))
            expected = self.expected_output(route, c.tier)
            if attempts and self.escalation_raises_budget:
                tracker.raise_to(self.budget_policy.for_tier(self._tier_for_rung(i)))
            # The first attempt always runs (capped); later ones must fit the remaining budget.
            if attempts and not tracker.can_afford(route, input_tokens, expected):
                budget_exhausted = True
                break
            max_tokens = tracker.max_tokens_for(input_tokens) or self.budget_policy.output_tokens[0]

            comp = self.provider.complete(
                Request(task.id, task.prompt, route, max_tokens, task.metadata)
            )
            cost = tracker.charge(route, comp.input_tokens, comp.output_tokens)
            self._observe_output(route, c.tier, comp.output_tokens)
            conf = score_confidence(comp)
            is_top = i == len(self.ladder) - 1
            # Refusals, truncations and empty answers are never accepted, even at the top rung.
            usable = comp.stop_reason not in ("refusal", "max_tokens") and comp.text.strip() != ""
            accept = usable and (conf >= route.accept_threshold or is_top)

            attempt = Attempt(
                route=route.key,
                model_id=route.model.model_id,
                effort=route.effort,
                input_tokens=comp.input_tokens,
                output_tokens=comp.output_tokens,
                max_tokens=max_tokens,
                cost_usd=round(cost, 8),
                confidence=round(conf, 4),
                stop_reason=comp.stop_reason,
                accepted=accept,
                latency_ms=comp.latency_ms,
                correct=comp.correct,
            )
            attempts.append(attempt)
            if best is None or conf >= best[0]:
                best = (conf, attempt, comp.text)
            if accept:
                accepted_at = len(attempts) - 1
                final_text = comp.text
                break

        if accepted_at is None:
            # Nothing cleared its threshold: return the most confident answer we paid for.
            conf, final_attempt, final_text = best
        else:
            final_attempt = attempts[accepted_at]
            conf = final_attempt.confidence

        decision = Decision(
            task_id=task.id,
            strategy=self.strategy,
            complexity=c.score,
            tier=c.label,
            budget_usd=initial_budget,
            attempts=attempts,
            final_route=final_attempt.route,
            answer=final_text,
            confidence=conf,
            total_cost_usd=round(tracker.cost_used, 8),
            total_tokens=tracker.tokens_used,
            latency_ms=round(sum(a.latency_ms for a in attempts), 1),
            early_exit=accepted_at is not None and final_attempt.route != self.ladder.top.key,
            budget_exhausted=budget_exhausted,
            correct=final_attempt.correct,
        )
        if self.ledger is not None:
            self.ledger.record(decision)
        return decision

    def route_many(self, tasks) -> list[Decision]:
        return [self.route(t) for t in tasks]


class SingleModelRouter(CostAwareRouter):
    """Baseline: every task goes to one route, no budgeting, no escalation."""

    strategy = "single-model"

    def __init__(self, provider: Provider, route: Optional[Route] = None, ledger=None) -> None:
        route = route or baseline_route()
        big = BudgetPolicy(
            output_tokens=(16_000,) * 3, total_tokens=(10**9,) * 3, cost_usd=(1e9,) * 3
        )
        super().__init__(
            provider,
            ladder=Ladder([Route(route.model, route.effort, 0.0, route.output_multiplier)]),
            budget_policy=big,
            start_rung_by_tier=(0, 0, 0),
            escalation_raises_budget=False,
            ledger=ledger,
        )
