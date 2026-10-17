import pytest

from costrouter.budget import BudgetPolicy, BudgetTracker, TokenBudget
from costrouter.complexity import ComplexityEstimator
from costrouter.confidence import score_confidence
from costrouter.models import HAIKU, OPUS, SONNET, Route, default_ladder
from costrouter.providers.base import Completion


def test_model_cost_per_million_tokens():
    assert OPUS.cost(1_000_000, 0) == pytest.approx(4.00)
    assert OPUS.cost(0, 1_000_000) == pytest.approx(20.00)
    assert HAIKU.cost(2_000, 1_000) == pytest.approx(0.0007)


def test_ladder_is_ordered_cheapest_first():
    ladder = default_ladder()
    prices = [r.model.output_per_mtok * r.output_multiplier for r in ladder.routes]
    assert prices == sorted(prices)
    assert ladder.top.accept_threshold == 0.0


def test_route_rejects_unknown_effort():
    with pytest.raises(ValueError):
        Route(SONNET, "turbo")


@pytest.mark.parametrize(
    "prompt,tier",
    [
        ("Classify the sentiment of this review: 'Love it!'", 0),
        ("What is the capital of Peru?", 0),
        ("Write a SQL query that returns the top customers and explain it.", 1),
        (
            "Design a distributed rate limiter that must tolerate partitions. Explain the "
            "trade-offs step by step and debug the race condition:\n```python\nx = 1\n```",
            2,
        ),
    ],
)
def test_complexity_tiers(prompt, tier):
    assert ComplexityEstimator().estimate(prompt).tier == tier


def test_budget_caps_output_to_remaining_tokens():
    t = BudgetTracker(TokenBudget(max_output_tokens=1000, max_total_tokens=1500, max_cost_usd=1))
    assert t.max_tokens_for(input_tokens=200) == 1000
    t.charge(Route(HAIKU, "low"), 400, 400)
    assert t.max_tokens_for(input_tokens=200) == 500


def test_budget_refuses_unaffordable_attempt():
    t = BudgetTracker(TokenBudget(4096, 100_000, max_cost_usd=0.01))
    assert t.can_afford(Route(HAIKU, "low"), 5_000, 500)
    assert not t.can_afford(Route(OPUS, "high"), 5_000, 2_000)


def test_budget_raise_never_shrinks():
    t = BudgetTracker(BudgetPolicy().for_tier(1))
    t.raise_to(BudgetPolicy().for_tier(0))
    assert t.budget == BudgetPolicy().for_tier(1)
    t.raise_to(BudgetPolicy().for_tier(2))
    assert t.budget == BudgetPolicy().for_tier(2)


def test_confidence_hard_failures_force_escalation():
    assert score_confidence(Completion("x", 1, 1, "refusal", 0.99)) == 0.0
    assert score_confidence(Completion("x", 1, 1, "max_tokens", 0.99)) == 0.0
    assert score_confidence(Completion("   ", 1, 1, "end_turn", 0.99)) == 0.0


def test_confidence_penalizes_hedging_and_defaults_when_missing():
    sure = score_confidence(Completion("Paris.", 1, 1, self_confidence=0.9))
    hedged = score_confidence(Completion("I'm not sure, possibly Paris.", 1, 1, self_confidence=0.9))
    assert hedged < sure
    assert score_confidence(Completion("Paris.", 1, 1)) == pytest.approx(0.6)
