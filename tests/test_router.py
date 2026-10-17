import pytest

from costrouter.analytics import Ledger, compare, summarize
from costrouter.budget import BudgetPolicy
from costrouter.complexity import estimate_tokens
from costrouter.providers.base import Completion
from costrouter.router import CostAwareRouter, Decision, SingleModelRouter, Task

EASY = "Classify the sentiment of this review: 'Love it!'"
HARD = (
    "Design a distributed rate limiter that must tolerate partitions. Explain the trade-offs "
    "step by step and debug the race condition:\n```python\nx = 1\n```"
)


class ScriptedProvider:
    """Returns a fixed confidence per route key; records every call."""

    def __init__(self, confidence: dict, output_tokens: int = 200, stop: dict | None = None):
        self.confidence = confidence
        self.output_tokens = output_tokens
        self.stop = stop or {}
        self.calls = []

    def count_tokens(self, prompt, route):
        return estimate_tokens(prompt)

    def complete(self, request):
        self.calls.append(request)
        key = request.route.key
        return Completion(
            text=f"answer from {key}",
            input_tokens=estimate_tokens(request.prompt),
            output_tokens=min(self.output_tokens, request.max_tokens),
            stop_reason=self.stop.get(key, "end_turn"),
            self_confidence=self.confidence.get(key, 0.5),
            correct=True,
        )


def test_confident_cheap_answer_exits_early():
    p = ScriptedProvider({"haiku-5.5:low": 0.95})
    d = CostAwareRouter(p).route(Task("t", EASY))
    assert d.tier == "easy"
    assert [a.route for a in d.attempts] == ["haiku-5.5:low"]
    assert d.early_exit and d.escalations == 0


def test_low_confidence_escalates_until_accepted():
    p = ScriptedProvider({"haiku-5.5:low": 0.4, "sonnet-5.5:medium": 0.9})
    d = CostAwareRouter(p).route(Task("t", EASY))
    assert [a.route for a in d.attempts] == ["haiku-5.5:low", "sonnet-5.5:medium"]
    assert d.final_route == "sonnet-5.5:medium"
    assert d.answer == "answer from sonnet-5.5:medium"
    assert not d.attempts[0].accepted and d.attempts[1].accepted
    assert d.total_cost_usd == pytest.approx(sum(a.cost_usd for a in d.attempts))


def test_hard_task_skips_cheap_rungs():
    p = ScriptedProvider({"opus-5.5:medium": 0.9})
    d = CostAwareRouter(p).route(Task("t", HARD))
    assert d.tier == "hard"
    assert d.attempts[0].route == "opus-5.5:medium"


def test_top_rung_always_accepted():
    p = ScriptedProvider({})  # every rung reports 0.5
    d = CostAwareRouter(p).route(Task("t", HARD))
    assert d.final_route == "opus-5.5:high"
    assert not d.early_exit


def test_refusal_at_top_falls_back_to_best_answer():
    p = ScriptedProvider({"opus-5.5:medium": 0.6}, stop={"opus-5.5:high": "refusal"})
    d = CostAwareRouter(p).route(Task("t", HARD))
    assert d.final_route == "opus-5.5:medium"
    assert d.confidence == 0.6


def test_budget_stops_escalation_without_raise():
    tiny = BudgetPolicy(cost_usd=(0.0002, 0.0002, 0.0002))
    p = ScriptedProvider({}, output_tokens=300)
    d = CostAwareRouter(p, budget_policy=tiny, escalation_raises_budget=False).route(Task("t", EASY))
    assert d.budget_exhausted
    assert [a.route for a in d.attempts] == ["haiku-5.5:low"]


def test_escalation_raises_budget_to_new_tier():
    tiny_easy = BudgetPolicy(cost_usd=(0.0002, 0.12, 0.60))
    p = ScriptedProvider({"sonnet-5.5:medium": 0.9}, output_tokens=300)
    d = CostAwareRouter(p, budget_policy=tiny_easy).route(Task("t", EASY))
    assert not d.budget_exhausted
    assert d.final_route == "sonnet-5.5:medium"


def test_max_tokens_comes_from_budget():
    p = ScriptedProvider({"haiku-5.5:low": 0.95})
    CostAwareRouter(p).route(Task("t", EASY))
    assert p.calls[0].max_tokens == BudgetPolicy().output_tokens[0]


def test_ledger_round_trip_and_summary(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    p = ScriptedProvider({"haiku-5.5:low": 0.95, "sonnet-5.5:medium": 0.9})
    router = CostAwareRouter(p, ledger=ledger)
    router.route_many([Task("a", EASY), Task("b", "Write a SQL query and explain it.")])

    loaded = Ledger.load(path)
    assert [d.task_id for d in loaded] == ["a", "b"]
    assert isinstance(loaded[0], Decision)
    assert loaded[0].to_dict() == ledger.decisions[0].to_dict()

    s = summarize(loaded)
    assert s["decisions"] == 2
    assert s["accuracy"] == 1.0
    assert s["cost_per_decision"]["mean"] == s["total_cost_usd"] / 2


def test_compare_against_single_model_baseline():
    tasks = [Task(str(i), EASY) for i in range(5)]
    p = ScriptedProvider({"haiku-5.5:low": 0.95})
    r = summarize(CostAwareRouter(p).route_many(tasks))
    b = summarize(SingleModelRouter(p).route_many(tasks))
    c = compare(r, b)
    assert c["savings_pct"] > 90  # haiku vs opus:high on trivially easy work
    assert c["accuracy_delta_pts"] == 0
