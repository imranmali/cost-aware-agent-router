"""Cost-per-decision analytics: a JSONL ledger plus summary and comparison reports."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

from .router import Decision


class Ledger:
    """Append-only record of routing decisions. In-memory, optionally mirrored to JSONL."""

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = Path(path) if path else None
        self.decisions: list[Decision] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")

    def record(self, decision: Decision) -> None:
        self.decisions.append(decision)
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(decision.to_dict()) + "\n")

    @staticmethod
    def load(path: str | Path) -> list[Decision]:
        with open(path) as f:
            return [Decision.from_dict(json.loads(line)) for line in f if line.strip()]


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(q * (len(s) - 1))))
    return s[idx]


def summarize(decisions: Iterable[Decision]) -> dict:
    ds = list(decisions)
    n = len(ds)
    if n == 0:
        return {"decisions": 0}
    costs = [d.total_cost_usd for d in ds]
    graded = [d.correct for d in ds if d.correct is not None]

    by_route = Counter(d.final_route for d in ds)
    spend_by_model: dict[str, float] = defaultdict(float)
    for d in ds:
        for a in d.attempts:
            spend_by_model[a.route] += a.cost_usd

    by_tier: dict[str, list[Decision]] = defaultdict(list)
    for d in ds:
        by_tier[d.tier].append(d)

    total = sum(costs)
    wasted = sum(a.cost_usd for d in ds for a in d.attempts if not a.accepted)
    return {
        "decisions": n,
        "total_cost_usd": total,
        "cost_per_decision": {
            "mean": total / n,
            "p50": _pct(costs, 0.50),
            "p95": _pct(costs, 0.95),
            "max": max(costs),
        },
        "accuracy": (sum(graded) / len(graded)) if graded else None,
        "early_exit_rate": sum(d.early_exit for d in ds) / n,
        "escalation_rate": sum(d.escalations > 0 for d in ds) / n,
        "avg_attempts": sum(len(d.attempts) for d in ds) / n,
        "budget_exhausted_rate": sum(d.budget_exhausted for d in ds) / n,
        "escalation_overhead_usd": wasted,
        "avg_latency_ms": statistics.fmean(d.latency_ms for d in ds),
        "final_route_share": {k: v / n for k, v in by_route.most_common()},
        "spend_share": {k: v / total for k, v in sorted(spend_by_model.items())} if total else {},
        "by_tier": {
            tier: {
                "decisions": len(group),
                "cost_per_decision": sum(d.total_cost_usd for d in group) / len(group),
                "accuracy": _acc(group),
            }
            for tier, group in sorted(by_tier.items(), key=lambda kv: ("easy", "medium", "hard").index(kv[0]))
        },
    }


def _acc(group: list[Decision]) -> Optional[float]:
    graded = [d.correct for d in group if d.correct is not None]
    return sum(graded) / len(graded) if graded else None


def compare(router: dict, baseline: dict) -> dict:
    """Savings and quality delta of the router against a baseline summary."""
    savings = 1 - router["total_cost_usd"] / baseline["total_cost_usd"]
    out = {
        "baseline_cost_usd": baseline["total_cost_usd"],
        "router_cost_usd": router["total_cost_usd"],
        "savings_pct": savings * 100,
        "baseline_cost_per_decision": baseline["cost_per_decision"]["mean"],
        "router_cost_per_decision": router["cost_per_decision"]["mean"],
    }
    if router.get("accuracy") is not None and baseline.get("accuracy") is not None:
        out["baseline_accuracy"] = baseline["accuracy"]
        out["router_accuracy"] = router["accuracy"]
        out["accuracy_delta_pts"] = (router["accuracy"] - baseline["accuracy"]) * 100
        # Cost per *correct* answer is the number that matters to the business.
        out["baseline_cost_per_correct"] = baseline["total_cost_usd"] / max(
            1e-9, baseline["accuracy"] * baseline["decisions"]
        )
        out["router_cost_per_correct"] = router["total_cost_usd"] / max(
            1e-9, router["accuracy"] * router["decisions"]
        )
    return out


def _usd(x: float) -> str:
    return f"${x:,.4f}" if x < 1 else f"${x:,.2f}"


def _pc(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def format_summary(s: dict, title: str = "Router") -> str:
    if not s.get("decisions"):
        return f"{title}: no decisions"
    cpd = s["cost_per_decision"]
    lines = [
        f"== {title} ==",
        f"decisions            {s['decisions']}",
        f"total cost           {_usd(s['total_cost_usd'])}",
        f"cost / decision      mean {_usd(cpd['mean'])}  p50 {_usd(cpd['p50'])}  p95 {_usd(cpd['p95'])}",
        f"accuracy             {_pc(s['accuracy'])}",
        f"early exit rate      {_pc(s['early_exit_rate'])}",
        f"escalation rate      {_pc(s['escalation_rate'])}  (avg attempts {s['avg_attempts']:.2f})",
        f"escalation overhead  {_usd(s['escalation_overhead_usd'])}",
        f"budget-capped        {_pc(s['budget_exhausted_rate'])}",
        "final route share    "
        + ", ".join(f"{k} {_pc(v)}" for k, v in s["final_route_share"].items()),
        "spend share          " + ", ".join(f"{k} {_pc(v)}" for k, v in s["spend_share"].items()),
        "by complexity tier:",
    ]
    for tier, t in s["by_tier"].items():
        lines.append(
            f"  {tier:<7} n={t['decisions']:<6} cost/decision {_usd(t['cost_per_decision'])}"
            f"  accuracy {_pc(t['accuracy'])}"
        )
    return "\n".join(lines)


def format_comparison(c: dict) -> str:
    lines = [
        "== Router vs baseline ==",
        f"baseline cost        {_usd(c['baseline_cost_usd'])}  ({_usd(c['baseline_cost_per_decision'])}/decision)",
        f"router cost          {_usd(c['router_cost_usd'])}  ({_usd(c['router_cost_per_decision'])}/decision)",
        f"savings              {c['savings_pct']:.1f}%",
    ]
    if "accuracy_delta_pts" in c:
        lines += [
            f"accuracy             {_pc(c['baseline_accuracy'])} -> {_pc(c['router_accuracy'])}"
            f"  ({c['accuracy_delta_pts']:+.1f} pts)",
            f"cost / correct       {_usd(c['baseline_cost_per_correct'])} -> {_usd(c['router_cost_per_correct'])}",
        ]
    return "\n".join(lines)
