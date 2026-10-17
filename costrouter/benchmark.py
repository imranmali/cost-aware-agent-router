"""Synthetic workload + side-by-side benchmark of the router against an always-top-model baseline."""

from __future__ import annotations

import random
from dataclasses import replace

from .analytics import Ledger, compare, summarize
from .models import Ladder, default_ladder
from .providers.simulated import SimulatedProvider
from .router import CostAwareRouter, SingleModelRouter, Task

# Prompt templates per true difficulty band. The router only ever sees the text.
TEMPLATES = {
    0: [
        "Classify the sentiment of this review as positive, negative or neutral: '{s}'",
        "Extract the email address from: 'Contact {n} at {n}@example.com for details.'",
        "Translate to French: '{s}'",
        "What is the capital of {c}?",
        "Label this support ticket with one category (billing, bug, account): '{s}'",
        "Convert {x} kilometers to miles.",
        "Answer yes or no: is {x} a prime number?",
    ],
    1: [
        "Summarize the following meeting notes in five bullet points and list three action items:\n{doc}",
        "Write a SQL query that returns the top 10 customers by revenue in {c} for the last quarter, "
        "joining orders and customers.",
        "Explain the difference between a process and a thread to a junior engineer, with one example each.",
        "Compare PostgreSQL and MongoDB for an analytics workload. Give pros and cons of each.",
        "Draft a polite email to {n} rescheduling our meeting and proposing two new times.",
        "Write a Python function that parses ISO-8601 dates and returns a datetime, with tests.",
    ],
    2: [
        "Design a distributed rate limiter for a multi-region API gateway. It must tolerate network "
        "partitions, should stay under 5ms p99, and must not double count. Explain the trade-offs "
        "step by step.",
        "Debug this race condition and find the root cause:\n```python\n{code}\n```\n"
        "Then propose a fix that works without a global lock.",
        "Prove that the following scheduling algorithm is optimal, or give a counterexample, and "
        "derive its time complexity:\n{doc}",
        "Plan the migration of a 40-service monolith-adjacent system to event sourcing. Identify "
        "security risks, edge cases, and a rollback strategy.\n1. Inventory\n2. Sequencing\n3. Cutover",
        "Refactor this module to remove the concurrency bug, keep the public API, and explain each "
        "change:\n```python\n{code}\n```",
    ],
}

_FILL = {
    "s": ["Great product, shipping was slow though.", "Terrible support experience.",
          "It works as described.", "Love it!", "The app keeps crashing on login."],
    "n": ["alice", "bob", "carol", "dan", "erin"],
    "c": ["Kenya", "Peru", "Norway", "Japan", "Canada"],
    "x": ["17", "42", "97", "120", "13"],
}
_DOC = "Q3 sync. Revenue up 8%. Hiring paused in EMEA. Infra costs rose 22% due to LLM usage. " * 6
_CODE = (
    "counter = 0\n"
    "def worker():\n"
    "    global counter\n"
    "    for _ in range(1000):\n"
    "        tmp = counter\n"
    "        counter = tmp + 1\n"
)

BANDS = {0: (0.02, 0.30), 1: (0.30, 0.62), 2: (0.62, 0.92)}
EXPECTED_OUTPUT = {0: 120, 1: 450, 2: 1100}
CONTEXT_TOKENS = {0: (0, 400), 1: (500, 3000), 2: (2000, 12000)}


def generate_workload(
    n: int = 2000,
    mix: tuple[float, float, float] = (0.6, 0.3, 0.1),
    seed: int = 7,
    mislabel_rate: float = 0.12,
) -> list[Task]:
    """A production-like mix of easy/medium/hard tasks.

    `mislabel_rate` of tasks are written with a neighbouring band's template, so the
    complexity estimator is wrong some of the time, as it would be on real traffic.
    """
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        band = rng.choices((0, 1, 2), weights=mix)[0]
        look = band
        if rng.random() < mislabel_rate:
            look = min(2, max(0, band + rng.choice((-1, 1))))
        tmpl = rng.choice(TEMPLATES[look])
        prompt = tmpl.format(
            s=rng.choice(_FILL["s"]), n=rng.choice(_FILL["n"]), c=rng.choice(_FILL["c"]),
            x=rng.choice(_FILL["x"]), doc=_DOC, code=_CODE,
        )
        lo, hi = BANDS[band]
        tasks.append(
            Task(
                id=f"t{i:05d}",
                prompt=prompt,
                metadata={
                    "difficulty": round(rng.uniform(lo, hi), 4),
                    "band": band,
                    "expected_output": EXPECTED_OUTPUT[band],
                    "context_tokens": rng.randint(*CONTEXT_TOKENS[band]),
                },
            )
        )
    return tasks


def run_benchmark(
    n: int = 2000,
    mix: tuple[float, float, float] = (0.6, 0.3, 0.1),
    seed: int = 7,
    ladder: Ladder | None = None,
    ledger_path: str | None = None,
    baseline_ledger_path: str | None = None,
) -> dict:
    tasks = generate_workload(n, mix, seed)
    provider = SimulatedProvider(seed=seed)

    router = CostAwareRouter(provider, ladder=ladder, ledger=Ledger(ledger_path))
    baseline = SingleModelRouter(provider, ledger=Ledger(baseline_ledger_path))
    r = summarize(router.route_many(tasks))
    b = summarize(baseline.route_many(tasks))
    return {"router": r, "baseline": b, "comparison": compare(r, b)}


def threshold_sweep(
    thresholds=(0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
    n: int = 2000,
    mix: tuple[float, float, float] = (0.6, 0.3, 0.1),
    seed: int = 7,
) -> list[dict]:
    """Cost/quality frontier: one confidence threshold applied to every non-top rung."""
    rows = []
    for t in thresholds:
        base = default_ladder("balanced")
        rungs = [replace(r, accept_threshold=t) for r in base.routes[:-1]] + [base.top]
        res = run_benchmark(n, mix, seed, ladder=Ladder(rungs))
        c = res["comparison"]
        rows.append({
            "threshold": t,
            "savings_pct": c["savings_pct"],
            "accuracy_delta_pts": c.get("accuracy_delta_pts"),
            "escalation_rate": res["router"]["escalation_rate"],
        })
    return rows


def mix_sweep(n: int = 2000, seed: int = 7, ladder: Ladder | None = None) -> list[dict]:
    """How savings depend on the share of easy traffic."""
    mixes = {
        "support-style (80/15/5)": (0.8, 0.15, 0.05),
        "easy-heavy (70/20/10)": (0.7, 0.2, 0.1),
        "typical (60/30/10)": (0.6, 0.3, 0.1),
        "balanced (40/40/20)": (0.4, 0.4, 0.2),
        "hard-heavy (20/40/40)": (0.2, 0.4, 0.4),
    }
    rows = []
    for name, mix in mixes.items():
        c = run_benchmark(n, mix, seed, ladder=ladder)["comparison"]
        rows.append({"mix": name, "savings_pct": c["savings_pct"],
                     "accuracy_delta_pts": c.get("accuracy_delta_pts")})
    return rows
