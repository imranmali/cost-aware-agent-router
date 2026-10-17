"""Cheap, local complexity estimation. No model call is spent deciding where to route."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Signals that a task needs multi-step reasoning or deep domain work.
HARD_MARKERS = (
    "prove", "proof", "design", "architect", "trade-off", "tradeoff", "optimi",
    "race condition", "concurren", "distributed", "debug", "root cause",
    "step by step", "derive", "formal", "security", "vulnerab", "migrat",
    "refactor", "algorithm", "complexity analysis", "edge case",
)
MEDIUM_MARKERS = (
    "summar", "explain", "compare", "write a", "sql", "query", "function",
    "draft", "rewrite", "analy", "plan", "outline", "list three", "pros and cons",
)
EASY_MARKERS = (
    "classify", "sentiment", "extract", "translate", "yes or no", "true or false",
    "spell", "capital of", "convert", "format", "label", "which category",
)

_CODE_RE = re.compile(r"```|def |class |function\s*\(|\{\s*\n|;\s*\n")
_CONSTRAINT_RE = re.compile(r"\b(must|should|ensure|without|at most|at least|only)\b", re.I)


def estimate_tokens(text: str) -> int:
    """~4 characters per token. Good enough for budgeting; use count_tokens for billing-grade numbers."""
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class Complexity:
    score: float  # 0 (trivial) .. 1 (hardest)
    tier: int  # 0 easy, 1 medium, 2 hard
    features: dict

    @property
    def label(self) -> str:
        return ("easy", "medium", "hard")[self.tier]


class ComplexityEstimator:
    """Weighted heuristic features -> score in [0, 1] -> tier.

    Deliberately simple and transparent; swap in a trained classifier by
    subclassing and overriding `features`/`score`.
    """

    def __init__(self, easy_below: float = 0.30, hard_above: float = 0.62) -> None:
        self.easy_below = easy_below
        self.hard_above = hard_above

    def features(self, prompt: str) -> dict:
        p = prompt.lower()
        tokens = estimate_tokens(prompt)
        return {
            "tokens": tokens,
            "hard_hits": sum(m in p for m in HARD_MARKERS),
            "medium_hits": sum(m in p for m in MEDIUM_MARKERS),
            "easy_hits": sum(m in p for m in EASY_MARKERS),
            "has_code": bool(_CODE_RE.search(prompt)),
            "questions": prompt.count("?"),
            "constraints": len(_CONSTRAINT_RE.findall(prompt)),
            "numbered_steps": len(re.findall(r"^\s*\d+[.)]", prompt, re.M)),
        }

    def score(self, f: dict) -> float:
        s = 0.30
        s += min(f["hard_hits"], 3) * 0.14
        s += min(f["medium_hits"], 3) * 0.06
        s -= min(f["easy_hits"], 2) * 0.12
        s += 0.10 if f["has_code"] else 0.0
        s += min(f["constraints"], 4) * 0.025
        s += min(f["numbered_steps"], 5) * 0.02
        s += min(max(f["questions"] - 1, 0), 3) * 0.03
        # Long inputs tend to be harder, with diminishing returns.
        s += min(f["tokens"] / 2000, 1.0) * 0.12
        return max(0.0, min(1.0, s))

    def estimate(self, prompt: str) -> Complexity:
        f = self.features(prompt)
        s = self.score(f)
        tier = 0 if s < self.easy_below else (2 if s >= self.hard_above else 1)
        return Complexity(score=round(s, 4), tier=tier, features=f)
