"""Deterministic simulator for benchmarking routing policies without spending money.

Each task carries a hidden difficulty in [0, 1]. Each route has a capability.
The chance of a correct answer is a logistic function of (capability - difficulty);
self-reported confidence is informative but noisy and somewhat overconfident on
smaller models, which is what makes early exit a real trade-off rather than free.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field

from ..complexity import ComplexityEstimator
from .base import Completion, EstimatingTokenCounter, Request

DEFAULT_CAPABILITY = {"haiku-5.5": 0.42, "sonnet-5.5": 0.70, "opus-5.5": 0.86}
EFFORT_BONUS = {"low": 0.0, "medium": 0.03, "high": 0.06, "xhigh": 0.08, "max": 0.10}
# Mean self-confidence when right / when wrong. Smaller models are less calibrated.
CALIBRATION = {
    "haiku-5.5": (0.87, 0.66),
    "sonnet-5.5": (0.88, 0.60),
    "opus-5.5": (0.90, 0.55),
}
LATENCY_MS_PER_OUTPUT_TOKEN = {"haiku-5.5": 4.0, "sonnet-5.5": 9.0, "opus-5.5": 14.0}


@dataclass
class SimulatedProvider(EstimatingTokenCounter):
    seed: int = 0
    capability: dict = field(default_factory=lambda: dict(DEFAULT_CAPABILITY))
    sharpness: float = 12.0
    estimator: ComplexityEstimator = field(default_factory=ComplexityEstimator)

    def _rng(self, *parts) -> random.Random:
        key = ":".join(str(p) for p in (self.seed, *parts)).encode()
        return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))

    def _difficulty(self, request: Request, rng: random.Random) -> float:
        if "difficulty" in request.metadata:
            return float(request.metadata["difficulty"])
        # Ad-hoc prompts: infer a plausible difficulty from the prompt itself.
        score = self.estimator.estimate(request.prompt).score
        return min(1.0, max(0.0, score + rng.gauss(0, 0.08)))

    def complete(self, request: Request) -> Completion:
        rng = self._rng(request.task_id, request.route.key)
        route = request.route
        name = route.model.name
        d = self._difficulty(request, rng)

        cap = self.capability.get(name, 0.6) + EFFORT_BONUS.get(route.effort, 0.0)
        p_correct = 1.0 / (1.0 + math.exp(-(cap - d) * self.sharpness))
        # One draw per task, shared by every route: retries are correlated, so a stronger
        # model only helps through its higher capability, never through a lucky re-roll.
        correct = self._rng(request.task_id, "luck").random() < p_correct

        right_mu, wrong_mu = CALIBRATION.get(name, (0.85, 0.6))
        conf = rng.gauss(right_mu, 0.07) if correct else rng.gauss(wrong_mu, 0.15)
        conf = min(1.0, max(0.0, conf))

        input_tokens = self.count_tokens(request.prompt, route) + int(
            request.metadata.get("context_tokens", 0)
        )
        base_out = float(request.metadata.get("expected_output", 300))
        out = int(base_out * route.output_multiplier * (1 + d) * rng.lognormvariate(0, 0.25))
        out = max(16, out)

        stop = "end_turn"
        if out > request.max_tokens:
            out, stop, correct = request.max_tokens, "max_tokens", False

        latency = 250 + out * LATENCY_MS_PER_OUTPUT_TOKEN.get(name, 10.0)
        return Completion(
            text=f"[{route.key}] simulated answer for {request.task_id}",
            input_tokens=input_tokens,
            output_tokens=out,
            stop_reason=stop,
            self_confidence=round(conf, 4),
            latency_ms=round(latency, 1),
            correct=correct,
        )
