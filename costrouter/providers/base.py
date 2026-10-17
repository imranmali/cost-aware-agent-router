"""Provider interface shared by the simulator and the live Claude provider."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

from ..complexity import estimate_tokens
from ..models import Route

SYSTEM_PROMPT = (
    "Answer the user's request. Then rate how confident you are that the answer is "
    "correct and complete, from 0.0 (guessing) to 1.0 (certain). Be calibrated: "
    "a lower score is useful, it sends the request to a stronger model."
)


@dataclass(frozen=True)
class Request:
    task_id: str
    prompt: str
    route: Route
    max_tokens: int
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str = "end_turn"
    self_confidence: Optional[float] = None
    latency_ms: float = 0.0
    # Only known when ground truth exists (simulator, labelled evals).
    correct: Optional[bool] = None


class Provider(Protocol):
    def complete(self, request: Request) -> Completion: ...

    def count_tokens(self, prompt: str, route: Route) -> int: ...


class EstimatingTokenCounter:
    """Mixin: local token estimate including the router's system prompt."""

    def count_tokens(self, prompt: str, route: Route) -> int:
        return estimate_tokens(prompt) + estimate_tokens(SYSTEM_PROMPT)
