"""Model catalog: prices, effort levels and the escalation ladder."""

from __future__ import annotations

from dataclasses import dataclass, field

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class ModelSpec:
    """A model and its per-million-token prices (USD)."""

    name: str
    model_id: str
    input_per_mtok: float
    output_per_mtok: float
    context_window: int = 1_000_000

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok
        ) / 1_000_000


@dataclass(frozen=True)
class Route:
    """One rung of the escalation ladder: a model run at a given effort."""

    model: ModelSpec
    effort: str = "medium"
    # Accept this rung's answer when confidence >= threshold (early exit).
    accept_threshold: float = 0.8
    # Rough output-token multiplier for this effort (thinking is billed as output).
    output_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.effort not in EFFORT_LEVELS:
            raise ValueError(f"unknown effort {self.effort!r}")

    @property
    def key(self) -> str:
        return f"{self.model.name}:{self.effort}"

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return self.model.cost(input_tokens, output_tokens)


# First-party Claude API list prices, USD per million tokens.
HAIKU = ModelSpec("haiku-5.5", "claude-haiku-5-5", 0.10, 0.50)
SONNET = ModelSpec("sonnet-5.5", "claude-sonnet-5-5", 2.00, 10.00)
OPUS = ModelSpec("opus-5.5", "claude-opus-5-5", 4.00, 20.00)

CATALOG = {m.name: m for m in (HAIKU, SONNET, OPUS)}


@dataclass
class Ladder:
    """Routes ordered cheapest -> most capable."""

    routes: list[Route] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.routes:
            raise ValueError("ladder needs at least one route")

    def __len__(self) -> int:
        return len(self.routes)

    def __getitem__(self, i: int) -> Route:
        return self.routes[i]

    @property
    def top(self) -> Route:
        return self.routes[-1]


def _ladder(haiku: float, sonnet: float, opus_medium: float) -> Ladder:
    return Ladder(
        [
            Route(HAIKU, "low", accept_threshold=haiku, output_multiplier=1.0),
            Route(SONNET, "medium", accept_threshold=sonnet, output_multiplier=1.6),
            # Same model as the top rung at lower effort: hard tasks try this first.
            Route(OPUS, "medium", accept_threshold=opus_medium, output_multiplier=1.6),
            Route(OPUS, "high", accept_threshold=0.0, output_multiplier=2.5),
        ]
    )


# Accept thresholds per non-top rung. Lower thresholds exit earlier: cheaper, less accurate.
PROFILES = {
    "balanced": (0.85, 0.80, 0.75),
    "aggressive": (0.80, 0.75, 0.75),
}


def default_ladder(profile: str = "balanced") -> Ladder:
    return _ladder(*PROFILES[profile])


def baseline_route() -> Route:
    """What many teams ship by default: every request to the top model at high effort."""
    return Route(OPUS, "high", accept_threshold=0.0, output_multiplier=2.5)
