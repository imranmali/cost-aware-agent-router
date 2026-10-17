"""Turn a completion into a single confidence number used for early exit."""

from __future__ import annotations

from .providers.base import Completion

HEDGES = (
    "i'm not sure", "i am not sure", "not certain", "i don't know", "i do not know",
    "it depends", "might be", "possibly", "unclear", "cannot determine",
    "can't determine", "insufficient information",
)


def score_confidence(c: Completion) -> float:
    """Blend the model's self-reported confidence with cheap output checks.

    Hard failures (refusal, truncation, empty answer) force escalation no
    matter what the model claims.
    """
    if c.stop_reason in ("refusal", "max_tokens") or not c.text.strip():
        return 0.0
    conf = c.self_confidence if c.self_confidence is not None else 0.6
    text = c.text.lower()
    hedges = sum(h in text for h in HEDGES)
    conf -= 0.08 * min(hedges, 3)
    return max(0.0, min(1.0, conf))
