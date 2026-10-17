"""Live provider backed by the Claude API.

Each call asks for a structured `{answer, confidence}` object so the router can
early-exit on the model's own confidence. Effort is passed through
`output_config.effort`; thinking is left at the model default (adaptive).
"""

from __future__ import annotations

import json
import time
from typing import Optional

from ..models import Route
from .base import SYSTEM_PROMPT, Completion, EstimatingTokenCounter, Request

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["answer", "confidence"],
    "additionalProperties": False,
}

# Server-side refusal fallback: a declined request is re-run on a fallback model
# inside the same call. Not offered for Haiku.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}


class AnthropicProvider(EstimatingTokenCounter):
    def __init__(
        self,
        client=None,
        refusal_fallbacks: bool = True,
        exact_token_counts: bool = False,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.refusal_fallbacks = refusal_fallbacks
        self.exact_token_counts = exact_token_counts

    def count_tokens(self, prompt: str, route: Route) -> int:
        if not self.exact_token_counts:
            return super().count_tokens(prompt, route)
        resp = self.client.messages.count_tokens(
            model=route.model.model_id,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.input_tokens

    def complete(self, request: Request) -> Completion:
        route = request.route
        params = dict(
            model=route.model.model_id,
            max_tokens=request.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": request.prompt}],
            output_config={
                "effort": route.effort,
                "format": {"type": "json_schema", "schema": ANSWER_SCHEMA},
            },
        )
        start = time.perf_counter()
        if self.refusal_fallbacks and route.model.model_id in _FALLBACK_MODELS:
            response = self.client.beta.messages.create(
                **params, betas=[FALLBACK_BETA], fallbacks="default"
            )
        else:
            response = self.client.messages.create(**params)
        latency_ms = (time.perf_counter() - start) * 1000

        usage = response.usage
        input_tokens = (
            usage.input_tokens
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
            + (getattr(usage, "cache_read_input_tokens", 0) or 0)
        )
        base = dict(
            input_tokens=input_tokens,
            output_tokens=usage.output_tokens,
            stop_reason=response.stop_reason or "end_turn",
            latency_ms=round(latency_ms, 1),
        )
        # Check stop_reason before reading content: refusals and truncation carry no usable answer.
        if response.stop_reason in ("refusal", "max_tokens"):
            return Completion(text="", **base)

        text = next((b.text for b in response.content if b.type == "text"), "")
        answer, confidence = _parse_answer(text)
        return Completion(text=answer, self_confidence=confidence, **base)


def _parse_answer(text: str) -> tuple[str, Optional[float]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text, None
    conf = data.get("confidence")
    if isinstance(conf, (int, float)):
        conf = max(0.0, min(1.0, float(conf)))
    else:
        conf = None
    return str(data.get("answer", "")), conf
