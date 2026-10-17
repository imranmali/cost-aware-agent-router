from .base import Completion, Provider, Request
from .simulated import SimulatedProvider

__all__ = ["Completion", "Provider", "Request", "SimulatedProvider", "AnthropicProvider"]


def __getattr__(name):
    # Imported lazily so the simulator works without the anthropic package installed.
    if name == "AnthropicProvider":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider
    raise AttributeError(name)
