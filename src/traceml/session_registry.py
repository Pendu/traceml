"""Per-model trace session state registry.

Design
------
- Each traced model gets its own SessionState via
  get_session(id(model))
- Module-level patches remain global (import-time, per D-08)
- Only data/state (queues, counters, buffers) is scoped
  per-session
- dict operations are atomic under CPython's GIL (safe for
  main thread + sampler thread access)
- _GLOBAL_TIME_QUEUE stays global (not per-model, per D-08)
"""

from collections import deque
from dataclasses import dataclass, field
from queue import Queue
from typing import Dict


@dataclass
class SessionState:
    """Per-model trace session state."""

    step: int = 0
    step_time_queue: Queue = field(
        default_factory=lambda: Queue(maxsize=2048)
    )
    step_buffer: deque = field(default_factory=deque)
    step_memory_queue: Queue = field(
        default_factory=lambda: Queue(maxsize=2048)
    )
    model_queue: Queue = field(default_factory=Queue)
    layer_forward_memory_queue: Queue = field(
        default_factory=lambda: Queue(maxsize=4096)
    )
    layer_backward_memory_queue: Queue = field(
        default_factory=lambda: Queue(maxsize=2048)
    )
    layer_forward_time_queue: Queue = field(
        default_factory=lambda: Queue(maxsize=4096)
    )
    layer_backward_time_queue: Queue = field(
        default_factory=lambda: Queue(maxsize=2048)
    )
    model_forward_memory_queue: Queue = field(
        default_factory=lambda: Queue(maxsize=128)
    )


_registry: Dict[int, SessionState] = {}


def get_session(model_id: int) -> SessionState:
    """Get or create session state for a model.

    Parameters
    ----------
    model_id : int
        Result of id(model) for the nn.Module being traced.

    Returns
    -------
    SessionState
        The session state for this model.
    """
    if model_id not in _registry:
        _registry[model_id] = SessionState()
    return _registry[model_id]


def remove_session(model_id: int) -> None:
    """Remove session state when model is no longer traced."""
    _registry.pop(model_id, None)


def active_sessions() -> int:
    """Return count of active sessions (for testing)."""
    return len(_registry)
