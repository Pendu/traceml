import os
import sys
from dataclasses import dataclass
from queue import Full, Queue
from typing import Dict, Optional

import torch
import torch.nn as nn

_temp_step_memory_buffer: Dict = {}

TRACEML_DISABLED = os.environ.get("TRACEML_DISABLED") == "1"


@dataclass
class StepMemoryEvent:
    """
    Peak GPU memory during a TraceML step.
    """

    step: int
    model_id: int
    device: str
    peak_allocated: float
    peak_reserved: float


class StepMemoryTracker:
    """
    Tracks peak CUDA memory across a train step.
    """

    def __init__(self, model: nn.Module):
        if TRACEML_DISABLED:
            return  # BYPASS: Do not attach any trackers

        self.model = model
        self.model_id = id(model)

        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

    def reset(self):
        """
        Reset CUDA peak memory counters at step start.
        """
        if TRACEML_DISABLED:
            return

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def record(self):
        """
        Record peak memory at step end.

        Semantics:
        - CUDA: record real peak allocated / reserved memory
        - Non-CUDA: emit a sentinel event with 0 values
          (means: step memory not applicable on this device)
        """
        if TRACEML_DISABLED:
            return

        if self.device.type == "cuda":
            peak_allocated = torch.cuda.max_memory_allocated(self.device)
            peak_reserved = torch.cuda.max_memory_reserved(self.device)
        else:
            peak_allocated = 0.0
            peak_reserved = 0.0

        evt = StepMemoryEvent(
            model_id=self.model_id,
            device=str(self.device),
            peak_allocated=float(peak_allocated),
            peak_reserved=float(peak_reserved),
            step=-1,  # filled during flush
        )
        _temp_step_memory_buffer[self.model_id] = evt


def get_step_memory_queue(
    model_id: Optional[int] = None,
) -> Queue:
    """Return the step memory queue for a model session.

    Parameters
    ----------
    model_id : int, optional
        If given, returns the session-scoped queue.
        If None, returns the first active session's queue.
    """
    from traceml.session_registry import (
        _registry,
        get_session,
    )

    if model_id is not None:
        return get_session(model_id).step_memory_queue
    for sess in _registry.values():
        return sess.step_memory_queue
    return Queue(maxsize=2048)


def flush_step_memory_buffer(
    model: nn.Module, step: int
) -> None:
    if TRACEML_DISABLED:
        return

    model_id = id(model)

    evt = _temp_step_memory_buffer.pop(model_id, None)
    if evt is None:
        return

    evt.step = step
    queue = get_step_memory_queue(model_id)
    try:
        queue.put_nowait(evt)
    except Full:
        print(
            "[TraceML:StepMemory] Queue full, "
            f"dropping event for model {evt.model_id}",
            file=sys.stderr,
        )
