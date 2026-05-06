"""TraceML host-to-device transfer auto-timer patch.

Patches ``torch.Tensor.to`` so every transfer made inside an active
``trace_step`` is timed and recorded as the ``_traceml_internal:h2d_time``
event. Outside an active step, the patch fast-paths to the original method.

Coverage notes:
- The patch fires on all three polymorphic forms of ``Tensor.to``:
  ``.to(device)`` (the canonical H2D case), ``.to(dtype)`` (same-device
  dtype conversion — recorded but not literally a host-to-device transfer),
  and ``.to(other_tensor)`` (match device/dtype of another tensor). The
  wire-name ``h2d_time`` therefore covers slightly more than literal H2D;
  the dominant case in real training is genuine H2D, so the over-recording
  is bounded and accepted.
- Convenience shortcuts (``.cuda()``, ``.cpu()``, ``.float()``, ``.half()``,
  ``.double()``, ``.bfloat16()``, ``.type()``, ``.type_as()``,
  ``.pin_memory()``) bypass ``torch.Tensor.to`` and reach C++ directly. They
  are NOT recorded by this patch. Migration guidance for users:
  prefer ``.to(device, non_blocking=True)``.
"""

from __future__ import annotations

import threading
from typing import Any

import torch

from traceml.utils.timing import timed_region

_TLS = threading.local()
_ORIG_TENSOR_TO = torch.Tensor.to


def _enabled() -> bool:
    return bool(getattr(_TLS, "_traceml_h2d_enabled", False))


def _traceml_tensor_to(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
    if not _enabled():
        return _ORIG_TENSOR_TO(self, *args, **kwargs)
    with timed_region(
        "_traceml_internal:h2d_time", scope="step", use_gpu=True
    ):
        return _ORIG_TENSOR_TO(self, *args, **kwargs)


def patch_h2d() -> None:
    """Patch ``torch.Tensor.to`` once. Safe to call multiple times."""
    if getattr(torch.Tensor, "_traceml_h2d_patched", False):
        return
    torch.Tensor.to = _traceml_tensor_to  # type: ignore[assignment]
    torch.Tensor._traceml_h2d_patched = True  # type: ignore[attr-defined]


class h2d_auto_timer:
    """Enables h2d timing during its scope.

    Assumes ``patch_h2d()`` has been called once at startup / runtime init.
    """

    def __enter__(self) -> "h2d_auto_timer":
        _TLS._traceml_h2d_enabled = True
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _TLS._traceml_h2d_enabled = False
        return False
