"""TraceML host-to-device transfer auto-timer patch.

Patches ``torch.Tensor.to`` so every transfer made inside an active
``trace_step`` is timed and recorded as the ``_traceml_internal:h2d_time``
event. Outside an active step, the patch fast-paths to the original method.

Coverage gap: convenience shortcuts (``.cuda()``, ``.cpu()``, ``.float()``,
``.half()``, etc.) bypass ``torch.Tensor.to`` and reach C++ directly. Users
relying on those shortcuts will not see h2d events. Migration guidance:
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
