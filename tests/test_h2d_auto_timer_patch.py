"""Tests for the h2d auto-timer patch.

These tests do **not** reload the patch module between cases. ``importlib``
reload re-executes ``_ORIG_TENSOR_TO = torch.Tensor.to`` after the prior
import already replaced ``torch.Tensor.to`` with the patched wrapper, which
causes a self-recursive lookup through the reloaded module's ``__globals__``
and produces ``RecursionError`` on the next ``.to()`` call.

Instead, we import the patch module once and treat ``patch_h2d()`` as the
idempotent install it claims to be — install once, then drive behavior via
the TLS gate. ``timed_region`` is stubbed with ``monkeypatch.setattr`` per the
``test_wrap_optimizer_wraps_real_instance_step`` convention in
``tests/test_initialization_and_wrappers.py``.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

import traceml.instrumentation.patches.h2d_auto_timer_patch as h2d


@pytest.fixture(autouse=True)
def _reset_h2d_state():
    """Install the patch (idempotent) and force the TLS gate off around
    every test in this module."""
    h2d.patch_h2d()
    h2d._TLS._traceml_h2d_enabled = False
    yield
    h2d._TLS._traceml_h2d_enabled = False


def _make_fake_timed_region(calls):
    """Return a ``timed_region``-shaped context manager that records calls."""

    @contextmanager
    def _fake(name, scope, use_gpu):
        calls.append((name, scope, use_gpu))
        yield

    return _fake


def test_patch_h2d_is_idempotent():
    """Second call to patch_h2d must be a no-op."""
    assert getattr(torch.Tensor, "_traceml_h2d_patched", False) is True
    first_method = torch.Tensor.to

    h2d.patch_h2d()  # already installed by fixture; must short-circuit

    assert torch.Tensor.to is first_method


def test_patch_does_not_record_when_disabled(monkeypatch):
    """Outside h2d_auto_timer, patched .to() must not call timed_region."""
    calls: list = []
    monkeypatch.setattr(h2d, "timed_region", _make_fake_timed_region(calls))

    t = torch.randn(4, 4)
    moved = t.to("cpu")

    assert moved.device.type == "cpu"
    assert calls == []


def test_patch_records_event_inside_activator(monkeypatch):
    """Inside h2d_auto_timer, patched .to() invokes timed_region with the
    expected name/scope/use_gpu signature."""
    calls: list = []
    monkeypatch.setattr(h2d, "timed_region", _make_fake_timed_region(calls))

    t = torch.randn(4, 4)
    with h2d.h2d_auto_timer():
        t.to("cpu")

    assert calls == [("_traceml_internal:h2d_time", "step", True)]


def test_activator_exit_resets_gate_on_exception(monkeypatch):
    """If user code raises inside the activator, the gate must be False on
    exit so subsequent ``.to()`` calls fast-path."""
    calls: list = []
    monkeypatch.setattr(h2d, "timed_region", _make_fake_timed_region(calls))

    with pytest.raises(RuntimeError, match="boom"):
        with h2d.h2d_auto_timer():
            raise RuntimeError("boom")

    assert h2d._enabled() is False

    t = torch.randn(2, 2)
    t.to("cpu")
    assert calls == []  # gate is False — must fast-path


@pytest.mark.parametrize("form", ["device", "dtype", "tensor_like"])
def test_polymorphic_to_forms_route_through_patch(monkeypatch, form):
    """All three .to() forms (device, dtype, tensor-like) hit the patch."""
    calls: list = []
    monkeypatch.setattr(h2d, "timed_region", _make_fake_timed_region(calls))

    t = torch.randn(3, 3)
    other = torch.randn(2, 2, dtype=torch.float64)

    with h2d.h2d_auto_timer():
        if form == "device":
            t.to("cpu")
        elif form == "dtype":
            t.to(torch.float16)
        elif form == "tensor_like":
            t.to(other)

    assert calls == [("_traceml_internal:h2d_time", "step", True)]


def test_model_to_outside_activator_records_nothing(monkeypatch):
    """model.to(device) at init time fires Tensor.to per parameter; the gate
    must keep all of them silent."""
    import torch.nn as nn

    calls: list = []
    monkeypatch.setattr(h2d, "timed_region", _make_fake_timed_region(calls))

    model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4))
    assert sum(1 for _ in model.parameters()) == 4  # 2 weight + 2 bias

    model.to("cpu")  # would fire Tensor.to 4 times — gate must mute all

    assert calls == []


def test_model_to_inside_activator_records_per_parameter(monkeypatch):
    """Counter-test: inside the activator, model.to fires N events (one per
    parameter)."""
    import torch.nn as nn

    calls: list = []
    monkeypatch.setattr(h2d, "timed_region", _make_fake_timed_region(calls))

    model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4))

    with h2d.h2d_auto_timer():
        model.to("cpu")

    assert len(calls) == 4
    assert all(
        c == ("_traceml_internal:h2d_time", "step", True) for c in calls
    )


def test_trace_step_opens_h2d_activator():
    """trace_step must open h2d_auto_timer alongside forward and backward.

    We assert directly on the patch module's TLS state rather than driving a
    real ``init(mode="auto")`` + ``.to()`` flow. That keeps the test focused
    on the trace_step modification (Task 8) and decoupled from init/sampler
    internals.
    """
    import torch.nn as nn

    import traceml.sdk.instrumentation as instrumentation

    captured = {}
    model = nn.Linear(2, 2)

    with instrumentation.trace_step(model):
        captured["enabled_inside"] = h2d._enabled()
    captured["enabled_after"] = h2d._enabled()

    assert captured["enabled_inside"] is True
    assert captured["enabled_after"] is False
