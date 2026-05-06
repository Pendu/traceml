"""Tests for the h2d auto-timer patch."""
from __future__ import annotations

import importlib
from contextlib import contextmanager

import pytest
import torch


def _reload_h2d_patch():
    import traceml.instrumentation.patches.h2d_auto_timer_patch as h2d
    return importlib.reload(h2d)


@pytest.fixture
def isolated_patch():
    """Ensure each test starts and ends with an unpatched torch.Tensor.to.

    Without this fixture, a test that installs the h2d patch would leak the
    monkey-patched ``torch.Tensor.to`` to subsequent tests in the same
    pytest process.
    """
    orig = torch.Tensor.to
    if hasattr(torch.Tensor, "_traceml_h2d_patched"):
        delattr(torch.Tensor, "_traceml_h2d_patched")
    yield
    torch.Tensor.to = orig
    if hasattr(torch.Tensor, "_traceml_h2d_patched"):
        delattr(torch.Tensor, "_traceml_h2d_patched")


def _make_fake_timed_region(calls):
    """Return a ``timed_region``-shaped context manager that records calls.

    Mirrors the convention in ``test_initialization_and_wrappers.py``:
    monkeypatch the patch module's ``timed_region`` import with this stub
    instead of inspecting ``_STEP_BUFFER`` globals.
    """

    @contextmanager
    def _fake(name, scope, use_gpu):
        calls.append((name, scope, use_gpu))
        yield

    return _fake


def test_patch_h2d_is_idempotent(isolated_patch):
    h2d = _reload_h2d_patch()

    h2d.patch_h2d()
    assert getattr(torch.Tensor, "_traceml_h2d_patched", False) is True
    first_method = torch.Tensor.to

    h2d.patch_h2d()  # second call must be a no-op
    assert torch.Tensor.to is first_method


def test_patch_does_not_record_when_disabled(isolated_patch, monkeypatch):
    """Outside h2d_auto_timer, patched .to() must not call timed_region."""
    h2d = _reload_h2d_patch()
    h2d.patch_h2d()

    calls: list = []
    monkeypatch.setattr(h2d, "timed_region", _make_fake_timed_region(calls))

    t = torch.randn(4, 4)
    moved = t.to("cpu")  # gate is False — should fast-path

    assert moved.device.type == "cpu"
    assert calls == []


def test_patch_records_event_inside_activator(isolated_patch, monkeypatch):
    """Inside h2d_auto_timer, patched .to() invokes timed_region with the
    expected name/scope/use_gpu signature."""
    h2d = _reload_h2d_patch()
    h2d.patch_h2d()

    calls: list = []
    monkeypatch.setattr(h2d, "timed_region", _make_fake_timed_region(calls))

    t = torch.randn(4, 4)
    with h2d.h2d_auto_timer():
        t.to("cpu")

    assert calls == [("_traceml_internal:h2d_time", "step", True)]


def test_activator_exit_resets_gate_on_exception(
    isolated_patch, monkeypatch
):
    """If user code raises inside the activator, the gate must be False on
    exit so subsequent ``.to()`` calls fast-path."""
    h2d = _reload_h2d_patch()
    h2d.patch_h2d()

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
def test_polymorphic_to_forms_route_through_patch(
    isolated_patch, monkeypatch, form
):
    """All three .to() forms (device, dtype, tensor-like) hit the patch."""
    h2d = _reload_h2d_patch()
    h2d.patch_h2d()

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


def test_model_to_outside_activator_records_nothing(
    isolated_patch, monkeypatch
):
    """model.to(device) at init time fires Tensor.to per parameter; the gate
    must keep all of them silent."""
    import torch.nn as nn

    h2d = _reload_h2d_patch()
    h2d.patch_h2d()

    calls: list = []
    monkeypatch.setattr(h2d, "timed_region", _make_fake_timed_region(calls))

    model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4))
    assert sum(1 for _ in model.parameters()) == 4  # 2 weight + 2 bias

    model.to("cpu")  # would fire Tensor.to 4 times — gate must mute all

    assert calls == []


def test_model_to_inside_activator_records_per_parameter(
    isolated_patch, monkeypatch
):
    """Counter-test: inside the activator, model.to fires N events (one per
    parameter)."""
    import torch.nn as nn

    h2d = _reload_h2d_patch()
    h2d.patch_h2d()

    calls: list = []
    monkeypatch.setattr(h2d, "timed_region", _make_fake_timed_region(calls))

    model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4))

    with h2d.h2d_auto_timer():
        model.to("cpu")

    assert len(calls) == 4
    assert all(
        c == ("_traceml_internal:h2d_time", "step", True) for c in calls
    )
