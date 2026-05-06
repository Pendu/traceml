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
