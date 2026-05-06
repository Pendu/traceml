"""Tests for the wrap_h2d manual instrumentation proxy."""

from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

import traceml.sdk.wrappers as wrappers


def _make_fake_timed_region(calls):
    @contextmanager
    def _fake(name, scope, use_gpu):
        calls.append((name, scope, use_gpu))
        yield

    return _fake


@pytest.fixture
def isolated_h2d_sentinel(monkeypatch):
    """Force ``torch.Tensor._traceml_h2d_patched`` to False around each test
    so ``wrap_h2d`` does not refuse based on a leaked auto-patch sentinel
    from a prior test in the same pytest process.
    """
    monkeypatch.setattr(
        torch.Tensor,
        "_traceml_h2d_patched",
        False,
        raising=False,
    )
    yield


def test_wrap_h2d_intercepts_to_call(isolated_h2d_sentinel, monkeypatch):
    """wrap_h2d(t).to(device) invokes timed_region with the h2d_time event."""
    calls: list = []
    monkeypatch.setattr(
        wrappers, "timed_region", _make_fake_timed_region(calls)
    )

    t = torch.randn(4, 4)
    moved = wrappers.wrap_h2d(t).to("cpu")

    assert isinstance(moved, torch.Tensor)
    assert moved.device.type == "cpu"
    assert calls == [("_traceml_internal:h2d_time", "step", True)]


def test_wrap_h2d_getattr_forwards_to_underlying_tensor(isolated_h2d_sentinel):
    """Attribute access on the proxy forwards to the wrapped tensor."""
    t = torch.randn(3, 5, dtype=torch.float64)
    w = wrappers.wrap_h2d(t)

    assert w.shape == t.shape
    assert w.dtype == t.dtype
    assert torch.equal(w.detach(), t.detach())


def test_wrap_h2d_refuses_when_auto_patch_active(monkeypatch):
    """wrap_h2d must raise if the global h2d patch is already installed.

    We monkeypatch the sentinel attribute directly rather than calling the
    real ``patch_h2d()`` so we don't mutate ``torch.Tensor.to`` for the rest
    of the test process.
    """
    monkeypatch.setattr(
        torch.Tensor,
        "_traceml_h2d_patched",
        True,
        raising=False,
    )

    t = torch.randn(2, 2)

    with pytest.raises(RuntimeError, match="host-to-device"):
        wrappers.wrap_h2d(t)


def test_wrap_h2d_rejects_non_tensor(isolated_h2d_sentinel):
    with pytest.raises(TypeError, match="expects a torch.Tensor"):
        wrappers.wrap_h2d([1, 2, 3])


def test_traceml_wrap_h2d_resolves_via_lazy_loader(isolated_h2d_sentinel):
    """``import traceml; traceml.wrap_h2d`` resolves through the lazy loader."""
    import traceml

    wrap = traceml.wrap_h2d
    t = torch.randn(2, 2)
    proxy = wrap(t)

    assert hasattr(proxy, "to")
    moved = proxy.to("cpu")
    assert isinstance(moved, torch.Tensor)


def test_traceml_init_accepts_patch_h2d_kwarg(monkeypatch):
    """traceml.init(..., patch_h2d=True) propagates to the resolved config.

    Stubs ``patch_h2d()`` so the test does not actually install the global
    patch. Same convention as
    ``test_init_auto_enables_all_supported_patches`` in
    ``tests/test_init_and_wrappers.py``.
    """
    import importlib

    import traceml
    import traceml.sdk.initial as initialization

    importlib.reload(initialization)

    import traceml.instrumentation.patches.h2d_auto_timer_patch as h2d_patch

    monkeypatch.setattr(h2d_patch, "patch_h2d", lambda: None)

    cfg = traceml.init(mode="selective", patch_h2d=True)
    assert cfg.patch_h2d is True
