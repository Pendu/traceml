"""Tests for per-model session isolation.

Validates that independent trace sessions in the same process
have independent step counters, queues, and lifecycle.
"""

import pytest

from traceml.session_registry import (
    SessionState,
    _registry,
    _weak_refs,
    active_sessions,
    get_session,
    register_model,
    remove_session,
)


@pytest.fixture(autouse=True)
def _clear_registry():
    """Ensure a clean registry for every test."""
    _registry.clear()
    _weak_refs.clear()
    yield
    _registry.clear()
    _weak_refs.clear()


class TestSessionRegistry:
    def test_get_session_creates_new(self):
        s = get_session(1)
        assert isinstance(s, SessionState)
        assert s.step == 0

    def test_get_session_returns_same(self):
        s1 = get_session(1)
        s2 = get_session(1)
        assert s1 is s2

    def test_independent_step_counters(self):
        s1 = get_session(1)
        s2 = get_session(2)
        for _ in range(5):
            s1.step += 1
        assert s1.step == 5
        assert s2.step == 0

    def test_independent_queues(self):
        s1 = get_session(1)
        s2 = get_session(2)
        s1.step_time_queue.put("event")
        assert s2.step_time_queue.empty()

    def test_session_cleanup(self):
        get_session(1)
        assert active_sessions() == 1
        remove_session(1)
        assert active_sessions() == 0
        fresh = get_session(1)
        assert fresh.step == 0

    def test_many_sessions(self):
        for i in range(10):
            s = get_session(i)
            s.step = i
        for i in range(10):
            assert get_session(i).step == i

    def test_gc_cleanup(self):
        """Model GC triggers automatic session removal."""
        import gc

        import torch.nn as nn

        model = nn.Linear(2, 2)
        mid = id(model)
        get_session(mid)
        register_model(model, mid)
        assert active_sessions() == 1
        del model
        gc.collect()
        assert active_sessions() == 0

    def test_gc_no_leak_after_many_models(self):
        """Many models created and discarded don't leak."""
        import gc

        import torch.nn as nn

        for _ in range(50):
            m = nn.Linear(1, 1)
            get_session(id(m))
            register_model(m, id(m))
        del m
        gc.collect()
        assert active_sessions() == 0

    def test_trace_step_uses_session(self):
        """Verify trace_step increments session step counter.

        This test requires decorators.py to use
        get_session(id(model)).
        """
        import torch.nn as nn

        from traceml.decorators import trace_step

        model = nn.Linear(2, 2)
        with trace_step(model):
            pass
        with trace_step(model):
            pass
        assert get_session(id(model)).step == 2
