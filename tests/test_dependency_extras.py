"""
Smoke tests for dependency extras separation.

Verifies that core ``import traceml`` works without heavy optional
dependencies (nicegui, plotly, ipython, ipywidgets) and that
pyproject.toml declares only minimal core dependencies.
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import patch


def test_core_import_succeeds():
    """Core modules import without error."""
    import traceml  # noqa: F401
    import traceml.decorators  # noqa: F401
    import traceml.transport.tcp_transport  # noqa: F401


def test_core_deps_importable():
    """All declared core deps are importable."""
    import rich  # noqa: F401
    import psutil  # noqa: F401
    import pynvml  # noqa: F401
    import numpy  # noqa: F401
    import msgspec  # noqa: F401


def test_nicegui_not_in_core_import():
    """``import traceml`` must not pull in nicegui."""
    blocked = {"nicegui": None, "nicegui.ui": None}
    with patch.dict(sys.modules, blocked):
        # Force reimport of the top-level package.
        mod = importlib.import_module("traceml")
        assert mod is not None


def test_ipython_not_in_core_import():
    """``import traceml`` must not pull in IPython."""
    blocked = {
        "IPython": None,
        "IPython.display": None,
    }
    with patch.dict(sys.modules, blocked):
        mod = importlib.import_module("traceml")
        assert mod is not None


def test_notebook_renderable_raises_without_ipython():
    """Renderer raises clear error when IPython is absent."""
    from traceml.renderers import user_time_renderer as mod

    original = mod.HTML
    try:
        mod.HTML = None
        renderer = mod.UserTimeRenderer()
        try:
            renderer.get_notebook_renderable()
            assert False, "Expected ImportError"
        except ImportError as exc:
            assert "pip install traceml-ai[notebook]" in str(exc)
    finally:
        mod.HTML = original


def test_pyproject_core_deps_minimal():
    """pyproject.toml core deps contain only lightweight packages."""
    pyproject = (
        Path(__file__).resolve().parent.parent / "pyproject.toml"
    )
    text = pyproject.read_text()

    # Extract the dependencies list (between first [ and ])
    in_deps = False
    deps = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("dependencies"):
            in_deps = True
            continue
        if in_deps:
            if stripped == "]":
                break
            # Strip quotes, commas, whitespace
            dep = stripped.strip('",').strip()
            if dep:
                deps.append(dep)

    assert len(deps) == 5, f"Expected 5 core deps, got {deps}"

    banned = [
        "scikit",
        "pandas",
        "nicegui",
        "plotly",
        "ipython",
        "ipywidgets",
    ]
    for dep in deps:
        low = dep.lower()
        for b in banned:
            assert b not in low, (
                f"Banned dep {b!r} found in core: {dep}"
            )
