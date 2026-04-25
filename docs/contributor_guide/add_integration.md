# How to add a new framework integration

> Internal contributor guide. Audience: one trusted co-founder / new engineer
> being onboarded to TraceML. Not for public docs.

This guide teaches you how to add a new framework integration — a thin adapter that lets users of HuggingFace `Trainer`, PyTorch Lightning, HuggingFace Accelerate, or similar training-loop frameworks get TraceML telemetry by adding **one line** instead of editing their training code. It assumes you have read the top-level `CLAUDE.md`, have a working `pip install -e ".[dev,torch]"` checkout, and have at least skimmed `add_sampler.md` so you understand what the rest of the pipeline does with the events you emit.

---
Feature type: framework integration
Risk level: medium (framework API changes break the integration)
Cross-cutting impact: training process only (no aggregator changes)
PyTorch coupling: indirect (via the framework)
Reference PRs: —
Companion reviewer guide: none yet
Last verified: 2026-04-25
---

## 1. Intro and mental model

### What is a "framework integration" in TraceML?

A **framework integration** is a thin lifecycle-event-to-TraceML-API translator. Its only job is to detect when a third-party training framework hits a meaningful boundary — training start, batch start, batch end, training end — and route that boundary to the four TraceML primitives:

1. `traceml.init(mode=...)` — installs the auto-instrumentation patches.
2. `trace_model_instance(model)` — attaches per-layer hooks for deep profile.
3. `trace_step(model)` — context-manages a single optimizer step.
4. `traceml.final_summary(...)` (or letting the runtime drain naturally) — ensures the telemetry pipeline flushes before the process exits.

The real work — sampling, queue draining, CUDA event resolution, TCP shipping, aggregation, rendering — lives in the samplers, patches, hooks, and aggregator layers. The integration just wires the framework's lifecycle to those primitives. That is why each existing integration is roughly 80–170 lines.

There are exactly two existing integrations as of `traceml-ai` v0.2.11:

- **HuggingFace `Trainer`** — `src/traceml/integrations/huggingface.py` (83 lines). Subclass-based: users replace `Trainer` with `TraceMLTrainer`.
- **PyTorch Lightning** — `src/traceml/integrations/lightning.py` (169 lines). Callback-based: users append `TraceMLCallback()` to `Trainer(callbacks=[...])`.

Both files live under `src/traceml/integrations/`, which has an empty `__init__.py` — see §6 for why.

### What an integration is NOT

- **Not a sampler.** Samplers poll a data source on every runtime tick (see `add_sampler.md`). Integrations do not poll; they react to framework callbacks. If you find yourself writing a `sample()` method inside an integration file, you are in the wrong subsystem.
- **Not a patch.** Patches mutate `nn.Module.__call__`, `Tensor.backward`, `DataLoader.__iter__`, etc. globally for every PyTorch user (see `add_patch.md`). Integrations never monkey-patch the framework's classes. They use the framework's documented hook surface — `Callback` for Lightning, subclassing `Trainer` for HuggingFace.
- **Not a hook on `nn.Module`.** Per-layer hooks attach via `register_forward_hook` / `register_full_backward_hook` and are managed by `trace_model_instance` ([W5](../deep_dive/code-walkthroughs.md#w5-per-layer-hooks--forwardbackward-time-and-memory-hooks)). Integrations route to `trace_model_instance`; they never attach hooks themselves.
- **Not a decorator.** `trace_time` (in `instrumentation.py:245`) is for user-defined regions. Integrations are zero-code; they don't decorate user functions.

### The four lifecycle events every integration must handle

| Event | TraceML primitive | When it must fire |
|---|---|---|
| **Init** | `traceml.init(mode=...)` (or rely on legacy auto-init) | Once, before training starts. |
| **Model attach** | `trace_model_instance(model)` | Once, when the framework hands you the *final* model — after DDP/FSDP wrapping. |
| **Step boundary** | `with trace_step(model): ...` (or fine-grained `timed_region` + `flush_step_events`) | Around every optimizer step in the training loop. |
| **Shutdown** | `traceml.final_summary(...)` if the user wants the JSON; otherwise the runtime drains on process exit | Once, after the last batch. |

The HuggingFace integration takes the high-level path: it wraps the parent `training_step` in a single `trace_step(model)` context manager. The Lightning integration takes the low-level path: it composes `timed_region` calls around the individual phases (forward, backward, optimizer) because Lightning gives it those granular hooks for free, and uses the lower-level `flush_step_events` + `TraceState.step` directly. Both are valid — see §3 for why each chose what it chose.

### The fail-open contract (not optional)

A framework integration must **never break training**. If the framework changes a hook signature, if `trace_model_instance` raises because the model isn't a real `nn.Module`, if `flush_step_events` chokes — the integration logs to stderr with `[TraceML]` prefix and lets the user's training step proceed unmodified. See [principles.md](principles.md) §1 for the cross-cutting fail-open rule; do not re-derive it.

The corollary: **every integration must honor `TRACEML_DISABLED=1`** as the universal bypass. Users who want to ship code with the integration permanently wired in but disable telemetry on a particular run set `TRACEML_DISABLED=1` and expect the integration to be functionally a no-op. Both existing integrations early-return on this flag at the top of every public method.

---

## 2. Before you start: decisions to make

Answer all of these before opening an editor. Write them in the PR description; reviewers will check.

- [ ] **Framework class to wrap.** Exactly which class is the user replacing or extending? (`transformers.Trainer`, `lightning.pytorch.Trainer`'s `Callback`, `accelerate.Accelerator`, ...). Use the fully-qualified import path.
- [ ] **Integration shape.** Subclass-based (replace the framework's class) or callback-based (register a callback)? Pick one explicitly. See §3 for the trade-offs.
- [ ] **Lifecycle hook mapping.** Write a four-row table mapping framework events to TraceML primitives. If you can't fill all four rows, you don't have enough hooks — pick a different integration shape.
- [ ] **Granularity choice.** High-level (`trace_step`) or low-level (`timed_region` calls around individual phases)? HuggingFace took high-level because `Trainer` has only `training_step`. Lightning took low-level because it has separate hooks.
- [ ] **Optional dependency declaration.** The framework MUST be in `pyproject.toml` under `[project.optional-dependencies]`. Pick the extra name and document the install path. See §6.
- [ ] **Model-attach timing.** When does the framework hand you the *final* model — after DDP/FSDP/Accelerator wrapping has happened? Lightning: `on_fit_start`. HuggingFace: lazily on the first `training_step`.
- [ ] **Configuration surface.** How does the user pass TraceML init args? Subclass: extra constructor kwargs. Callback: callback constructor kwargs.
- [ ] **DDP / FSDP / Accelerator awareness.** Does the framework manage distributed launch on its own? The integration should not fight the framework's launch.
- [ ] **Init policy default.** `mode="auto"` (install all patches), `mode="manual"` (no patches), or `mode="selective"` (only the patches the integration explicitly needs)? Default for new integrations is `mode="auto"`.
- [ ] **`final_summary` policy.** Default: do nothing on shutdown. Only call `final_summary()` when the user explicitly opts in (constructor flag), because it blocks for up to 30 s by default.

---

## 3. Anatomy of an existing integration (annotated walkthrough)

We will walk through `lightning.py` end-to-end first. Reasons for that choice:

- It is the more complex of the two — 169 lines vs. 83 — so every primitive shows up at least once.
- It uses **callback-based** integration, which is the future-proof shape every modern framework offers.
- It demonstrates the fine-grained timing path: instead of one `trace_step(model)` block, it composes `timed_region` calls around each phase the framework exposes.

Then we'll point out the differences in `huggingface.py` (subclass-based, high-level `trace_step`).

File: `src/traceml/integrations/lightning.py`.

### 3.1. Imports and module-scope state

```python
import os
import sys

from lightning.pytorch.callbacks import Callback

from traceml.decorators import TraceState
from traceml.utils.flush_buffers import flush_step_events
from traceml.utils.step_memory import StepMemoryTracker
from traceml.utils.timing import (
    TimeEvent,
    TimeScope,
    record_event,
    timed_region,
)

TRACEML_DISABLED = os.environ.get("TRACEML_DISABLED") == "1"
```

Three things to notice:

- **`from lightning.pytorch.callbacks import Callback` is at module top level — and that is a problem.** This is the one place where the existing Lightning integration is *not* a model for new code: importing `lightning.py` requires the user to have `lightning` installed. The HuggingFace integration does it correctly (see §3.7); follow the HuggingFace pattern for new integrations. See §6 and §9 pitfall #1.
- **`TRACEML_DISABLED` is captured at import time.** This is fine for the current implementation because the env var is set by the CLI before any module loads. Don't try to "improve" it by re-reading on every method call; one capture per process is correct.
- **No top-level side effects.** Importing `lightning.py` does not `traceml.init()`, doesn't attach hooks, doesn't patch anything. Construction-time work happens in `__init__`; runtime work happens in the callback methods.

### 3.2. The callback class skeleton

```python
class TraceMLCallback(Callback):
    """
    Official TraceML Callback for PyTorch Lightning.
    ...
    """

    def __init__(self):
        super().__init__()
        self._traceml_step_ctx = None
        self._forward_ctx = None
        self._backward_ctx = None
        self._optimizer_ctx = None

        self._mem_tracker = None
        self._opt_step_occurred = False
```

Observations:

- **Takes no required arguments.** The class is registered by the user as `TraceMLCallback()` — zero config. Adding arguments is fine but every argument should have a safe default.
- **Every context manager is held as an instance attribute.** The Lightning hook API gives you start and end as separate callbacks (`on_train_batch_start` / `on_before_backward`), so the only way to keep a `with` block open across them is to call `__enter__` in one method and `__exit__` in another.
- **`_opt_step_occurred`** is the gradient-accumulation tracker. Document this kind of framework-specific idiom in the callback docstring.

### 3.3. Step boundary opening

```python
def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
    if TRACEML_DISABLED:
        return
    self._traceml_step_ctx = timed_region(
        "_traceml_internal:step_time", scope="step", use_gpu=False
    )
    self._traceml_step_ctx.__enter__()
    ...
    self._forward_ctx = timed_region(
        "_traceml_internal:forward_time", scope="step"
    )
    self._forward_ctx.__enter__()
```

The shape:

1. **`TRACEML_DISABLED` early return.** Always the first line.
2. **Open the outer `step_time` region.** This corresponds to what `trace_step()` does in `instrumentation.py:147-149` — same `name`, same `scope`, same `use_gpu=False`.
3. **Reset the memory tracker.** Every step is a fresh memory snapshot.
4. **Open the inner `forward_time` region.** This will be closed in `on_before_backward`.
5. **Each step opens a `try/except` around side effects.**

### 3.4. Phase transitions

```python
def on_before_backward(self, trainer, pl_module, loss):
    if TRACEML_DISABLED:
        return
    if self._forward_ctx is not None:
        try:
            self._forward_ctx.__exit__(None, None, None)
        except Exception:
            pass
        self._forward_ctx = None
    self._backward_ctx = timed_region(
        "_traceml_internal:backward_time", scope="step"
    )
    self._backward_ctx.__enter__()
```

Two patterns to copy:

- **Always check the context-manager attribute against `None` before closing it.** A misordered Lightning version (or a custom training step that bypasses `on_before_backward`) can leave the slot empty.
- **`try/except: pass` on `__exit__`.** Closing a context manager that has already been closed once must not raise. Silent swallow is correct here because the `__exit__` is bookkeeping, not the user's work.

### 3.5. Step boundary closing

```python
def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    if TRACEML_DISABLED:
        return
    # Safety: end any active context managers
    for ctx_attr in ("_forward_ctx", "_backward_ctx",
                     "_optimizer_ctx", "_traceml_step_ctx"):
        ctx = getattr(self, ctx_attr, None)
        if ctx is not None:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
            setattr(self, ctx_attr, None)

    if not self._opt_step_occurred:
        # synthetic optimizer event for grad-accum batches
        ...

    if self._mem_tracker is not None:
        try:
            self._mem_tracker.record()
        except Exception as e:
            print(f"[TraceML] record failed: {e}", file=sys.stderr)

    TraceState.step += 1
    try:
        flush_step_events(pl_module, TraceState.step)
    except Exception as e:
        print(f"[TraceML] flush failed: {e}", file=sys.stderr)
```

This is the most load-bearing method in the integration. Five things happen, in order:

1. **Safety net for leaked context managers.**
2. **Synthetic optimizer event for grad-accum batches.**
3. **Record the memory tracker.**
4. **Bump the global step counter.** This is `TraceState.step` from `decorators.py` — the same counter `trace_step()` increments.
5. **Flush the step-time buffer.**

The order matters. **Bump the step counter before flushing**, because `flush_step_events` stamps the current step on every event it emits.

### 3.6. What `lightning.py` does *not* do (and what new integrations should add)

The existing Lightning callback omits four things you should include in a new integration:

- **No `traceml.init()` call.** The legacy auto-init path means the patches are installed by the time `trace_step`/`trace_model_instance` are first imported. New integrations should call `traceml.init(mode=...)` explicitly in their `setup` / `on_fit_start`.
- **No `trace_model_instance(pl_module)` call.** The Lightning callback predates the deep-profile layer hooks; deep-profile users have to decorate manually. New integrations must call `trace_model_instance(model)` once after the framework has finalized model wrapping.
- **No DDP / rank gating.** The Lightning callback ignores rank entirely because it only emits step-level events, which the per-rank `Database` handles correctly. New integrations must consider rank explicitly.
- **No `final_summary()` call.** New integrations should expose a constructor flag to optionally call `final_summary()` before shutdown.

These omissions are not bugs — they reflect the integration's age — but they are the four most common review comments on new integrations. Address them up front.

### 3.7. The HuggingFace pattern (subclass-based, high-level)

File: `src/traceml/integrations/huggingface.py`. The whole file:

```python
import logging
import os
from typing import Any, Dict, Optional

from traceml.decorators import trace_model_instance, trace_step

logger = logging.getLogger(__name__)
TRACEML_DISABLED = os.environ.get("TRACEML_DISABLED") == "1"

try:
    from transformers import Trainer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    Trainer = object  # Fallback for type hinting


class TraceMLTrainer(Trainer if HAS_TRANSFORMERS else object):
    def __init__(self, *args, traceml_enabled=True,
                 traceml_kwargs=None, **kwargs):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "TraceMLTrainer requires 'transformers' to be installed. "
                "Please run `pip install transformers`."
            )

        super().__init__(*args, **kwargs)
        self.traceml_enabled = traceml_enabled
        self.traceml_kwargs = traceml_kwargs
        self._traceml_hooks_attached = False

    def training_step(self, model, inputs, *args, **kwargs):
        if TRACEML_DISABLED or not self.traceml_enabled:
            return super().training_step(model, inputs, *args, **kwargs)

        if self.traceml_enabled:
            if self.traceml_kwargs is not None and (
                not self._traceml_hooks_attached
                or id(model) != getattr(self, "_attached_model_id", None)
            ):
                try:
                    trace_model_instance(model, **self.traceml_kwargs)
                    self._attached_model_id = id(model)
                    self._traceml_hooks_attached = True
                except Exception as e:
                    logger.error(
                        f"[TraceML] Failed to initialize model tracing: {e}"
                    )

            with trace_step(model):
                return super().training_step(model, inputs, *args, **kwargs)

        return super().training_step(model, inputs, *args, **kwargs)
```

The differences from Lightning:

- **Optional import, top-level guarded.** `try: from transformers import Trainer except ImportError` with a sentinel `Trainer = object` for the type hint. **This is the pattern to copy for new integrations.**
- **`class TraceMLTrainer(Trainer if HAS_TRANSFORMERS else object):`** — conditional base class. When the framework is missing, the class inherits from `object` so the import path is still valid; the `__init__` raises before doing anything else.
- **One `trace_step` block, not five `timed_region` blocks.** HuggingFace's `Trainer` doesn't expose `on_before_backward`; you only get `training_step`. So we wrap the whole thing in `trace_step(model)` and let the auto-installed forward / backward patches do the phase split.
- **Lazy model attach.** `Trainer.__init__` runs *before* `Accelerator.prepare()` and DDP wrapping; the model handed to `__init__` is not the model that will be trained. So model-attach is delayed to the first `training_step`, with a re-check on `id(model)` in case the model object gets swapped.

### 3.8. The empty `__init__.py`

`src/traceml/integrations/__init__.py` is intentionally empty (one line). Reason: importing `traceml.integrations` must not pull in `lightning` or `transformers`. The contract: **users opt into one specific integration by name.**

---

## 4. Step-by-step: adding a new integration

We'll build a hypothetical **`accelerate.py`** integration for HuggingFace Accelerate. Accelerate is a real, growing framework that wraps DDP / FSDP / DeepSpeed launch behind a single `Accelerator` object. Users call `accelerator.prepare(model, optimizer, dataloader)` and then write a hand-rolled training loop. There is no callback API, so the integration shape is "wrap the `Accelerator`, expose the same public methods, intercept the ones we care about."

The end state: users replace `from accelerate import Accelerator` with `from traceml.integrations.accelerate import TraceMLAccelerator`, and the rest of their code is unchanged.

### Step 1 — Create the integration file

Path: `src/traceml/integrations/accelerate.py`. Follow the HuggingFace pattern:

```python
"""
TraceML integration for HuggingFace Accelerate.

Lifecycle mapping
-----------------
- Accelerator.__init__   -> traceml.init(mode=...)
- Accelerator.prepare(model, ...) -> trace_model_instance(prepared_model)
- Accelerator.backward(loss)      -> step boundary inferred by user wrapping
                                     their loop in `with traceml_acc.trace():`
- Accelerator.end_training (or process exit) -> optional final_summary
"""

from __future__ import annotations
import logging
import os
from contextlib import contextmanager
from typing import Any, Optional

import traceml
from traceml.api import trace_model_instance, trace_step

logger = logging.getLogger(__name__)
TRACEML_DISABLED = os.environ.get("TRACEML_DISABLED") == "1"

try:
    from accelerate import Accelerator
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False
    Accelerator = object


class TraceMLAccelerator(Accelerator if HAS_ACCELERATE else object):
    """
    A subclass of `accelerate.Accelerator` that auto-instruments TraceML.
    """

    def __init__(self, *args, traceml_enabled: bool = True,
                 traceml_init_mode: str = "auto",
                 traceml_kwargs: Optional[dict] = None, **kwargs):
        if not HAS_ACCELERATE:
            raise ImportError(
                "TraceMLAccelerator requires 'accelerate' to be installed. "
                "Please run `pip install traceml-ai[hf]`."
            )

        super().__init__(*args, **kwargs)
        self.traceml_enabled = traceml_enabled
        self._traceml_init_mode = traceml_init_mode
        self._traceml_kwargs = traceml_kwargs
        self._traceml_hooks_attached = False

        if not (TRACEML_DISABLED or not self.traceml_enabled):
            try:
                traceml.init(mode=self._traceml_init_mode)
            except RuntimeError as e:
                logger.warning(f"[TraceML] init skipped: {e}")
            except Exception as e:
                logger.error(f"[TraceML] init failed: {e}")

    def prepare(self, *args, **kwargs):
        prepared = super().prepare(*args, **kwargs)

        if TRACEML_DISABLED or not self.traceml_enabled:
            return prepared

        try:
            from torch import nn
            tuple_out = prepared if isinstance(prepared, tuple) else (prepared,)
            for obj in tuple_out:
                if isinstance(obj, nn.Module):
                    if (not self._traceml_hooks_attached
                            and self._traceml_kwargs is not None):
                        trace_model_instance(obj, **self._traceml_kwargs)
                        self._traceml_hooks_attached = True
                    break
        except Exception as e:
            logger.error(f"[TraceML] trace_model_instance failed: {e}")

        return prepared

    @contextmanager
    def trace_step(self):
        if TRACEML_DISABLED or not self.traceml_enabled:
            yield
            return

        from torch import nn
        with trace_step(nn.Module()):
            yield

    def end_training(self) -> None:
        if hasattr(super(), "end_training"):
            super().end_training()  # type: ignore[misc]
```

That file is roughly 130 lines. The shape is the same as `huggingface.py`.

### Step 2 — Pick the optional-dependency extra

Open `pyproject.toml`. Accelerate is already declared under the `hf` extra:

```toml
hf = [
    "transformers",
    "accelerate>=0.26.0",
]
```

If you were adding a new framework, add a new extra:

```toml
trl = [
    "trl>=0.7.0",
]
```

Pin the lower bound to a tested version. Do **not** pin an upper bound unless you have a concrete reason. Document the tested upper bound in the module docstring rather than enforcing it in pyproject.

### Step 3 — Wire the explicit init call

The `accelerate.py` integration above calls `traceml.init(mode=...)` in `__init__`. This is the explicit-init pattern that new integrations should adopt; the existing Lightning integration relies on the legacy auto-init path through `from traceml.decorators import TraceState`, which is silent and version-fragile.

Three init-mode choices:

| `mode=` | What it does | When to use |
|---|---|---|
| `"auto"` | Installs all auto-patches (forward, backward, dataloader). | Default. Pick this unless you know the framework already times those phases. |
| `"manual"` | Installs no auto-patches. The integration is responsible for all timing. | Use when the framework has its own perf callback that you don't want to double-count. |
| `"selective"` | Installs only the named patches. | Use when you want some auto-instrumentation but the framework already handles e.g. dataloader timing. |

**Idempotency:** `traceml.init()` raises `RuntimeError` on a second call with a different config. The wrapper `try/except RuntimeError` in our `__init__` swallows that case as a warning, because users routinely call `traceml.init()` themselves before instantiating the integration.

### Step 4 — Wire model-attach to the right point

The model handed to `Accelerator.__init__` is *not* the model that will be trained — `prepare()` wraps it in DDP/FSDP/DeepSpeed. So we must attach per-layer hooks in `prepare()`, not in `__init__`. Compare to:

- HuggingFace `Trainer`: `prepare()` is hidden inside `train()`; we attach lazily on the first `training_step`, with `id(model)` re-check.
- Lightning: `pl_module` arrives already-wrapped on `on_fit_start`; we attach there.
- Accelerate: explicit `prepare()` call from user; we attach there.

The general rule: **attach on the latest event that hands you the post-wrapping model.**

### Step 5 — Wire the step boundary

The `accelerate.py` integration exposes `trace_step()` as a method users must explicitly enter. This is necessary because Accelerate has no batch callback. Users opt in:

```python
for batch in dataloader:
    with accelerator.trace_step():
        ...
```

This is a small bit of friction but unavoidable given Accelerate's API. **Document the friction prominently in the module docstring**, because "single-line integration" is a marketing claim and we shouldn't break it silently.

For a callback-based integration (Lightning, HF `TrainerCallback`), step boundary detection is automatic via `on_train_batch_start` / `on_train_batch_end` and the user adds nothing.

### Step 6 — Wire the shutdown path

The `accelerate.py` integration above does not call `final_summary()`. That's the right default. If you want to expose an opt-in flag:

```python
def __init__(self, *args, traceml_final_summary: bool = False, **kwargs):
    ...
    self._traceml_final_summary = traceml_final_summary

def end_training(self) -> None:
    if hasattr(super(), "end_training"):
        super().end_training()
    if self._traceml_final_summary and not TRACEML_DISABLED:
        try:
            traceml.final_summary(timeout_sec=30.0)
        except Exception as e:
            logger.error(f"[TraceML] final_summary failed: {e}")
```

The `timeout_sec=30.0` default means an unresponsive aggregator will block the user's process for half a minute. That is why this is opt-in.

### Step 7 — Don't touch `__init__.py`

`src/traceml/integrations/__init__.py` stays empty. Users will write:

```python
from traceml.integrations.accelerate import TraceMLAccelerator
```

Do not add a re-export in `integrations/__init__.py`. See §6 for why.

### Step 8 — Update `pyproject.toml` if needed

Already done in Step 2 if you added a new extra.

### Step 9 — Document the integration

Add a short docs page under `docs/integrations/`. Include:

- Install command (`pip install traceml-ai[<extra>]`).
- One-line replacement example.
- Tested framework version range.
- A pointer to the smoke-test command for this integration.

### Step 10 — Add a smoke test

See §8.

---

## 5. Common patterns and exemplars

When writing a new integration, find the closest existing one in column 2 and copy its structure.

| Pattern | Copy from |
|---|---|
| Callback-based (framework provides hook lifecycle) | `lightning.py::TraceMLCallback` |
| Subclass-based (override the framework's training class) | `huggingface.py::TraceMLTrainer` |
| Optional-import with conditional base class | `huggingface.py:13-22` (`HAS_TRANSFORMERS`, `Trainer = object` fallback) |
| `TRACEML_DISABLED` early return on every public method | `lightning.py:41, 67, 84, 95, 106, 119` |
| Lazy model-attach with `id(model)` re-check | `huggingface.py:62-71` |
| Eager model-attach after framework wrapping | hypothetical `accelerate.py::prepare` (Step 4) |
| Cross-method context manager via `__enter__` / `__exit__` on `self.<ctx>` | `lightning.py:43-46` opens, `lightning.py:71-75` closes |
| Safety net for leaked context managers | `lightning.py:122-134` |
| Synthetic event for misaligned framework callbacks (grad-accum) | `lightning.py:140-154` |
| Step counter advance + flush | `lightning.py:165-169` (`TraceState.step += 1; flush_step_events`) |
| High-level `trace_step(model)` (one block per step) | `huggingface.py:80-81` |
| Low-level `timed_region(...)` (one block per phase) | `lightning.py:43-46, 61-64, 78-81, 100-103` |
| Explicit `traceml.init(mode=...)` in `__init__` | hypothetical `accelerate.py` (Step 3); not yet in production integrations |

### Notable helpers to reuse

- `trace_step(model)` from `traceml.api` — the highest-level step boundary.
- `timed_region(name, scope, use_gpu)` from `traceml.utils.timing` — the primitive that `trace_step` is built on.
- `trace_model_instance(model, **kwargs)` from `traceml.api` — attaches per-layer hooks for deep profile.
- `flush_step_events(model, step)` from `traceml.utils.flush_buffers` — drains `_STEP_BUFFER` into `_STEP_TIME_QUEUE`. Call exactly once per step boundary, *after* incrementing `TraceState.step`.
- `TraceState.step` from `traceml.decorators` — the global step counter. Increment exactly once per optimizer step.
- `record_event(TimeEvent(...))` from `traceml.utils.timing` — write a synthetic event directly. Use only when you need to fill an alignment gap.
- `traceml.init(mode=..., patch_*=...)` from `traceml.api` — explicit init.
- `traceml.final_summary(timeout_sec=..., ...)` from `traceml.api` — opt-in end-of-run summary. Blocking; pick a timeout you can afford.

---

## 6. Optional dependency rules

This is the section that matters most for integrations. Get this wrong and you break TraceML for users who don't have your framework installed.

### 6.1. The framework is always optional

Every integrated framework MUST be declared under `[project.optional-dependencies]` in `pyproject.toml`, never under `dependencies`. This is non-negotiable. The `traceml-ai` PyPI package is small and dependency-light by design.

### 6.2. Top-level `import lightning` breaks TraceML

Every integration must guard its framework import with `try/except ImportError`, exactly as `huggingface.py:13-19` does:

```python
try:
    from transformers import Trainer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    Trainer = object  # Fallback for type hinting
```

The naked `from lightning.pytorch.callbacks import Callback` at the top of `lightning.py:4` is the one place where the existing Lightning integration fails this rule. **New integrations must use the guarded pattern.**

### 6.3. Conditional base class

```python
class TraceMLAccelerator(Accelerator if HAS_ACCELERATE else object):
    ...
```

When the framework is missing, `Accelerator = object` and the class inherits from `object` — the import path is still valid, the class is defined, but `__init__` raises a clear `ImportError`:

```python
def __init__(self, ...):
    if not HAS_ACCELERATE:
        raise ImportError(
            "TraceMLAccelerator requires 'accelerate' to be installed. "
            "Please run `pip install traceml-ai[hf]`."
        )
```

This three-line pattern — `try/except ImportError` at top, conditional base class, ImportError in `__init__` — is the entire optional-dependency contract for an integration.

### 6.4. The empty `__init__.py` is intentional

`src/traceml/integrations/__init__.py` is one byte (a newline) and stays that way. If we re-exported `TraceMLTrainer` from there, then `import traceml.integrations` would crash on machines without `transformers`. By keeping the package's `__init__.py` empty, we force users to opt into a specific integration:

```python
# Right:
from traceml.integrations.huggingface import TraceMLTrainer

# Wrong (would import every integration's framework):
from traceml.integrations import TraceMLTrainer
```

### 6.5. `HAS_<FRAMEWORK>` flag conventions

- Naming: `HAS_<FRAMEWORK>` in SCREAMING_SNAKE_CASE. `HAS_TRANSFORMERS`, `HAS_LIGHTNING`, `HAS_ACCELERATE`, `HAS_TRL`.
- Module-level constant, set inside the `try`/`except`.
- Used in the conditional base class and in the `ImportError` guard in `__init__`. Don't sprinkle `if HAS_X:` checks throughout method bodies.

### 6.6. Don't catch `ModuleNotFoundError` separately

`ImportError` covers `ModuleNotFoundError`. But **don't catch `Exception`** in the import guard. If the framework's top-level import raises something exotic (a CUDA driver mismatch, a version-skew assertion), let it propagate.

---

## 7. Overhead budget

### Targets

Integrations are at the edge of the hot path — they intercept every batch boundary on the user's training loop. The budget:

| Layer | Target | Notes |
|---|---|---|
| Public method overhead when `TRACEML_DISABLED=1` | sub-microsecond | One env-var check at module load, one `if` at method entry, immediate return. No allocation. |
| Step-boundary callback (e.g. `on_train_batch_start`) | < 50 µs | Open one or more `timed_region` contexts; build a `StepMemoryTracker`; reset it. |
| Phase-boundary callback (e.g. `on_before_backward`) | < 10 µs | Close one context, open another. Pure Python. |
| Step-end callback (`on_train_batch_end`) | < 100 µs | Close contexts, record memory tracker, increment counter, flush events. |
| Model-attach on first step (deep profile) | < 50 ms once | Walks the model graph and registers per-layer hooks. One-time cost. |

See [principles.md](principles.md) §5 for the cross-cutting overhead budget.

### Hot-path rules

1. **Module-level `TRACEML_DISABLED` capture.** Capture once at import, compare on every method call.
2. **No allocations in the disabled path.** When `TRACEML_DISABLED` is True, every public method should be a single `if` and a return.
3. **No blocking I/O.** The integration runs on the user's training thread.
4. **No `torch.cuda.synchronize()`.**
5. **Reuse `StepMemoryTracker` instances if you can.**
6. **Don't introspect the framework on every call.**

---

## 8. Testing

### Existing test patterns

- `tests/test_hf_trainer.py` — end-to-end smoke test for `TraceMLTrainer`. Skips when `transformers` is not installed.
- No dedicated test file for `TraceMLCallback` exists today.

### What a new integration's test should cover

At minimum:

1. **Construction without the framework.** `import traceml.integrations.<name>` does not raise on a machine without the framework installed. Instantiating the class raises a clear `ImportError`.
2. **`TRACEML_DISABLED=1` short-circuit.** With the env var set, every public method is functionally a passthrough.
3. **Step counter advancement.** After N batches, `TraceState.step` increased by exactly N (or N×accum_steps for grad accumulation).
4. **End-to-end smoke.** Train for a small number of steps with a tiny model; assert no exceptions.
5. **Optional: deep profile.** If your integration supports `trace_model_instance`, smoke-test that per-layer hooks attach.

### Minimal test template

Put this in `tests/test_<framework>_integration.py`:

```python
"""
Tests for TraceMLAccelerator.
"""

import pytest

try:
    import torch
    from accelerate import Accelerator
    from torch import nn
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False


@pytest.fixture(autouse=True)
def _reset_traceml(monkeypatch, tmp_path):
    monkeypatch.setenv("TRACEML_LOGS_DIR", str(tmp_path))
    monkeypatch.setenv("TRACEML_SESSION_ID", "test_session")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    yield


def test_import_without_accelerate_does_not_raise():
    """Module import is safe even on a box without accelerate."""
    import importlib
    mod = importlib.import_module("traceml.integrations.accelerate")
    assert hasattr(mod, "TraceMLAccelerator")


@pytest.mark.skipif(not HAS_ACCELERATE, reason="accelerate not installed")
def test_disabled_short_circuits(monkeypatch):
    monkeypatch.setenv("TRACEML_DISABLED", "1")
    import importlib
    import traceml.integrations.accelerate as mod
    importlib.reload(mod)

    accelerator = mod.TraceMLAccelerator(traceml_enabled=True)
    model = nn.Linear(4, 4)
    prepared = accelerator.prepare(model)

    assert isinstance(prepared, nn.Module)
    assert not getattr(accelerator, "_traceml_hooks_attached", False)


@pytest.mark.skipif(not HAS_ACCELERATE, reason="accelerate not installed")
def test_step_counter_advances():
    from traceml.decorators import TraceState

    before = TraceState.step
    accelerator = TraceMLAccelerator(traceml_enabled=True)
    model = nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model, optimizer = accelerator.prepare(model, optimizer)

    n_steps = 3
    for _ in range(n_steps):
        with accelerator.trace_step():
            x = torch.randn(2, 4)
            loss = model(x).sum()
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

    assert TraceState.step >= before + n_steps
```

### Smoke-test discipline

Every integration PR ships a manual smoke test:

```bash
pip install -e ".[dev,torch,hf]"
python -m examples.<your_framework>_smoke
TRACEML_PROFILE=run traceml watch examples/<your_framework>_smoke.py
```

Expected: training completes, telemetry rows visible in the live UI.

---

## 9. Common pitfalls

Numbered, with symptom and fix.

1. **Symptom:** `import traceml` (or `import traceml.integrations.<x>`) crashes for a user without the framework installed with `ModuleNotFoundError: No module named 'lightning'`.
   **Cause:** Top-level `from lightning.pytorch.callbacks import Callback` without an `try/except ImportError` guard.
   **Fix:** Use the guarded pattern (`huggingface.py:13-19`). See §6.

2. **Symptom:** Per-layer hooks attach to the *unwrapped* model; layer-time / layer-memory rows show up but the layer names are odd, or hooks fire at unexpected times.
   **Cause:** You called `trace_model_instance(model)` before the framework wrapped the model in DDP/FSDP/Accelerator.
   **Fix:** Move the attach call to a later lifecycle event. Lightning: `on_fit_start`. HuggingFace: lazily on first `training_step` with `id(model)` re-check. Accelerate: in `prepare()`, on the returned model.

3. **Symptom:** `TRACEML_DISABLED=1` is set but the integration still does work — log lines, hook attach attempts, init calls.
   **Cause:** The integration's public methods don't check `TRACEML_DISABLED` first.
   **Fix:** Add `if TRACEML_DISABLED: return` at the top of every public method.

4. **Symptom:** On 8-GPU DDP, the dashboard shows 8x duplicate per-layer rows or the per-layer hook attaches on every rank.
   **Cause:** You called `trace_model_instance(model)` unconditionally on every rank, but the per-layer hooks fire on all 8 ranks for the same logical layer.
   **Fix:** Layer-time / memory samplers *want* per-rank rows. Don't gate the attach by rank; instead, gate the rendering by rank in the renderer.

5. **Symptom:** Training finishes successfully but the summary JSON is empty.
   **Cause:** The integration never called `flush_step_events`, so events accumulated in `_STEP_BUFFER` and never reached the queue.
   **Fix:** Call `flush_step_events(model, TraceState.step)` exactly once per step boundary, in the step-end callback, *after* incrementing `TraceState.step`.

6. **Symptom:** Integration works on framework v2.6, breaks on v2.7 with `TypeError: on_train_batch_start() takes 4 positional arguments but 5 were given`.
   **Cause:** Framework changed a callback signature between releases.
   **Fix:** Pin the *lower* bound in `pyproject.toml` to a tested version. Document the *tested upper bound* in the module docstring. Use `*args, **kwargs` in callback signatures when you don't need the specific arguments.

7. **Symptom:** Users report that "the integration says lightning isn't installed" but they *do* have lightning installed.
   **Cause:** Your `try/except ImportError` is too broad — it caught something else and silently set `HAS_LIGHTNING = False`.
   **Fix:** Catch `ImportError` only.

8. **Symptom:** User calls `traceml.init(mode="manual")` themselves before instantiating your integration; the integration's `__init__` then raises `RuntimeError`.
   **Cause:** Your integration's `__init__` calls `traceml.init(mode="auto")` unconditionally; the user's earlier `manual` init conflicts.
   **Fix:** Wrap the init call in `try/except RuntimeError` and treat "already initialized" as a warning, not an error. The user's explicit choice wins.

9. **Symptom:** Steps and optimizer events are misaligned in the renderer.
   **Cause:** Gradient accumulation. The framework calls `on_before_optimizer_step` once per N micro-batches, but you treat every batch as a step.
   **Fix:** Either (a) emit a synthetic `optimizer_step` event on accumulating batches (Lightning's pattern) or (b) advance `TraceState.step` only when an optimizer step actually happens (HuggingFace's pattern).

10. **Symptom:** Model-attach happens on a model that is about to be wrapped by DDP/FSDP. Hooks attach to the wrong instance.
    **Cause:** Wrong lifecycle event for the attach. See pitfall #2.

11. **Symptom:** `flush_step_events` runs *before* `TraceState.step += 1`. All step-time events are stamped with the previous step number.
    **Cause:** Wrong order in the step-end callback.
    **Fix:** Always `TraceState.step += 1` *first*, then `flush_step_events(model, TraceState.step)`.

12. **Symptom:** Aggregator logs "received payload from rank 1 but TCP server expected only rank 0" or similar rank-mismatch errors.
    **Cause:** The framework manages distributed launch with its own rank assignment.
    **Fix:** Don't try to "fix" rank inside the integration. The transport layer reads `RANK` / `LOCAL_RANK` / `WORLD_SIZE` from the environment; the framework sets these correctly before calling user code.

13. **Symptom:** `final_summary()` blocks for 30 s on every training run, even when the user didn't ask for a summary.
    **Cause:** You called `final_summary()` unconditionally on shutdown.
    **Fix:** Make it opt-in via a constructor flag.

14. **Symptom:** `id(model)` re-check (HuggingFace pattern) keeps re-attaching hooks every step.
    **Cause:** Accelerator / DDP swaps the model object between steps in weird edge cases.
    **Fix:** Use a sentinel attribute on the model itself (`model._traceml_attached = True`) instead of comparing `id(model)`.

---

## 10. Checklist before opening a PR

1. [ ] New integration file at `src/traceml/integrations/<name>.py`.
2. [ ] Optional `import` pattern: `try: import <framework> / HAS_<FRAMEWORK> = True / except ImportError: HAS_<FRAMEWORK> = False / <Class> = object`.
3. [ ] Conditional base class: `class TraceMLX(<Class> if HAS_X else object):`.
4. [ ] `ImportError` raised in `__init__` if the framework is missing, with a clear `pip install traceml-ai[<extra>]` message.
5. [ ] All four lifecycle events handled: init, model attach, step boundary, finalize.
6. [ ] `TRACEML_DISABLED` captured at module load; checked at the top of every public method.
7. [ ] Step counter advance + flush in the right order: `TraceState.step += 1` *then* `flush_step_events(model, TraceState.step)`.
8. [ ] Module-level docstring documents: framework class wrapped, integration shape (subclass / callback), tested framework version range, install command, lifecycle mapping table.
9. [ ] Function docstrings on every public method.
10. [ ] DDP / multi-rank behavior considered explicitly.
11. [ ] `pyproject.toml` extras updated. New extra named or existing extra extended.
12. [ ] `__init__.py` of `integrations/` left empty. No re-exports.
13. [ ] Tests in `tests/test_<name>_integration.py` covering: import without framework, `TRACEML_DISABLED` short-circuit, step counter advance, end-to-end smoke. `pytest` passes.
14. [ ] Smoke test: minimal example that uses the framework + the integration, completes, produces telemetry. Documented in PR description with command and expected output.
15. [ ] Smoke test against framework version range — at least the oldest supported and the latest released.
16. [ ] CHANGELOG entry: `Added: <framework> integration via traceml.integrations.<name>`.
17. [ ] [principles.md](principles.md) compliance verified: fail-open, overhead budget, wire-compat, logging convention.
18. [ ] `pre-commit run --all-files` clean.
19. [ ] Commit message short, single-line, no `Co-Authored-By` trailers.

---

## 11. Appendix

### 11.1. Detecting framework version at runtime

Many pitfalls (§9 #6, #14) come from version drift. Query the framework's version on construction and stash it on the manifest:

```python
import lightning.pytorch as pl
self._framework_version = pl.__version__
```

The convention: document tested versions in the module docstring, log a warning when the runtime version falls outside the tested range:

```python
TESTED_VERSIONS = ("2.6.0", "2.7.0")
if pl.__version__ not in TESTED_VERSIONS:
    logger.warning(
        f"[TraceML] Lightning {pl.__version__} is outside the tested "
        f"range {TESTED_VERSIONS}. The integration may behave unexpectedly."
    )
```

### 11.2. The lifecycle ordering invariant

Across every integration:

```
init  →  model attach  →  (step boundary  →  step boundary  → ...)  →  finalize
```

Constraints:

- `init` must complete before *any* `trace_step` opens.
- `model attach` must complete after the framework finishes wrapping (DDP/FSDP/Accelerator), and before the first step boundary.
- Each `step boundary` is `TraceState.step += 1` *then* `flush_step_events`.
- `finalize` (if explicit) must happen after the last step boundary.

### 11.3. The "manual decoration" fallback

If your integration breaks (framework version drift, unsupported feature, custom training loop), the user can always drop your integration and decorate manually. Document this fallback in your integration's docstring:

```python
"""
Fallback
--------
If TraceMLAccelerator does not work for your setup, you can drop the
integration and use the underlying primitives directly:

    import traceml
    from traceml.api import trace_step, trace_model_instance

    traceml.init(mode="auto")

    accelerator = Accelerator()
    model, optimizer = accelerator.prepare(model, optimizer)
    trace_model_instance(model)

    for batch in dataloader:
        with trace_step(model):
            loss = model(batch).loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
"""
```

### 11.4. When NOT to add a new integration

Sometimes a framework already has good first-class instrumentation hooks that overlap with what TraceML offers. Before writing a new integration, ask:

- Can the user drop the four-line manual-decoration pattern (§11.3) into their training script with no friction?
- Does the framework have an existing perf / profiling callback you would double-count?
- Is the framework popular enough to justify the maintenance treadmill?

If the answers are "yes / yes / no", the right move is a docs page showing the manual-decoration pattern, not a new integration.

---

## Gaps and ambiguities encountered while writing this guide

These are places where the current source does not fully pin down a contract.

- **Lightning callback's lack of explicit `traceml.init()`.** The existing `lightning.py` relies on the legacy auto-init path. New integrations should call `traceml.init(mode=...)` explicitly. Whether to retrofit the Lightning callback to do the same is an open question — backward compatibility for v0.2.x users argues against it.

- **Lightning callback's lack of `trace_model_instance`.** Deep-profile layer hooks are not attached by `TraceMLCallback`. Lightning users in `deep` profile have to decorate manually. This is a real feature gap.

- **No DDP / rank gating in the existing integrations.** Both `TraceMLTrainer` and `TraceMLCallback` are completely rank-agnostic.

- **`final_summary` policy is undecided.** Neither existing integration calls it. The 30 s default timeout is hostile for an automatic call.

- **Optional dependency boundary inside `traceml`.** The current rule is convention, not enforcement. A linter rule that catches top-level un-guarded `import lightning` in `traceml/integrations/*.py` would close the gap.

- **No shared test fixtures for integrations.** `tests/test_hf_trainer.py` rolls its own setup. There is no `MockFramework` or `FakeAccelerator` fixture to validate integration shape without installing the real framework.

- **Framework version test matrix.** PRs are tested against whatever versions the dev box has. There is no CI matrix that exercises multiple framework versions.

- **Accelerate's lack of a callback API.** The hypothetical `accelerate.py` integration exposes `accelerator.trace_step()` as a method users must explicitly enter. This is friction. If Accelerate ever ships a `TrainerCallback`-equivalent, the integration shape should switch to callback-based.
