# How to add a new instrumentation patch

---
Feature type: instrumentation patch
Risk level: high
Cross-cutting impact: multiple subsystems
PyTorch coupling: deep
Reference PRs: #87 (H2D `.to()` timing), prior art for `forward_auto_timer_patch.py`, `backward_auto_timer_patch.py`, `dataloader_patch.py`
Companion reviewer guide: `review_patch.md`
---

This guide teaches you how to add a new auto-instrumentation patch to TraceML. It assumes you have read `add_sampler.md` (because every patch ultimately feeds an existing sampler), the top-level `CLAUDE.md`, and the PR #87 review file at `traceml/Notes/PR_87_review_through_walkthroughs.md`. That review is the freshest exemplar — go re-read it before opening an editor for any patch PR. Section 2.1 of that file (the four-patch consistency table) is the single most useful artifact for grading whether your design is consistent with the family.

---

## 1. Intro and mental model

### What is "a patch" in TraceML?

A **patch** monkey-patches a function on a PyTorch internal (or, rarely, a neighboring library) so that calls to that function are wrapped in `timed_region(...)`. The wrapper records CPU wall-clock times and, when CUDA is available, two CUDA events (start and end) on the current stream. The result is a `TimeEvent` appended to the module-global `_STEP_BUFFER` deque in `src/traceml/utils/timing.py`.

That is the entire job description. A patch:

1. **Replaces** a target callable on a class or module with a wrapper.
2. **Gates** the wrapper on a thread-local enable flag so instrumentation only fires inside `trace_step()`.
3. **Emits** one (or more) `TimeEvent`s per call via `timed_region`.
4. **Preserves** the original behavior on the fast path and on every error path.

Patches do not own any storage, any sampler, any database, any rendering, and no transport. They sit at the very front of the pipeline. Everything downstream — buffer drain, queue handoff, GPU resolution, aggregation, TCP, SQLite, renderer — is reused from the existing infrastructure. This is why a new patch is a roughly 100–150 LOC change in `src/traceml/utils/patches/` plus a registration line, plus tests.

For the end-to-end data flow your TimeEvent rides on, see [pipeline_walkthrough.md](pipeline_walkthrough.md). Stations 1–3 of Appendix D in the PR #87 review file walk it concretely for the H2D event.

### What a patch is NOT

- Not a sampler. Patches do not poll. They are reactive — they fire when the patched function is called.
- Not a hook. `nn.Module.register_forward_hook` is a **public PyTorch API** for inserting a callable around a specific module instance. Patches mutate the class itself, globally, for every instance in the process. Different axis. See `add_sampler.md` §3 for why we patch `nn.Module.__call__` instead of using `register_forward_hook`.
- Not a decorator. `trace_time` (in `instrumentation.py`) lets the user opt in around their own function. Patches are zero-code: the user gets timing by calling `traceml.init(...)`, not by editing their training loop.

### Decision matrix: patch vs. hook vs. decorator

This is the first decision. PR #87 patched `torch.Tensor.to`; an alternative was an `nn.Module` hook on `Module.to`. Each axis has different trade-offs.

| Mechanism             | Target                     | Scope             | Use when                                                                                          |
|-----------------------|----------------------------|-------------------|---------------------------------------------------------------------------------------------------|
| **Patch**             | a class or module function | global, every call| The op is a PyTorch primitive every model uses (`nn.Module.__call__`, `Tensor.backward`, `DataLoader.__iter__`, `Tensor.to`). |
| **Per-instance hook** | a single `nn.Module` instance | one model, one site | The signal is per-layer (forward time per layer, backward memory per layer). See `add_sampler.md` §5 and W5 for layer hooks. |
| **Decorator**         | a user function            | one user-defined region | The signal is custom user code — `trace_time` exists for this. Not infrastructure work. |

For the H2D case, `Module.to` was the wrong patch surface because **`nn.Module.to` calls `tensor.to` once per parameter** via `_apply` — see PR #87 review §3.2. Patching at the lower-level `Tensor.to` site is the only way to capture batch H2Ds, host-pinned dataloader transfers, and explicit `.to(device)` from the user, all under one event name. The trade-off is that `Tensor.to` is also called by `_apply` traversal, which forces you to filter (a new axis introduced by PR #87, see §5).

Heuristic: **patch when the target is one method on a class that every PyTorch user touches**, and the signal you want is "every call to that method, gated to inside `trace_step()`." Anything per-instance or per-layer is a hook. Anything per-user-function is a decorator.

For the patches/timing-primitives full walkthrough see [W4](../deep_dive/code-walkthroughs.md#w4-patches-timing-primitives-how-zero-code-instrumentation-actually-works); for hooks, [W5](../deep_dive/code-walkthroughs.md#w5-per-layer-hooks-forwardbackward-time-and-memory-hooks); for the user-facing API, [W3](../deep_dive/code-walkthroughs.md#w3-user-facing-api-decorators-instrumentation-wrappers).

### The instrumentation-site contract

Every patch in TraceML honors the same external contract:

- **One or more `TimeEvent`s per invocation**, with a `name` field of the form `_traceml_internal:<your_name>` (string).
- **Recorded on the current CUDA stream** via `start_evt.record()` / `end_evt.record()` issued inside `timed_region.__enter__` and `__exit__`. `timed_region` does this for you — never call `.record()` yourself.
- **Written to `_STEP_BUFFER`** (the per-rank, single-thread deque in `timing.py:111`) for `scope=TimeScope.STEP` events. Drained on step boundary by `flush_step_time_buffer(step)` (`timing.py:163-180`).
- **Resolved later, never inside the patch**. `event.elapsed_time` is called by `StepTimeSampler` via `TimeEvent.try_resolve` (`timing.py:65-90`). The patch never calls `torch.cuda.synchronize()` and never calls `event.synchronize()`.
- **Aggregated downstream** by `(name, device, is_gpu)` in `StepTimeSampler._build_step_payload`. The `name` is the key. Renaming the name across releases is a wire-format break — see §6.

That is it. If your patch honors this contract, downstream code does not need to change.

### The TLS gate principle

Instrumentation only fires inside `trace_step()`. Every patch except `dataloader_patch` defines a thread-local enable flag and consults it on entry. The flag is raised by a context-manager activator (e.g. `forward_auto_timer`) entered inside `trace_step` in `src/traceml/instrumentation.py:150`:

```python
with forward_auto_timer(), backward_auto_timer():
    ...
```

Why bother with the gate? Because the patched function is also called during **model setup, evaluation mode toggling, validation, gradient probes, and checkpoint load** — none of which are training steps. PR #87 chose this same pattern via `_H2D_TLS._traceml_h2d_enabled`: setup-time `model.to("cuda")` runs through the patched function but short-circuits to the original because the flag is False outside `trace_step()`.

`dataloader_patch.py` is the only existing patch without a TLS gate, because `DataLoader.__iter__` is **only** called inside the training loop — construction-time iteration is not a real-world pattern. See the four-patch table in §5.

### Fail-open, always

The single non-negotiable: a patch must **never break training**. If `timed_region` setup fails, if `get_cuda_event()` returns nothing, if your filter raises — fall back to the original function, log to stderr with a `[TraceML]` prefix, and return whatever the original would have returned. This is the exact same rule that `principles.md` codifies for samplers. See [principles.md](principles.md) for the cross-cutting fail-open / overhead / wire-compat rules; do not re-derive them in your patch.

---

## 2. Before you start: decisions to make

Answer all of these before opening an editor. Write them in the PR description; reviewers will check.

- [ ] **Target function.** Exactly which class / module + method are you patching? Use the fully qualified path (`torch.Tensor.to`, `torch.optim.Optimizer.step`, `torch.distributed.all_reduce`). If you are patching multiple targets in one module (PR #87's `backward` patches `Tensor.backward` AND `autograd.backward`), name both.
- [ ] **Decision matrix outcome.** Patch vs. hook vs. decorator — pick one, explain why. PR #87 picked patch over hook because parameter-traversal via `_apply` makes hooks too coarse. Document the equivalent reasoning.
- [ ] **`name` field.** What string will the `TimeEvent` carry? The convention is `_traceml_internal:<your_name>` (lowercase, underscore-separated, descriptive). The name is the wire-level contract — every renderer that wants to display your timing pattern-matches on this string. Pick carefully; renaming later is a breaking change. See §6.
- [ ] **`scope`.** `TimeScope.STEP` (almost always), `TimeScope.GLOBAL` (rare; for events outside `trace_step` like checkpoint save), or something else (you would need to extend the `TimeScope` enum, which is out of scope for one patch). Pick `STEP` unless you have a written reason.
- [ ] **`use_gpu`.** True if the patched op runs on the CUDA stream and you want stream-side wall-clock; False if it is CPU-only (e.g., `dataloader_patch` patches `__iter__` which is dispatcher-side Python). `True` allocates two CUDA events from the pool per call — overhead matters here, see §7.
- [ ] **Target filter.** Does every call to the target carry the signal you want, or do you need to skip uninteresting calls? PR #87 introduces the **target filter** axis: only `.to()` calls whose destination is a CUDA device are timed; CPU-only `.to()` and dtype-only `.to(float16)` are short-circuited. None of the pre-PR-87 patches had this concept because every forward / every backward / every loader iter is interesting. If your patch needs a filter, document its truth table (and write the test from §8).
- [ ] **TLS depth tracking.** Does the patched call **nest** naturally? A forward call recurses through every submodule's `__call__`. A backward call recurses through retain-graph + higher-order grad. Both need a depth counter that fires `timed_region` only at depth 0 (see `forward_auto_timer_patch.py:_depth`, `backward_auto_timer_patch.py:_depth`). A `.to()` call does not nest — `tensor.to` is a leaf op. PR #87 correctly omits depth tracking. Document the call shape and pick.
- [ ] **Wrapper class for manual API.** Will users also want a per-instance manual wrapper (`wrap_<name>(...)`)? PR #87 added `_WrappedH2D` and `wrap_h2d()` to `src/traceml/wrappers.py` for the manual / selective modes. If yes, it adds a class to `wrappers.py`, an `_ensure_<name>_wrapper_allowed()` guard, and an entry in `__all__`. Decide up front; the wrapper is half the PR.
- [ ] **Patch composition.** Does another patch already wrap the same target? `forward_auto_timer_patch.py` patches `nn.Module.__call__`. If a future patch also wraps `__call__` (say, a separate `nn.Module` mode-toggle timing patch), the second patch would see the first patch's wrapper as "the original." Order of installation in `_apply_requested_patches` matters. Default: do not double-wrap a class method.
- [ ] **Idempotency sentinel.** What attribute will guard against double patching? The convention is `<target>._traceml_<name>_patched = True`. The check happens at the top of your `patch_<name>()` function and bails immediately. See `forward_auto_timer_patch.py:43-47`.
- [ ] **Reversibility.** Tests need to undo the patch, otherwise patches from one test leak into the next. Store the original (`_ORIG_<TARGET> = <target>`) at module scope so a teardown fixture can restore it. Do **not** rely on `importlib.reload` (PR #87 review §3.6 — it is fragile because the module's `_ORIG_<TARGET>` capture can be polluted by leaked state from earlier tests).
- [ ] **Profile gating.** Should the patch always install (every profile that uses `init(mode="auto")`), or only some? Default: always-on, since `TraceMLInitConfig` decides via `mode`/`patch_<name>` rather than via the CLI profile string. Patches are not gated on `TRACEML_PROFILE`.

---

## 3. Anatomy of an existing patch (annotated walkthrough)

We will walk through `forward_auto_timer_patch.py` end-to-end. Reasons for this choice:

- It is the cleanest exemplar — 64 lines total, no target filter, no wrapper class.
- It exercises every primitive a patch needs: original storage, TLS gate, depth tracking, `timed_region`, idempotent install, context-manager activator.
- The shape it uses is the shape every other patch (including PR #87's H2D patch) follows.

File: `src/traceml/utils/patches/forward_auto_timer_patch.py`.

### 3.1. Imports and module-scope captures

```python
import threading

import torch.nn as nn

from traceml.utils.timing import timed_region

_TLS = threading.local()
_ORIG_MODULE_CALL = nn.Module.__call__
```

Three things to notice:

- **`threading.local()` is a module-level singleton.** Every thread that enters the patched method gets its own view of `_traceml_forward_enabled` and `_traceml_forward_depth`. This matters in DDP: each rank is a separate process, so threading is mostly moot, but DataLoader workers are separate processes too (forked with `fork` or spawned with `spawn`); inside any one process, multiple threads still need isolation.
- **`_ORIG_MODULE_CALL` is captured at import time.** If your patch's module is imported *after* some other code has already mutated `nn.Module.__call__`, you will capture the *mutated* function as "the original" and chain. That is why patch-installation order in `initialization.py::_apply_requested_patches` is fixed and why the initialization module is imported lazily, not at top-level of `traceml`.
- **No top-level side effects.** Importing `forward_auto_timer_patch` does not patch anything. `patch_forward()` does.

### 3.2. The TLS-gated wrapper

```python
def _enabled() -> bool:
    return bool(getattr(_TLS, "_traceml_forward_enabled", False))


def _depth() -> int:
    return int(getattr(_TLS, "_traceml_forward_depth", 0))


def _set_depth(v: int) -> None:
    setattr(_TLS, "_traceml_forward_depth", v)


def _traceml_module_call(self: nn.Module, *args, **kwargs):
    if not _enabled():
        return _ORIG_MODULE_CALL(self, *args, **kwargs)

    # Only time the OUTERMOST forward to avoid submodule spam
    if _depth() > 0:
        return _ORIG_MODULE_CALL(self, *args, **kwargs)

    _set_depth(_depth() + 1)
    try:
        with timed_region(
            "_traceml_internal:forward_time", scope="step", use_gpu=True
        ):
            return _ORIG_MODULE_CALL(self, *args, **kwargs)
    finally:
        _set_depth(_depth() - 1)
```

The shape is always:

1. **Fast-path bail-out.** If TLS gate is False, return the original. No allocation, no try/except, one attribute lookup. This branch runs every time the user instantiates a model, every time a `model.eval` mode toggle triggers a forward, every time a DataLoader worker process loads a batch. It must be free.
2. **Depth check** (when applicable). If we are already inside a timed forward, fall through to the original — submodule forwards are noise. Without this, a 100-layer ResNet emits 100 events per step.
3. **Increment depth, time, decrement.** The `try/finally` is mandatory: if the user's forward raises, the depth counter must still decrement, otherwise the next step will skip timing entirely.
4. **`timed_region(name, scope, use_gpu)`.** This is your only call into `traceml.utils.timing`. It yields, you call the original, the context manager handles event allocation, recording, and pushing to `_STEP_BUFFER`.

The patch never:

- Calls `torch.cuda.synchronize()`. That would force the training thread to wait on the GPU and destroy overhead budget.
- Calls `event.synchronize()` or `event.elapsed_time()`. That is the sampler's job in `TimeEvent.try_resolve`.
- Allocates a CUDA event directly. `timed_region` calls `get_cuda_event()` from the pool; recycling happens when the sampler resolves the event.
- Catches user exceptions. The user's forward is allowed to raise; the `try/finally` only manages depth, not error logging.

### 3.3. The install / register function

```python
def patch_forward() -> None:
    """Patch nn.Module.__call__ once."""
    if getattr(nn.Module, "_traceml_forward_patched", False):
        return
    nn.Module.__call__ = _traceml_module_call  # type: ignore[assignment]
    nn.Module._traceml_forward_patched = True
```

Three rules:

- **Idempotent.** A second call after the first must do nothing. The sentinel attribute (`_traceml_forward_patched`) is the lock. Without it, `init()` called twice in the same process would chain the wrapper around itself — every forward call would emit two events, then four, and so on.
- **Sentinel goes on the patched class itself**, not on the patch module. This makes the sentinel survive `importlib.reload`, which is what causes the test-isolation pain in PR #87 §3.6: tests that reload the patch module do not reset the sentinel on the class, so the second-test installation silently no-ops with the first test's wrapper still in place.
- **`# type: ignore[assignment]` is house style** when patching a method defined on a torch class. mypy / ruff complain otherwise.

### 3.4. The context-manager activator

```python
class forward_auto_timer:
    """
    Context manager that enables forward timing during its scope.
    Assumes patch_forward() has been called once at startup/runtime init.
    """

    def __enter__(self):
        _TLS._traceml_forward_enabled = True
        return self

    def __exit__(self, exc_type, exc, tb):
        _TLS._traceml_forward_enabled = False
        _TLS._traceml_forward_depth = 0
        return False
```

This is the activator — a separate object from the patch. Its job: raise the TLS flag on entry, lower it on exit. **Always reset the depth counter on exit.** If the user's training step raises mid-forward, the depth counter might be left at a non-zero value; resetting it on context exit prevents the next step from starting at depth 1 and skipping timing entirely.

The activator is consumed in `trace_step` in `instrumentation.py:147-154`:

```python
with timed_region(
    "_traceml_internal:step_time", scope="step", use_gpu=False
):
    with forward_auto_timer(), backward_auto_timer():
        if _should_auto_install_optimizer_timing():
            ensure_optimizer_timing_installed()
        yield
        step_completed = True
```

PR #87 adds `h2d_auto_timer()` to that nested `with` line. That single line edit is the only `instrumentation.py` change in the PR.

### 3.5. What the file does NOT contain

- No filter function. Every `nn.Module.__call__` is timed (when enabled and at depth 0). The filter axis is PR #87's contribution.
- No wrapper class. `wrap_forward` exists in `wrappers.py` for users in manual mode, but it is a separate concern from the patch. It mutates one model instance in place; the patch mutates the class. PR #87's `_WrappedH2D` is the equivalent for `.to()`.
- No registration. The patch lives in `src/traceml/utils/patches/forward_auto_timer_patch.py` but is wired to the runtime through `src/traceml/initialization.py`. See §4 step 4.

That is the entire pattern. Read PR #87's `h2d_auto_timer_patch.py` against this template. Every difference is intentional and called out in the PR review's §2.1 four-patch table.

---

## 4. Step-by-step: adding a new patch

We will build a hypothetical `optimizer_step_patch.py` that times every call to `torch.optim.Optimizer.step`. The reason for this choice:

- It is a real future patch (today's optimizer timing is per-instance via `wrap_optimizer` and the global `optimizer_hook`).
- It maps onto the existing four-patch family axes cleanly: TLS-gated, no nesting (so no depth), no target filter (every optimizer step is interesting), CUDA-stream-relevant.
- It demonstrates the wrapper-class question (when do you also need a per-instance API?).

The end state:

- New file `src/traceml/utils/patches/optimizer_step_patch.py`.
- One-line edit to `src/traceml/instrumentation.py`.
- Three-line edit to `src/traceml/initialization.py` (config field + `_apply_requested_patches`).
- New `OptimizerStepWrapper` in `src/traceml/wrappers.py` (only if you also want a manual API).
- New tests in `tests/test_optimizer_step_patch.py`.

### Step 1 — Create the patch file

Path: `src/traceml/utils/patches/optimizer_step_patch.py`. Follow the same shape as `forward_auto_timer_patch.py` (§3) — module-level original capture, TLS local, fast-path bail, `timed_region` slow path inside `try/finally`, idempotent `patch_optimizer_step()` install, context-manager activator class. The intentional deletions vs. the forward template:

- No `_depth() / _set_depth()` helpers — `Optimizer.step` does not nest.
- No depth check in the wrapper.
- No depth reset in `__exit__`.

If a future variant of this patch needed depth (say, an optimizer's `zero_grad` that itself calls `step`), copy the depth pattern from `forward_auto_timer_patch.py` verbatim.

### Step 2 — Wire into `trace_step()` in `instrumentation.py`

Open `src/traceml/instrumentation.py`. Find the `with` chain in `trace_step()` (line 150 today):

```python
with forward_auto_timer(), backward_auto_timer():
```

Add the new activator:

```python
from traceml.utils.patches.optimizer_step_patch import optimizer_auto_timer

# inside trace_step:
with forward_auto_timer(), backward_auto_timer(), optimizer_auto_timer():
    ...
```

Order does not matter for the activators (they each manage their own TLS flag), but keep the order consistent with file order: forward → backward → dataloader → h2d → optimizer is the convention.

The PR #87 review file calls this out in §1.3 — the one-line edit should look exactly like:

```python
with forward_auto_timer(), backward_auto_timer(), h2d_auto_timer(), optimizer_auto_timer():
```

### Step 3 — Wire into `initialization.py`

Open `src/traceml/initialization.py`. Three edits:

#### 3a. Add the field to `TraceMLInitConfig`:

```python
@dataclass(frozen=True)
class TraceMLInitConfig:
    mode: TraceMLInitMode
    patch_dataloader: bool
    patch_forward: bool
    patch_backward: bool
    patch_optimizer_step: bool   # <-- NEW
    source: str = "user"
```

Decide on a default. PR #87's `patch_h2d: bool = False` is review-flagged (§3.10) as asymmetric with the other fields; copy the asymmetry only if you have written down why. For this patch, **no default** — the constructor always passes it explicitly, matching `patch_dataloader`, `patch_forward`, `patch_backward`.

#### 3b. Update `same_effective_configuration`:

```python
def same_effective_configuration(self, other: "TraceMLInitConfig") -> bool:
    return (
        self.mode == other.mode
        and self.patch_dataloader == other.patch_dataloader
        and self.patch_forward == other.patch_forward
        and self.patch_backward == other.patch_backward
        and self.patch_optimizer_step == other.patch_optimizer_step  # NEW
    )
```

If you forget this, two `init()` calls with different `patch_optimizer_step` values silently agree. Bug surface; do not skip.

#### 3c. Extend `_build_config`:

For `mode='auto'`, set `patch_optimizer_step=True`. For `mode='manual'`, set to False. For `mode='selective'`, accept the user-provided override. Update the `not any(...)` check at the end so passing only `patch_optimizer_step=True` counts as a valid selective config.

#### 3d. Extend `_apply_requested_patches`:

```python
if config.patch_optimizer_step:
    from traceml.utils.patches.optimizer_step_patch import (
        patch_optimizer_step,
    )

    patch_optimizer_step()
```

Lazy import. The patch module imports `torch`; we want `import traceml` to stay torch-free until init is called.

#### 3e. Update `init()` and `start()` signatures:

Add `patch_optimizer_step: Optional[bool] = None` to both signatures and thread through to `_build_config`.

That is the entire `initialization.py` change. Six lines of real code, plus docstring updates.

### Step 4 — (Optional) Add the manual wrapper

If you also want a per-instance API for users in `mode="manual"` or `mode="selective"` who do not want global patching, add a wrapper class to `src/traceml/wrappers.py`. PR #87 added `_WrappedH2D` and `wrap_h2d`; for an optimizer wrapper, follow the same shape.

Read `wrappers.py:247-286` (`wrap_optimizer`) and copy. Note that PR #87's `_WrappedH2D` is a **proxy** (the wrapped object is held as `self._obj`) because `.to()` accepts arbitrary containers; `wrap_optimizer` is **in-place mutation** because the user has the optimizer object already and identity matters (`GradScaler.step(optimizer)` requires the real instance). Pick the right pattern for your target — see W3 lines 655–697.

PR #87 review §3.4 documents a critical race condition for this pattern: the duplicate-guard runs at wrap time, but the global patch can install **after** the user wrapped. The fix is to re-check inside the wrapper's hot method (`_WrappedH2D.to()`):

```python
def to(self, *args, **kwargs) -> Any:
    if getattr(
        torch.Tensor, "_traceml_h2d_patched", False
    ):
        # Auto-patch is active — let it time, do not double-count
        return self._obj.to(*args, **kwargs)
    with timed_region(...):
        return self._obj.to(*args, **kwargs)
```

Same applies to your wrapper. Always re-read the global sentinel inside the hot method, never trust the constructor-time check alone.

### Step 5 — Public API

Open `src/traceml/api.py` and add a 3-line stub for `wrap_optimizer_step`:

```python
def wrap_optimizer_step(*args, **kwargs):
    from traceml.wrappers import wrap_optimizer_step as _impl
    return _impl(*args, **kwargs)
```

Add to `src/traceml/__init__.py`:

```python
from traceml.api import wrap_optimizer_step
__all__ += ["wrap_optimizer_step"]
```

Lazy-import is the pattern (W3 lines 523–543). Eager import would re-pull torch into every `import traceml`.

### Step 6 — Tests

Create `tests/test_optimizer_step_patch.py`. Minimal coverage in §8.

### Step 7 — Smoke test

```bash
pip install -e ".[dev,torch]"
traceml watch examples/<a small example>.py --profile run
```

Confirm the run completes, no `[TraceML]` errors on stderr, and the live UI shows an `optimizer_step` row alongside `forward_time` / `backward_time`.

If running on multi-GPU:

```bash
traceml watch example.py --nproc-per-node 2
```

Confirm timings are per-rank, no duplication.

### Step 8 — Reverse / unpatch (tests only)

Patches are global mutations to a class object. Tests need a teardown fixture that restores the original method. Do **not** rely on `importlib.reload` (PR #87 §3.6).

```python
import pytest
import torch


@pytest.fixture(autouse=True)
def _restore_optimizer_step():
    saved = torch.optim.Optimizer.step
    saved_sentinel = getattr(
        torch.optim.Optimizer, "_traceml_optimizer_patched", None
    )
    yield
    torch.optim.Optimizer.step = saved
    if saved_sentinel is None and hasattr(
        torch.optim.Optimizer, "_traceml_optimizer_patched"
    ):
        delattr(torch.optim.Optimizer, "_traceml_optimizer_patched")
    else:
        torch.optim.Optimizer._traceml_optimizer_patched = saved_sentinel
```

Snapshot the method AND the sentinel; restore both on teardown. Module-level `_ORIG_OPTIMIZER_STEP` does not help here — that captures whatever was on the class **at module-import time**, which may already be a mutated wrapper left by a prior test.

A cleaner, project-level pattern would be a session-scoped autouse fixture in `tests/conftest.py` that snapshots all patch surfaces at session start and restores at session end. This does not exist today — flagged in §11.

---

## 5. Common patterns and exemplars

Reference table. When writing a new patch, find the closest existing one in column 2 and copy.

| Pattern                                          | Copy from                                                                       |
|--------------------------------------------------|---------------------------------------------------------------------------------|
| Patch a class method, no nesting                 | `dataloader_patch.py`                                                           |
| Patch a class method, with nesting (depth gate)  | `forward_auto_timer_patch.py`                                                   |
| Patch multiple targets in one module             | `backward_auto_timer_patch.py` (Tensor.backward + autograd.backward)            |
| Add a target filter (only some calls interesting)| PR #87's `h2d_auto_timer_patch.py::_is_cuda_target`                             |
| Wrapper class: in-place mutation (preserves identity) | `wrappers.py::wrap_forward`, `wrappers.py::wrap_optimizer`                |
| Wrapper class: proxy (does not preserve identity)| PR #87's `wrappers.py::_WrappedH2D`, existing `wrappers.py::_WrappedDataLoaderFetch`, `_WrappedBackwardHandle` |
| Idempotency sentinel on the class itself         | `forward_auto_timer_patch.py::nn.Module._traceml_forward_patched`               |
| Idempotency sentinel on the module               | `backward_auto_timer_patch.py::torch._traceml_backward_patched`                 |
| Context-manager activator                        | `forward_auto_timer_patch.py::forward_auto_timer`                               |

### The four-patch consistency table

This is the single most useful artifact for patch authors. Your new patch should be checkable against every column. If a column says "YES" for the existing patches and "NO" for yours, you owe a written justification in the PR description.

| Aspect                          | `forward`                | `backward`                          | `dataloader`             | `h2d` (PR #87)               | YOUR PATCH |
|---------------------------------|--------------------------|-------------------------------------|--------------------------|------------------------------|------------|
| Target class/method             | `nn.Module.__call__`     | `Tensor.backward` + `autograd.backward` | `DataLoader.__iter__` | `Tensor.to`                  | ?          |
| Calls outside training?         | YES                      | YES                                 | NO                       | YES                          | ?          |
| Needs TLS enable flag?          | YES                      | YES                                 | NO                       | YES                          | ?          |
| Has TLS enable flag?            | YES                      | YES                                 | N/A                      | YES                          | ?          |
| Calls nest naturally?           | YES                      | YES                                 | NO                       | NO                           | ?          |
| Needs depth tracking?           | YES                      | YES                                 | NO                       | NO                           | ?          |
| Has depth tracking?             | YES                      | YES                                 | N/A                      | N/A                          | ?          |
| Sentinel attribute              | `nn.Module._traceml_forward_patched` | `torch._traceml_backward_patched` | `DataLoader._traceml_patched` | `torch.Tensor._traceml_h2d_patched` | ? |
| `use_gpu` in `timed_region`     | True                     | True                                | False                    | True                         | ?          |
| `scope`                         | step                     | step                                | step                     | step                         | ?          |
| Call-site filter?               | NO                       | NO                                  | NO                       | YES (`_is_cuda_target`)      | ?          |
| Sampler routing                 | StepTimeSampler          | StepTimeSampler                     | StepTimeSampler          | StepTimeSampler              | ?          |
| Manual wrapper available?       | `wrap_forward`           | `wrap_backward`                     | `wrap_dataloader_fetch`  | `wrap_h2d`                   | ?          |
| Wrapper style                   | in-place                 | proxy (`_WrappedBackwardHandle`)    | proxy (`_WrappedDataLoaderFetch`) | proxy (`_WrappedH2D`)   | ?          |
| Error handling                  | try/finally on depth     | try/finally on depth                | try/StopIteration in loop | try inside `timed_region`   | ?          |
| CUDA event lifecycle            | pool acquire → record → resolve via sampler → release | same                              | N/A (`use_gpu=False`)    | same as forward/backward     | ?          |

Fill in your patch's row in the PR description. Any "deviation" column needs a sentence of explanation.

### Notable helpers to reuse

- `timed_region(name, scope, use_gpu)` — `src/traceml/utils/timing.py:184`. Your only call into the timing core. Always use it; never roll your own `time.time()` + `record_event` flow — you will skip queue handling.
- `get_cuda_event()` / `return_cuda_event()` — `src/traceml/utils/cuda_event_pool.py`. **Only used inside `timed_region`, not directly from a patch.** If you find yourself calling these in your patch, you have re-implemented `timed_region` and should stop.
- `TimeScope.STEP` / `TimeScope.GLOBAL` — `src/traceml/utils/timing.py:31`. Prefer the enum over the string literal `"step"`. PR #87 review §3.7 flags string-vs-enum inconsistency within a single PR — pick one and hold the line. Convention forming around the enum.
- `_raise_duplicate_instrumentation` — `src/traceml/wrappers.py:35`. Used inside `_ensure_<name>_wrapper_allowed()` helpers in `wrappers.py`.

---

## 6. Wire-format and contract rules

### 6.1. The `name` is the contract

`TimeEvent.name` is the only thing TraceML uses to identify your patch's events through the pipeline. `StepTimeSampler._build_step_payload` aggregates by `(name, device, is_gpu)`. Renderers pattern-match on the string. Once a `name` lands in user dashboards, **renaming it is a wire-format break**.

Rules:

- New `name` strings use the prefix `_traceml_internal:`. The colon is part of the convention; do not invent your own separator.
- Lowercase, underscore-separated, descriptive. `optimizer_step`, `h2d_time`, `forward_time` — not `OptStep` or `host-to-device`.
- Once introduced, **never rename**. Add a new name if the semantics change; deprecate the old one over a release cycle.
- Wire-compat rules at large live in [principles.md](principles.md). Do not re-derive them here.

### 6.2. The `device` field

`TimeEvent.device` is set by `timed_region`:

```python
device = f"cuda:{torch.cuda.current_device()}"  # use_gpu=True path
device = "cpu"                                  # CPU-only path
```

Aggregation happens by `(name, device, is_gpu)`. This means two CUDA devices (rank 0 / rank 1 on the same node, or `cuda:0` vs `cuda:1` in single-process multi-GPU) produce separate aggregation keys.

Patches do not set `device` themselves — `timed_region` does. Do not override.

### 6.3. The pending-CUDA-event invariant

This is the single most important rule and the most common bug.

- `gpu_start.record()` and `gpu_end.record()` are called by `timed_region`'s `__enter__` and `__exit__`. They enqueue a timestamp-capture op onto the current CUDA stream and return immediately. **They do not block.**
- `gpu_start.elapsed_time(gpu_end)` is **never** called inside the patch. It is called by `StepTimeSampler` via `TimeEvent.try_resolve` — asynchronously, on a separate thread, after `gpu_end.query()` returns True (`timing.py:65-90`).
- A patch must **never** call `torch.cuda.synchronize()`, `event.synchronize()`, `event.wait()`, or `event.elapsed_time()`. Any of these forces the training thread to wait on the GPU.

If you violate this, your patch will appear to work in tests (CPU fallback, events resolve trivially) but introduce 10+ ms stalls per call on a real GPU.

### 6.4. CUDA event pool discipline

`cuda_event_pool.py` defines a 2000-slot deque-backed pool. `timed_region` acquires two events on entry, the sampler releases them on resolution. A patch must:

- **Never** call `torch.cuda.Event(enable_timing=True)` directly. Use `get_cuda_event()` (which does this for you on pool miss).
- **Never** call `return_cuda_event()` directly. The sampler owns release.
- **Never** retain a reference to a CUDA event past `timed_region`'s exit. The `TimeEvent` holds the references; the sampler clears them on `try_resolve`.

If a patch leaks events (e.g., emits events on a code path that never gets flushed because it bypasses `record_event`), the pool drains and falls back to per-call `torch.cuda.Event` allocation — a driver call, measurable overhead.

### 6.5. Wire compatibility for new patches

A new patch introduces a new `name` string. Adding a new name is **not** a breaking change — old aggregators and old renderers will see new rows in `StepTimeTable` and ignore them.

Removing or renaming an existing `name` IS a breaking change. Bump the minor version and update CHANGELOG. See [principles.md](principles.md) for versioning rules.

---

## 7. Overhead budget and performance

### Targets

- **Fast path (TLS gate False).** Sub-microsecond. The fast path runs for every `nn.Module.__call__` during model setup, every dataloader-worker forward, every `tensor.to()` during checkpoint load. Two attribute lookups and a Python function call is the budget. **No `try/except` on the steady-state branch.**
- **Slow path (TLS gate True, depth 0, filter passes).** Microseconds per call for `timed_region` overhead — two CUDA event acquisitions, two `record()` calls, one `TimeEvent` allocation, one `deque.append`. Treat sub-10-µs as acceptable.
- **Aggregate.** Patches should add **<1%** to per-step wall-clock at representative model sizes. The cross-cutting overhead budget lives in [principles.md](principles.md); do not re-derive it here.

### Hot-path rules

1. **Branch to the original on the fast path with O(1) cost.** Single TLS attribute lookup, single conditional, single original call. No allocation, no `try/except`.
2. **Filter cheaply.** If your patch needs a target filter (PR #87's `_is_cuda_target`), the filter must itself be sub-microsecond. Avoid string operations, regex, anything that allocates.
3. **Reuse CUDA events from the pool.** `timed_region` does this for you.
4. **Never `torch.cuda.synchronize()` from a patch** (see §6.3).
5. **Never log on the fast path.** `print(file=sys.stderr)` from inside `_enabled() == False` would be fired every forward of every layer of every model in the process. Logging is for error paths only.
6. **No psutil / NVML / network calls in a patch.** If you need that data, it is a sampler concern, not a patch concern.

### Micro-benchmarking

There is no formal benchmark harness for patches today. The proposed v0.2.9 benchmark workflow will provide one. Until then, roll a `timeit`. Run on the dev box. Confirm the gate-off path is within ~100 ns of baseline. If not, look for an inadvertent allocation in the fast path.

---

## 8. Testing

### Existing test patterns

There is no `tests/test_*_patch.py` for the existing three patches as a single dedicated unit test. The closest references are:

- `tests/test_h2d_timing.py` — 392 lines, 30 tests for PR #87. The reference for patch-level testing. Covers TLS on/off transitions, filter truth tables, idempotency, wrapper guard, init config across modes. Read it cover to cover before writing your own.
- `tests/test_seq_counter.py` — Database / sender shape tests.
- `tests/test_grad_accum.py` — integration-style test for `trace_step` with gradient accumulation.
- `tests/test_msgpack_roundtrip.py` — wire encoder/decoder tests.

### What a new patch's test should cover

At minimum:

1. **Idempotency.** Calling `patch_<name>()` twice does not double-wrap.
2. **Fast path correctness when TLS gate is False.** Calling the patched method outside `<your>_auto_timer()` returns the same value as the original, raises the same exception class, and emits no `TimeEvent`.
3. **Slow path correctness when TLS gate is True.** Inside `<your>_auto_timer()`, emits exactly one `TimeEvent` with the right `name`, `scope`, and populated `cpu_start` / `cpu_end`. If `use_gpu=True` and CUDA available, `gpu_start` / `gpu_end` are present.
4. **Target filter truth table** (if applicable). For every documented "interesting" / "uninteresting" input, assert the right behavior.
5. **Depth gate** (if applicable). Recursive calls inside the same patched method emit one event for the outermost call only.
6. **Error path.** When the original raises, the wrapper re-raises and leaves TLS / depth state clean.
7. **Construction-time safety.** Calling the patched method during model construction (TLS False) does not emit events and does not crash.
8. **Init config integration.** `init(mode="auto")` installs your patch. `init(mode="manual")` does not. `init(mode="selective", patch_<name>=True)` installs it; `patch_<name>=False` does not.

### Minimal test template

The template uses two autouse fixtures: one snapshots/restores the patched method and its sentinel, and one drains `_STEP_BUFFER` between tests. Then it tests idempotency (`patch_<name>()` is the same after first and second call), fast-path silence (no events when TLS is False), slow-path emission (one event when TLS is True), and error-path state cleanup. See `tests/test_h2d_timing.py` for the full reference.

The fixture pattern:

```python
@pytest.fixture(autouse=True)
def _restore_optimizer_step():
    saved_method = torch.optim.Optimizer.step
    had_sentinel = hasattr(
        torch.optim.Optimizer, "_traceml_optimizer_patched"
    )
    saved_sentinel = getattr(
        torch.optim.Optimizer, "_traceml_optimizer_patched", None
    )
    yield
    torch.optim.Optimizer.step = saved_method
    if had_sentinel:
        torch.optim.Optimizer._traceml_optimizer_patched = saved_sentinel
    else:
        if hasattr(torch.optim.Optimizer, "_traceml_optimizer_patched"):
            delattr(torch.optim.Optimizer, "_traceml_optimizer_patched")
```

Snapshot the method AND the sentinel attribute. Without the sentinel restore, later tests' `patch_optimizer_step()` calls would silently no-op against whatever this test left in place (PR #87 review §3.6).

### What you cannot test on a CPU-only CI

CUDA events fall through to the no-event branch in `timed_region` when `torch.cuda.is_available()` is False. So:

- The `gpu_start.record()` / `gpu_end.record()` calls are not exercised.
- `try_resolve()` returns immediately with `resolved=True` and no GPU time.
- The CUDA event pool is not exercised.
- `event.elapsed_time` is not exercised.

Manual GPU validation on a Lightning Studio with a CUDA box is mandatory before merge — see the checklist in §10.

---

## 9. Common pitfalls

Numbered, with symptom + cause + fix. PR #87's review file is the source for most of these. If you hit one, read the cited section.

1. **Symptom:** Telemetry shows `_traceml_internal:<your_name>` events firing during model setup (before any `trace_step` is entered).
   **Cause:** Your patch is missing the TLS enable flag, or the activator is not inside `trace_step`.
   **Fix:** Add the TLS gate following `forward_auto_timer_patch.py`. Wire the activator into `instrumentation.py:150`.

2. **Symptom:** `n_calls` for your event is 100× what you expect on the first event of every step.
   **Cause:** The patched function is called by a higher-level PyTorch helper that loops over parameters, layers, or buffers — like `nn.Module.to` calling `tensor.to` once per parameter via `_apply`. Depth tracking does not help, because each call is at depth 0 (sequential, not nested).
   **Fix:** Add a target filter that excludes the loop's caller pattern. For PR #87 the fix is `if isinstance(self, torch.nn.Parameter): return _ORIG_TENSOR_TO(...)` or a TLS flag set by patching `nn.Module._apply`.
   **Reference:** PR #87 §3.2.

3. **Symptom:** D2D copies (`cuda_tensor.to("cuda:1")`) are reported as H2D-equivalent in your patch.
   **Cause:** Your filter checks the destination but not the source. D2D uses `cudaMemcpyPeer` over NVLink/PCIe — different hardware path, different bottleneck class.
   **Fix:** Source-device check at the top of the wrapper: if the source is already on the target's device class, short-circuit.
   **Reference:** PR #87 §3.1.

4. **Symptom:** Docstring says "this measures X up to the queue boundary" but the actual code measures full transfer time.
   **Cause:** Drift between the design assumption (synchronous CPU-side timing) and the actual implementation (CUDA events on the stream resolved post-hoc).
   **Fix:** Read what `try_resolve()` actually computes in `timing.py:79`, then rewrite the docstring.
   **Reference:** PR #87 §3.3.

5. **Symptom:** A user calls `wrap_<name>(x)` before `init(mode="auto")` and then `<x>.<method>()` is double-counted.
   **Cause:** Your wrapper's `_ensure_<name>_wrapper_allowed()` check runs at construction time. The global patch installs later. The check is now stale.
   **Fix:** Re-check the global sentinel inside the wrapper's hot method, and short-circuit when the global patch is active.
   **Reference:** PR #87 §3.4.

6. **Symptom:** Users who wrap a batch dict / list with `wrap_<name>(...)` get `TypeError: '_Wrapped<X>' object is not subscriptable` when they try `wrapped["x"]` or `iter(wrapped)` before calling `.<method>()`.
   **Cause:** Python bypasses `__getattr__` for special methods.
   **Fix:** Define `__len__`, `__iter__`, `__getitem__`, `__contains__` explicitly on the wrapper class. Mirror `_WrappedDataLoaderFetch.__len__` in `wrappers.py:134`.
   **Reference:** PR #87 §3.5.

7. **Symptom:** `scope="step"` (string) and `scope=TimeScope.STEP` (enum) are used interchangeably across the codebase, sometimes within a single PR.
   **Cause:** `TimeScope` is a `str`-subclass enum, so both work. The inconsistency creates a typo surface.
   **Fix:** Pick one. Convention forming around the enum.
   **Reference:** PR #87 §3.7.

8. **Symptom:** Substring match in your filter accepts garbage device strings (`"foo-cuda-bar"` is True, `"CUDA:0"` is False with `"cuda" in s`).
   **Cause:** `"cuda" in str_arg` is over-broad and case-sensitive.
   **Fix:** Use `torch.device(str_arg).type == "cuda"` inside a try/except for `RuntimeError`/`TypeError`.
   **Reference:** PR #87 §3.9.

9. **Symptom:** CUDA event pool drains over a long run; per-step overhead climbs.
   **Cause:** Your patch emits events on a code path that bypasses the sampler's `try_resolve` → `return_cuda_event` → pool reuse cycle.
   **Fix:** Use `timed_region` as your only timing entry point. Never `torch.cuda.Event` directly.
   **Reference:** PR #87 §4.1.

10. **Symptom:** Two patches both wrap the same target class method, and the second patch sees the first patch's wrapper as "the original".
    **Cause:** `_ORIG_<TARGET> = <target>` at module scope captures whatever is on the class at import time.
    **Fix:** Do not double-wrap a class method. If you genuinely need two independent timing axes on the same target, fold them into one patch with one wrapper that emits two `TimeEvent`s with different `name`s.

11. **Symptom:** Async `tensor.to(..., non_blocking=True)` shows the same measured time as `non_blocking=False`.
    **Cause:** This is correct behavior. CUDA events record timestamps on the stream; `elapsed_time` reads them after the GPU has completed both `start` and `end`. The async DMA is *between* them on the stream.
    **Fix:** Update the docstring (see pitfall 4). The metric is right; the docstring caused the confusion.
    **Reference:** PR #87 §3.3.

12. **Symptom:** GPU events recorded by your patch resolve on the wrong stream and produce nonsensical `elapsed_time`.
    **Cause:** Your patched op runs on a stream different from `torch.cuda.current_stream()`.
    **Fix:** `timed_region` records events on whatever `torch.cuda.current_stream()` is at entry/exit. If your patched op *itself* changes the stream during execution, capture the stream at entry and pass it to event `record()` calls explicitly. Today this requires extending `timed_region`; flag it on the issue tracker.

13. **Symptom:** Your patched function is reentered from inside its own timed region.
    **Cause:** TraceML hooks may legitimately call other patched functions from inside `nn.Module.__call__`. Without re-entry guards, you double-count.
    **Fix:** Add a depth counter; only emit at depth 0.

14. **Symptom:** `importlib.reload(your_patch_module)` in one test causes the next test's `patch_<name>()` to capture the wrong "original".
    **Cause:** `_ORIG_<TARGET>` re-executes at module-import; reload captures whatever `<target>` is right now, which may already be a leaked wrapper.
    **Fix:** Do not `importlib.reload`. Use the snapshot-and-restore fixture from §8.
    **Reference:** PR #87 §3.6.

---

## 10. Checklist before opening a PR

Copy this into your PR description. Tick each box.

### Patch implementation

- [ ] New file in `src/traceml/utils/patches/<name>_auto_timer_patch.py` (or `<name>_patch.py` for non-timer patches).
- [ ] Module-level `_ORIG_<TARGET>` capture; no top-level side effects.
- [ ] Module-level `threading.local()` if a TLS gate is needed.
- [ ] Wrapper function: fast-path bail-out on TLS False; depth check (if applicable); `timed_region(...)` slow path; `try/finally` for any depth state.
- [ ] `patch_<name>()` is idempotent via a sentinel attribute.
- [ ] Context-manager activator class (`<name>_auto_timer`) raises TLS on `__enter__`, lowers it AND resets depth on `__exit__`.

### Wiring

- [ ] One-line add to `instrumentation.py::trace_step`'s nested `with`.
- [ ] `TraceMLInitConfig` extended with the new `patch_<name>` field; `same_effective_configuration` updated.
- [ ] `_build_config` extended to handle the new field in all three modes.
- [ ] `_apply_requested_patches` lazy-imports and calls `patch_<name>()`.
- [ ] `init()` and `start()` signatures threaded through.
- [ ] (Optional) `wrappers.py` adds `_Wrapped<Name>` + `wrap_<name>` + `_ensure_<name>_wrapper_allowed`. Wrapper's hot method re-checks the global sentinel.
- [ ] (Optional) `api.py` adds 3-line stub.
- [ ] (Optional) `__init__.py` exports.

### Schema and contract

- [ ] `name` follows the `_traceml_internal:<your_name>` convention.
- [ ] Filled in your patch's row in the §5 four-patch consistency table and pasted into the PR description.
- [ ] No `synchronize()`, no `elapsed_time()`, no `event.wait()` anywhere in the patch.
- [ ] CUDA events go through the pool via `timed_region` only.

### Tests

- [ ] `tests/test_<name>_patch.py` with the eight coverage areas from §8.
- [ ] Snapshot-and-restore fixture for the patched method + sentinel.
- [ ] `_STEP_BUFFER` drained between tests via autouse fixture.
- [ ] No `importlib.reload` in tests.

### Performance

- [ ] Bench the fast path off / fast path on / TLS gate on, separately. Confirm the gate-off path stays within ~100 ns of baseline.
- [ ] If the proposed v0.2.9 benchmarking workflow is in place, run it. Otherwise note "no benchmark harness yet" and bench locally.

### Smoke tests

- [ ] `traceml watch examples/<small example>.py --profile run` completes without crashes and shows your event in the live UI.
- [ ] Multi-GPU smoke: `traceml watch ... --nproc-per-node 2` shows the event per-rank.
- [ ] `--mode summary` smoke: end-of-run summary includes the event in `aggregator/summaries/step_time_diagnosis.py` output.

### Release artifacts

- [ ] CHANGELOG entry under the right release.
- [ ] Version bump if a wire-format `name` is genuinely new (minor bump); patch-level bump if everything else.
- [ ] PR description references PR #87's review file (`traceml/Notes/PR_87_review_through_walkthroughs.md`) so the reviewer can read the four-patch consistency table on their first pass.
- [ ] `pre-commit run --all-files` clean.

---

## 11. Appendix

### 11.1. Reversing / unpatching a patch in tests

The clean pattern is the snapshot-and-restore fixture from §8. The fragile pattern is `importlib.reload(your_patch_module)` followed by `patch_<name>()` again — this is what PR #87's tests do, and PR #87 review §3.6 calls it out as a real test-isolation hazard.

A better project-level pattern (not yet implemented) lives in a `tests/conftest.py` shared autouse fixture that snapshots every `(class, method, sentinel)` triple at session start and restores at session end. If you find yourself writing this in your patch's PR, propose it as a follow-up PR. Do not roll it inside your patch's test file — keep it tightly scoped to your patch.

### 11.2. Composing patches

What if a future patch wraps a function that another patch already wraps? Three options, in order of preference:

1. **Add a new event type to the existing patch.** Refactor the existing patch to emit two events when both signals are interesting. Single source of truth for the patched method.
2. **Split your concern out of the existing patch.** If the new signal is genuinely separate, let the existing patch detect a marker (`hasattr(self, "_dynamo_orig_callable")`) and emit a different `name`. Still single-wrapper.
3. **Wrap the wrapper.** Capture `_ORIG_<TARGET>` after the first patch has installed, install your second wrapper around it, ensure your sentinel guards against re-entry. This is **discouraged** because the import order between the two patch modules now matters.

If you cannot use option 1 or 2, write a long PR description explaining why; option 3 is a maintenance hazard.

### 11.3. When your patch needs information that `timed_region` does not capture

Today's `TimeEvent` schema is fixed in `timing.py:43-63`. If your patch needs to attribute a timing to additional metadata (transfer size, target device subindex, parameter count), there are two paths:

1. **Encode metadata into `name`.** Reasonable for a small, fixed set (`h2d_time_pinned`, `h2d_time_unpinned`). Not reasonable for arbitrary integers.
2. **Extend `TimeEvent`.** Add an optional metadata field. This is a wire-format change — bumps the minor version, requires migration on the aggregator side, and forces you to update every renderer that pattern-matches on the schema. Out of scope for one patch.

For new metadata, prefer option 1.

### 11.4. Patches and DDP

Patches are per-process. In DDP, every rank installs the patch independently during its `init()`. Each rank's `_STEP_BUFFER` is local; the rank's own `StepTimeSampler` drains it; the rank's `DBIncrementalSender` ships rows over TCP. The aggregator's `RemoteDBStore` keeps rows separate by rank.

Implication: a patch does not need to do anything special for DDP. Do **not** add `if rank == 0` guards inside the patch. If your patch only makes sense on rank 0, the right place to gate is the **sampler** that drains your events, not the patch itself. See `add_sampler.md` §4 step 6 for the rank-aware patterns.

### 11.5. Patches and DataLoader workers

DataLoader workers are separate processes (forked or spawned). They import TraceML normally. If `init()` is called in the main process, the workers do **not** inherit the patch state — `fork` does, `spawn` does not.

Today this is not a problem because no existing patch does meaningful work inside workers. If your patch is the first one whose target genuinely fires inside DataLoader workers, you have new work to do — consider whether your patch needs `worker_init_fn`-based re-installation.

---

## Gaps and ambiguities encountered while writing this guide

These are places where the current source is underspecified. Flag them in review if your patch lands near any of these.

- **No formal benchmark harness for patches.** §7's overhead numbers are folklore. The proposed v0.2.9 benchmarking workflow is supposed to provide one, but is not yet in the test suite. Until then, every patch author rolls a `timeit` and asserts "looks fine."
- **No shared `tests/conftest.py` for patch isolation.** Every existing patch PR re-rolls its own snapshot/restore logic, often partially. PR #87's reliance on `importlib.reload` is the most fragile instance. The proposed shared fixture in §11.1 would be 30 lines.
- **`_ORIG_<TARGET>` capture happens at module-import time.** If a future workflow installs patches dynamically, the originals are not what you think they are. The fix is to capture inside `patch_<name>()` rather than at module scope, but every existing patch does the latter.
- **`TimeScope` is open for extension but no extension protocol exists.** If your patch needs a new scope (`epoch`-level, `validation`-level), you have to extend the enum, update `record_event` to know about it, and update every consumer.
- **The four-patch table assumes patches are TimeEvent-emitters via `timed_region`.** Patches that emit something else (memory events, custom telemetry) are unmapped. If you find yourself writing a patch that doesn't go through `timed_region`, you are probably writing a hook + sampler instead — re-read §1.
- **No reverse / uninstall API.** There is no `unpatch_forward()`. Once installed, patches live for the process lifetime. Tests work around this with snapshot/restore fixtures.
- **Wrapper-class proxy pattern is repeated four times** without a base class. Each rolls its own `__getattr__`. The dunder-forwarding gap (§9 pitfall 6) recurs because there is no shared base to enforce it. A `_TraceMLProxyBase` with comprehensive `__len__` / `__iter__` / `__getitem__` / `__contains__` forwarders would prevent the bug; it doesn't exist yet.
