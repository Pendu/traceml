# Integrations

The `integrations` subsystem provides drop-in adapters that let TraceML
hook into popular PyTorch training stacks without requiring users to
manually wrap their training loop with the primitives from
[`decorators.py`](decorators.md). It ships two adapters today:
`TraceMLTrainer`, a subclass of Hugging Face's `transformers.Trainer`,
and `TraceMLCallback`, a callback for PyTorch Lightning's `Trainer`.
Both handle step-boundary detection, model-hook attachment, and graceful
bypass when TraceML is disabled, so users get the full telemetry surface
by swapping a single class or adding a single callback.

## Role in the architecture

Integrations sit one layer above `decorators.py` in the stack described
in the [architecture overview](../architecture.md). The decorator module
exposes the primitives (`trace_step` as a step-boundary context manager,
`trace_model_instance` for attaching layer-level hooks, plus the shared
`TraceState` counter), and the integrations subsystem is where those
primitives are wired into the lifecycle callbacks of an external
framework. Users who drive their own loop can call the decorators
directly; users who live inside Hugging Face `Trainer` or Lightning
`Trainer` get the same coverage for free.

Because the integrations only call into already-fail-open primitives,
they inherit the project's fail-open philosophy: a hook-attach or timing
error is logged, but it never aborts a user's training run. They also
do not talk to the aggregator, transport, or samplers directly. Step
boundaries and model hooks they install are observed by the runtime
loop and the event-driven samplers described in
[`samplers.md`](samplers.md); the plumbing that ships those events to
the aggregator is owned by [`runtime/`](runtime.md) and
[`transport/`](transport.md).

The integrations are an opt-in path: if a user prefers manual
instrumentation, nothing here runs. The subsystem is deliberately thin —
most of the heavy lifting (timing regions, memory trackers, step flush,
hook installation) is delegated to [`utils/`](utils.md) helpers so that
the framework-specific files stay focused on lifecycle wiring.

## TraceMLTrainer (Hugging Face)

`TraceMLTrainer` lives in `src/traceml/integrations/huggingface.py` and
is a thin subclass of `transformers.Trainer`. It overrides a single
method — `training_step(self, model, inputs, *args, **kwargs)` — and
wraps the call to `super().training_step(...)` inside a
`trace_step(model)` context manager. That is the entire step-boundary
mechanism: `trace_step` handles advancing the `TraceState.step` counter,
opening a timed region, and emitting the step-end flush that the
samplers rely on.

The subclass accepts two extra keyword arguments beyond the stock
`Trainer` signature:

- `traceml_enabled: bool = True` — a per-instance kill-switch. When
  `False`, `training_step` short-circuits to the parent implementation
  with no instrumentation at all.
- `traceml_kwargs: Optional[Dict[str, Any]] = None` — when provided,
  these kwargs are forwarded to
  `trace_model_instance(model, **traceml_kwargs)` to enable Deep-Dive
  layer-level hooks.

### Lazy model-hook attachment

Model-hook attachment is intentionally **lazy**. Hooks are not installed
in `__init__`; instead, the first call to `training_step` attaches them.
This matters because Hugging Face's `Trainer` wraps and moves the model
after construction (for DDP, `accelerate`, mixed precision, device
placement, etc.), and the wrapped/moved model is the one whose forward
and backward we actually need to observe.

`TraceMLTrainer` tracks two pieces of state to keep attachment
idempotent:

- `_traceml_hooks_attached: bool` — set once hooks are installed
  successfully.
- `_attached_model_id: int` — the Python `id()` of the model object that
  hooks were installed on.

If a later `training_step` call is invoked with a different model object
(`id(model) != self._attached_model_id`), the trainer reinstalls the
hooks on the new object. This covers edge cases where the framework
swaps the model mid-run (resume from checkpoint, re-wrapping,
precision-policy change).

### Error handling

The `trace_model_instance` call is wrapped in a `try/except` that logs
through the standard `logging` module under the `[TraceML]` prefix but
never re-raises. If hook attachment fails, the user still gets step-
level timing from `trace_step`; only the Deep-Dive layer view degrades.
The `super().training_step(...)` call itself is never wrapped — user
exceptions propagate normally so that training fails loudly on real
bugs.

If `transformers` is not installed, constructing `TraceMLTrainer` raises
`ImportError` with an instructive message pointing at
`pip install transformers`. If `TRACEML_DISABLED=1` is set in the
environment, `training_step` skips every instrumentation branch and
delegates straight to the parent — no hooks, no `trace_step`, no
attempts to mutate TraceML state.

## TraceMLCallback (PyTorch Lightning)

`TraceMLCallback` lives in `src/traceml/integrations/lightning.py` and
is a standard `lightning.pytorch.callbacks.Callback`. Instead of
overriding `training_step`, it wires six Lightning lifecycle hooks to
build a detailed phase-level timeline of each step:

| Lightning hook | What the callback does |
|---|---|
| `on_train_batch_start` | Opens the overall step timing region, resets the `StepMemoryTracker` for the active `pl_module`, and opens the forward-pass timing region. |
| `on_before_backward` | Closes the forward region and opens the backward region. |
| `on_after_backward` | Closes the backward region. |
| `on_before_optimizer_step` | Sets `_opt_step_occurred = True` and opens the optimizer-step timing region. |
| `on_before_zero_grad` | Closes the optimizer-step region (Lightning calls `zero_grad` right after `optimizer.step`). |
| `on_train_batch_end` | Safety-closes any still-open regions, emits a zero-duration optimizer event when the step was a gradient-accumulation micro-batch, records step memory, advances `TraceState.step`, and calls `flush_step_events`. |

All timing regions are created through `timed_region(...)` from
[`utils/timing.py`](utils.md), with internal event names such as
`_traceml_internal:step_time`, `_traceml_internal:forward_time`,
`_traceml_internal:backward_time`, and
`_traceml_internal:optimizer_step`, all at `scope="step"`. Step memory
is captured through `StepMemoryTracker` from
[`utils/step_memory.py`](utils.md), and the final flush is performed by
`flush_step_events(pl_module, TraceState.step)` from
[`utils/flush_buffers.py`](utils.md).

### Gradient accumulation

Lightning calls `on_train_batch_start` / `on_train_batch_end` once per
micro-batch but only fires the optimizer hooks on the step boundary
that actually runs `optimizer.step`. The callback treats **every
micro-batch as a step** so that fine-grained forward and backward
timings are preserved. On accumulating micro-batches where no real
optimizer step runs, it synthesizes a zero-duration
`_traceml_internal:optimizer_step` event via `record_event(...)` with
`cpu_start=0.0`, `cpu_end=0.0`, `gpu_time_ms=0.0`, `resolved=True`, and
`scope=TimeScope.STEP`.

!!! note
    The dummy event exists purely to keep step alignment across metric
    streams intact — the dashboards assume every metric has the same
    set of step indices, and a missing event would desynchronize the
    view. The `device="cpu"` tag on the dummy event is meaningless;
    only the presence and step index matter.

### Step counter ownership

The callback is the authoritative owner of `TraceState.step` under
Lightning: no `trace_step` context manager is used. Instead,
`on_train_batch_end` increments `TraceState.step += 1` and then
flushes. This mirrors what `trace_step` does for a hand-rolled loop,
just unrolled across Lightning's lifecycle hooks.

### Safety-close on batch end

`on_train_batch_end` iterates over `_forward_ctx`, `_backward_ctx`,
`_optimizer_ctx`, and `_traceml_step_ctx` and unconditionally exits any
that are still open. This guards against edge cases where a Lightning
hook raises before the natural region close — for example, a custom
`LightningModule.backward` that throws before `on_after_backward` is
reached. Each exit is individually wrapped in `try/except` so that one
stuck region cannot prevent the others from closing.

## Optional dependencies

Both integration modules treat their host framework as an **optional
dependency**:

- The Hugging Face adapter guards the import: `from transformers import
  Trainer` is wrapped in `try/except ImportError`, with a
  `HAS_TRANSFORMERS` flag and a fallback `Trainer = object` alias so
  the module still imports when `transformers` is missing.
  `TraceMLTrainer`'s class declaration uses
  `Trainer if HAS_TRANSFORMERS else object` as its base, and `__init__`
  raises a clear `ImportError` if the user tries to instantiate without
  `transformers` installed.
- The Lightning adapter currently imports
  `from lightning.pytorch.callbacks import Callback` unconditionally at
  module top. Users on an install without Lightning should avoid
  importing `traceml.integrations.lightning`; the package's top-level
  `__init__.py` does not force the import, so the base install remains
  importable even when `lightning` is absent.

The install extras in `pyproject.toml` follow the same split:
`pip install traceml-ai[hf]` pulls `transformers`, and
`pip install traceml-ai[lightning]` pulls `lightning`. The base
`traceml-ai` install works without either.

!!! note
    Per-module import failures never cascade into `traceml.decorators`,
    the CLI, or the aggregator. A user who installs the base package
    and never touches integrations sees no import errors from this
    subsystem.

## Disable flag

Both modules read `TRACEML_DISABLED` at module-import time:

```python
TRACEML_DISABLED = os.environ.get("TRACEML_DISABLED") == "1"
```

When the flag is set to `"1"`:

- `TraceMLTrainer.training_step` short-circuits to
  `super().training_step(...)` immediately. No hook attachment, no
  `trace_step`, no memory tracking.
- Every `TraceMLCallback` hook returns early before touching timing
  regions, the memory tracker, or `TraceState`.

This is a hard bypass — the user can ship `TraceMLTrainer` or
`TraceMLCallback` in production code and disable all instrumentation by
flipping a single environment variable, with zero overhead past the
function-call prologue. The same `TRACEML_DISABLED` convention is
honored by [`decorators.py`](decorators.md) (`_traceml_disabled()`) and
by the CLI's `--disable-traceml` flag, so the three entry points
present a unified kill-switch.

Because the flag is read once at import, changing it mid-process has
no effect. The intended workflow is to set it in the environment
before launching the training script (or via the CLI flag, which the
launcher translates into `TRACEML_DISABLED=1` for child processes).

## Design notes

- **Never break user training.** All instrumentation side effects are
  wrapped in `try/except`. Hook-attach failures, memory-tracker
  failures, and flush failures are logged (via the standard library
  `logging` module in `huggingface.py`, and via
  `print(..., file=sys.stderr)` in `lightning.py`) but never
  propagated. User code exceptions from `super().training_step(...)`
  or from Lightning's own hooks are not caught — they must bubble up
  so training fails loudly.
- **Idempotent model attachment.** `TraceMLTrainer` tracks
  `_traceml_hooks_attached` and `_attached_model_id` so hooks are
  installed exactly once per unique model object. If the framework
  re-wraps the model (e.g. between resume checkpoints), the id check
  triggers a re-attach rather than double-attaching, which would
  otherwise double-count forward/backward events.
- **Lazy attachment, not eager.** Installing hooks in `__init__` would
  catch the un-wrapped base model and miss DDP/accelerator wrappers.
  Deferring to the first `training_step` guarantees we see the final
  model as the framework will actually call it.
- **Thin adapters.** The integration modules do not implement timing,
  memory tracking, event encoding, or flushing themselves. All of that
  is delegated to `trace_step`, `trace_model_instance`, `timed_region`,
  `record_event`, `StepMemoryTracker`, and `flush_step_events`. Adding
  a new framework integration is primarily a matter of locating its
  step-boundary callbacks and forwarding them to the same primitives.
- **Rank awareness is inherited.** Neither adapter special-cases
  `RANK` / `LOCAL_RANK`. The primitives they call already produce
  rank-tagged telemetry through the runtime's DB senders; rank
  filtering (e.g. rank-0-only aggregation) happens at the
  [aggregator](../architecture.md) layer.
- **No aggregator coupling.** The integrations are pure in-process
  instrumentation. They work identically whether the aggregator is
  running (normal `traceml watch` flow) or absent (library mode with
  the sender disabled). This keeps them testable in isolation.

## Cross-references

- [`decorators.md`](decorators.md) — the primitives (`trace_step`,
  `trace_model_instance`, `TraceState`) that both adapters wrap.
- [`samplers.md`](samplers.md) — consumers of the step boundaries and
  hooks installed by these adapters.
- [`utils.md`](utils.md) — `timed_region`, `record_event`,
  `StepMemoryTracker`, `flush_step_events`, and the underlying hook
  and patch helpers.
- [`runtime.md`](runtime.md) — the per-rank agent that owns sampler
  orchestration and the DB sender that ships telemetry to the
  aggregator.
- [`../architecture.md`](../architecture.md) — process layout,
  telemetry data flow, and design principles.
