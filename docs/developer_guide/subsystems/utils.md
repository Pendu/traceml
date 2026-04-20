# Utils

The `utils` subsystem is TraceML's **foundation layer** — a grab-bag of low-level helpers that everything else composes. Forward and backward hooks for layer-level telemetry, monkey-patches that auto-instrument `DataLoader`/`nn.Module.__call__`/`Tensor.backward` without touching user code, a CUDA event pool that recycles timing primitives across thousands of ticks, a step-memory tracker that brackets peak-memory stats around an optimizer step, and a `timed_region` context manager that captures a CPU+GPU-safe duration with a single `with` statement. None of these modules know about TCP, the aggregator, or rendering; they exist solely so decorators, samplers, and integrations have a uniform, fail-open toolkit to build on.

## Role in the architecture

The rest of TraceML is organized around telemetry pipelines: samplers produce rows, the runtime ships them, the aggregator stores them, renderers paint them. `utils` sits **underneath** that pipeline. It is the layer that actually touches PyTorch internals — `nn.Module.__call__`, `torch.Tensor.backward`, `register_full_backward_hook`, `torch.cuda.Event`, `torch.cuda.max_memory_allocated` — and it turns those into small, composable primitives that higher layers consume without caring about PyTorch's version-to-version quirks. If a future PyTorch release changes a hook signature, only `utils` needs to change.

Two higher-level layers are the main consumers. **Decorators** (`src/traceml/decorators.py`) use `timed_region`, the auto-timer context managers, `StepMemoryTracker`, and the hook/patch attachment helpers to turn a `with trace_step():` block into a fully instrumented training step. **Samplers** (`src/traceml/samplers/`) read from the shared queues that `utils` exposes — each hook module owns a `queue.Queue` that the corresponding sampler drains on every tick. The sampler never inspects the hook's state directly; it only consumes `queue.get_nowait()`. This one-way queue contract is what keeps the hot path lock-free and the sampler decoupled from PyTorch-internal data structures.

Because `utils` is the only layer that mutates PyTorch globals (monkey-patching `nn.Module.__call__`, `DataLoader.__iter__`, `torch.Tensor.backward`, `torch.autograd.backward`), every patch is **idempotent** and guarded by a sentinel attribute on the patched object — `DataLoader._traceml_patched`, `nn.Module._traceml_forward_patched`, `torch._traceml_backward_patched`. Hook attachment helpers follow the same pattern using a per-model `{id(model): True}` registry, so calling `attach_*_hooks(model)` twice is a no-op. The assumption everywhere is that these primitives will be invoked defensively from many call sites (decorators, integrations, user code) and must never stack.

## Directory layout

```text
src/traceml/utils/
├── cuda_event_pool.py        # Reusable torch.cuda.Event pool
├── timing.py                 # TimeEvent, StepTimeBatch, timed_region, queues
├── step_memory.py            # StepMemoryTracker + step_memory_queue
├── layer_parameter_memory.py # One-shot per-layer parameter-byte inspection
├── entry_hook.py             # EXECUTION_LAYER thread-local forward/backward tagger
├── flush_buffers.py          # flush_step_events() — one call per step boundary
├── formatting.py             # fmt_mem_new, fmt_time_ms, fmt_percent helpers
├── shared_utils.py           # get_hookable_modules, subtree_param_bytes, on-cuda check
├── base_trace_event.py       # BaseTraceEvent dataclass
├── ast_analysis/             # Static scan of user script (separate concern)
├── hooks/
│   ├── layer_forward_time_hooks.py
│   ├── layer_backward_time_hooks.py
│   ├── layer_forward_memory_hook.py
│   ├── layer_backward_memory_hook.py
│   ├── model_forward_memory_hook.py
│   └── optimizer_hook.py
└── patches/
    ├── dataloader_patch.py
    ├── forward_auto_timer_patch.py
    └── backward_auto_timer_patch.py
```

## Hooks

The `utils/hooks/` package contains six PyTorch-hook modules. Every one of them follows the same pattern: a pre-hook records a start marker into a thread-local/per-model buffer, a post-hook pops the marker, computes the delta, and appends an event to a per-model list. At a step boundary the runtime calls a `flush_*_buffers(model, step)` function that drains the per-model list, wraps it in a step-scoped dataclass (`LayerForwardTimeStepEvent`, `LayerBackwardMemoryEvents`, …), and pushes it onto a bounded `queue.Queue`. The corresponding sampler drains that queue.

**Layer forward time.** `attach_layer_forward_time_hooks(model)` walks `get_hookable_modules(model)` and attaches a `LayerForwardTimePreHook` + `LayerForwardTimePostHook` pair to each leaf. The pre-hook stamps `time.perf_counter()` and, if the model is on CUDA, acquires a CUDA event from the pool and `record()`s it. The post-hook pops the matching start record (FIFO, so shared/re-entered modules pair correctly) and emits a `LayerForwardTimeEvent` with both CPU and GPU timings. GPU timings are intentionally *not* resolved here — `try_resolve()` on the event is called later by the sampler so the hook never blocks for a CUDA sync.

**Layer backward time.** `attach_layer_backward_time_hooks(model)` is the mirror of the forward version but uses `register_full_backward_pre_hook` + `register_full_backward_hook`. Same FIFO pairing logic, same `LayerBackwardTimeEvent` with deferred GPU resolution. The flush function emits a `LayerBackwardTimeStepEvent` per step.

**Layer forward memory.** `attach_layer_forward_memory_hooks(model)` registers a single post-hook per leaf that walks the output (tensor, list, tuple, or dict) and sums `t.numel() * t.element_size()` for every tensor it finds. This is activation memory: how many bytes of forward output each layer produced on this step. Results are buffered as `(layer_name, bytes)` tuples and flushed as a `LayerForwardMemoryEvents` step snapshot.

**Layer backward memory.** `attach_layer_backward_memory_hooks(model)` registers a full backward hook per leaf that sums the bytes of `grad_output`. Same buffer-then-flush pattern; produces `LayerBackwardMemoryEvents` per step.

**Model forward memory.** `attach_model_forward_memory_hooks(model)` attaches a pre/post hook pair to the **root** module (not leaves). The pre-hook calls `torch.cuda.reset_peak_memory_stats()`; the post-hook reads `max_memory_allocated` / `max_memory_reserved` and stores a `ModelForwardMemoryEvent`. This gives a clean measurement of forward-only peak memory, separate from the full-step peak captured by `StepMemoryTracker`.

**Optimizer step.** `install_optimizer_time_hooks()` uses the public-but-underdocumented `torch.optim.optimizer.register_optimizer_step_pre_hook` / `register_optimizer_step_post_hook` registries. Unlike the layer hooks, these attach **once per process** and apply to every optimizer — there is no per-optimizer attach/detach. The pair emits a `TimeEvent` named `_traceml_internal:optimizer_step` into the shared step-time queue. `ensure_optimizer_timing_installed()` makes the call idempotent via a sentinel attribute on `torch.optim.Optimizer`.

Every hook attacher uses a `_*_hook_registry: Dict[int, bool]` keyed on `id(model)` so attaching twice to the same model is a silent no-op. Hook filtering is centralized: `get_hookable_modules(model, include_names, exclude_names, leaf_only)` in `shared_utils.py` yields the `(name, module)` pairs that match the filter — substring match on the qualified module name, with a default of leaves only.

### Hook-to-sampler pairing

| Hook module | Queue | Sampler (in `src/traceml/samplers/`) |
|---|---|---|
| `layer_forward_time_hooks.py` | `layer_forward_time_queue` | `LayerForwardTimeSampler` |
| `layer_backward_time_hooks.py` | `layer_backward_time_queue` | `LayerBackwardTimeSampler` |
| `layer_forward_memory_hook.py` | `layer_forward_memory_queue` | `LayerForwardMemorySampler` |
| `layer_backward_memory_hook.py` | `layer_backward_memory_queue` | `LayerBackwardMemorySampler` |
| `model_forward_memory_hook.py` | `model_forward_memory_queue` | `ModelForwardMemorySampler` |
| `optimizer_hook.py` | `_STEP_TIME_QUEUE` (shared) | `StepTimeSampler` |

The optimizer hook does not own its own queue; it reuses the shared step-time queue in `timing.py` because its events are semantically identical to any other timed region — just one more `TimeEvent` in the step batch.

### Execution-layer tagging

`utils/entry_hook.py` is a special, lightweight hook used for crash attribution, not telemetry. `attach_execution_entry_hooks(model)` registers a `ForwardEntryHook` (forward pre-hook) and a `BackwardEntryHook` (tensor-level backward hook, attached lazily via the forward post-hook) on every leaf. The sole job of these hooks is to update `EXECUTION_LAYER.current` — a thread-local string like `"forward_encoder.layer_3"` or `"backward_decoder.attn"`. When the user script crashes, the runtime's `report_crash()` reads this value and writes it to `torchrun_error.log`, so post-mortems can tell whether the failure happened during forward or backward and in which layer. The hook does nothing else — no queues, no allocations beyond the string assignment.

## Patches

Three modules in `utils/patches/` perform **monkey-patching** of PyTorch globals. Patches are used, rather than hooks, whenever the instrumentation point is a built-in method rather than an `nn.Module` — `DataLoader.__iter__` has no hook, and wrapping `nn.Module.__call__` is cleaner than registering one forward-pre-hook per module when you only want the outermost timing.

**Dataloader patch.** `patch_dataloader()` replaces `DataLoader.__iter__` with a generator that wraps each `next(it)` call in a `timed_region("_traceml_internal:dataloader_next", use_gpu=False)`. This measures pure CPU time spent waiting for the loader (worker-process startup, collate-fn, host-to-device copy initiation) and is the signal that tells users "your dataloader is the bottleneck." The patch is idempotent via `DataLoader._traceml_patched` and is applied once at runtime start.

**Forward patch.** `patch_forward()` replaces `nn.Module.__call__` with `_traceml_module_call`, which optionally wraps the call in a `timed_region("_traceml_internal:forward_time", use_gpu=True)`. Critically, the timing is gated by a thread-local flag **and a depth counter**: only the outermost forward is timed, so calling `self.submodule(x)` inside a parent module's `forward` does not double-count. The flag is controlled by the `forward_auto_timer` context manager — `patch_forward()` installs the wrapper globally but does nothing until a `trace_step` (or equivalent) enters `forward_auto_timer`. This two-stage design means the patch can be installed once at runtime init without inflating every unrelated `nn.Module.__call__` elsewhere in the user's process.

**Backward patch.** `patch_backward()` replaces both `torch.Tensor.backward` and `torch.autograd.backward` with wrappers that behave identically to the forward patch: gated by a thread-local flag (`backward_auto_timer`), outer-only via a depth counter, emits a `_traceml_internal:backward_time` event. The two entry points are patched together because user code can call either form.

### Patch attributes at a glance

| Patch | Target | Sentinel attribute | Enabled by |
|---|---|---|---|
| `patch_dataloader` | `DataLoader.__iter__` | `DataLoader._traceml_patched` | Always on once installed (no thread-local gate) |
| `patch_forward` | `nn.Module.__call__` | `nn.Module._traceml_forward_patched` | `forward_auto_timer` context (thread-local) |
| `patch_backward` | `torch.Tensor.backward` + `torch.autograd.backward` | `torch._traceml_backward_patched` | `backward_auto_timer` context (thread-local) |

The dataloader patch is the odd one out: it has no enable-flag gate because the cost of `timed_region` with `use_gpu=False` is a single `time.time()` pair and a queue `put_nowait`, which is negligible even on unbracketed loaders. The forward and backward patches are gated because wrapping every `nn.Module.__call__` in a no-op timer would be measurable across thousands of submodule invocations.

### Reversal contract

None of the three patch modules expose an `unpatch_*()`. The reversal contract is:

- Patches are installed **once per Python interpreter** and live until the process exits.
- Since the runtime lives and dies with the process, there is no "turn off TraceML mid-run" operation — you set `TRACEML_DISABLED=1` and restart.
- The patched wrappers fast-path to the original function when their thread-local enable flag is false, so even with the patch installed, un-bracketed training code incurs only a `getattr` + boolean check per call.

!!! warning "Why monkey-patch at all?"
    `DataLoader` has no hook API, and wrapping `nn.Module.__call__` is the only way to reliably time the top-level forward regardless of how the user wrote their training loop. Monkey-patching is a deliberate, minimized trade-off: the patches mutate exactly three global symbols, each guarded by a sentinel, each documented, and each a no-op when the thread-local flag is off.

## CUDA event pool

`utils/cuda_event_pool.py` provides a process-global pool of reusable `torch.cuda.Event` objects. The pool exists because creating a CUDA event is not free — it allocates a CUDA handle and registers it with the driver. At TraceML's hook density (two events per leaf per forward, two per backward, plus step-level events) a deep model could easily allocate tens of thousands of events per step. Reusing them turns this into a deque pop/append.

**API.**

- `get_cuda_event() -> torch.cuda.Event` — pops from the pool, or creates a new `Event(enable_timing=True)` if empty.
- `return_cuda_event(evt)` — appends back to the pool. No-op if the pool is full or `evt` is `None`.
- `clear_event_pool()` — drops all pooled events (used in shutdown paths / test isolation).

**Pool semantics.**

- Thread-safe via a single `threading.Lock`.
- Bounded by `max_size=2000` in the global instance. If a sampler is slow to resolve (GPU backlog), unreturned events are simply recreated; if callers return more than the cap, overflow is silently discarded.
- Events are returned only **after** they have been successfully resolved — i.e. `gpu_end.query()` returned `True` and `gpu_start.elapsed_time(gpu_end)` has been recorded. See `TimeEvent.try_resolve()` in `timing.py` and the analogous methods in the layer-time hooks for the recycling pattern.
- The pool holds `torch.cuda.Event` objects; each event reference is ~small but the underlying CUDA handle is the real cost being amortized.

**Memory budget.** A CUDA event is a few hundred bytes of Python plus a driver handle. At 2000 cap the pool is on the order of a megabyte of Python heap, and the driver-side footprint is what CUDA requires for 2000 timing events — negligible next to model activations.

## Memory tracking helpers

Two modules handle memory bookkeeping, and they operate at different granularities.

**Step-level: `step_memory.py`.** `StepMemoryTracker(model)` is a thin wrapper around `torch.cuda.reset_peak_memory_stats()` / `max_memory_allocated()` / `max_memory_reserved()`. Call `.reset()` at the top of a step and `.record()` at the bottom; the latter builds a `StepMemoryEvent(step=-1, ...)` and stashes it in `_temp_step_memory_buffer[model_id]`. The step index is `-1` because the tracker doesn't know it yet — `flush_step_memory_buffer(model, step)` is called by the runtime at the step boundary, sets the correct step, and enqueues the event onto `step_memory_queue`. On non-CUDA devices the tracker emits a sentinel event with zeros rather than skipping, so the downstream sampler sees a well-formed row in every environment.

**Layer-level parameter memory: `layer_parameter_memory.py`.** `collect_layer_parameter_memory(model)` is a one-shot synchronous inspection: it walks `model.named_modules()`, skips containers, and returns `{leaf_name: total_bytes}` summing each leaf's directly owned parameters. Unlike the activation/gradient memory hooks, this is a **static** property of the model — it doesn't change step-to-step — and it is measured once (per model instance or architecture) and cached downstream. The module also exposes `model_queue`, a `Queue` used by samplers to request inspection of newly registered models without holding references to them.

**Shared helpers: `shared_utils.py`.** Three functions collaborate on deciding which modules to hook:

- `get_hookable_modules(model, include_names, exclude_names, leaf_only)` — the single yield-based traversal used by every hook attacher.
- `subtree_param_bytes(module)` — recursively sums `p.numel() * p.element_size()` for a module and all descendants; memoized via `subtree_param_cache` keyed on the module object.
- `should_hook(module, min_memory_threshold)` — threshold-based decision used by future selective-hooking paths: if any child subtree exceeds the threshold, descend; otherwise hook at this level.

`model_is_on_cuda(model)` is a short-circuit check (first parameter or buffer wins) used by the hook attachers to decide whether to acquire CUDA events in the first place.

## Timing helpers

`utils/timing.py` is the unified timing primitive. Every other component — decorators, dataloader patch, forward/backward patches, optimizer hook — ultimately funnels into `timed_region`.

**`TimeEvent`.** A single timing measurement. Holds `cpu_start`, `cpu_end`, optional `gpu_start`/`gpu_end` CUDA events, an optional resolved `gpu_time_ms`, a `step` index, and a `scope` (`STEP` or `GLOBAL`). `try_resolve()` is a non-blocking call: it returns `False` if the GPU event is still pending, and on success it computes the elapsed time, returns both CUDA events to the pool, and marks the event resolved. The sampler polls `try_resolve()` each tick until it succeeds — this is the mechanism that lets hooks be allocation-light and synchronization-free.

**`StepTimeBatch`.** A group of `TimeEvent`s collected during a single optimizer step. The runtime's per-step flush wraps all buffered events into one batch and enqueues the batch in `_STEP_TIME_QUEUE` — one queue item per step, not per event.

**Two queues, two scopes.** `_STEP_TIME_QUEUE` receives `StepTimeBatch` objects (one per optimizer step); `_GLOBAL_TIME_QUEUE` receives `TimeEvent` objects directly (init, checkpoint, anything outside the step loop). `record_event(evt)` routes based on `evt.scope`: STEP events accumulate in `_STEP_BUFFER` and are flushed via `flush_step_time_buffer(step)`; GLOBAL events are enqueued immediately. Both queues have `maxsize=2048` and drop with a stderr warning on overflow — never block the training thread.

**`timed_region(name, scope, use_gpu)`.** The core context manager. Guarantees:

- User code always runs. Even if timing setup raises, the `yield` is not skipped.
- Exceptions from the user's block are not swallowed — `finally` only runs the teardown, which is itself wrapped in a try/except.
- GPU timing is best-effort. If `torch.cuda.is_available()` is false or CUDA event acquisition fails, the region degrades to CPU-only timing silently.
- On `TRACEML_DISABLED=1`, the manager yields immediately with no setup — zero-overhead fast path.

```python
with timed_region("my_region", scope=TimeScope.STEP, use_gpu=True):
    result = expensive_call()
```

Under the hood, `timed_region` acquires two CUDA events from the pool (if on GPU), records the start, yields, records the end, and posts a `TimeEvent` to `record_event()` — which either buffers (STEP) or enqueues (GLOBAL). The actual GPU elapsed time is computed later by the sampler via `try_resolve()`.

## Step-boundary flush

`utils/flush_buffers.py` ties the hooks and trackers together at the step boundary. `flush_step_events(model, step)` is a single call that drains every per-step buffer: forward/backward memory, forward/backward time, model forward memory, step memory tracker, step-time batch. The runtime — specifically `trace_step` at the end of the `with` block — invokes this once per step, passing the current step index. Downstream samplers see a consistent, step-stamped view: either a given step has data in every queue, or it has data in none.

## Formatting helpers

`utils/formatting.py` is renderer-facing: `fmt_mem_new(bytes)` → "123 MB", `fmt_time_ms(v)` → "µs/ms/s/min" adaptive scaling, `fmt_percent(x)` → "45.2%", `fmt_mem_triple(used, reserved, total)` → "used/reserved/total MB". Binary units (1024 base). No `utils` caller uses these — they're imported by renderers (`src/traceml/renderers/`) and by the aggregator's display drivers. They live in `utils` because they're stateless pure functions and have no better home.

All formatters are written to **never raise**: non-numeric input returns `"N/A"` (or `"—"` for zero/negative durations), and unit-scaling clips at `TB` so pathological inputs degrade gracefully. The unit ladder is shared between `fmt_mem_new`, `format_memory`, `fmt_mem_ratio`, and `fmt_mem_triple` via the module-level `_MEMORY_UNITS` list, keeping a single source of truth for the binary-unit progression.

## AST analysis

`utils/ast_analysis/` is a distinct sub-package with a different mission from the rest of `utils`. Every other module in `utils` is a **runtime** helper — it runs inside the training process while the model is executing. `ast_analysis` is **static**: it parses a user script with Python's `ast` module, walks the tree, and returns structured findings about what the script is doing, without ever importing or executing user code.

It exists to power recommendations and benchmarks: "your script uses `optim.AdamW` and `fp16`" or "you're using `DataLoader` without `pin_memory`" or "this script looks like it uses DDP with NCCL." The public API is `analyze_script(path) -> CodeFindings`, plus the narrower `detect_strategy_hint` and `scan_for_optimizer` helpers. Every entry point is guaranteed to not raise on ordinary failures — syntax errors, missing files, unexpected AST shapes are all captured into `CodeFindings.parse_errors` and the caller receives a partially populated result.

The analyzer also follows shallow local imports: if `train.py` does `from model import MyNet`, the scanner will open sibling `model.py` in the same directory and merge its findings. This surfaces dataloader/optimizer/distributed patterns hidden in helper modules without requiring the user to pass every file explicitly. It does not recurse across packages or follow dotted imports — deliberately shallow to keep analysis bounded.

This sub-package has no runtime dependencies on PyTorch, no shared state with the rest of `utils`, and no queues or buffers. It's included under `utils` only because its consumers (the recommendation engine, manifest writers) are disparate and there is no better home; it could be extracted cleanly if needed.

## Design notes

**Fail-open at every layer.** Every hook `__call__`, every patch wrapper, every flush function is wrapped in a broad `try/except` that logs to `stderr` with a `[TraceML]` prefix and returns. A broken hook cannot break training; a broken flush cannot break training; even the teardown of `timed_region` has its own inner try/except so "absolutely nothing here may break training." The one intentional exception is that user code exceptions inside `timed_region` are never swallowed — they propagate after the timing teardown runs in the `finally`.

**Idempotent attachment.** Every hook attacher checks `_*_hook_registry[id(model)]` before doing any work; every patch checks `_traceml_patched` / `_traceml_forward_patched` / `_traceml_backward_patched` on the patched object. Calling `attach_layer_forward_time_hooks(model)` ten times attaches hooks exactly once. This is load-bearing because integrations (Lightning, HuggingFace) and user decorators may both try to attach, and the contract is that everyone calls unconditionally and the first caller wins.

**Queue-based hand-off.** Hooks do not call into samplers; samplers do not inspect hook state. The contract is a bounded `queue.Queue`, one per hook module, with `put_nowait` on the producer side (drop with a stderr warning on full) and `get_nowait` on the consumer side. This decouples the PyTorch-internal thread that runs hooks from the TraceML sampler thread and keeps the hot path lock-free.

**Deferred GPU resolution.** No hook or patch ever calls `torch.cuda.synchronize()`. CUDA events are recorded on the hot path, returned to the pool on the cold path, and `elapsed_time()` is only invoked once `event.query()` returns `True`. This is the single most important overhead discipline in `utils`: instrumentation must not serialize the CUDA stream.

**Zero-overhead disable.** `TRACEML_DISABLED=1` short-circuits `timed_region` (immediate yield), `StepMemoryTracker` (no-op init), `record_event` (early return), and `flush_step_events` (early return). Patches are still installed but their thread-local enable flags are never set, so their wrappers fall through to the original implementation on every call. The disabled path is a few `os.environ.get` and attribute checks — an order of magnitude below sampler overhead.

**Single-device assumption (V1).** Several modules declare in their docstrings that they assume a model on a single device. Multi-device (model-parallel) support is a known gap: `subtree_param_bytes` would sum across devices, `LayerForwardTimeStepEvent.device` would be a list rather than a scalar, and memory hooks would need per-device buffers. This is deliberately scoped out of V1.

## Cross-references

- [Decorators](decorators.md) — the primary consumer. `trace_step` uses `timed_region`, `forward_auto_timer`, `backward_auto_timer`, `StepMemoryTracker`, and `flush_step_events`. `trace_model_instance` calls the `attach_layer_*_hooks` and `attach_execution_entry_hooks` helpers.
- [Samplers](samplers.md) — the other primary consumer. Every layer sampler drains a queue exposed by a module in `utils/hooks/`; `StepTimeSampler` reads `_STEP_TIME_QUEUE` from `utils/timing.py`; `StepMemorySampler` reads `step_memory_queue` from `utils/step_memory.py`.
- [Integrations](integrations.md) — HuggingFace and Lightning adapters invoke the same decorator path and therefore transitively depend on `utils`; they also call `patch_dataloader`, `patch_forward`, and `patch_backward` at framework bootstrap time.
- [Runtime](runtime.md) — calls `ensure_optimizer_timing_installed()` and the patch installers during startup; owns the sampler thread that drains the `utils` queues.
- [Architecture overview](../architecture.md) — where `utils` sits in the three-process picture and why the hook→queue→sampler split is what it is.
