# Samplers

Samplers are TraceML's telemetry collectors. Each sampler owns a narrow
slice of what "bottleneck visibility" means — step time, step memory,
per-layer forward/backward time and memory, process metrics, system
metrics, captured stdout/stderr — and turns it into rows written to an
in-memory `Database`. A dedicated runtime thread drives them on a fixed
tick, and an incremental sender ships newly appended rows to the
aggregator. Samplers are best-effort by contract: a sampler exception
logs and the next tick simply retries.

## Role in the architecture

TraceML training processes are agents. Inside each torchrun worker,
the `TraceMLRuntime` builds the set of samplers appropriate to the
profile (`watch`, `run`, `deep`), wires a `DBIncrementalSender` onto
every sampler's database, and runs one sampler thread per rank. On
each tick the runtime calls `sampler.sample()` for every registered
sampler, flushes the local writer, collects one ready payload per
sender, and hands the whole batch to a single
`TCPClient.send_batch()` call. The sampler layer is therefore the
*only* place in the codebase that actually produces telemetry —
everything downstream (aggregator, renderers, dashboards) is a read
path.

Samplers never talk to the aggregator themselves. They only write to
the `Database` attached to them at construction time. The
`DBIncrementalSender` attached in `TraceMLRuntime._attach_senders()`
owns the "new rows since last flush" counter and is the component
that actually serializes rows and calls into transport. This split
matters: samplers stay synchronous, single-purpose, and unaware of
TCP state or aggregator liveness. If the aggregator is unreachable,
the sender's error is logged and the next tick proceeds normally;
the sampler keeps filling its bounded local database.

The sampler layer is also the first place rank identity becomes
observable in the data. `TraceMLRuntime` stamps each sender with
`sampler.sender.rank = self.local_rank`, and a few samplers (notably
`SystemSampler` and `StdoutStderrSampler`) gate behavior on
local-rank-0 to avoid duplicating host-wide telemetry across ranks.
That per-rank stamping is what lets the aggregator's `RemoteDBStore`
keep each rank's data separate without any deduplication logic of
its own.

## Data in / data out

The "data in" side of a sampler varies by kind. Periodic samplers
(system, process) simply poll the OS or a library (`psutil`,
`pynvml`, `torch.cuda.*`) every tick. Event-driven samplers (step
time, step memory, model forward memory, and all four layer
samplers) read from producer queues that are filled by PyTorch hooks
and patches living in [`utils/`](utils.md). The hooks fire during
forward, backward, and step boundaries; the sampler's job is to
drain its queue, resolve any asynchronous GPU timings, and aggregate
per-call events into a single per-step record.

The "data out" side is uniformly a dict produced by
`schema.*.to_wire()`. All wire payloads go through
`BaseSampler._add_record(payload, table_name)` which forwards into
`Database.add_record(table, payload)`. Column/field layouts are
schema-owned and deliberately stable across releases — renderers
read these same dicts on the aggregator side. The most important
fields per sampler:

- **StepTimeSampler → `StepTimeTable`** — `seq`, `timestamp`,
  `step`, and `events[event_name][device] = {is_gpu, duration_ms,
  n_calls}`. One row per resolved step. Repeated regions in the
  same step (e.g. gradient accumulation) are summed.
- **StepMemorySampler → `step_memory`** — `seq`, `ts`, `model_id`,
  `device`, `step`, `peak_alloc`, `peak_resv` (bytes). One row per
  forward pass ended.
- **ModelForwardMemorySampler → `model_forward_memory`** —
  `timestamp`, `model_id`, `device`, `peak_allocated_mb`,
  `peak_reserved_mb`. One row per model-level forward event.
- **LayerForwardTimeSampler / LayerBackwardTimeSampler →
  `LayerForward|BackwardTimeTable`** — `seq`, `ts`, `model_id`,
  `step`, `device`, and parallel lists `layers`, `cpu_ms`,
  `gpu_ms`, `n_calls`. One row per fully resolved step; CPU and GPU
  time summed across calls.
- **LayerForwardMemorySampler / LayerBackwardMemorySampler →
  `LayerForward|BackwardMemoryTable`** — `seq`, `ts`, `model_id`,
  `step`, `device`, and parallel lists `layers`, `layer_bytes`. One
  row per drained event; repeated observations within a step folded
  with MAX.
- **LayerMemorySampler → `LayerMemoryTable`** — `seq`, `ts`,
  `model_index`, `model_signature`, `total_param_bytes`,
  `layer_count`, `layers`, `layer_bytes`. One row *per unique model
  architecture*, deduplicated by signature.
- **ProcessSampler → `ProcessTable`** — `seq`, `ts`, `pid`, `cpu`,
  `cpu_cores`, `ram_used`, `ram_total`, `gpu_available`,
  `gpu_count`, `gpu` (nested: `{device, mem_used, mem_reserved,
  mem_total}`).
- **SystemSampler → `SystemTable`** — `seq`, `ts`, `cpu`,
  `ram_used`, `ram_total`, `gpu_available`, `gpu_count`, `gpus`
  (list of fixed-order arrays `[util, mem_used, mem_total,
  temperature, power_usage, power_limit]`).
- **StdoutStderrSampler → `stdout_stderr`** — `{line}` per captured
  non-empty stdout/stderr line.

Bytes are bytes, milliseconds are milliseconds, and watts are
watts. Units live in the schema docstrings under
`src/traceml/samplers/schema/`. The `to_wire()` representations are
the canonical contract with downstream code — renderers reconstruct
the dataclasses via `from_wire()` only when they need type-checked
access.

## Sampler catalog

Every concrete sampler lives under `src/traceml/samplers/` and
subclasses `BaseSampler`. The runtime picks the set at startup based
on profile.

| Sampler | File | Profile | Purpose |
|---|---|---|---|
| `SystemSampler` | `system_sampler.py` | all (rank 0 only in DDP) | Host CPU / RAM + per-GPU utilization, memory, temperature, power via `psutil` + NVML |
| `ProcessSampler` | `process_sampler.py` | all | Process-attributed CPU %, RSS, and single-device CUDA memory via `psutil` + `torch.cuda` |
| `StdoutStderrSampler` | `stdout_stderr_sampler.py` | CLI mode, all ranks | Drains captured stdout/stderr; writes per-rank log file; mirrors rank-0 lines to DB |
| `StepTimeSampler` | `step_time_sampler.py` | `run`, `deep` | One timing row per step; aggregates `(event_name, device)` across repeated regions; resolves GPU events asynchronously |
| `StepMemorySampler` | `step_memory_sampler.py` | `run`, `deep` | Peak allocated / reserved CUDA memory per step, drained from the step-memory queue |
| `ModelForwardMemorySampler` | `model_forward_memory_sampler.py` | (available, not auto-enabled) | Peak forward-pass memory at the whole-model level |
| `LayerMemorySampler` | `layer_memory_sampler.py` | `deep` | One-time per-architecture static parameter memory per layer; deduplicates by model signature |
| `LayerForwardMemorySampler` | `layer_forward_memory_sampler.py` | `deep` | Per-layer activation memory observed during forward; MAX-aggregated within a step |
| `LayerBackwardMemorySampler` | `layer_backward_memory_sampler.py` | `deep` | Per-layer memory observed during backward; MAX-aggregated within a step |
| `LayerForwardTimeSampler` | `layer_forward_time_sampler.py` | `deep` | Per-layer CPU+GPU execution time during forward; summed across repeated calls |
| `LayerBackwardTimeSampler` | `layer_backward_time_sampler.py` | `deep` | Per-layer CPU+GPU execution time during backward; summed across repeated calls |

Supporting (non-sampler) modules alongside the catalog:

- `base_sampler.py` — abstract `BaseSampler`, holds DB + sender +
  logger construction.
- `layer_time_common.py` — shared `aggregate_layer_time_payload()`
  and `all_layer_events_resolved()` for the two layer-time samplers.
- `layer_memory_common.py` — shared
  `aggregate_layer_memory_payload_max()` for the two layer-memory
  samplers.
- `runtime_context.py` — resolves `RANK`, `LOCAL_RANK`,
  `WORLD_SIZE`, `TRACEML_SESSION_ID`, `TRACEML_LOGS_DIR` into a
  frozen `SamplerRuntimeContext`.
- `system_manifest.py` — one-time per-session
  `system_manifest.json` writer used by `SystemSampler` to record
  hardware identity.
- `utils.py` — `drain_queue_nowait()`,
  `append_queue_nowait_to_deque()`, atomic JSON writer, session
  directory helper.
- `schema/` — frozen dataclass contracts for every wire payload.

## Base class & lifecycle

Every sampler inherits from `BaseSampler` (`base_sampler.py`). The
base class takes three parameters:

- `sampler_name` — logical identity; used for the `Database` owner,
  transport payloads, and error-logger name.
- `table_name` — default table for `_add_record()`. Samplers that
  write to a single table set it here; samplers with multiple
  tables omit it and pass a per-call override.
- `max_rows_per_flush` — sender-side transport cap. `-1` (default)
  means "send all new rows since last flush"; layer samplers use
  `5` to cap burst size, and `SystemSampler` uses `1` to smooth
  bursty NVML reads.

On construction, the base class creates:

1. a `Database(sampler_name)` — the bounded append-only in-memory
   table store,
2. a `DBIncrementalSender(db, sampler_name)` — the "new rows since
   last flush" ship-it-to-aggregator helper, and
3. a named `get_error_logger(sampler_name)` — the fail-open
   stderr+file logger.

It exposes one concrete helper,
`_add_record(payload, table_name=None)`, and one `@abstractmethod`,
`sample()`, which concrete samplers implement. The `sample()`
contract is explicit: *collect one best-effort telemetry sample,
swallow your own exceptions, never interfere with training.*

Samplers are **registered** exclusively by
`TraceMLRuntime._build_samplers()`. There is no discovery, no plugin
registry, and no config file — the method is a straight list-builder
keyed on `profile` and `mode`. Every profile always includes
`ProcessSampler`. `SystemSampler` is included on every rank 0 (or
every process in non-DDP). `StdoutStderrSampler` is added only when
`mode == "cli"`. The `run` and `deep` profiles add the two step
samplers. The `deep` profile additionally adds the five layer
samplers. To add a sampler, implement a class in
`src/traceml/samplers/`, export it from `__init__.py`, and add it
to the appropriate branch in `_build_samplers()`.

Once built, samplers go through three lifecycle moments:

1. **Attach senders** — `TraceMLRuntime._attach_senders()` walks
   every sampler, skips any that set `sender = None` (non-rank-0
   `StdoutStderrSampler` does this), and injects the shared
   `TCPClient` and this rank's `local_rank` into `sampler.sender`.
2. **Tick** — a dedicated daemon thread
   `TraceMLSampler(rank=N)` loops forever, calling `sample()` on
   each sampler, flushing `db.writer`, then collecting one
   `sender.collect_payload()` per sender and batching them into a
   single `TCPClient.send_batch()` call. Sleep between ticks is
   `TRACEML_INTERVAL` (default 1.0 s).
3. **Final tick** — when `stop()` fires the stop event, the loop
   runs one last tick before joining. This guarantees rows
   accumulated between the last normal tick and teardown are
   shipped.

A typical tick for an event-driven sampler looks like this:

```python
def sample(self) -> None:
    try:
        self._ingest_queue()           # drain producer queue
        while self._local_buffer:
            event = self._local_buffer[0]
            if not self._step_is_resolved(event):
                break                   # GPU events not ready yet
            self._local_buffer.popleft()
            payload = aggregate_layer_time_payload(event.layers)
            sample = LayerForwardBackwardTimeSample(..., payload=payload)
            self._add_record(sample.to_wire())
    except Exception as e:
        self.logger.error(f"[TraceML] {self.sampler_name} error: {e}")
```

Note the two distinct "wait-and-retry" conditions: (a) the queue may
be empty, in which case the loop exits and the next tick drains it;
and (b) the earliest buffered step may have unresolved CUDA events,
in which case we `break` and leave it at the head of the local FIFO
for the next tick to try again.

!!! note "`sample()` must always return quickly"
    The tick runs every sampler serially on a single thread. A slow
    sampler delays every other sampler. Samplers that depend on
    asynchronous work (GPU event resolution, queue draining) return
    early when the work isn't ready and rely on the next tick to
    retry.

## Design notes

**Overhead budget.** Every sampler is expected to cost orders of
magnitude less than training itself. Periodic samplers do one
`psutil` or NVML call per tick. Event-driven samplers hold their hot
path to a `queue.get_nowait()` drain plus a sorted list walk.
Aggregation helpers reuse plain dicts and parallel lists rather than
any heavier object graph. The runtime tick is coarse (1 s by
default), and bounded deques in `Database` evict oldest rows at
fixed `maxlen` so memory never grows unboundedly even if the
aggregator disappears.

**CUDA event pool reuse.** Layer and step timing samplers receive
`TimeEvent`-like objects that wrap CUDA `Event` handles from a
reusable pool allocated in [`utils/`](utils.md). The samplers call
`evt.try_resolve()` on each event and only aggregate when
`all(try_resolve())` returns true; unresolved events stay in a
local FIFO (`_pending` / `_local_buffer`) and are retried next
tick. This keeps `torch.cuda.synchronize()` out of the hot path
entirely.

**Incremental storage pattern.** `BaseSampler` pairs a `Database`
with a `DBIncrementalSender`. The database is append-only with an
O(1) append counter. The sender reads `db.append_count()`, compares
against its own last-sent counter, and ships only the new rows.
This is what allows the runtime to call `collect_payload()` every
tick without the sampler needing to know whether anything changed —
the sender returns `None` when there's nothing new, and batching at
the `TCPClient` level skips zero-payload entries.

**Aggregation semantics matter.** Time is additive; memory is not.
`aggregate_layer_time_payload()` sums CPU and GPU durations across
repeated layer calls within a step (a layer invoked twice in the
same forward gets its time summed). `aggregate_layer_memory_payload_max()`
takes the MAX per layer instead, because memory is a capacity
metric and repeated observations don't add. `StepTimeSampler`
applies the same additive rule at the event level, keying
aggregation on `(event_name, device, is_gpu)`. If you add a new
aggregator helper, follow the same rule of thumb: durations sum,
capacities max.

**Fail-open behavior.** Every `sample()` implementation in the tree
wraps its real work in `try: ... except Exception as e:
self.logger.error(...)`. The runtime additionally wraps each
`sampler.sample` call in `_safe(...)` so even an exception that
escapes a sampler's own handler cannot kill the sampler thread.
`writer.flush`, `collect_payload`, and `send_batch` are all
similarly wrapped. The behavior contract is: *telemetry loss is
acceptable; training crashes because of telemetry are not.*

**Rank awareness.** Samplers are themselves rank-unaware by
construction — they write local data and stamp no rank field on it.
Rank identity is injected by `TraceMLRuntime._attach_senders()`
onto the sender (`sampler.sender.rank = self.local_rank`), and by a
handful of samplers that consult `resolve_runtime_context()` for
policy decisions:

- `SystemSampler` is only constructed on local-rank-0 (host-wide
  metrics would otherwise be duplicated).
- `ProcessSampler` uses the context to pick the right CUDA device
  index for this rank, and gates CUDA access on
  `dist.is_initialized()` to avoid hanging in DDP startup.
- `StdoutStderrSampler` writes per-rank log files on every rank but
  only mirrors lines into the DB (and therefore over the wire) on
  local-rank-0.

**Schema stability.** Wire formats are a public contract between
samplers and aggregator-side renderers. Schemas live under `schema/`
as `@dataclass(frozen=True)` classes with explicit `to_wire()` /
`from_wire()` methods. New fields are forward-compatible
(`from_wire` ignores unknown keys and defaults missing ones).
Column renames require coordinated changes on both sides — these
should go through a migration notice per the project's
backward-compatibility rule.

**One-time vs. periodic.** Most samplers emit one row per tick (or
per resolved step). Two samplers behave differently:
`LayerMemorySampler` deduplicates by architecture signature and so
emits at most one row per distinct model definition observed across
the whole run; and `SystemSampler` side-effects
`system_manifest.json` exactly once per session via
`write_system_manifest_if_missing()` during its `__init__`, giving
the aggregator a stable hardware record independent of the
time-series stream.

!!! warning "Hooks live outside the samplers module"
    Layer and step samplers *consume* telemetry produced by PyTorch
    hooks and patches in [`utils/`](utils.md). If you're adding a
    new layer-level metric, the hook goes in `utils/hooks/`, the
    queue accessor goes with it, and the sampler just drains that
    queue. Do not put hook code inside `samplers/`.

## Cross-references

- [Database](database.md) — bounded append-only tables,
  `DBIncrementalSender`, rank-aware `RemoteDBStore`.
- [Runtime](runtime.md) — per-rank agent that builds, attaches, and
  ticks samplers.
- [Utils](utils.md) — PyTorch hooks and patches that feed the
  event-driven samplers, plus the CUDA event pool.
- [Decorators](decorators.md) — `trace_model_instance` attaches the
  layer hooks that produce events for the four layer samplers;
  `trace_step` marks the step boundaries that the step samplers
  aggregate against.
- [Transport](transport.md) — `TCPClient.send_batch()` is what ships
  the sender payloads to the aggregator.
- [Architecture](../architecture.md) — system-wide data flow,
  process model, and design principles.
