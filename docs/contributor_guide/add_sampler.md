# How to add a new sampler

This guide teaches you how to add a new telemetry sampler to TraceML. It
assumes you have the repository checked out, can run the test suite, and have
read `CLAUDE.md` at the repo root. Everything below is anchored in the
`src/traceml/` source tree as of the current branch.

---

## 1. Intro & mental model

### What is a "sampler" in TraceML?

A **sampler** is a small, focused object that collects one narrow kind of
telemetry and writes it into an in-memory `Database` owned by that sampler.
Samplers run *inside each training rank* (the torchrun worker process that
executes the user's script), on a single background thread driven by
`TraceMLRuntime._tick()`. The aggregator is a separate process that ingests
what samplers send, via TCP.

A sampler does three things and nothing else:

1. **Observe** one data source (psutil, pynvml, a hook queue, a PyTorch
   internal).
2. **Record** one row into its local `Database` per useful observation.
3. **Fail open** on any error — log it, move on, never crash training.

Transport, rank-awareness, deduplication, rendering, and summary generation
are *not* the sampler's problem. The `DBIncrementalSender` handles transport
cursoring, the `TCPClient` handles wire framing, the aggregator's
`RemoteDBStore` handles cross-rank assembly, and renderers handle UI.

!!! note "One sampler = one data domain"
    Do not combine "cpu + gpu + disk io" into a single sampler. Split them.
    Keeping each sampler focused on one data source keeps failure modes
    isolated and makes the profile flags (`watch` / `run` / `deep`) meaningful.

### Two axes of classification

Every existing sampler fits somewhere on these two axes — find where yours
sits before writing code.

**Axis A — How the sampler is driven:**

| Style          | Description                                                                 | Exemplars                                                    |
|----------------|-----------------------------------------------------------------------------|--------------------------------------------------------------|
| **Periodic**   | `sample()` is called every runtime tick; the sampler polls a live source.   | `SystemSampler`, `ProcessSampler`                            |
| **Event-driven** | Data arrives on a `queue.Queue` (or `deque`) populated by a hook/patch; `sample()` drains the queue and emits one row per event (or per resolved step). | `StepTimeSampler`, `LayerForwardTimeSampler`, `StepMemorySampler` |

**Axis B — What scope the data is attributed to:**

| Scope       | Row identity                                        | Exemplars                           |
|-------------|-----------------------------------------------------|-------------------------------------|
| **System**  | one row per host tick                               | `SystemSampler`                     |
| **Process** | one row per (rank, tick)                            | `ProcessSampler`                    |
| **Step**    | one row per optimizer step                          | `StepTimeSampler`, `StepMemorySampler` |
| **Layer**   | one row per (step, layer) or per (step, model)      | `LayerForwardTimeSampler`, `LayerForwardMemorySampler` |
| **Other**   | one row per stdout/stderr line, etc.                | `StdoutStderrSampler`               |

Picking the right cell in this 2×N matrix is 80% of the design. The remaining
20% is wire schema.

### Data flow (single tick)

```
                    ┌─────────────────────────────────────────┐
                    │  Training rank (torchrun worker)        │
                    │                                         │
user code ─hooks──▶ │  hook queue ─┐                          │
                    │              ▼                          │
                    │     YourSampler.sample()                │
                    │              │                          │
                    │              ▼                          │
                    │        Database (deque-per-table)       │
                    │              │                          │
                    │              │  (get_append_count)      │
                    │              ▼                          │
                    │   DBIncrementalSender.collect_payload   │
                    │              │                          │
                    │              ▼                          │
                    │        TCPClient.send_batch  ───────────┼──┐
                    └─────────────────────────────────────────┘  │
                                                                 ▼
                    ┌─────────────────────────────────────────┐
                    │  Aggregator process                     │
                    │     TCPServer → RemoteDBStore →         │
                    │     renderers → display driver          │
                    └─────────────────────────────────────────┘
```

Source anchors:
- `src/traceml/runtime/runtime.py::TraceMLRuntime._tick` — the loop.
- `src/traceml/database/database.py::Database.add_record` — the write point.
- `src/traceml/database/database_sender.py::DBIncrementalSender.collect_payload`
  — what actually gets sent and why only new rows.
- `src/traceml/transport/tcp_transport.py` — the wire.

### The fail-open contract (not optional)

!!! danger "A sampler exception must never break training."
    Every `sample()` body wraps its work in `try/except Exception` and logs
    via `self.logger.error(...)`. The runtime's `_safe()` wrapper (in
    `runtime.py`) is a second belt around the first pair of suspenders —
    don't rely on it alone. If your sampler raises, the next tick still
    calls it. If it corrupts its own internal state, that's your problem to
    detect and reset.

---

## 2. Before you start: decisions to make

Before opening an editor, answer all of these. Write them in the PR
description.

- [ ] **Data domain.** What exactly are you measuring? One sentence.
- [ ] **Driver style.** Periodic (poll every tick) or event-driven (drain a
      queue)? If event-driven, who pushes to the queue and when?
- [ ] **Rank scope.** All ranks, rank-0 only, or something weirder? Host
      metrics are typically rank-0-only to avoid N-way duplication. Per-step
      and per-process metrics are typically per-rank.
- [ ] **Profile.** Which of `watch`, `run`, `deep` should enable it?
      Prefer extending an existing profile; adding a new profile is Section 11.
- [ ] **Schema.** What fields per row? Types? Which are optional?
      Remember: flat dicts of primitives, nothing fancy.
- [ ] **Memory budget.** Accept `Database.DEFAULT_MAX_ROWS` (3000) unless
      you have a reason. If you emit one row per step on a 100k-step job,
      3000 is fine — old rows will evict and the sender's append counter is
      robust to that.
- [ ] **Overhead target.** Periodic samplers should be <1 ms per tick on a
      healthy host. Event-driven samplers should be O(queue depth) with
      cheap per-event work.
- [ ] **Disable path.** Does `TRACEML_DISABLED=1` silence your data source
      at the producer side (e.g. the hook never fires), or do you need a
      guard inside the sampler? Most samplers don't need a guard because
      the runtime isn't started at all when disabled.

---

## 3. Anatomy of a sampler (anchored in code)

We'll walk through **`ProcessSampler`** end-to-end. Reasons for this choice:

- It's a clean **periodic** sampler — `sample()` is called on a timer; no
  queue draining to explain yet.
- It covers all four non-trivial infrastructure pieces: `BaseSampler`
  subclass, init-time source setup, rank-aware CUDA handling (`local_rank`),
  and defensive partial-failure handling (missing GPU, psutil exception).
- It demonstrates the "one domain per sampler" rule — process-relative CPU,
  RAM, and GPU memory, and *only* what this process consumes.

File: `src/traceml/samplers/process_sampler.py`.

### 3.1. Class declaration and constructor

```python
from traceml.samplers.base_sampler import BaseSampler
from traceml.samplers.runtime_context import resolve_runtime_context
from traceml.samplers.schema.process import ProcessGPUMetrics, ProcessSample


class ProcessSampler(BaseSampler):
    def __init__(self) -> None:
        super().__init__(
            sampler_name="ProcessSampler",
            table_name="ProcessTable",
        )

        self.sample_idx = 0

        self._init_process()
        self._init_ram()
        self._warmup_cpu()
        self._init_ddp_context()
        self._init_gpu_state()
```

Observations:

- **Takes no parameters.** All samplers are constructed by the runtime with
  zero args. Configuration comes from environment variables resolved via
  `resolve_runtime_context()` — not from constructor parameters.
- **Calls `super().__init__` with two strings.** `sampler_name` is the
  logical identity for this sampler; it shows up in the wire payload
  (`{"sampler": ..., ...}`), the error log names, and the database label.
  `table_name` is the default table `self._add_record(...)` will write into.
- **Everything else happens in private `_init_*` helpers.** This is the
  house style. Keep `__init__` readable — a one-line overview of all the
  subsystems this sampler touches.
- **`self.sample_idx` is a per-sampler monotonic counter.** Included in
  every row so downstream code can detect dropped samples. Not mandatory,
  but every sampler uses it.

Source of defaults: `BaseSampler.__init__` (see
`src/traceml/samplers/base_sampler.py`). Reading it once is worth it — it's
40 lines and it is the contract.

### 3.2. What `BaseSampler` gives you for free

After `super().__init__(...)`, your instance has:

| Attribute         | Type                     | Purpose                                                     |
|-------------------|--------------------------|-------------------------------------------------------------|
| `self.sampler_name` | `str`                  | Identity string used everywhere.                            |
| `self.table_name`   | `Optional[str]`        | Default target table for `_add_record()`.                   |
| `self.logger`       | `logging.Logger`       | Child of `traceml.<sampler_name>`. Writes to rotating file.|
| `self.db`           | `Database`             | Your private bounded in-memory table store.                 |
| `self.sender`       | `DBIncrementalSender`  | Ships new rows to the aggregator via TCP (wired later).     |
| `self.enable_send`  | `bool`                 | Set to `False` to skip network emission on a particular sampler. |

And the helper:

```python
def _add_record(self, payload: dict[str, Any], table_name: Optional[str] = None) -> None
```

which delegates to `self.db.add_record(target_table, payload)`.

!!! tip "Never write to `self.db` directly in normal code"
    Use `self._add_record(payload)`. It picks up the default table name,
    raises a clear `ValueError` if you forgot to set one, and keeps the
    write-site uniform across samplers. Direct `self.db.add_record(...)` is
    acceptable only when you're writing to *multiple* tables from the same
    sampler, and even then, pass `table_name=` to `_add_record` instead.

### 3.3. Source initialization — the defensive pattern

```python
    def _init_process(self) -> None:
        try:
            self.pid = os.getpid()
            self.process = psutil.Process(self.pid)
        except Exception as e:
            self.logger.error(
                f"[TraceML] WARNING: Failed to attach to process {os.getpid()}: {e}"
            )
            self.pid = -1
            self.process = None
```

The pattern is identical for every source the sampler touches:

1. Try to attach / query the source.
2. On failure, log to `self.logger` with a `[TraceML]` prefix.
3. **Set the instance attribute to a sentinel** (`None`, `0.0`, `-1`,
   empty list, whatever makes `sample()` safe later).

Do not `raise` from `__init__`. If the source is unavailable, the sampler
should still exist; `sample()` will just emit degraded rows.

### 3.4. The rank-aware CUDA dance

```python
    def _init_ddp_context(self) -> None:
        ctx = resolve_runtime_context()
        self.local_rank = ctx.local_rank
        self.world_size = ctx.world_size
        self.rank = ctx.rank
        self.is_ddp_intended = ctx.is_ddp_intended
        if self.is_ddp_intended and self.local_rank == -1:
            self.local_rank = 0
```

Key points:

- `resolve_runtime_context()` reads `RANK`, `LOCAL_RANK`, `WORLD_SIZE`,
  `TRACEML_SESSION_ID`, `TRACEML_LOGS_DIR` from the environment. torchrun
  sets the DDP vars.
- `is_ddp_intended` is true iff `WORLD_SIZE > 1` — we read this from the
  launcher, *not* from `dist.is_initialized()`, because distributed init
  hasn't necessarily happened yet when samplers come online.
- The `_cuda_safe_to_touch()` helper (later in the file) is the right way
  to check if CUDA can be used: it returns `True` if single-process, else
  checks `dist.is_initialized()`. Copy this pattern for any sampler that
  reads CUDA state.

### 3.5. The `sample()` body

```python
    def sample(self) -> None:
        self.sample_idx += 1
        try:
            gpu_metrics = self._sample_gpu()
            gpu_available = bool(self.gpu_available) if self.gpu_available is not None else False
            gpu_count = int(self.gpu_count) if self.gpu_available else 0

            sample = ProcessSample(
                sample_idx=self.sample_idx,
                timestamp=time.time(),
                pid=self.pid,
                cpu_percent=self._sample_cpu(),
                cpu_logical_core_count=self.cpu_count,
                ram_used=self._sample_ram(),
                ram_total=self.ram_total,
                gpu_available=gpu_available,
                gpu_count=gpu_count,
                gpu=gpu_metrics,
            )
            self._add_record(sample.to_wire())

        except Exception as e:
            self.logger.error(f"[TraceML] Process sampling error: {e}")
```

The shape of every periodic sampler's `sample()` method is:

1. `self.sample_idx += 1`.
2. Wrap the entire body in `try/except Exception`.
3. Collect from each sub-source; each sub-source method has its *own*
   try/except so one failure doesn't kill the whole row.
4. Build a `@dataclass(frozen=True)` schema object.
5. Call `self._add_record(sample.to_wire())` (note: wire-format dict, not
   the dataclass itself — see Section 6 on schema rules).
6. Outer `except` logs and returns. No re-raise. Ever.

### 3.6. What `BaseSampler` does *not* give you

- No lifecycle hooks. There is no `start()` / `stop()` / `close()` method
  on `BaseSampler`. The runtime constructs the sampler, calls `sample()`
  repeatedly, and drops the reference at shutdown.
- No periodicity control. The runtime decides the tick cadence
  (`TRACEML_INTERVAL`, default 1.0 s). If your sampler needs a different
  cadence, either rate-gate internally (track `time.time()` and skip calls)
  or accept that it runs on the runtime's interval.
- No persistence. `DatabaseWriter` is attached to the Database by default,
  but its on/off behavior is controlled by `config.enable_logging` — not by
  the sampler.

---

## 4. Step-by-step: adding a new sampler

We'll build a **`GpuUtilizationSampler`** — a periodic, rank-0-only sampler
that emits per-GPU utilization percentages. This is intentionally close in
shape to `SystemSampler` but simpler (no power/temp/memory, just utilization).

The end state: one row per tick on rank 0, containing a list of
`(gpu_idx, util_percent, mem_used_bytes)` tuples for every NVML-visible GPU.

### Step 1 — Create the sampler file

Path: `src/traceml/samplers/gpu_utilization_sampler.py`.

```python
"""
GPU utilization sampler for TraceML.

Emits one row per runtime tick (rank-0 only) containing per-GPU utilization
and allocated-memory figures, queried via NVML.

Design notes
------------
- Rank-0-only: GPU utilization is a host-level metric (all ranks share the
  same NVML view), so only one rank should emit it to avoid N-way duplicates.
- NVML-only: does not touch torch.cuda. Safe to sample before / during
  distributed init.
- Failure policy: one failing GPU does not prevent sampling others; NVML
  init failure disables the sampler but does not raise.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from pynvml import (
    NVMLError,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
)

from traceml.samplers.base_sampler import BaseSampler
from traceml.samplers.runtime_context import resolve_runtime_context


class GpuUtilizationSampler(BaseSampler):
    """
    Periodic per-GPU utilization sampler (rank-0 only).
    """

    def __init__(self) -> None:
        super().__init__(
            sampler_name="GpuUtilizationSampler",
            table_name="GpuUtilizationTable",
            max_rows_per_flush=1,
        )
        self.sample_idx = 0
        self._ctx = resolve_runtime_context()
        self._init_nvml()

        # Detach sender on non-rank-0 to avoid duplicate host metrics on the
        # wire. The sampler still exists on other ranks but emits nothing
        # over TCP.
        if self._ctx.is_ddp_intended and self._ctx.local_rank != 0:
            self.sender = None

    def _init_nvml(self) -> None:
        self.gpu_available = False
        self.gpu_count = 0
        try:
            nvmlInit()
            self.gpu_count = int(nvmlDeviceGetCount())
            self.gpu_available = self.gpu_count > 0
        except NVMLError as e:
            self.logger.error(
                f"[TraceML] GpuUtilizationSampler NVML init failed: {e}"
            )
        except Exception as e:
            self.logger.error(
                f"[TraceML] GpuUtilizationSampler unexpected init error: {e}"
            )

    def _sample_one(self, gpu_idx: int) -> Dict[str, Any]:
        """
        Sample a single GPU. Zero-filled on failure to preserve index alignment.
        """
        try:
            handle = nvmlDeviceGetHandleByIndex(gpu_idx)
            util = nvmlDeviceGetUtilizationRates(handle)
            mem = nvmlDeviceGetMemoryInfo(handle)
            return {
                "gpu_idx": int(gpu_idx),
                "util_percent": float(util.gpu),
                "mem_used_bytes": float(mem.used),
            }
        except Exception as e:
            self.logger.error(
                f"[TraceML] GpuUtilizationSampler GPU {gpu_idx} failed: {e}"
            )
            return {
                "gpu_idx": int(gpu_idx),
                "util_percent": 0.0,
                "mem_used_bytes": 0.0,
            }

    def sample(self) -> None:
        self.sample_idx += 1
        if not self.gpu_available:
            return

        try:
            gpus: List[Dict[str, Any]] = [
                self._sample_one(i) for i in range(self.gpu_count)
            ]
            self._add_record(
                {
                    "seq": self.sample_idx,
                    "ts": time.time(),
                    "gpu_count": int(self.gpu_count),
                    "gpus": gpus,
                }
            )
        except Exception as e:
            self.logger.error(
                f"[TraceML] GpuUtilizationSampler sample failed: {e}"
            )
```

That file is 77 lines. Nothing surprising. Let's walk through the choices.

### Step 2 — Pick the `BaseSampler` subclass

There is only one base class: `BaseSampler` in
`src/traceml/samplers/base_sampler.py`. Every sampler inherits directly
from it. There is no "periodic base" vs "event-driven base" hierarchy —
the distinction lives inside `sample()`.

### Step 3 — Define the schema

We inlined a plain `dict` above. For a production sampler, follow the
existing pattern: add a dataclass in `src/traceml/samplers/schema/`.

```python
# src/traceml/samplers/schema/gpu_utilization.py
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class GpuUtilizationEntry:
    gpu_idx: int
    util_percent: float
    mem_used_bytes: float

    def to_wire(self) -> Dict[str, Any]:
        return {
            "gpu_idx": self.gpu_idx,
            "util_percent": self.util_percent,
            "mem_used_bytes": self.mem_used_bytes,
        }


@dataclass(frozen=True)
class GpuUtilizationSample:
    sample_idx: int
    timestamp: float
    gpu_count: int
    gpus: List[GpuUtilizationEntry]

    def to_wire(self) -> Dict[str, Any]:
        return {
            "seq": self.sample_idx,
            "ts": self.timestamp,
            "gpu_count": self.gpu_count,
            "gpus": [g.to_wire() for g in self.gpus],
        }
```

Schema conventions (see `schema/process.py` and `schema/system.py` for
prior art):

- Frozen dataclasses.
- Always expose `to_wire() -> Dict[str, Any]` and `from_wire(data)`.
- Wire keys are short (`seq`, `ts`, `cpu`, ...) because they ride on every
  payload; Python names in the dataclass are descriptive.
- Primitive types only on the wire. `float` / `int` / `str` / `bool` /
  `list` / `dict` of those. No numpy scalars (cast them). No dataclass
  instances (call `to_wire()`).

### Step 4 — Wire into `TraceMLRuntime._build_samplers()`

Open `src/traceml/runtime/runtime.py` and find `_build_samplers`. At the
time of writing it looks like this (truncated):

```python
    def _build_samplers(self) -> List[BaseSampler]:
        samplers: List[BaseSampler] = []

        # Host/system metrics only once (rank 0) in DDP
        if not (self.is_ddp and self.local_rank != 0):
            samplers.append(SystemSampler())

        samplers.append(ProcessSampler())

        if self.mode == "cli":
            samplers.append(StdoutStderrSampler())

        if self.profile in ["run", "deep"]:
            samplers += [
                StepTimeSampler(),
                StepMemorySampler(),
            ]

        if self.profile == "deep":
            samplers += [
                LayerMemorySampler(),
                LayerForwardMemorySampler(),
                LayerBackwardMemorySampler(),
                LayerForwardTimeSampler(),
                LayerBackwardTimeSampler(),
            ]

        return samplers
```

Decide which profile(s) enable it. GPU utilization is cheap and useful at
every level, so we add it alongside `SystemSampler` in the rank-0 host
metrics branch. But since the sampler itself already detaches its sender
on non-rank-0 (Step 1), we can unconditionally instantiate it on every
rank if we prefer a per-rank trace file — see Section 5 on the "non-rank-0
detach" pattern.

Two reasonable patterns:

**A. Gate at the runtime level (simpler, like `SystemSampler`):**

```python
        if not (self.is_ddp and self.local_rank != 0):
            samplers.append(SystemSampler())
            samplers.append(GpuUtilizationSampler())   # <-- add here
```

**B. Let the sampler decide (more like `StdoutStderrSampler`):**

```python
        samplers.append(GpuUtilizationSampler())
```

The sampler's `self.sender = None` on non-rank-0 ensures the wire stays
clean even though it instantiates on every rank. Option B is preferable
when you want the sampler to still write a per-rank log file via
`DatabaseWriter`, which `DatabaseWriter` does for any rank that has a
`Database` (it's driven by `config.enable_logging`). For pure
aggregator-side consumption, Option A is simpler.

Pick **Option A** for `GpuUtilizationSampler` (no per-rank file desired):

```diff
             samplers.append(SystemSampler())
+            samplers.append(GpuUtilizationSampler())

         samplers.append(ProcessSampler())
```

And add the import at the top:

```diff
 from traceml.samplers.layer_forward_time_sampler import LayerForwardTimeSampler
 from traceml.samplers.layer_memory_sampler import LayerMemorySampler
 from traceml.samplers.process_sampler import ProcessSampler
+from traceml.samplers.gpu_utilization_sampler import GpuUtilizationSampler
 from traceml.samplers.stdout_stderr_sampler import StdoutStderrSampler
```

That's the entire registration. There is no decorator, no plugin manifest,
no entry point. If you don't edit `_build_samplers`, your sampler is dead
code.

### Step 5 — Set sender knobs

In `__init__` we passed `max_rows_per_flush=1`. Here's the cheat sheet:

| `max_rows_per_flush` | Meaning                                               | When to use                                                                 |
|----------------------|-------------------------------------------------------|-----------------------------------------------------------------------------|
| `-1` (default)       | Send all new rows since the last flush.               | Event-driven samplers with bursty input (e.g. `StepTimeSampler`).           |
| `1`                  | Send only the single latest row per flush.            | Periodic "current state" samplers where intermediate ticks are redundant if sends fall behind (e.g. `SystemSampler`, our GPU sampler). |
| `5`                  | Send up to 5 rows per flush.                          | Rate-capping a hot producer (e.g. layer samplers).                          |
| Other N>0            | At most N rows per flush.                             | Narrow use; document in the sampler.                                        |

`DBIncrementalSender` tracks a monotonic per-table append cursor
(`_last_sent_seq`). If flushes fall behind and the deque evicts rows,
the sender still advances the cursor correctly — it sends whatever is
currently in the deque and marks everything as seen. That means
`max_rows_per_flush=1` is genuinely "latest N rows per table, drop the
rest" behavior, which is exactly what you want for periodic state.

### Step 6 — Handle rank awareness

Three patterns exist in the codebase. Pick one explicitly:

1. **Gate at `_build_samplers` (our Option A above).**
   Sampler is only instantiated on rank 0. No further work needed.
   Used by: `SystemSampler`.

2. **Instantiate everywhere; detach sender on non-rank-0.**
   ```python
   if self._ctx.is_ddp_intended and self._ctx.local_rank != 0:
       self.sender = None
   ```
   Database still gets populated (so local file logs still work), but
   `collect_payload()` on a `None` sender is skipped by `_tick`. Used by:
   `StdoutStderrSampler`.

3. **Per-rank data everywhere.**
   Do nothing special — every rank emits rows with `rank=local_rank`, and
   the aggregator's `RemoteDBStore` keeps them separate. Used by:
   `ProcessSampler`, step and layer samplers.

!!! warning "Never use `rank == 0` as a proxy for `local_rank == 0`"
    On multi-node runs, `RANK` is the global rank and `LOCAL_RANK` is the
    rank on the local host. Host metrics (CPU, RAM, GPU utilization) are
    per-host, not per-global-rank. Always filter on
    `ctx.local_rank == 0` for host metrics.

### Step 7 — Handle the disable path

For the vast majority of samplers, you don't need to do anything.

When `traceml --disable-traceml` is used, the CLI sets `TRACEML_DISABLED=1`
and then bypasses the executor entirely — training runs via plain
`torchrun`. The runtime isn't started, so samplers are never instantiated.

When `TRACEML_DISABLED=1` is set *inside* user code via environment
manipulation (unusual), producers still check the flag:
`src/traceml/utils/step_memory.py`, `src/traceml/utils/timing.py`,
`src/traceml/utils/flush_buffers.py`, and the integrations all early-return
when disabled. Their queues stay empty, so event-driven samplers naturally
produce nothing.

You'd only add an explicit `TRACEML_DISABLED` guard inside a sampler if
that sampler has side effects you want to suppress even when the runtime is
somehow still running — very rare. Skip it unless you have a concrete
reason.

Profile gating is a separate concern and is handled entirely in
`_build_samplers()`. If you want `GpuUtilizationSampler` only on `deep`:

```python
        if self.profile == "deep":
            if not (self.is_ddp and self.local_rank != 0):
                samplers.append(GpuUtilizationSampler())
```

---

## 5. Common patterns & their exemplars

Reference table. When you're writing a new sampler, find the closest
existing one in column 2, copy its structure, then adapt.

| Pattern                                       | Copy from                                                                 |
|-----------------------------------------------|---------------------------------------------------------------------------|
| Periodic polling with psutil                  | `ProcessSampler._sample_cpu`, `_sample_ram`                               |
| Periodic polling with pynvml                  | `SystemSampler._sample_gpus`                                              |
| Event-driven via hook queue (resolve-then-emit) | `LayerForwardTimeSampler`, `LayerBackwardTimeSampler`                   |
| Event-driven, drain-all-and-emit              | `LayerForwardMemorySampler`, `LayerBackwardMemorySampler`, `StepMemorySampler` |
| Step-boundary-triggered with CUDA event resolution | `StepTimeSampler` (see `evt.try_resolve()`)                          |
| One-time manifest write in `__init__`         | `SystemSampler.__init__` → `write_system_manifest_if_missing`             |
| Architecture-dedup by MD5 signature           | `LayerMemorySampler._compute_signature` + `seen_signatures: Set[str]`     |
| Non-rank-0 sender detach                      | `StdoutStderrSampler`, pattern 2 in Section 4 Step 6                      |
| Rate-capped flush                             | `LayerForward*Sampler` / `LayerBackward*Sampler` with `max_rows_per_flush=5` |
| Latest-only periodic                          | `SystemSampler` with `max_rows_per_flush=1`                               |
| Single-sample on change (not every tick)      | `LayerMemorySampler` (architecture signature dedup)                       |

### Notable helpers to reuse

- `drain_queue_nowait(queue)` and `append_queue_nowait_to_deque(queue, deque)`
  in `src/traceml/samplers/utils.py` — use these, don't re-roll
  `while not q.empty(): ...` loops (they're racy).
- `resolve_runtime_context()` in
  `src/traceml/samplers/runtime_context.py` — any env-derived state
  (session id, logs dir, ranks) should come from here.
- `ensure_session_dir(...)` in `src/traceml/samplers/utils.py` — if your
  sampler writes a file alongside its DB (see `StdoutStderrSampler`).
- `write_json_atomic(path, payload)` in `src/traceml/samplers/utils.py` —
  for one-time manifest writes.
- `get_cuda_event()` / `return_cuda_event()` in
  `src/traceml/utils/cuda_event_pool.py` — reuse CUDA events; do not
  `torch.cuda.Event(enable_timing=True)` in a hot loop.

---

## 6. Schema design rules

### 6.1. Flatness and primitive types

- Rows are `dict[str, <primitive-or-list-of-primitives>]`. Nested dicts are
  tolerated (see `ProcessSample.gpu` which is `Optional[dict]`), but one
  level is the limit. No arbitrary nesting.
- **Primitive types only.** `int`, `float`, `str`, `bool`, `list` of those,
  `None`. No dataclasses (call `to_wire()`), no enums (use the string
  value), no numpy scalars (cast via `float(...)` / `int(...)`), no
  `torch.Tensor` (call `.item()` or cast).
- Optional fields: use `None` and document it. Do not use sentinel
  floats like `-1.0` unless there's already precedent in a neighboring
  sampler.

!!! warning "Numpy scalars serialize — but don't"
    `msgspec.msgpack` can encode `numpy.float64`, but it's non-portable and
    silently expensive. Cast to native Python numbers at the sampler
    boundary. The cost is a single C-level conversion; the payoff is a
    clean wire payload that any decoder can read.

### 6.2. Timestamp convention

- `"ts"`: unix epoch seconds as `float` (from `time.time()`). Used by every
  numeric sampler.
- `"timestamp"`: same meaning as `"ts"`, spelled out. Some samplers use
  this longer form. Both are acceptable; prefer `"ts"` for new code because
  that's what `SystemSample.to_wire()` and `ProcessSample.to_wire()` use.
- ISO-8601 UTC strings are used for CLI-level artifacts (manifests), **not**
  for telemetry rows. Don't use them in sampler payloads.

### 6.3. Column naming

- `snake_case`.
- Short wire keys (`seq`, `ts`, `cpu`, `ram_used`) are acceptable and
  precedented; long Python attribute names are fine in the dataclass.
- Domain-prefix when ambiguous: `gpu_mem_used`, not `mem_used`, when the
  same row contains CPU and GPU memory.
- Units in the name or the docstring — not both. Prefer a docstring.

### 6.4. Rank column

You generally do **not** need to include `rank` or `local_rank` in your
row. The wire payload wrapper
(`DBIncrementalSender.collect_payload`) already stamps `"rank"` on the
outer envelope:

```python
{
    "rank": self.rank,
    "sampler": self.sampler_name,
    "timestamp": time.time(),
    "tables": tables_payload,
}
```

If your sampler *actually* has per-row rank attribution (e.g. rows from
different ranks interleaved in one table), include an explicit `rank`
field — but this is rare; it's normally cleaner to let the envelope
carry rank and the aggregator's `RemoteDBStore` to shard by rank.

### 6.5. Backward compatibility

We have users on v0.2.3. The aggregator and renderers must continue to
decode old wire payloads.

Rules:

- **Never remove or rename an existing key.** Add new keys instead.
- **Never change the type of an existing key.** A `float` stays a `float`
  forever.
- **New keys must be optional on the consumer side.** Renderers should use
  `payload.get("new_field", default)`, not `payload["new_field"]`.
- If you genuinely need a breaking schema change, introduce a new table
  name and deprecate the old one over a release cycle.

Rationale: wire compatibility keeps comparing old and new runs possible
(`traceml compare`), and keeps installed `traceml-ai` clients from breaking
when the aggregator side upgrades.

---

## 7. Overhead budget & performance

### Targets

- A full tick of all samplers in the `watch` profile should cost
  **well under 1%** of a typical step. On a 100 ms/step job at 1 s sampler
  interval, that's a generous 1 ms budget per tick. Most periodic samplers
  spend far less.
- Event-driven samplers should be O(queue depth). Draining 1k queued
  events per tick should still stay under ~1 ms of Python-level work.
- Per-row emission cost: target **sub-microsecond** for building the row
  dict (`dataclass.to_wire()` is fine). The `deque.append()` is O(1).

### Hot-path rules

1. **No allocations in inner loops if you can avoid it.** Build
   intermediate dicts outside the hottest inner loops when iterating per
   GPU / per layer.
2. **Reuse CUDA events from the pool.** See
   `src/traceml/utils/cuda_event_pool.py`. Never
   `torch.cuda.Event(enable_timing=True)` inside a hook; always
   `get_cuda_event()` and `return_cuda_event(evt)`.
3. **No psutil calls inside the training step.** psutil is safe inside
   `sample()` (background thread) but unsafe as a patch inside the user's
   forward pass.
4. **No blocking I/O inside `sample()`.** The rotating-file logger is
   asynchronous enough in practice, but avoid explicit `open()`, `fsync()`,
   or network calls.
5. **Never call `torch.cuda.synchronize()` from a sampler.** Synchronization
   serializes the GPU and can introduce 10+ ms stalls per tick.
6. **Bound every queue at the producer.** Layer hooks use
   `queue.Queue(maxsize=4096)` and handle `queue.Full` by dropping (see
   `src/traceml/utils/hooks/layer_forward_time_hooks.py`). Follow that
   pattern.

### Micro-benchmarking

There is no formal benchmark harness for samplers as of this writing.
When in doubt, drop a throwaway `timeit`:

```python
import timeit
s = GpuUtilizationSampler()
n = 10_000
dt = timeit.timeit(s.sample, number=n) / n
print(f"{dt*1e6:.2f} µs per sample")
```

Run on the dev box, confirm sub-millisecond, move on. If you want a more
thorough check, look at `tests/bench_tcp_drain.py` and
`tests/benchmark_hook_opt.py` for existing benchmark shapes.

!!! tip "Threading"
    All samplers run sequentially on a single `TraceMLSampler(rank=N)`
    thread (see `TraceMLRuntime._sampler_loop`). Do not introduce your own
    threading inside a sampler. If you need async work (file I/O,
    network), that is a runtime concern and is out of scope for a sampler.

---

## 8. Testing

### Existing test patterns

There is no dedicated sampler-test file in `tests/` at the time of writing.
The closest references are:

- `tests/test_seq_counter.py` — exercises the `Database`,
  `DBIncrementalSender`, `DatabaseWriter` contract end-to-end using
  `unittest.mock.MagicMock` as the transport. Excellent reference for how
  to assert on outgoing payloads without booting the real TCP stack.
- `tests/test_msgpack_roundtrip.py` — asserts wire compatibility across
  the encoder/decoder pair.
- `tests/test_hf_trainer.py` — an integration-style test that boots the
  HuggingFace Trainer integration and checks `TraceState.step` incremented.
  This is an end-to-end smoke test, not a unit test; use sparingly.
- `tests/test_trend_core.py` — pure-function tests for analytics modules.
  Good shape to copy for deterministic sampler logic.
- `tests/test_grad_accum.py` — integration-style test for gradient
  accumulation in the `trace_step` decorator path.
- `tests/test_compare_missing.py` — tests the `traceml compare` CLI path.

### What a new sampler's test should cover

At minimum:

1. **Construction.** The sampler can be instantiated without touching the
   network, a real GPU, or torch being initialized.
2. **Single-sample success.** After calling `sample()` once, exactly one
   row lands in the expected table with the expected schema.
3. **Source failure.** When the data source raises
   (mock it to raise), `sample()` does not raise; an error is logged; no
   row is emitted (or a zero-filled row is emitted, per your design).
4. **Idempotence across many ticks.** Calling `sample()` N times appends
   N rows (or fewer, if the sampler dedups / rate-caps; assert your
   invariant explicitly).

### Minimal test template

Put this in `tests/test_gpu_utilization_sampler.py`:

```python
"""
Tests for GpuUtilizationSampler.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    """
    Make resolve_runtime_context() succeed without a live torchrun.
    """
    monkeypatch.setenv("TRACEML_LOGS_DIR", str(tmp_path))
    monkeypatch.setenv("TRACEML_SESSION_ID", "test_session")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    yield


def _build_sampler(n_gpus=2):
    """
    Construct the sampler with NVML fully mocked.
    """
    from traceml.samplers import gpu_utilization_sampler as mod

    util = MagicMock(gpu=42.0)
    mem = MagicMock(used=1024.0)

    with patch.object(mod, "nvmlInit", return_value=None), patch.object(
        mod, "nvmlDeviceGetCount", return_value=n_gpus
    ), patch.object(
        mod, "nvmlDeviceGetHandleByIndex", side_effect=lambda i: i
    ), patch.object(
        mod, "nvmlDeviceGetUtilizationRates", return_value=util
    ), patch.object(
        mod, "nvmlDeviceGetMemoryInfo", return_value=mem
    ):
        s = mod.GpuUtilizationSampler()
        s.sample()
        return s


class TestGpuUtilizationSampler:
    def test_constructs_with_nvml_init_failure(self, monkeypatch):
        from pynvml import NVMLError

        from traceml.samplers import gpu_utilization_sampler as mod

        def _boom(*a, **kw):
            raise NVMLError(1)

        with patch.object(mod, "nvmlInit", side_effect=_boom):
            s = mod.GpuUtilizationSampler()

        assert s.gpu_available is False
        assert s.gpu_count == 0
        # sample() on a degraded sampler is a no-op (plus a log line).
        s.sample()
        rows = s.db.get_table("GpuUtilizationTable") or []
        assert len(rows) == 0

    def test_single_sample_emits_one_row(self):
        s = _build_sampler(n_gpus=2)
        rows = list(s.db.get_table("GpuUtilizationTable"))
        assert len(rows) == 1
        row = rows[0]
        assert row["seq"] == 1
        assert row["gpu_count"] == 2
        assert len(row["gpus"]) == 2
        assert row["gpus"][0]["util_percent"] == 42.0
        assert row["gpus"][1]["mem_used_bytes"] == 1024.0

    def test_one_failing_gpu_does_not_break_others(self):
        from traceml.samplers import gpu_utilization_sampler as mod

        def _mem(i):
            if i == 0:
                raise RuntimeError("GPU 0 exploded")
            return MagicMock(used=2048.0)

        with patch.object(mod, "nvmlInit", return_value=None), patch.object(
            mod, "nvmlDeviceGetCount", return_value=2
        ), patch.object(
            mod, "nvmlDeviceGetHandleByIndex", side_effect=lambda i: i
        ), patch.object(
            mod,
            "nvmlDeviceGetUtilizationRates",
            return_value=MagicMock(gpu=99.0),
        ), patch.object(
            mod, "nvmlDeviceGetMemoryInfo", side_effect=_mem
        ):
            s = mod.GpuUtilizationSampler()
            s.sample()  # must not raise

        rows = list(s.db.get_table("GpuUtilizationTable"))
        assert len(rows) == 1
        # Failed GPU still appears, zero-filled.
        assert rows[0]["gpus"][0] == {
            "gpu_idx": 0,
            "util_percent": 0.0,
            "mem_used_bytes": 0.0,
        }
        assert rows[0]["gpus"][1]["mem_used_bytes"] == 2048.0
```

Things to notice:

- The autouse fixture sets the five env vars `resolve_runtime_context()`
  reads so sampler construction doesn't fail on a dev box without torchrun.
- NVML calls are patched at the module level (`mod.nvmlInit`, etc.) — so
  the import-level symbols the sampler actually calls are the ones we
  override. This is why the sampler imports NVML symbols by name from
  `pynvml` rather than calling `pynvml.nvmlInit()`: it makes patching
  trivial.
- Every assertion is on the Database contents, not on `self.sender` —
  the sender is wired up by the runtime, not by `__init__`. During unit
  tests the sender exists but has no transport attached.

---

## 9. Common pitfalls

Numbered, with symptom and fix. If you hit one of these, check here first.

1. **Symptom:** My sampler compiles and imports, but no rows ever reach
   the aggregator.
   **Cause:** You forgot to add it to `TraceMLRuntime._build_samplers()`.
   **Fix:** Add it. There is no auto-discovery.

2. **Symptom:** Training dies with a `KeyError` or `RuntimeError` mentioning
   my sampler.
   **Cause:** `sample()` raised. The runtime's `_safe()` will catch it, but
   if a nested frame re-raises, training gets unhappy.
   **Fix:** Wrap the body of `sample()` in `try/except Exception` and log
   via `self.logger.error(...)`. Fail open is the law.

3. **Symptom:** Aggregator appears to freeze or log-line flood;
   `DBIncrementalSender` log shows huge payload sizes.
   **Cause:** Sampler emits too many rows per tick (e.g. one row per
   forward invocation of a large model).
   **Fix:** Set `max_rows_per_flush=5` (or similar). The sender will
   send the latest N per tick and drop older ones. The cursor still
   advances correctly across evictions.

4. **Symptom:** `ValueError: <name> has no default table name configured.`
   **Cause:** You called `self._add_record(payload)` without passing
   `table_name=` and without setting `table_name` in `super().__init__`.
   **Fix:** Always pass `table_name` to `super().__init__`, or pass
   `table_name=...` to each `_add_record()` call.

5. **Symptom:** Out-of-memory on long runs.
   **Cause:** Your sampler is fed by an *unbounded* external queue (e.g.
   you created a `queue.Queue()` without `maxsize`), and the producer
   runs faster than `sample()` drains.
   **Fix:** Bound the queue at the producer (`Queue(maxsize=4096)`) and
   handle `queue.Full` by dropping. See
   `src/traceml/utils/hooks/layer_forward_time_hooks.py` for the pattern.
   The `Database` is already bounded via `deque(maxlen=max_rows)`; the
   *upstream* queue is what can explode.

6. **Symptom:** Users report overhead even with `watch` profile on
   production.
   **Cause:** A sampler you added runs unconditionally, ignoring the
   profile.
   **Fix:** Gate it explicitly in `_build_samplers()`:
   `if self.profile == "deep": samplers.append(...)`.

7. **Symptom:** `msgspec.EncodeError` on the wire, aggregator crashes on
   decode.
   **Cause:** You put a non-primitive in a row — a dataclass instance,
   numpy scalar, tensor, set, or deeply nested dict.
   **Fix:** Always call `to_wire()` on dataclass payloads. Cast numpy
   types via `float(x) / int(x)`. Convert sets to sorted lists.

8. **Symptom:** Duplicate rows on the aggregator side, one per rank, for
   host metrics.
   **Cause:** You instantiated a host-level sampler on every rank without
   detaching the sender or gating in `_build_samplers`.
   **Fix:** Either gate on `not (self.is_ddp and self.local_rank != 0)`
   at construction, or set `self.sender = None` on non-rank-0 in
   `__init__`. Pick one; don't do both.

9. **Symptom:** Tests pass locally, `pip install traceml-ai` fails for a
   user without a GPU.
   **Cause:** You imported `pynvml` at the top of a module that's loaded
   unconditionally (e.g. `runtime.py`), and pynvml's Python wrapper
   misbehaves without NVML drivers.
   **Fix:** Keep pynvml imports inside the sampler file (not in
   `runtime.py`) — the sampler module is only imported if the sampler is
   instantiated. Guard NVML *initialization* (`nvmlInit()`) with
   try/except. Do not add new top-level imports of hardware-specific
   libraries in `runtime.py` or `executor.py`.

10. **Symptom:** On DDP runs, `torch.cuda.*` calls from the sampler hang
    or raise "CUDA not initialized".
    **Cause:** You touched CUDA before `dist.init_process_group()` ran.
    **Fix:** Copy `ProcessSampler._cuda_safe_to_touch` — check
    `dist.is_available()` and `dist.is_initialized()` first; return `None`
    (degraded row) otherwise. CUDA samples will come in a few ticks later,
    once distributed init finishes.

11. **Symptom:** `sample_idx` skips values on the aggregator side.
    **Cause:** You increment `self.sample_idx` inside a conditional, so
    ticks where nothing is emitted don't bump the counter. That's actually
    fine — see `LayerForwardTimeSampler`, which only bumps when a step is
    fully resolved and persisted. Just make sure you document the
    semantics.

---

## 10. Checklist before opening a PR

1. [ ] New sampler file in `src/traceml/samplers/<name>_sampler.py`.
2. [ ] Subclasses `BaseSampler`; takes no constructor args.
3. [ ] `super().__init__(sampler_name=..., table_name=..., max_rows_per_flush=...)` called.
4. [ ] `sample()` is wrapped in `try/except Exception` with a
       `self.logger.error(...)` on failure. No re-raises.
5. [ ] Schema dataclass added in `src/traceml/samplers/schema/` with
       `to_wire()` and `from_wire()`.
6. [ ] Registered in `TraceMLRuntime._build_samplers()` under the correct
       profile branch.
7. [ ] Import added at the top of `src/traceml/runtime/runtime.py`.
8. [ ] Rank-awareness decision made explicitly (gated at runtime, sender
       detached, or per-rank — pick one).
9. [ ] Module-level docstring describes: what it measures, periodic vs
       event-driven, rank scope, guarantees.
10. [ ] Function docstrings for `sample()` and each non-trivial helper.
11. [ ] Tests in `tests/test_<name>_sampler.py` covering construction,
        one-sample success, source-failure safety. `pytest` passes.
12. [ ] Local smoke test: `pip install -e ".[dev,torch]"` then
        `traceml watch examples/<a small example>.py` — the training run
        completes, no stack traces on stderr, rows visible in the live UI
        (for CLI mode) or in `logs/<session>/aggregator/telemetry` (for
        summary mode).
13. [ ] If running multi-GPU, verify with
        `traceml watch example.py --nproc-per-node 2` — no duplicate
        host-level rows, each rank's per-rank rows still show up.
14. [ ] No new top-level dependencies added to `pyproject.toml` without
        explicit sign-off. psutil / pynvml / torch are acceptable; anything
        else needs a conversation.
15. [ ] `pre-commit run --all-files` clean (black, ruff, isort, codespell).
16. [ ] Commit message is short and in the house style: no
        `Co-Authored-By` trailers, single-line subject, optional body.

---

## 11. Appendix: adding a new profile

Rare. Almost all samplers fit under `watch`, `run`, or `deep`. Only add a
new profile if the sampler genuinely does not belong in any of those
(e.g. a pilot experimental tracer you want gated behind an explicit
opt-in).

The plumbing touches three files:

1. **`src/traceml/cli.py`.** In `build_parser()`, add a new subparser next
   to `watch_parser`, `run_parser`, `deep_parser`, and call
   `_add_launch_args(new_parser)`. In `main()`, route to
   `run_with_tracing(args, profile="<new_profile>")`.

2. **`src/traceml/runtime/runtime.py`.** In `_build_samplers()`, add a
   branch:
   ```python
   if self.profile == "<new_profile>":
       samplers += [YourSampler(), ...]
   ```

3. **`src/traceml/runtime/executor.py`.** No change required. The profile
   string flows through `TRACEML_PROFILE`, which `read_traceml_env()`
   already reads.

Also update:

- `src/traceml/runtime/settings.py::TraceMLSettings.profile` docstring if
  the default changes (don't change the default).
- The top-level `README.md` and `docs/` if you want users to know.

!!! note "Profile names are strings, not enums"
    Keep them short and unambiguous. `trace`, `deep`, `run`, `watch` are
    fine. Avoid underscores or punctuation; these strings also appear in
    manifests and summary JSON.

---

## Gaps / ambiguities encountered while writing this guide

These are places where the current source does not fully pin down a
contract. Flag these in code review if your sampler lands near them:

- **`DatabaseWriter` lifecycle.** `BaseSampler` constructs a `Database`
  which in turn constructs a `DatabaseWriter`. The writer's flush cadence
  and on-disk behavior is controlled by `traceml.runtime.config.config`
  and activated in `runtime.py::_tick` via `db.writer.flush()`. I did not
  document it here because sampler authors don't normally touch it. If
  your sampler produces rows that must not be written to disk, there is
  currently no first-class way to opt out other than not calling
  `db.writer` or setting `config.enable_logging = False` globally.
- **`BaseSampler` lifecycle hooks.** The base class has no `start()` /
  `stop()`. A few real samplers (e.g. `StdoutStderrSampler`) do setup in
  `__init__` that arguably belongs in a lifecycle method. I noted the
  constraint in Section 3.6 and copied the house pattern.
- **`model_forward_memory_sampler.py` is not wired into any profile.**
  It exists but is unused. I mentioned it in the prompt context but did
  not walk through it as an exemplar; treat it as a draft.
- **Sampler-level test fixtures.** There is no shared `conftest.py` for
  sampler tests, no `MockSender` fixture, no `Database` fixture. My test
  template rolls its own mocks. If sampler tests proliferate, a shared
  `tests/samplers/conftest.py` would pay for itself.
- **Overhead budgets.** The "<1% of step time" number is folklore from
  the PR that introduced `max_rows_per_flush=1` for `SystemSampler`; it's
  not codified in a benchmark harness. Treat it as the target, not a
  test gate.
- **`run_name`-style profile extension.** `profile` is currently a plain
  string compared with `==` in `_build_samplers`. If we ever want
  overlapping profile flags (`--profile=deep,net`) we would need to switch
  to a set or bitmask. Out of scope for adding a single sampler, worth
  knowing about if you find yourself copy-pasting profile branches.
