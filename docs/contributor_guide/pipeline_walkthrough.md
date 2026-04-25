# Pipeline walkthrough — one telemetry event, end-to-end

> Internal contributor reference. Audience: anyone touching TraceML code.
> Use this as the shared mental model that every per-feature contributor
> guide refers back to.

If you can trace a single telemetry event from the moment it's born inside the user's training process to the moment it renders as a number on someone's terminal, you understand TraceML. This document follows one event through eight stations. The deep coverage of each station lives in the [W6–W10](../deep_dive/code-walkthroughs.md) walkthroughs and in [PR #87 Appendix D](../deep_dive/pr_reviews/pr-87-h2d-timing.md) — this is the condensed map.

**Last verified:** 2026-04-25.

---

## The event we follow

A user calls `tensor.to("cuda")` inside `trace_step()` while running with TraceML in `--profile run`:

```python
import traceml
traceml.init(mode="auto")    # installs patches

with traceml.trace_step(model):
    batch = next(iter(loader))
    batch = batch.to("cuda")    # ← this is the call we trace
    out = model(batch)
    loss = out.loss
    loss.backward()
    optimizer.step()
# trace_step exits → step boundary → flush
```

The patched `torch.Tensor.to` (PR #87) emits exactly one `TimeEvent` with `name="_traceml_internal:h2d_time"`. That event has eight stations to traverse before it shows up as a row in the user's terminal.

---

## The eight stations

```
[ Training rank ]
  Station 1 · timed_region creates a TimeEvent → appended to _STEP_BUFFER (deque)
  Station 2 · trace_step exits → flush_step_time_buffer drains buffer into _STEP_TIME_QUEUE
  Station 3 · StepTimeSampler.sample() drains queue, resolves CUDA events, aggregates, writes to Database
  Station 4 · DBIncrementalSender ships only new rows via TCPClient (msgpack framed)
                                            ──── TCP wire ────
[ Aggregator ]
  Station 5 · TCPServer receives frames, routes payload to RemoteDBStore (rank-aware)
  Station 6 · SQLite projection writer (per-sampler) fans out the payload into query-friendly tables
  Station 7 · Renderer reads SQLite (windowed query), produces Rich/Plotly output
  Station 8 · Display driver flushes to terminal (Rich Live) or browser (NiceGUI)
```

Stations 1–4 live in the **training process** (one per rank). Stations 5–8 live in the **aggregator process** (one, long-running). The TCP wire between Station 4 and Station 5 is the only inter-process communication.

---

## Station 1 — `timed_region` creates a `TimeEvent`

**File:** [`traceml/src/traceml/utils/timing.py`](../../src/traceml/utils/timing.py) · **Walkthrough:** [W4](../deep_dive/code-walkthroughs.md#w4-patches--timing-primitives--how-zero-code-instrumentation-actually-works).

The patched `.to()` runs inside `with timed_region("_traceml_internal:h2d_time", scope="step", use_gpu=True):`. The context manager records `cpu_start = time.time()`, allocates two CUDA events from the pool (`get_cuda_event()`), records `start_evt.record()` on the current stream, yields to the wrapped function, then in `finally` records `end_evt.record()`, captures `cpu_end`, and constructs a `TimeEvent` dataclass holding both CPU timestamps and pending CUDA events. The event is appended to the module-global `_STEP_BUFFER` deque.

**Two design points worth holding in cache:**

- **CPU and GPU timestamps both captured.** GPU events are accurate; CPU is the fallback when CUDA is unavailable, and a sanity cross-check.
- **GPU events are recorded but NOT resolved here.** `start.record()` and `end.record()` enqueue timestamp ops on the stream; calling `elapsed_time(start, end)` would require synchronization, which would serialize training and destroy the overhead budget. Resolution is deferred to Station 3.

`_STEP_BUFFER` is a plain `deque`, written only from the training thread. Thread-safety is by convention, not by lock — see [PR #87 §3.2](../deep_dive/pr_reviews/pr-87-h2d-timing.md) for what happens when this convention is violated.

---

## Station 2 — `trace_step` exits → flush to the queue

**File:** [`traceml/src/traceml/utils/timing.py`](../../src/traceml/utils/timing.py) (`flush_step_time_buffer`) · **Walkthrough:** [W3](../deep_dive/code-walkthroughs.md#w3-user-facing-api--decorators-instrumentation-wrappers).

When the user exits the `with traceml.trace_step(model):` block, `flush_step_time_buffer(step)` runs. It drains the entire `_STEP_BUFFER` into a single `StepTimeBatch(step=N, events=[...])` and `put_nowait`s the batch onto `_STEP_TIME_QUEUE` — a `queue.Queue(maxsize=2048)` that's the cross-thread handoff primitive.

**Two design points:**

- **Step batching is load-bearing.** All `TimeEvent`s from the same step (forward, backward, h2d_time, optimizer) ride in one batch. The sampler downstream collapses N invocations of the same `name` in the same step into one row with `n_calls` and `sum_ms`.
- **`put_nowait` + drop-on-full.** Under sustained overflow, batches are logged-and-dropped rather than blocking training. Fail-open in action.

CUDA events in the batch are still pending. Training thread is already running step N+1.

---

## Station 3 — `StepTimeSampler.sample()` resolves and aggregates

**File:** [`traceml/src/traceml/samplers/step_time_sampler.py`](../../src/traceml/samplers/step_time_sampler.py) · **Walkthrough:** [W6](../deep_dive/code-walkthroughs.md#w6-samplers--schemas--turning-hook-events-into-structured-rows).

The sampler runs on a separate thread inside the training process — the runtime's background sampler loop, polling at `TRACEML_INTERVAL` (default 1.0 s). Each tick, `sample()` does three things:

**3a.** Drain `_STEP_TIME_QUEUE` into a local FIFO (`self._pending`). The split is deliberate — the sampler doesn't hold the global queue while waiting on GPU resolution.

**3b.** For each pending batch, call `evt.try_resolve()` on every event. `try_resolve()` checks `gpu_end.query()` (non-blocking) — `True` if the GPU has completed the event, `False` otherwise. If `True`, the sampler reads `gpu_start.elapsed_time(gpu_end)`, returns both events to the pool (`return_cuda_event`), and marks the event resolved. If `False`, the sampler bails and retries next tick. **No `cuda.synchronize()` ever fires**; this is how TraceML times CUDA ops without blocking training.

**3c.** Aggregate the resolved batch by `(name, device, is_gpu)`: sum durations, count invocations. Build a row like `{"step": N, "events": {"_traceml_internal:h2d_time": {"cuda:0": {"is_gpu": True, "duration_ms": 0.41, "n_calls": 1}}, ...}}` and write it to the in-memory `Database` (table: `StepTimeTable`) via `BaseSampler._add_record()`.

**Design point worth holding:** `n_calls` and `sum_ms` are both first-class. `sum_ms / n_calls` recovers the per-call mean; `sum_ms` alone is "how much wall-time this `name` cost this step." Renderer choices about which to display matter — see [PR #87 §3.2](../deep_dive/pr_reviews/pr-87-h2d-timing.md) for the overcount bug.

---

## Station 4 — `DBIncrementalSender` ships only new rows

**File:** [`traceml/src/traceml/database/database_sender.py`](../../src/traceml/database/database_sender.py) · **Walkthrough:** [W7](../deep_dive/code-walkthroughs.md#w7-database--sender--bounded-in-memory-store-and-incremental-tcp-shipping).

The runtime's tick calls `sender.collect_payload()` after every sampler. The sender tracks a monotonic per-table append cursor (`_last_sent_seq`). It reads `db.get_append_count(table)`, computes the delta, slices the new rows, builds an envelope `{"rank": ..., "sampler": ..., "timestamp": ..., "tables": {...}}`, encodes via `msgspec.msgpack`, and ships through the TCP client.

**Design points:**

- **Only-new-rows.** If the deque has evicted rows since the last flush, the cursor still advances to the current position. Eviction equals "drop"; no reshipping.
- **`max_rows_per_flush` cap.** Configurable per sampler. `1` = latest-only (good for periodic state). `-1` = all-new-rows (good for event-driven). `5` = rate-cap (good for hot producers).
- **Failed send drops the batch but advances the cursor.** Aggregator unreachable means lost rows, not a stall. Fail-open.

The wire format is length-prefixed msgpack frames. Header is 4 bytes big-endian unsigned int (frame length); payload is the msgpack-encoded envelope. See [Q10](../deep_dive/learning-qa.md#q10-what-is-tcp-concretely-and-whats-a-port) for the TCP framing rationale.

---

## Station 5 — `TCPServer` receives, routes to `RemoteDBStore`

**File:** [`traceml/src/traceml/transport/tcp_transport.py`](../../src/traceml/transport/tcp_transport.py), [`traceml/src/traceml/database/remote_database_store.py`](../../src/traceml/database/remote_database_store.py) · **Walkthrough:** [W8](../deep_dive/code-walkthroughs.md#w8-transport--tcp-serverclient-msgpack-framing-ddp-rank-detection), [W9](../deep_dive/code-walkthroughs.md#w9-aggregator-core--tcp-receive-frame-dispatch-sqlite-writes).

The aggregator's TCP server runs a single accept thread plus one recv thread per connected rank. Each frame is decoded via `msgspec.msgpack`. The envelope's `rank` and `sampler` route the inner `tables` payload into `RemoteDBStore`, which lazily creates a `Database(rank, sampler_name)` on first ingestion from each (rank, sampler) pair. `RemoteDBStore` is the in-memory rank-aware shadow of every sampler's local `Database`.

**Design points:**

- **Single-threaded ingest** — the aggregator runs the recv-and-dispatch loop on one thread. Reader threads (renderers, SQLite writers) do not block ingest; ingest is the only writer.
- **Rank-aware by construction.** `Database(rank, sampler_name)` keeps rank 0, rank 1, ... separate. Per-rank views and "worst-rank wins" aggregations both work over the same store.
- **Allow-list for in-memory retention.** `_REMOTE_STORE_SAMPLERS` (currently the four layer samplers) is the only retained-in-memory set. Other samplers go straight to SQLite (Station 6) without persisting in `RemoteDBStore`. This is a memory-pressure trade-off; renderers that need live in-memory rows belong on the allow-list, the rest read SQLite.

---

## Station 6 — SQLite projection writer fans out the payload

**File:** [`traceml/src/traceml/aggregator/sqlite_writers/`](../../src/traceml/aggregator/sqlite_writers/) · **Walkthrough:** [W9](../deep_dive/code-walkthroughs.md#w9-aggregator-core--tcp-receive-frame-dispatch-sqlite-writes).

For every received envelope, the aggregator's writer dispatcher routes to the per-sampler projection writer. Each writer (`system.py`, `process.py`, `step_time.py`, `step_memory.py`, `stdout_stderr.py`) defines an `init_schema(conn)` that runs at aggregator startup (`CREATE TABLE IF NOT EXISTS ...`, indexes) and a write method that consumes the wire payload row-by-row, projecting flat fields into columns and nested structures into child tables (FK-linked) or JSON blobs.

The SQLite database opens in **WAL mode** at startup, which lets the single writer thread coexist with many concurrent reader threads (renderers querying for windowed views). The database file lives at `logs/<session>/aggregator/telemetry.sqlite` and survives the run for `traceml compare` to consume later.

**Design points:**

- **Single-writer thread.** Same thread as ingest. Per-payload write must be sub-millisecond; use prepared statements, no `SELECT` inside the writer.
- **Schema additions only.** New columns added via `ALTER TABLE ... ADD COLUMN ... DEFAULT NULL`. Never drop, never rename, never change type. v0.2.3 users have existing databases.
- **`recv_ts_ns` and the sampler-side `ts` both retained.** Lag detection (recv-vs-sample latency) and timeline correctness (when the event actually happened) need different timestamps.

This is the bridge between sampler and renderer. If a sampler emits rows but no projection writer exists, renderers can't query the data — `RemoteDBStore` is the in-memory escape hatch for the four allow-listed samplers, not a general solution. See [add_sqlite_projection.md](add_sqlite_projection.md) for the author's guide to writing one.

---

## Station 7 — Renderer reads SQLite, produces output

**File:** [`traceml/src/traceml/renderers/`](../../src/traceml/renderers/) · **Walkthrough:** [W10](../deep_dive/code-walkthroughs.md#w10-display-drivers--renderers--terminal-and-web-ui-from-sql).

Each renderer is a `BaseRenderer` subclass with a `*Computer` that opens a short-lived SQLite connection per call, runs a bounded query (`SELECT ... ORDER BY id DESC LIMIT N`), groups in Python, builds a frozen dataclass result, and returns it. The renderer formats the result for its medium: `get_panel_renderable()` returns a Rich `Panel` for the CLI driver; `get_dashboard_renderable()` returns the dataclass directly for the NiceGUI driver to render.

**Design points:**

- **Read-only.** Renderers never write back to `RemoteDBStore` or SQLite. Mutating a data source from the read path produces weird intermittent bugs.
- **Bounded queries with stale-cache fallback.** Every renderer caps the query (`window_size * lookback_factor`); if the read fails or returns empty, the renderer returns the last good result for `stale_ttl_s` seconds before giving up.
- **Compute / schema / render split.** Mature renderers separate the SQL reduction (`compute.py`) from the dataclass schema (`schema.py`) from the display (`renderer.py`). Lets the dashboard and CLI share compute, and lets unit tests assert on the dataclass without booting Rich or NiceGUI.

---

## Station 8 — Display driver flushes to the user

**File:** [`traceml/src/traceml/aggregator/display_drivers/`](../../src/traceml/aggregator/display_drivers/) · **Walkthrough:** [W10](../deep_dive/code-walkthroughs.md#w10-display-drivers--renderers--terminal-and-web-ui-from-sql).

The display driver runs a tick loop at `render_interval_sec` (default 2.0 s). Each tick, it iterates over its registered renderers, calls the appropriate method (`get_panel_renderable()` for CLI, `get_dashboard_renderable()` for NiceGUI), and updates the display. The CLI driver uses Rich `Live` for in-place terminal updates; the NiceGUI driver pushes typed payloads to per-client subscribers, which mutate Plotly figures in-place.

The driver is selected by the `--mode` flag in the CLI (`cli` / `dashboard` / `summary`), routed through the `_DISPLAY_DRIVERS` dispatcher in `trace_aggregator.py`. The summary driver has no live UI — it only produces an end-of-run summary card.

**Design points:**

- **Tick-rate-limited.** The renderer is bounded query → dataclass → format. The driver's tick interval (2.0 s) gates how often the user sees updates. Too frequent = wasted CPU; too infrequent = laggy UI.
- **Per-renderer error isolation.** If `get_panel_renderable()` raises, the driver shows a red "Render Error" panel for that section; other renderers keep working.
- **Profile gates which renderers exist.** `watch` profile = minimal (system + process + stdout/stderr). `run` adds step-time + step-memory. `deep` adds the four layer renderers.

---

## What this means for each per-feature guide

Every contributor guide in this folder describes a feature that lives at one or two stations. Knowing the station is half the work:

| Feature | Lives at station | Guide |
|---|---|---|
| Sampler | 3 (and 1–2 if event-driven) | [add_sampler.md](add_sampler.md) |
| Patch | 1 (and 2 via `flush_step_time_buffer`) | [add_patch.md](add_patch.md) |
| SQLite projection writer | 6 | [add_sqlite_projection.md](add_sqlite_projection.md) |
| Renderer | 7 | [add_renderer.md](add_renderer.md) |
| Display driver | 8 | [add_renderer.md](add_renderer.md) §12 (advanced) |
| Diagnostic | post-Station 8 (consumes SQLite at end-of-run) | [add_diagnostic.md](add_diagnostic.md) |
| CLI flag / `--mode` | upstream of the whole pipeline (configures all eight) | [add_cli.md](add_cli.md) |

The feature you're adding owns one or two stations and does not touch the others. If you find yourself modifying multiple stations in the same PR, that's a flag — either you're reaching beyond the feature's natural scope, or you're doing a wire-format change (which touches Stations 4–6 by definition; see the phase-2 `change_wire_format.md` guide).

---

## Two checkpoint questions

If you can answer these without re-reading, you have the mental model.

**Q1.** Why does TraceML use a three-step buffer chain — `_STEP_BUFFER` (deque, training thread) → `_STEP_TIME_QUEUE` (`queue.Queue`, cross-thread) → `self._pending` (deque, sampler thread) — instead of writing directly from `timed_region` to the queue?

*Answer sketch:* the deque is sub-microsecond; `queue.Queue.put` is microseconds (lock + condvar). On the hot path, the deque keeps the patch overhead invisible. The handoff to the cross-thread queue happens once per step, at `trace_step` exit, so the cost is amortized. The sampler-side `_pending` deque is a third buffer because the sampler may need multiple ticks to drain pending CUDA events without holding the cross-thread queue.

**Q2.** If you replaced `try_resolve()`'s non-blocking `event.query()` with `torch.cuda.synchronize()` at the top of `sample()`, the code would still work and be simpler. What's the cost?

*Answer sketch:* `cuda.synchronize()` blocks the calling thread (the sampler thread) until all stream operations complete. That's harmless to the sampler. But `synchronize` is GPU-global — it also blocks any GPU work the **training thread** has queued. Now every sampler tick drains the GPU pipeline, training throughput drops measurably, and the overhead budget collapses. Non-blocking `event.query()` reads the timestamp without forcing the GPU to drain.

---

## Cross-references

- **Deep walkthroughs:** [W6 (samplers)](../deep_dive/code-walkthroughs.md#w6-samplers--schemas--turning-hook-events-into-structured-rows), [W7 (DB / sender)](../deep_dive/code-walkthroughs.md#w7-database--sender--bounded-in-memory-store-and-incremental-tcp-shipping), [W8 (transport)](../deep_dive/code-walkthroughs.md#w8-transport--tcp-serverclient-msgpack-framing-ddp-rank-detection), [W9 (aggregator core)](../deep_dive/code-walkthroughs.md#w9-aggregator-core--tcp-receive-frame-dispatch-sqlite-writes), [W10 (renderers / drivers)](../deep_dive/code-walkthroughs.md#w10-display-drivers--renderers--terminal-and-web-ui-from-sql).
- **Concept Q&A:** [Q10 (TCP)](../deep_dive/learning-qa.md#q10-what-is-tcp-concretely-and-whats-a-port), [Q15 (CUDA streams)](../deep_dive/learning-qa.md#q15-what-is-a-cuda-stream-and-how-does-it-differ-from-a-cpu-thread).
- **PyTorch internals:** [P48 (`_call_impl`)](../deep_dive/pytorch-qa.md#p48-what-is-_call_impl-and-why-does-traceml-monkey-patch-around-it-instead-of-just-using-public-hooks), [P49 (hook firing order)](../deep_dive/pytorch-qa.md#p49-whats-the-exact-firing-order-of-forward_pre_hook-forward_hook-backward_pre_hook-backward_hook), [P51 (`torch.cuda.*` API stability)](../deep_dive/pytorch-qa.md#p51-which-torchcuda-apis-does-traceml-rely-on-and-how-stable-are-they-across-pytorch-versions-relevant-to-the-pytorch-coupling-constraint-in-claudemd).
- **The full pedagogical version:** [PR #87 Appendix D](../deep_dive/pr_reviews/pr-87-h2d-timing.md) — Stations 1–3 are written in detail there with checkpoint questions; Stations 4–8 above complete what that document defers.
- **Cross-cutting rules:** [principles.md](principles.md).
