# Aggregator

The aggregator is TraceML's out-of-process telemetry server. The CLI spawns it
before any training rank starts, and it runs for the lifetime of the job as a
sibling of the `torchrun` worker group. Its job is to receive telemetry from
every rank over TCP, maintain a unified rank-aware view of what training is
doing, drive whichever display driver the user asked for (terminal, web
dashboard, or summary-only), and — when the run ends — emit the final end-of-run
summary. All of this is done on a best-effort basis: if the aggregator dies, or
if ingestion falls behind, the training process continues uninterrupted.

## Role in the architecture

The aggregator is one of three cooperating processes during a TraceML run. The
[CLI](cli.md) parses `traceml watch|run|deep`, sets up the `TRACEML_*`
environment contract, and then calls `start_aggregator_process()` in
`src/traceml/cli.py` which `Popen`s
`src/traceml/aggregator/aggregator_main.py` in a new session. Only after the
CLI has polled the TCP port and confirmed the aggregator is listening does it
spawn the training process via `torchrun`. This ordering matters: the ranks
assume the aggregator is already accepting connections when they boot their
local [transport](transport.md) clients. If the aggregator fails to come up,
the CLI fails fast before any training work starts.

Once running, the aggregator owns a `TCPServer` bound to `TRACEML_TCP_HOST` /
`TRACEML_TCP_PORT` (default `127.0.0.1:29765`). Each rank opens a client to this
endpoint and begins shipping length-prefixed msgpack frames as its samplers
produce rows. The aggregator's main loop waits on the TCP server for new data,
drains batches into two sinks (a transitional in-memory `RemoteDBStore` for the
remaining legacy renderers, and an asynchronous SQLite writer that acts as the
full history sink), then ticks the display driver on a fixed cadence. The
[display driver](display-drivers.md) owns UI layout and renderer orchestration;
the aggregator itself does not know about panels, sections, or widgets.

Fail-open is the defining behavior. Every ingestion path, every UI tick, and
every shutdown step is wrapped in a `_safe()` helper that catches exceptions,
logs them through the configured error logger, and returns. Training is never
waiting on the aggregator — the TCP server uses non-blocking sends on the rank
side, so even if the aggregator stops draining, the worst case is that the
rank's send queue fills and newer telemetry is dropped. If the aggregator
crashes outright, the CLI logs the exit in the manifest, the terminal UI is
torn down, and training keeps running to completion. The final summary is
best-effort as well: it only runs when history persistence is enabled and a
valid database path was configured.

## Data in / data out

**In — telemetry frames from ranks.** Each message arriving on the TCP socket
is either a single dict payload or a batch (`list[dict]`). Both shapes carry a
`sampler` name and a `rank` field. The aggregator feeds every decodable message
into `SQLiteWriterSimple.ingest()`, which serializes to `raw_messages` and, for
a projected subset of samplers (system, process, step time, step memory,
stdout/stderr), expands into structured tables for faster query patterns. A
small whitelist of layer-level samplers
(`LayerMemorySampler`, `LayerForwardMemorySampler`,
`LayerBackwardMemorySampler`, `LayerForwardTimeSampler`,
`LayerBackwardTimeSampler`) is additionally mirrored into the in-memory
`RemoteDBStore` because a few deep-profile renderers still read from that path;
SQLite-backed reads are intended to replace this over time.

**Out — display frames to renderers.** The aggregator does not render anything
itself. It exposes the `RemoteDBStore` to the configured display driver at
construction time, and invokes `display_driver.tick()` on every interval. The
driver pulls read-only views of the store (or queries the SQLite history) and
composes the UI. The [renderers](renderers.md) layer is where data is turned
into Rich tables, Plotly charts, or whatever the driver needs.

**Out — summary artifacts.** At shutdown, and optionally on-demand via the
`FinalSummaryService`, the aggregator produces two canonical artifacts under
the session root: `final_summary.json` (structured payload) and
`final_summary.txt` (the bordered text card that also prints to stdout). Legacy
`<db_path>_summary_card.{json,txt}` files are written alongside the DB for
backward compatibility. The CLI-level manifest (start, running, complete
states) is owned by the CLI, not by the aggregator; the aggregator only writes
its own `aggregator_error.log` under the session directory when it fails.

## Key classes / modules

- **`TraceMLAggregator`** (`src/traceml/aggregator/trace_aggregator.py`) —
  The top-level owner. Constructs the TCP server, `RemoteDBStore`,
  `SQLiteWriterSimple`, `FinalSummaryService`, and display driver, then runs
  the drain + tick loop on a daemon thread.
- **`aggregator_main.main()`**
  (`src/traceml/aggregator/aggregator_main.py`) — Process entrypoint. Reads
  `TRACEML_*` env vars, builds `TraceMLSettings`, installs SIGINT/SIGTERM
  handlers, starts the aggregator, and waits on the shutdown event.
- **`SQLiteWriterSimple`** (`src/traceml/aggregator/sqlite_writer.py`) —
  Asynchronous single-writer SQLite persistence layer with bounded queue,
  periodic flushing, WAL journaling, and a `flush_now()` barrier used by the
  on-demand summary path. Enabled when `TRACEML_HISTORY_ENABLED=1`.
- **`FinalSummaryService`** (`src/traceml/aggregator/summary_service.py`) —
  File-based request/response service that watches the session root for a
  summary request, flushes history, regenerates artifacts, and publishes a
  response JSON. Polled once per aggregator tick.
- **`generate_summary()`** (`src/traceml/aggregator/final_summary.py`) —
  Orchestrates the per-domain summary cards (system, process, step time, step
  memory) into one structured payload plus a single printed text card. Writes
  both legacy and canonical artifacts.
- **Summary card generators** (`src/traceml/aggregator/summaries/`) — Per-
  domain builders (`system.py`, `process.py`, `step_time.py`,
  `step_memory.py`) that read from SQLite and produce structured payloads and
  text cards for the final summary.
- **SQLite projection writers** (`src/traceml/aggregator/sqlite_writers/`) —
  Per-sampler modules that define `accepts_sampler()`, `init_schema()`,
  `build_rows()`, and `insert_rows()` hooks the main writer calls when
  projecting selected samplers into structured tables.

## Entry points

The aggregator is always spawned as its own Python process. The CLI computes
the path to `aggregator/aggregator_main.py` relative to the installed package
and runs it with `subprocess.Popen(cmd, env=env, start_new_session=True)`. The
new session matters for signal handling — the aggregator installs its own
SIGINT/SIGTERM handlers and must not receive the CLI's Ctrl-C cascade through
a shared process group.

Inside `main()`, the flow is:

1. `setup_error_logger(is_aggregator=True)` — configures the aggregator-side
   error logger that writes to the session directory.
2. `read_traceml_env()` — parses the `TRACEML_*` environment contract into a
   plain dict. Accepts `TRACEML_UI_MODE` preferentially, falling back to
   `TRACEML_MODE` for backward compatibility. Supported modes are `cli`,
   `dashboard`, and `summary`.
3. Session directory setup — resolves `<logs_dir>/<session_id>/aggregator` and
   prepares a `telemetry` DB path.
4. `TraceMLSettings` is constructed and handed to `TraceMLAggregator`.
5. `agg.start()` starts the TCP server, SQLite writer, display driver, and
   loop thread in that order. Startup failures propagate so the CLI can fail
   fast rather than leave a half-initialized aggregator running.
6. The main thread blocks on `stop_event.wait()`. SIGINT/SIGTERM set the
   event; the `finally` block then stops the aggregator with a 5-second
   timeout and — if history was enabled — calls `generate_summary(...,
   print_to_stdout=True)` so the final card lands on the user's terminal.

On fatal errors, a best-effort `aggregator_error.log` is appended under the
session root, a short message is printed to stderr so the user sees something
even if the UI already tore down, and the process exits with code 1. On clean
shutdown, it prints `[TraceML] Aggregator stopped.` and exits 0.

## Design notes

**Fail-open toward training.** The aggregator is a sibling process, not a
parent. Training ranks connect to it over TCP but never block waiting for it.
Inside the aggregator, ingestion and UI paths are guarded by `_safe()`, and
both the `RemoteDBStore` sink and the SQLite sink are invoked independently
per message so a failure in one does not prevent the other from receiving the
payload. The SQLite writer's queue is bounded; when it fills, newer messages
are dropped (telemetry-first policy). This is the right tradeoff: TraceML is
trying to find bottlenecks in real training, and a telemetry gap is always
preferable to a stalled training step.

**TCP server responsibilities are narrow.** The aggregator's TCP server accepts
rank connections, buffers incoming frames, and exposes `poll()` and
`wait_for_data()`. The main loop blocks on `wait_for_data(timeout=interval)`
rather than sleeping a fixed interval, so the aggregator drains new messages
as soon as they arrive and end-to-end latency is near-zero. The display tick
is still rate-limited to at most one per `render_interval_sec` so the UI
cadence stays smooth. See [transport](transport.md) for the wire format.

**Rank-awareness is a first-class concern.** In DDP jobs, every rank is
producing its own telemetry stream. The `RemoteDBStore` is a rank-aware
wrapper around [Database](database.md): it lazily creates a
`Database(rank, sampler_name)` on first ingestion from that `(rank, sampler)`
pair. SQLite storage carries the `rank` column explicitly. This separation
matters because aggregating across ranks without knowing which rank produced
which row would hide exactly the kind of stragglers and load imbalance users
want to find. Renderers and summary generators get to decide whether to show
per-rank detail, a specific rank (rank 0 by default for single-GPU-like
views), or an aggregate.

**Summary emission timing.** There are two summary paths. The normal path runs
inside `TraceMLAggregator.stop()`: after the loop thread has joined, the
display driver has been torn down, and the SQLite writer has been stopped,
`generate_summary()` reads from the final DB state and prints the bordered
card. The on-demand path is the `FinalSummaryService`: it watches for a
request file under the session root, blocks on `flush_now()` to force any
queued telemetry into SQLite, then writes fresh summary artifacts and a
response file the requesting process can read. The on-demand path lets
integrations and wrapper scripts request a summary without tearing down the
aggregator — it is intentionally low-frequency and should not be called per
step.

!!! note "Transitional RemoteDBStore"
    The `RemoteDBStore` is explicitly transitional. Most renderers already
    read from SQLite-backed history; a small subset of layer-level deep-profile
    views still depend on the in-memory path. The long-term plan is to retire
    `RemoteDBStore` entirely once those renderers migrate. Today, the aggregator
    filters messages by sampler name (`_REMOTE_STORE_SAMPLERS`) so only the
    layer-level payloads are mirrored into the in-memory store — everything
    else goes straight to SQLite.

## Cross-references

- [Display drivers](display-drivers.md) — CLI vs. NiceGUI vs. summary-only
  output mediums the aggregator drives.
- [Transport](transport.md) — TCP wire format and rank detection.
- [Database](database.md) — `Database` and `RemoteDBStore` contracts the
  aggregator reads and writes.
- [Renderers](renderers.md) — read-only views the display drivers compose.
- [CLI](cli.md) — how the aggregator process is spawned and supervised.
- [Architecture overview](../architecture.md) — the three-process model and
  telemetry flow diagram.
