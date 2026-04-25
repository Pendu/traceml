# How to add a new SQLite projection writer

This guide teaches you how to bridge a sampler's wire payload to the query-friendly SQLite tables that renderers and diagnostics read from. It assumes you have already worked through `add_sampler.md` (upstream of this guide) and skimmed `add_renderer.md` §7 (downstream consumer of this guide), and that `pip install -e ".[dev,torch]"` works in your checkout.

---
Feature type: SQLite projection writer
Risk level: medium
Cross-cutting impact: one subsystem (aggregator persistence), but read by every windowed renderer downstream
PyTorch coupling: none
---

## 1. Intro and mental model

### What is "an SQLite projection writer" in TraceML?

A **projection writer** is a small module under `src/traceml/aggregator/sqlite_writers/` that takes one decoded sampler wire payload and turns it into rows in one or more **query-friendly** SQLite tables. Three files compose the contract:

- `sqlite_writers/<sampler>.py` — your projection writer (this guide).
- `aggregator/sqlite_writer.py` — the core writer thread that owns the SQLite connection and dispatches each payload through the registered projection writers (`_PROJECTION_WRITERS` list at the top of the file).
- `aggregator/trace_aggregator.py::TraceMLAggregator._drain_tcp` — the ingest path that calls `self._sqlite_writer.ingest(msg)` for every message arriving over TCP.

### Where projection writers sit in the pipeline

```
training rank: sampler.sample()
  -> Database.add_record(payload)
  -> DBIncrementalSender.collect_payload
  -> TCPClient.send_batch
                                 │
                                 ▼  TCP
aggregator: TCPServer.poll()
  -> TraceMLAggregator._drain_tcp
       ├── RemoteDBStore.ingest      (legacy live store, see §11 allow-list)
       └── SQLiteWriterSimple.ingest (queue → writer thread)
                                       │
                                       ▼
            SQLiteWriterSimple._flush_once
              -> raw_messages           (payload_mp BLOB, always)
              -> _PROJECTION_WRITERS:
                   for writer in writers:
                     if writer.accepts_sampler(sampler):
                         writer.build_rows(payload_dict, recv_ts_ns)
                         writer.insert_rows(conn, ...)
                                       │
                                       ▼
                renderer (next tick): sqlite3.connect(db_path).execute(...)
```

Cross-link: full ingest path is [W9](../deep_dive/code-walkthroughs.md#w9-aggregator-core-tcp-receive-frame-dispatch-sqlite-writes); the end-to-end pipeline is `pipeline_walkthrough.md`. Don't re-derive it here.

### Why projection writers exist (vs. just storing the payload as JSON)

Every payload is already preserved verbatim as MessagePack in `raw_messages` (see `sqlite_writer.py::_init_schema`). So why add a projection at all?

1. **Renderer query shape.** Renderers want `SELECT ... FROM step_time_samples WHERE rank = ? ORDER BY step DESC LIMIT 100`, not "decode every blob in `raw_messages` and filter in Python." `add_renderer.md` §7 explicitly mandates SQLite reads for any windowed view. This guide is how that SQLite gets populated.
2. **Cross-rank joins on `step`.** The renderer side joins per-rank timing on step number to detect stragglers. That requires `step` to be a first-class indexed column.
3. **Storage compactness.** Denormalized columns cost less than JSON blobs once the renderer needs to read tens of thousands of rows.
4. **Backward compat with v0.2.3 SQLite databases.** Users on PyPI may have on-disk telemetry from prior versions; the schema is part of the wire-format-stability contract (see `principles.md`).

### The writer's contract (do not break)

A projection writer is:

- **Stateless.** It owns no Python state across calls. `init_schema`, `accepts_sampler`, `build_rows`, `insert_rows` are pure module-level functions. (Look at any of the five existing writers — none of them declare a class.)
- **Single-payload at a time** for `build_rows`. The dispatcher batches the *output* across many payloads, but each `build_rows` call sees exactly one decoded payload dict.
- **Idempotent at the row level only in the trivial sense.** Re-receiving the same wire payload **will** insert duplicate rows. Deduplication is the upstream sender's job (`DBIncrementalSender` tracks `_last_sent_seq`); the writer trusts what arrives. Don't add `INSERT OR REPLACE` cleverness — it hides upstream bugs.
- **Single-threaded.** The dispatcher runs your writer on the `TraceML-SQLiteWriter` thread (see `sqlite_writer.py:128`). Renderers read on different threads via short-lived connections. WAL mode lets this coexist; see §7.
- **Fail-open.** A writer exception is caught one frame up in `SQLiteWriterSimple._collect_flush_rows` (`continue` on `Exception`). Your payload is dropped, the next payload still gets through. **Do not raise from `build_rows` or `insert_rows`** — log and return empty. See `principles.md` for the cross-cutting rule.

### WAL mode and the single-writer thread

The SQLite connection is opened in `SQLiteWriterSimple._connect` with:

```python
conn = sqlite3.connect(self._cfg.path, isolation_level=None,
                       check_same_thread=False)
conn.execute("PRAGMA journal_mode=WAL;")
conn.execute(f"PRAGMA synchronous={self._cfg.synchronous};")  # NORMAL
conn.execute("PRAGMA cache_size=-2000;")
conn.execute("PRAGMA foreign_keys=ON;")
```

WAL means:

- **One writer at a time** — the aggregator's writer thread.
- **Many readers in parallel** — every renderer opens its own short-lived connection per `_compute()` call (see `add_renderer.md` §7 "Thread safety"). Readers do not block the writer.
- **No `with conn:` blocks in the writer** — `_write_flush_rows` issues `BEGIN;` / `COMMIT;` explicitly because `isolation_level=None` is autocommit.

Cross-link: [W9](../deep_dive/code-walkthroughs.md#w9-aggregator-core-tcp-receive-frame-dispatch-sqlite-writes) covers the WAL setup in detail.

---

## 2. Before you start: decisions to make

Answer these before opening an editor. Write the answers in the PR description.

- [ ] **Sampler exists?** Is the upstream sampler already shipping rows to the aggregator? If not, open `add_sampler.md` first; this guide assumes payloads already arrive over TCP.
- [ ] **Wire schema fixed?** Read the sampler's `schema/<name>.py` and know exactly which keys you'll consume. Schema changes after the writer ships are a wire-compat event (see §6).
- [ ] **One table or many?** Flat envelopes → one table (`process.py`, `step_memory.py`). Envelope + child list → parent + child table linked by FK or by shared `(rank, sample_ts_s, seq)` (`system.py`). Envelope + dynamic event map → JSON blob column (`step_time.py::events_json`). Decision rule in §2 below.
- [ ] **Primary key.** Always `INTEGER PRIMARY KEY AUTOINCREMENT`. The implicit `id` is the stable sort key for renderers (`ORDER BY id DESC`).
- [ ] **Indexes.** Walk the renderer's expected `WHERE` and `ORDER BY` clauses. The default composite for time-series tables is `(rank, <step or gpu_idx>, sample_ts_s, id)`; see every existing writer for the pattern.
- [ ] **JSON blob vs first-class columns.** First-class columns win for anything the renderer filters or aggregates. Use a JSON blob only when the field set is genuinely unbounded (`step_time.py` events, where event names like `forward`, `backward`, `optimizer`, `dataloader`, `h2d` are not pinned by the schema).
- [ ] **`recv_ts_ns` vs `sample_ts_s`.** Both go in. `recv_ts_ns` is the aggregator's `time.time_ns()` at ingest (`sqlite_writer.py:344`); `sample_ts_s` is the sampler's `time.time()` at row creation. The first is for ingest-lag detection; the second is the timeline the renderer plots.
- [ ] **Backward compat.** Are there v0.2.3 users with existing SQLite DBs? Default answer: yes. Plan an additive-only schema (see §6).

---

## 3. Anatomy of an existing projection writer

We walk through `system.py` end-to-end. It's the cleanest exemplar because it's the only writer that emits **two** linked tables and shows the full set of patterns: envelope decoding, child-row expansion, defensive `.get(...)` everywhere, type coercion, executemany. Read it once (~410 lines) before continuing.

File: `src/traceml/aggregator/sqlite_writers/system.py`.

### 3.1. Module-level constants and the sampler-name guard

```python
SAMPLER_NAME = "SystemSampler"


def accepts_sampler(sampler: Optional[str]) -> bool:
    """Return True if this projection writer handles the given sampler."""
    return sampler == SAMPLER_NAME
```

Three observations:

- **`SAMPLER_NAME` matches the sampler's `sampler_name` argument** to `BaseSampler.__init__`. Get this wrong and the dispatcher will never call your writer; payloads route via the envelope `"sampler"` field (`sqlite_writer.py::_extract_rank_sampler`). Special case to remember: `StdoutStderrSampler` ships with `SAMPLER_NAME = "Stdout/Stderr"` — the wire string, not the class name.
- **`accepts_sampler` is the dispatcher hook.** It's called once per payload in `_collect_flush_rows` (`sqlite_writer.py:350`). Keep it cheap — it's on the hot path.
- **Module-level functions, not a class.** All five existing writers follow this shape. Don't introduce a class just to "look more OO" — there's no state to carry.

### 3.2. `init_schema(conn)` — DDL on first run

```python
def init_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS system_samples (
            id                     INTEGER PRIMARY KEY AUTOINCREMENT,
            recv_ts_ns             INTEGER NOT NULL,
            rank                   INTEGER,
            sample_ts_s            REAL,
            seq                    INTEGER,
            cpu_percent            REAL,
            ram_used_bytes         REAL,
            ...
            gpu_power_peak_w       REAL
        );
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_system_samples_rank_ts
        ON system_samples(rank, sample_ts_s, id);
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS system_gpu_samples (...);
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_system_gpu_samples_rank_gpu_ts
        ON system_gpu_samples(rank, gpu_idx, sample_ts_s, id);
    """)
```

What's happening:

- **`CREATE TABLE IF NOT EXISTS`** — idempotent. Running the aggregator against a DB that already has the table is a no-op.
- **`CREATE INDEX IF NOT EXISTS`** — same. The index name (`idx_<table>_<cols>`) is the convention.
- **No `DROP`, no `ALTER` at this stage.** Schema migration is in §6.
- **Called from `SQLiteWriterSimple._run` once per process startup** (`sqlite_writer.py:454`). If your `init_schema` raises, the entire writer thread aborts and no history is persisted for the run. Do not raise.

### 3.3. `build_rows(payload_dict, recv_ts_ns)` — the reduction

```python
def build_rows(payload_dict, recv_ts_ns):
    out = {"system_samples": [], "system_gpu_samples": []}

    sampler = payload_dict.get("sampler")
    if not accepts_sampler(str(sampler) if sampler is not None else None):
        return out

    rank_raw = payload_dict.get("rank")
    try:
        rank = int(rank_raw) if rank_raw is not None else None
    except Exception:
        rank = None

    tables = payload_dict.get("tables")
    if not isinstance(tables, dict):
        return out

    for rows in tables.values():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            # ... extract per-row fields with .get() ...
            # ... coerce types defensively ...
            # ... append parent and child rows ...

    return out
```

Patterns to internalize:

1. **Always start with the empty `out` dict shaped like the SQL tables** you write to. That contract is what `_collect_flush_rows` consumes (`sqlite_writer.py:357`).
2. **Re-check `accepts_sampler`.** The dispatcher already checks, but the guard keeps the writer robust to direct calls in tests.
3. **Defensive coercion at every field.** `int(x) if isinstance(x, int) else None` is the house pattern. A malformed payload should produce `NULL` columns, not raise. The wire format is "trust but verify" — clients on v0.2.3 may emit slightly different shapes than you expect.
4. **Walk `tables.values()`, not `tables[<table_name>]`.** The envelope carries one or more table names; the projection writer doesn't care what they're called. All five existing writers iterate values.
5. **Tuples, not dicts, at the end.** The output rows are positional tuples in the exact column order of the eventual `INSERT`. Type mismatches between `build_rows` and `insert_rows` are silent — §9 has the war story.

### 3.4. The two-table FK-implicit pattern

`system.py` emits both `system_samples` (one parent row per snapshot) and `system_gpu_samples` (one child row per GPU per snapshot). They share `(rank, sample_ts_s, seq)`. Note what's **not** there: a foreign key column. The link is implicit via the shared key triple. Renderers that need to join do:

```sql
SELECT p.rank, p.cpu_percent, g.gpu_idx, g.util
FROM system_samples p
JOIN system_gpu_samples g
  ON p.rank = g.rank AND p.sample_ts_s = g.sample_ts_s AND p.seq = g.seq;
```

This works because both rows are inserted in the same flush transaction (see `_write_flush_rows`, `BEGIN;` / `COMMIT;` around all writes). They're durable together or not at all.

If your sampler emits a child list where rows benefit from an explicit FK to the parent's `id`, you need a different pattern: insert parent, read `cursor.lastrowid`, insert children with that FK. None of the five existing writers does this — they all use the implicit-key pattern. Discuss before introducing the FK pattern; it forces row-by-row inserts instead of `executemany`.

### 3.5. `insert_rows(conn, rows_by_table)` — the SQL boundary

```python
def insert_rows(conn, rows_by_table):
    system_rows = rows_by_table.get("system_samples", [])
    if system_rows:
        conn.executemany(
            "INSERT INTO system_samples(recv_ts_ns, rank, sample_ts_s, "
            "seq, cpu_percent, ram_used_bytes, ram_total_bytes, ...) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
            system_rows,
        )

    gpu_rows = rows_by_table.get("system_gpu_samples", [])
    if gpu_rows:
        conn.executemany(...)
```

What to copy:

- **Guard each table with `if rows:`** — `executemany([])` is technically fine but the check makes the empty-payload path obvious.
- **`executemany`, not a Python loop of `execute`.** Batched binds are ~10× faster on flushes >100 rows.
- **Explicit column list in the `INSERT`.** Never `INSERT INTO foo VALUES (...)` without a column list — when you add a column in a later release (see §6), the positional INSERT silently misaligns and corrupts the table. Every existing writer names columns explicitly.
- **Parameter count must match the column list.** `system_samples` has 17 placeholders for 17 columns. The `id` column is omitted because it's `INTEGER PRIMARY KEY AUTOINCREMENT` — SQLite assigns it.
- **No transaction control here.** `BEGIN` / `COMMIT` happen one frame up, in `_write_flush_rows` (`sqlite_writer.py:382`). All projection writers' `insert_rows` calls run inside one transaction per flush.

### 3.6. What `sqlite_writers/__init__.py` does

Nothing. It's a one-line file. Writers are *not* auto-discovered; the `_PROJECTION_WRITERS` list at the top of `sqlite_writer.py` is the registration mechanism (§4 Step 4).

---

## 4. Step-by-step: adding a new projection writer

We'll build a writer for the hypothetical `GpuUtilizationSampler` from `add_sampler.md` §4. The sampler emits one row per tick on rank 0 with this wire payload:

```python
{
    "rank": 0,
    "sampler": "GpuUtilizationSampler",
    "timestamp": 1745568000.123,
    "tables": {
        "GpuUtilizationTable": [
            {
                "seq": 42, "ts": 1745568000.123, "gpu_count": 2,
                "gpus": [
                    {"gpu_idx": 0, "util_percent": 87.0, "mem_used_bytes": 12_000_000_000.0},
                    {"gpu_idx": 1, "util_percent": 91.0, "mem_used_bytes": 11_500_000_000.0},
                ],
            },
        ]
    }
}
```

We need a parent table (one row per envelope) and a child table (one row per GPU per envelope), both indexed for the renderer's `WHERE rank = ? AND gpu_idx = ? ORDER BY id DESC LIMIT N` query (see `add_renderer.md` §4 step 2 for the matching renderer schema).

### Step 1 — Create the writer file

Path: `src/traceml/aggregator/sqlite_writers/gpu_utilization.py`. Follow the same shape as `system.py`:

- Module docstring stating sampler, tables, storage units, expected wire-payload shape.
- `SAMPLER_NAME = "GpuUtilizationSampler"`.
- `accepts_sampler(sampler)` returns `sampler == SAMPLER_NAME`.
- `init_schema(conn)` creates `gpu_util_samples` parent table + `gpu_util_per_gpu` child table + composite indexes `(rank, sample_ts_s, id)` and `(rank, gpu_idx, sample_ts_s, id)`.
- `build_rows(payload_dict, recv_ts_ns)` returns `{"gpu_util_samples": [...], "gpu_util_per_gpu": [...]}` with defensive `.get()` and `isinstance` coercion. Skip child rows missing `gpu_idx`.
- `insert_rows(conn, rows_by_table)` issues two `executemany` calls, one per table, with explicit column lists.

### Step 2 — Pick parent/child columns to match the renderer's queries

The renderer in `add_renderer.md` §4 does `ORDER BY id DESC LIMIT window_size * 64` and groups by `(rank, gpu_idx)`. Therefore:

- The `id` column is required for stable descending sort across ranks.
- `(rank, gpu_idx, sample_ts_s, id)` is the index the renderer needs.
- `seq` is included so the renderer can spot dropped samples on the envelope side.
- `recv_ts_ns` is included for ingest-lag forensics — not the renderer's primary timeline but invaluable when "the dashboard is stuck."

Run `EXPLAIN QUERY PLAN` against the renderer's actual SQL once the writer is wired up; if the plan says `SCAN`, your index is wrong.

### Step 3 — Register the writer in `sqlite_writer.py`

Open `src/traceml/aggregator/sqlite_writer.py` and add the import + the list entry. The file currently looks like this around line 41:

```python
from traceml.aggregator.sqlite_writers import process as process_sql_writer
from traceml.aggregator.sqlite_writers import (
    stdout_stderr as stdout_stderr_sql_writer,
)
from traceml.aggregator.sqlite_writers import (
    step_memory as step_memory_sql_writer,
)
from traceml.aggregator.sqlite_writers import step_time as step_time_sql_writer
from traceml.aggregator.sqlite_writers import system as system_sql_writer

_PROJECTION_WRITERS = [
    system_sql_writer,
    process_sql_writer,
    step_time_sql_writer,
    step_memory_sql_writer,
    stdout_stderr_sql_writer,
]
```

Add:

```diff
+from traceml.aggregator.sqlite_writers import (
+    gpu_utilization as gpu_utilization_sql_writer,
+)
 from traceml.aggregator.sqlite_writers import process as process_sql_writer
 ...

 _PROJECTION_WRITERS = [
     system_sql_writer,
     process_sql_writer,
     step_time_sql_writer,
     step_memory_sql_writer,
     stdout_stderr_sql_writer,
+    gpu_utilization_sql_writer,
 ]
```

That single list is the entire registration. There is no decorator, no plugin manifest, no entry point. If you don't add it, **payloads route to `raw_messages` only and your tables never get rows** — the failure is silent because the dispatcher just skips you.

### Step 4 — Verify ingest end-to-end

The aggregator path that calls your writer:

1. Worker rank emits the envelope via `DBIncrementalSender.collect_payload`.
2. `TCPClient` ships it.
3. `TCPServer.poll()` returns it inside `TraceMLAggregator._drain_tcp` (`trace_aggregator.py:248`).
4. `self._sqlite_writer.ingest(msg)` enqueues it (`trace_aggregator.py:258`).
5. `SQLiteWriterSimple._run` wakes every `flush_interval_sec=0.5` and calls `_flush_once`.
6. `_collect_flush_rows` decodes the payload and asks every writer in `_PROJECTION_WRITERS`:

   ```python
   for writer in _PROJECTION_WRITERS:
       if not writer.accepts_sampler(sampler):
           continue
       rows_by_table = writer.build_rows(payload_dict=payload_dict,
                                         recv_ts_ns=recv_ts_ns)
       ...
   ```

7. `_write_flush_rows` opens a transaction, inserts into `raw_messages`, then iterates `_PROJECTION_WRITERS` again and calls `insert_rows(conn, projection_rows[writer])`.

To smoke-verify, run a real session and inspect the DB:

```bash
traceml watch examples/mnist.py --mode summary --history-enabled
sqlite3 logs/<session>/aggregator/telemetry  ".tables"
sqlite3 logs/<session>/aggregator/telemetry \
  "SELECT COUNT(*) FROM gpu_util_samples; \
   SELECT COUNT(*) FROM gpu_util_per_gpu;"
```

Both counts should be > 0 within seconds of training start. If `gpu_util_samples` is empty but `raw_messages` has rows where `sampler = 'GpuUtilizationSampler'`, your writer isn't registered or `accepts_sampler` doesn't match the sampler's `sampler_name`.

### Step 5 — Add a downstream renderer (optional, but normal)

If you're adding a writer, you almost always have a renderer in mind. See `add_renderer.md` §4 — the `GpuUtilizationRenderer` walkthrough there reads from exactly the schema you just defined.

---

## 5. Common patterns and exemplars

| Wire-payload shape | Writer pattern | Copy from |
|---|---|---|
| Flat envelope, one row per sample | Single table | `process.py`, `step_memory.py` |
| Envelope + child list (per-GPU, per-device, ...) | Parent table + child table; share `(rank, sample_ts_s, seq)` for the implicit join | `system.py` |
| Envelope + dynamic event map (event names not pinned) | Single table with a `TEXT` JSON blob column for the dynamic part; first-class columns for stable fields | `step_time.py::events_json` |
| Append-only log lines | Narrow table: `(id, recv_ts_ns, rank, sample_ts_s, line)` | `stdout_stderr.py` |
| Sampler whose envelope timestamp *is* the row timestamp | Pull `payload_dict.get("timestamp")`, not `row.get("ts")` | `stdout_stderr.py:124` |

### Naming-convention reference

| Element | Convention | Example |
|---|---|---|
| Module file | `<sampler_short_name>.py` | `system.py` |
| `SAMPLER_NAME` constant | matches `BaseSampler.__init__(sampler_name=...)` | `"SystemSampler"` |
| Parent table | `<domain>_samples` | `system_samples`, `process_samples` |
| Child table | `<domain>_<child>_samples` or `<domain>_<child>` | `system_gpu_samples`, `gpu_util_per_gpu` |
| Index | `idx_<table>_<cols joined by _>` | `idx_system_gpu_samples_rank_gpu_ts` |
| Bookkeeping columns (always include) | `id`, `recv_ts_ns`, `rank`, `sample_ts_s`, `seq` | every existing writer |

---

## 6. Schema rules

Cross-link: `principles.md` for the cross-cutting wire-compat rule. This section is the SQLite-specific application of that rule.

### 6.1. Table and column naming

- snake_case, lowercase. `process_samples` not `ProcessSamples`.
- Column names match wire-payload keys when feasible. The wire ships `cpu` and `ram_used`; the projection stores them as `cpu_percent` and `ram_used_bytes` because units belong in the column name once you're past the wire (see `system.py` `# Storage units` docstring at the top). Keep the unit suffix consistent: `_bytes`, `_percent`, `_w`, `_c`, `_ms`.
- Primary key is always `id INTEGER PRIMARY KEY AUTOINCREMENT`.
- Required identity columns on every projection table: `recv_ts_ns INTEGER NOT NULL`, `rank INTEGER`, `sample_ts_s REAL`, `seq INTEGER`. These are what downstream renderers and diagnostics rely on. Don't drop them "to save space"; the cost is negligible and every renderer in `renderers/` uses one or more of them.

### 6.2. Type rules

SQLite is dynamically typed but be explicit:

- `INTEGER` for ranks, counts, `seq`, `step`, `gpu_idx`, ns timestamps.
- `REAL` for any float (percent, bytes when fractional, seconds, ms). Note: existing writers store byte counts as `REAL` even though they could be `INTEGER` — that's the precedent. Stay consistent.
- `TEXT` for `device`, log lines, JSON blobs.
- `BLOB` only for raw MessagePack (used in `raw_messages`, not in any projection).

`NOT NULL` only on columns you know will *always* be present (`id`, `recv_ts_ns`, `line` in stdout). Everything else is nullable so malformed payloads degrade to NULL columns rather than dropped rows.

### 6.3. Backward compatibility — additive only

Users on v0.2.3 (and any future released minor version) have on-disk SQLite databases. A schema change must not break those.

**Allowed:**

- `CREATE TABLE IF NOT EXISTS <new_table>` — add a whole new table.
- `ALTER TABLE <existing> ADD COLUMN <new_col> <TYPE> DEFAULT NULL` — add a column. Renderers reading the new column must use defensive access. See `principles.md` for the renderer-side rule.
- New indexes via `CREATE INDEX IF NOT EXISTS`.

**Not allowed (without a new table name + deprecation cycle):**

- `DROP COLUMN`. Even SQLite's recent `ALTER TABLE ... DROP COLUMN` is a wire-format-break event. Don't.
- Renaming a column. Same disaster — old renderers break.
- Changing a column's type. `REAL → INTEGER` will lose precision. Add a new column with the new type instead.

### 6.4. Migration triggers and where they live

The current writers do **not** run migrations. `init_schema` calls only `CREATE TABLE IF NOT EXISTS`. If you add a column in a later release, that DDL must run defensively at startup. The pattern (not yet present in the codebase — see Gaps):

```python
def _ensure_column(conn, table, column, ddl_type):
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        try:
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} "
                f"{ddl_type} DEFAULT NULL;"
            )
        except sqlite3.OperationalError:
            # Race: another aggregator instance won. Tolerable.
            pass


def init_schema(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS ... (existing cols);")
    _ensure_column(conn, "gpu_util_samples", "new_field", "REAL")
```

Until at least one writer ships this pattern, copy it from this guide when you need it and link the PR back here.

### 6.5. Schema versioning

The codebase has no `schema_version` table or convention as of 2026-04-25 (verified via `grep -ri schema_version src/traceml/`). The implicit version is "whatever PyPI release wrote this DB." That's fine while we're additive-only. If the day comes that we need a non-additive change, introducing a `schema_meta(key TEXT PRIMARY KEY, value TEXT)` table with a `version` row is the obvious step. Flag this in the gaps section if you find yourself reaching for it.

---

## 7. Overhead and concurrency

Cross-link: `principles.md` for the cross-cutting overhead budget.

### 7.1. Per-payload writer cost

Every wire payload routes through every writer's `accepts_sampler` and through the matching writer's `build_rows`. The writer is on the flush thread, so it doesn't block training, but it does compete with ingest throughput.

Targets:

- `accepts_sampler`: a string equality. Effectively free.
- `build_rows`: target **<100 µs** for a single payload of ~10 child rows. The existing writers are all comfortably under this.
- `insert_rows`: dominated by `executemany` SQLite cost; target **<2 ms** per flush across all writers combined. Scale with batch size, not per-writer.

If your writer takes 5+ ms per payload, the queue (`max_queue=50_000`, `sqlite_writer.py:97`) will fill on a busy run and messages drop. Profile with `cProfile` over a 30-second smoke run before merging.

### 7.2. Concurrency model

- **One writer thread.** `TraceML-SQLiteWriter`, started in `SQLiteWriterSimple._thread`. All your writer code runs there.
- **Many reader threads.** Renderers open short-lived `sqlite3.connect(db_path)` per `_compute()` call (see `add_renderer.md` §3.2 `_connect()` and §7 thread-safety). WAL mode lets readers proceed in parallel with the single writer.
- **Connection lifetime.** The writer's connection is created once in `_run` and held for the lifetime of the aggregator process. Readers must not share that connection — they create their own.

### 7.3. No `SELECT` in the writer

A projection writer's job is to write. If you find yourself reaching for `conn.execute("SELECT ...")` inside `build_rows`, stop and reconsider — every read inside the writer thread is a read that didn't happen on a renderer thread, plus it makes the writer transactionally weirder.

The exception is schema introspection (`PRAGMA table_info(...)`) for the migration pattern in §6.4. That's fine because it runs once at init, not per payload.

### 7.4. Idempotency / dedup

Don't add `INSERT OR REPLACE` or `INSERT ... ON CONFLICT`. The upstream sender (`DBIncrementalSender`) tracks `_last_sent_seq` and guarantees no row is sent twice under normal operation (see [W7](../deep_dive/code-walkthroughs.md#w7-database-sender-bounded-in-memory-store-and-incremental-tcp-shipping)). Adding "smart" dedup at the writer hides upstream bugs and makes the dispatcher's behavior harder to reason about.

If duplicate rows show up in the DB, the bug is upstream, not in your writer.

### 7.5. Transaction batching

You don't control transactions — `_write_flush_rows` wraps every flush in one `BEGIN; ... COMMIT;`. That means up to `max_flush_items=20_000` payloads land in one transaction, which is the fast path. Don't try to add nested transactions or savepoints; you'll deadlock with the outer transaction.

---

## 8. Testing

There are no projection-writer tests in `tests/` as of 2026-04-25 (verified via `ls traceml/tests/`; closest is `test_seq_counter.py` for the upstream sender). Adding the first one is a net improvement — don't skip because there's no precedent.

### 8.1. Minimal test template

Path: `tests/test_gpu_utilization_sql_writer.py`.

```python
"""
Tests for the GpuUtilizationSampler SQLite projection writer.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from traceml.aggregator.sqlite_writers import (
    gpu_utilization as writer,
)


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    """Open a temp SQLite DB and run init_schema once."""
    db_path = tmp_path / "test.db"
    c = sqlite3.connect(str(db_path), isolation_level=None)
    c.execute("PRAGMA journal_mode=WAL;")
    writer.init_schema(c)
    yield c
    c.close()


def _sample_payload(rank: int = 0, seq: int = 1) -> dict:
    return {
        "rank": rank,
        "sampler": "GpuUtilizationSampler",
        "timestamp": 1745568000.0,
        "tables": {
            "GpuUtilizationTable": [
                {
                    "seq": seq, "ts": 1745568000.0, "gpu_count": 2,
                    "gpus": [
                        {"gpu_idx": 0, "util_percent": 87.0,
                         "mem_used_bytes": 12_000_000_000.0},
                        {"gpu_idx": 1, "util_percent": 91.0,
                         "mem_used_bytes": 11_500_000_000.0},
                    ],
                }
            ]
        },
    }


class TestGpuUtilizationProjection:
    def test_init_schema_idempotent(self, conn):
        # Running init_schema twice must not raise.
        writer.init_schema(conn)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            )
        }
        assert "gpu_util_samples" in tables
        assert "gpu_util_per_gpu" in tables

    def test_accepts_sampler_match(self):
        assert writer.accepts_sampler("GpuUtilizationSampler") is True
        assert writer.accepts_sampler("SystemSampler") is False
        assert writer.accepts_sampler(None) is False

    def test_build_rows_happy_path(self):
        rows = writer.build_rows(_sample_payload(), recv_ts_ns=1234)
        assert len(rows["gpu_util_samples"]) == 1
        assert len(rows["gpu_util_per_gpu"]) == 2
        # Parent: (recv_ts_ns, rank, sample_ts_s, seq, gpu_count)
        assert rows["gpu_util_samples"][0] == (
            1234, 0, 1745568000.0, 1, 2,
        )

    def test_build_rows_wrong_sampler_returns_empty(self):
        bad = _sample_payload()
        bad["sampler"] = "SystemSampler"
        rows = writer.build_rows(bad, recv_ts_ns=0)
        assert rows == {"gpu_util_samples": [], "gpu_util_per_gpu": []}

    def test_build_rows_malformed_payload_does_not_raise(self):
        # tables is not a dict
        assert writer.build_rows(
            {"sampler": "GpuUtilizationSampler", "tables": "garbage"},
            recv_ts_ns=0,
        ) == {"gpu_util_samples": [], "gpu_util_per_gpu": []}

        # gpus is not a list
        bad = _sample_payload()
        bad["tables"]["GpuUtilizationTable"][0]["gpus"] = "garbage"
        rows = writer.build_rows(bad, recv_ts_ns=0)
        # Parent row still inserted; no children.
        assert len(rows["gpu_util_samples"]) == 1
        assert len(rows["gpu_util_per_gpu"]) == 0

    def test_insert_rows_round_trip(self, conn):
        rows = writer.build_rows(_sample_payload(rank=1, seq=42),
                                 recv_ts_ns=999)
        conn.execute("BEGIN;")
        writer.insert_rows(conn, rows)
        conn.execute("COMMIT;")

        parent = conn.execute(
            "SELECT rank, seq, gpu_count FROM gpu_util_samples;"
        ).fetchall()
        assert parent == [(1, 42, 2)]

        children = conn.execute(
            "SELECT gpu_idx, util_percent, mem_used_bytes "
            "FROM gpu_util_per_gpu ORDER BY gpu_idx;"
        ).fetchall()
        assert children == [
            (0, 87.0, 12_000_000_000.0),
            (1, 91.0, 11_500_000_000.0),
        ]

    def test_idempotency_double_insert(self, conn):
        # Re-feeding the same payload doubles the rows. This is the
        # documented behavior — dedup is the sender's job, not ours.
        rows = writer.build_rows(_sample_payload(), recv_ts_ns=0)
        conn.execute("BEGIN;")
        writer.insert_rows(conn, rows)
        writer.insert_rows(conn, rows)
        conn.execute("COMMIT;")
        n_parent = conn.execute(
            "SELECT COUNT(*) FROM gpu_util_samples;"
        ).fetchone()[0]
        assert n_parent == 2  # double-write is intentional
```

### 8.2. Schema-migration test (when you add a column)

When a later release adds a column via the `_ensure_column` pattern in §6.4, the test that catches breakage is the one that opens an old-schema DB, calls `init_schema`, and asserts the new column was added.

### 8.3. Smoke test against a real session

```bash
pip install -e ".[dev,torch]"
traceml run examples/mnist.py --mode summary
sqlite3 logs/<session>/aggregator/telemetry \
    "SELECT name FROM sqlite_master WHERE type='table' \
     ORDER BY name;"
```

The list should include all five existing projection tables plus yours. Then:

```bash
sqlite3 logs/<session>/aggregator/telemetry \
    "SELECT COUNT(*) FROM gpu_util_samples;"
```

> 0 within seconds. If 0, your writer isn't registered or `accepts_sampler` doesn't match. Cross-check with:

```bash
sqlite3 logs/<session>/aggregator/telemetry \
    "SELECT DISTINCT sampler FROM raw_messages;"
```

If `GpuUtilizationSampler` shows up there but your projection table is empty, the dispatcher path is broken — re-check the `_PROJECTION_WRITERS` list edit and the `SAMPLER_NAME` string.

---

## 9. Common pitfalls

Numbered, with symptom and fix.

1. **Symptom:** The writer file imports fine, `init_schema` runs, but no rows ever appear in the projection tables. `raw_messages` does have rows from the sampler.
   **Cause:** Forgot to add the writer to `_PROJECTION_WRITERS` in `sqlite_writer.py:52`. The dispatcher only iterates that list.
   **Fix:** Add the import and the list entry. There is no auto-discovery.

2. **Symptom:** Rows inserted, but every value is in the wrong column (e.g. `seq` shows up as `gpu_count`).
   **Cause:** Tuple parameter count or order in `insert_rows` doesn't match the column list in the `INSERT`. SQLite binds positionally and silently.
   **Fix:** Re-derive the tuple shape from `build_rows` and the `INSERT` columns side-by-side. Add a round-trip test like §8.1 `test_insert_rows_round_trip` — it catches this immediately.

3. **Symptom:** `sqlite3.OperationalError: duplicate column name` on aggregator startup after deploying a new release.
   **Cause:** You added an `ALTER TABLE ... ADD COLUMN` without guarding against re-runs. SQLite's `ALTER TABLE ADD COLUMN` is not idempotent the way `CREATE TABLE IF NOT EXISTS` is.
   **Fix:** Use the `_ensure_column` helper (§6.4). Check `PRAGMA table_info` first; only ALTER if the column is missing.

4. **Symptom:** Renderer is slow on a session with millions of rows. `EXPLAIN QUERY PLAN` shows `SCAN`.
   **Cause:** No index on the column the renderer's `WHERE` / `ORDER BY` references. Works fine in dev with 100 rows, dies on production.
   **Fix:** Add a composite index matching the renderer's predicate. The default is `(rank, <step or gpu_idx>, sample_ts_s, id)`.

5. **Symptom:** Aggregator log shows "SQLiteWriter flush failed: ... no such column ...".
   **Cause:** You renamed a column "to clean up the schema." A user's v0.2.3 DB still has the old name.
   **Fix:** Revert the rename. Add a new column instead. Document the old column as deprecated in the writer's docstring; remove it in a major version bump only.

6. **Symptom:** Dashboard freezes; aggregator log shows readers blocking the writer.
   **Cause:** WAL mode wasn't enabled (someone removed `PRAGMA journal_mode=WAL` from `_connect`), or readers hold a transaction open across compute.
   **Fix:** Verify `journal_mode=WAL` at startup (`sqlite_writer.py:248`). For renderers, use `with sqlite3.connect(...) as conn:` + short-lived connections per `_compute` (see `add_renderer.md` §3.2).

7. **Symptom:** Writer raises an exception on a malformed payload; the aggregator log shows `"SQLiteWriter flush failed: <your KeyError>"`.
   **Cause:** You used `payload_dict["x"]` instead of `payload_dict.get("x")`, or you didn't `isinstance(...)` check before coercing.
   **Fix:** Defensive `.get(...)` and `isinstance` everywhere. The writer must degrade to NULL columns, not raise. Compare to `system.py:208` for the pattern.

8. **Symptom:** Test passes, real session has zero rows.
   **Cause:** `SAMPLER_NAME` doesn't match the sampler's wire-level name. Easy to miss for samplers like `StdoutStderrSampler` whose `sampler_name="Stdout/Stderr"` (with the slash) doesn't match the class name.
   **Fix:** Read the sampler's `__init__` and copy the exact string.

9. **Symptom:** Cross-rank renderer sees rows in non-deterministic order.
   **Cause:** No stable secondary sort. `ORDER BY sample_ts_s` alone ties on equal timestamps from different ranks.
   **Fix:** Always include `id` as the tiebreaker: `ORDER BY sample_ts_s, id` or `ORDER BY id` for monotonic ingest order. The writer's job is to make sure `id` is `INTEGER PRIMARY KEY AUTOINCREMENT`; the renderer's job is to actually sort by it.

10. **Symptom:** A future renderer that needs to filter on a nested field can't, because that field is buried in `events_json`.
    **Cause:** Premature use of a JSON blob column. `step_time.py` uses `events_json` because the event names genuinely are not pinned. If your nested fields *are* known at schema-design time, project them into first-class columns.
    **Fix:** Refactor to first-class columns before users on PyPI have v0.2.3 DBs containing your blob.

11. **Symptom:** Aggregator process hangs at shutdown; writer thread won't exit.
    **Cause:** A bug in `init_schema` raised after partial DDL and the connection is in a weird state.
    **Fix:** Always test `init_schema` against a temp DB in a unit test before merging.

12. **Symptom:** Test fixture leaks SQLite connections; subsequent tests in the same process see "database is locked."
    **Cause:** The fixture opens `sqlite3.connect(...)` and never closes it.
    **Fix:** Use a `pytest.fixture` with a `yield` + `c.close()` in teardown (see §8.1 `conn` fixture). Or use the context-manager form `with sqlite3.connect(...) as c:`.

13. **Symptom:** Schema OK, build_rows OK, but `INSERT` raises `NOT NULL constraint failed`.
    **Cause:** You declared a column `NOT NULL` and the payload occasionally produces `None` for it.
    **Fix:** Either drop the `NOT NULL` (preferred for everything except `id` and `recv_ts_ns`), or filter the row out in `build_rows` (skip with `continue`) when the required field is missing.

---

## 10. Checklist before opening a PR

Copy into the PR description.

1. [ ] New writer file at `src/traceml/aggregator/sqlite_writers/<name>.py`.
2. [ ] Module-level docstring describes: which sampler, which tables, storage units, expected wire-payload shape (copy the docstring header from `system.py`).
3. [ ] `SAMPLER_NAME` constant matches the sampler's wire-level `sampler_name` exactly.
4. [ ] `accepts_sampler(sampler)`, `init_schema(conn)`, `build_rows(payload_dict, recv_ts_ns)`, `insert_rows(conn, rows_by_table)` — all four functions present with these exact signatures.
5. [ ] `init_schema` uses `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS`. Idempotent.
6. [ ] All projection tables include the bookkeeping columns `id INTEGER PRIMARY KEY AUTOINCREMENT`, `recv_ts_ns INTEGER NOT NULL`, `rank INTEGER`, `sample_ts_s REAL`, `seq INTEGER`.
7. [ ] At least one composite index covering the renderer's `WHERE` / `ORDER BY`. Verified with `EXPLAIN QUERY PLAN` — no `SCAN` for the renderer's hot query.
8. [ ] `build_rows` is fail-open: every field uses `.get(...)` + `isinstance` coercion, returns `{table: []}` on malformed inputs, never raises.
9. [ ] `insert_rows` uses explicit column lists in every `INSERT INTO ... VALUES (?, ...)` statement. Parameter count matches column list.
10. [ ] Writer registered in `sqlite_writer.py`'s `_PROJECTION_WRITERS` list. Import added.
11. [ ] Unit tests in `tests/test_<name>_sql_writer.py` covering: `init_schema` idempotence, `accepts_sampler` truth table, `build_rows` happy path, wrong-sampler returns empty, malformed payload does not raise, `insert_rows` round-trip.
12. [ ] If schema migrates (additive columns vs prior release): migration test against a synthetic old-schema DB.
13. [ ] WAL mode confirmed at aggregator startup (still set in `sqlite_writer.py::_connect`; you didn't accidentally remove it).
14. [ ] No `SELECT` inside `build_rows` or `insert_rows` (writer-only role; schema introspection in `init_schema` is the only exception).
15. [ ] Smoke run: `traceml run examples/mnist.py --mode summary` → `sqlite3 logs/<session>/aggregator/telemetry "SELECT COUNT(*) FROM <your_table>;"` returns > 0.
16. [ ] Multi-rank smoke run if applicable: rows from each rank present, sortable by `(rank, sample_ts_s, id)` deterministically.
17. [ ] CHANGELOG entry naming the new SQLite tables and indexes (so downstream consumers / external integrations know the schema grew).
18. [ ] `pre-commit run --all-files` clean (black, ruff, isort, codespell).
19. [ ] Commit message short, single-line, no `Co-Authored-By` (per `traceml/CLAUDE.md`).

---

## 11. Appendix

### 11.1. The `RemoteDBStore` allow-list — when does a sampler skip projection?

`TraceMLAggregator._REMOTE_STORE_SAMPLERS` (`trace_aggregator.py:78`) is a frozenset of sampler names that bypass SQLite for live rendering and stay in the in-memory `RemoteDBStore`:

```python
_REMOTE_STORE_SAMPLERS = frozenset(
    {
        "LayerMemorySampler",
        "LayerForwardMemorySampler",
        "LayerBackwardMemorySampler",
        "LayerForwardTimeSampler",
        "LayerBackwardTimeSampler",
    }
)
```

These layer samplers do not have projection writers yet. Their renderers read `RemoteDBStore` directly. The migration path documented in `add_renderer.md` §7 is "every new renderer reads from SQLite," which means: when a layer sampler gets a projection writer, its renderer should be ported to read SQLite, and the sampler should be removed from `_REMOTE_STORE_SAMPLERS`.

If you're adding a writer for one of these samplers, the migration is a two-PR sequence:

1. Add the projection writer (this guide). Rows now land in both `RemoteDBStore` (legacy) and SQLite (new). No renderer change.
2. Port the renderer to read SQLite. Drop the sampler from `_REMOTE_STORE_SAMPLERS`.

Cross-link: the live-store path is in [W9](../deep_dive/code-walkthroughs.md#w9-aggregator-core-tcp-receive-frame-dispatch-sqlite-writes) and `add_renderer.md` §7.

### 11.2. Multi-rank join contract

Renderers that aggregate across ranks (e.g. `StepCombinedRenderer`) join per-rank rows on `step`. The writer-side contract is:

- `step` is an indexed column on every step-keyed table (`step_time_samples`, `step_memory_samples`).
- `rank` is always nullable but written when present.
- The renderer's join shape is `... WHERE step IN (?, ?, ...) GROUP BY step, rank`. The `idx_step_time_samples_step_rank` index in `step_time.py:103` exists precisely to support this query shape. Copy that index pattern when your sampler is step-keyed.

### 11.3. Future direction — schema versioning as a first-class table

A `schema_meta(key TEXT PRIMARY KEY, value TEXT)` table with a `schema_version` row would let `init_schema` detect old DBs and run migrations conditionally instead of blanket `_ensure_column`. We haven't needed it yet because every change so far has been additive. When it lands, this guide gets a §6.6.

### 11.4. The v0.2.x → v1.0 migration path

Currently theoretical. Two options:

1. **Side-by-side:** v1.0 ships a new aggregator that writes a v1 schema to a different DB filename. v0.2.x DBs remain readable by the v0.2.x renderers (which won't be packaged with v1.0). Users who upgrade lose nothing but can't view their old runs in the new UI.
2. **In-place migration:** v1.0 detects v0.2.x DBs at first open and runs a one-shot `_ensure_column`-style migration. Higher complexity, higher value. Requires the `schema_version` table (§11.3).

Pick when we have a concrete v1 design.

---

## Gaps and ambiguities encountered while writing this guide

These are places where the current source under-specifies the contract. Flag in code review if your writer lands near them.

- **No schema versioning.** Verified via `grep -ri schema_version src/traceml/`: there is no `schema_version` table, no `PRAGMA user_version` use, no version constant in any writer. The implicit version is "whatever release wrote the file." This is fine while we're additive-only; the day we need a non-additive change, we need §11.3.
- **No migration helper.** `_ensure_column` is documented in §6.4 but not present in the codebase. The first writer that needs to add a column post-release will be the test case. Until then, treat the helper as "house style yet to be written."
- **No projection-writer tests in `tests/`.** Verified via `ls traceml/tests/`. The test template in §8 is the proposed shape; expect refinement when the first one lands.
- **`stdout_stderr.py` uses the envelope timestamp, others use the row timestamp.** `stdout_stderr.py:124` reads `payload_dict.get("timestamp")` for `sample_ts_s`, while every other writer reads `row.get("ts")`. The reason is that stdout lines don't carry per-row timestamps in the wire schema. Future contributors should not "fix" this inconsistency without first reading the sampler's wire schema.
- **`init_schema` never re-runs DDL after the first call within a process.** It runs once at writer startup (`sqlite_writer.py::_run`). If you change the schema in source and reuse an existing DB file, the changes are not applied until you add an `_ensure_column` path or delete the file. Document this behavior in your PR; it's surprising on first encounter.
- **Foreign keys are enabled (`PRAGMA foreign_keys=ON`) but no writer uses `REFERENCES`.** All cross-table links are implicit via `(rank, sample_ts_s, seq)`. If a future writer wants explicit FKs, the framework supports it but no exemplar shows the pattern (and the `executemany` performance cost of FK checks needs measuring before adoption).

Add new findings to this section in your PR.
