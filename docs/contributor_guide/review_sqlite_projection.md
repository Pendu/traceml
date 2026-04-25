# How to review an SQLite-projection-writer PR

> Internal contributor guide. Audience: one trusted co-founder / new engineer
> reviewing TraceML PRs. Companion to `add_sqlite_projection.md`. Not for public docs.

This guide teaches you how to review a PR that adds or modifies an SQLite projection writer in `src/traceml/aggregator/sqlite_writers/`. It assumes you have already read `add_sqlite_projection.md` (the author's guide) and have a working mental model of W7 (database + sender) and W9 (aggregator core). The seven-step workflow in §1 is the meta-pattern that future reviewer guides (`review_sampler.md`, `review_renderer.md`, ...) reuse — only §3 onward is projection-writer-specific.

---
Feature type: SQLite projection writer
Risk level: medium (silent corruption is the dominant failure mode; training is unaffected, but every renderer downstream reads what you accept)
Cross-cutting impact: aggregator persistence layer; consumed by every windowed renderer
PyTorch coupling: none
Reference reviews: — (no shipped projection-writer PRs as of 2026-04-25)
Companion author guide: `add_sqlite_projection.md`
Last verified: 2026-04-25
---

## 1. The meta-review-workflow (applies to every TraceML PR)

Every projection-writer review walks the same seven steps in order. Skipping any of them is how a flawed PR ships:

1. **Anchor** the PR diff to the relevant W-walkthroughs and Q/P entries. Read the PR through your existing mental models, not line-by-line.
2. **Run the projection-writer-family consistency check.** Build the table from §3 of this guide and grade the new writer against the existing five writers on each axis. Discrepancies are either justified deviations (document them) or bugs.
3. **Apply the projection-writer-class failure-mode catalogue** from §4. Each category maps to a known bug shape. Walk the diff with each shape in mind.
4. **Apply the four meta-questions** from §5: new schema axis, shared infrastructure interaction, wire-name as contract, invariant preservation.
5. **Write a verification gate** for each concern: a 3–10 line reproduction recipe with a clear pass/fail criterion. "I think this is buggy" becomes "here's the script that proves it." See §6.
6. **Draft comments at the right granularity** — line comment for specific code suggestions, PR-level comment for behavioural / architectural concerns. Hold parking-lot items back. See §7.
7. **Land the verdict** — approve / approve-with-changes / block. Criteria in §8.

The reviewer's job ends with a 2–3 sentence executive summary the maintainer can read without opening the diff. That goes in the verdict (§8).

This same seven-step shape applies to patch PRs, sampler PRs, renderer PRs, transport PRs — only the consistency table and the failure-mode catalogue change.

---

## 2. Step 1 — Anchor the PR to your walkthroughs

The first thing you do with a projection-writer PR is **not** open the diff. Open [`traceml_learning_code_walkthroughs.md`][W9] and re-read W7 §"DBIncrementalSender — `_last_sent_seq` discipline" and W9 §"`SQLiteWriterSimple._flush_once` and the `_PROJECTION_WRITERS` dispatch loop." Two reasons:

- The writer family has documented invariants (single-writer thread, WAL mode, fail-open `build_rows`, no `BEGIN`/`COMMIT` inside the writer, additive schema). You'll be checking the diff against those invariants, so they need to be in cache.
- A projection-writer PR will touch a small, stereotyped set of files. If you read the PR file-by-file without that map, you'll waste review budget. Most PRs of this shape touch 3–5 files.

### How to anchor

For each file in the diff, write down (in your review notes, not the PR yet):

| File pattern | W-section | What kind of change should this be? |
|---|---|---|
| `src/traceml/aggregator/sqlite_writers/<name>.py` (NEW) | [W9 §"Projection writers"][W9] | The substantive change — schema, `build_rows`, `insert_rows`. |
| `src/traceml/aggregator/sqlite_writer.py` | [W9 §"Dispatch loop"][W9] | Mechanical — one import, one list entry. |
| `src/traceml/aggregator/sqlite_writers/__init__.py` | none directly | Almost certainly unchanged — the `__init__` is a one-liner. If touched, ask why. |
| `src/traceml/aggregator/trace_aggregator.py` | [W9 §"`_drain_tcp` allow-list"][W9] | Touched only when the new writer is for a sampler currently in `_REMOTE_STORE_SAMPLERS` (see Appendix 11.1 of the author guide). |
| `src/traceml/samplers/<sampler>.py` or `samplers/schema/*.py` | [W6][W6] | Should be unchanged — the writer consumes wire schema, doesn't define it. If touched, this PR is doing two things; ask the author to split. |
| `tests/test_<name>_sql_writer.py` (NEW) | none directly | Surface coverage — see §4 for what must be tested. |
| `CHANGELOG.md` | none directly | New tables and indexes named so downstream consumers know the schema grew. |

If a file in the diff doesn't fit the table, that's a flag — the PR is doing something architecturally novel, and you should ask why before proceeding.

The point: **after anchoring, you should have one substantive file to read deeply, one mechanical registration edit to skim, and one new test file to grade against §4.** Anything else is suspicious.

[W6]: ../deep_dive/code-walkthroughs.md#w6-samplers-schemas-turning-hook-events-into-structured-rows
[W7]: ../deep_dive/code-walkthroughs.md#w7-database-sender-bounded-in-memory-store-and-incremental-tcp-shipping
[W9]: ../deep_dive/code-walkthroughs.md#w9-aggregator-core-tcp-receive-frame-dispatch-sqlite-writes

---

## 3. Step 2 — The projection-writer-family consistency table

Every projection writer slots into a small set of axes. The reviewer's job is to fill in the row for the new writer and grade each cell against the existing five writers.

### 3.1 The columns (these don't change PR-to-PR)

| Axis | What it asks | Where to verify |
|---|---|---|
| **Module location** | `aggregator/sqlite_writers/<sampler>.py`? | File path in the diff. |
| **`SAMPLER_NAME`** | Module-level constant? Matches the sampler's wire-level `sampler_name` exactly? | Top of the new module + the sampler's `__init__` (or `BaseSampler.__init__(sampler_name=...)`). |
| **Four functions** | `accepts_sampler`, `init_schema`, `build_rows`, `insert_rows` all present with house signatures? | New module top-level. |
| **`init_schema` idempotency** | `CREATE TABLE IF NOT EXISTS`? `CREATE INDEX IF NOT EXISTS`? No `DROP`/`ALTER`? | Read `init_schema` end-to-end. |
| **Bookkeeping columns** | `id INTEGER PRIMARY KEY AUTOINCREMENT`, `recv_ts_ns INTEGER NOT NULL`, `rank INTEGER`, `sample_ts_s REAL`, `seq INTEGER`? | First five columns of every projection table. |
| **Composite index** | At least one matching the renderer's `WHERE`/`ORDER BY`? Verified with `EXPLAIN QUERY PLAN` (no `SCAN`)? | Index DDL + the renderer's hot SQL. |
| **`build_rows` defensiveness** | `payload_dict.get(...)` everywhere? `isinstance` coercion? Returns empty dict on malformed input, never raises? | Read `build_rows` end-to-end. |
| **Output shape** | Tuples (positional) in exact column order of the eventual `INSERT`? | Compare the `out[<table>].append((...))` to the column list in `INSERT`. |
| **`insert_rows` form** | Explicit column list in every `INSERT INTO ... VALUES (?, ...)`? Param count matches column list? `executemany`, not loop of `execute`? | Read `insert_rows` end-to-end. |
| **Transaction** | No `BEGIN`/`COMMIT` in the writer (parent transaction only)? | grep the new module for `BEGIN` / `COMMIT`. |
| **Migration** | If column added vs. prior release: `_ensure_column` style or new table? | Search for `ALTER TABLE` and `PRAGMA table_info`. |
| **Dispatcher registration** | Added to `_PROJECTION_WRITERS` list in `sqlite_writer.py`? Import added? | `sqlite_writer.py` top-of-file diff. |
| **Tests** | `init_schema` idempotence, `accepts_sampler` truth table, `build_rows` happy/wrong-sampler/malformed, `insert_rows` round-trip? | New `tests/test_<name>_sql_writer.py`. |

### 3.2 The current state (April 2026)

| Axis | `system` | `process` | `step_time` | `step_memory` | `stdout_stderr` |
|---|---|---|---|---|---|
| Module | `sqlite_writers/system.py` | `sqlite_writers/process.py` | `sqlite_writers/step_time.py` | `sqlite_writers/step_memory.py` | `sqlite_writers/stdout_stderr.py` |
| `SAMPLER_NAME` | `"SystemSampler"` | `"ProcessSampler"` | `"StepTimeSampler"` | `"StepMemorySampler"` | `"Stdout/Stderr"` (note: slash, not class name) |
| Four functions | YES | YES | YES | YES | YES |
| `init_schema` idempotent | `CREATE TABLE IF NOT EXISTS` ×2, `CREATE INDEX IF NOT EXISTS` ×2 | `CREATE TABLE IF NOT EXISTS` ×1, `CREATE INDEX IF NOT EXISTS` ×2 | `CREATE TABLE IF NOT EXISTS` ×1, `CREATE INDEX IF NOT EXISTS` ×2 | `CREATE TABLE IF NOT EXISTS` ×1, `CREATE INDEX IF NOT EXISTS` ×2 | `CREATE TABLE IF NOT EXISTS` ×1, `CREATE INDEX IF NOT EXISTS` ×2 |
| Bookkeeping columns | id, recv_ts_ns, rank, sample_ts_s, seq | id, recv_ts_ns, rank, sample_ts_s, seq | id, recv_ts_ns, rank, sample_ts_s, seq | id, recv_ts_ns, rank, sample_ts_s, seq | id, recv_ts_ns, rank, sample_ts_s (no `seq` — append-only stream) |
| Composite index | `(rank, sample_ts_s, id)` parent; `(rank, gpu_idx, sample_ts_s, id)` child | `(rank, sample_ts_s, id)` + `(pid, sample_ts_s, id)` | `(rank, step, sample_ts_s, id)` + `(step, rank, id)` | `(rank, step, sample_ts_s, id)` + `(step, rank, id)` | `(rank, id)` + `(sample_ts_s, id)` |
| `build_rows` defensiveness | `.get` everywhere, `isinstance` coercion | same | same | same | same; additionally skips lines where `line_raw is None` or `not line` |
| Output shape | Tuples positional, parent 17 cols / child 11 cols | Tuple positional, 15 cols | Tuple positional, 6 cols | Tuple positional, 9 cols | Tuple positional, 4 cols |
| `insert_rows` form | `executemany`, explicit column list, two tables | `executemany`, explicit column list | `executemany`, explicit column list | `executemany`, explicit column list | `executemany`, explicit column list |
| Transaction in writer | NO | NO | NO | NO | NO |
| Migration | none yet (greenfield) | none | none | none | none |
| Dispatcher registration | YES (`_PROJECTION_WRITERS`) | YES | YES | YES | YES |
| Tests | NONE in `tests/` (gap — see author guide §8) | NONE | NONE | NONE | NONE |
| Sampler-time field source | `row.get("ts")` | `row.get("ts")` | `row.get("timestamp")` | `row.get("ts")` | `payload_dict.get("timestamp")` (envelope-level) |

When reviewing, **add a column** for the new writer and walk every row. Three outcomes per cell:

- Matches the family — note it and move on.
- Differs from the family — demand a justification in the PR description or a comment in the writer file. The `Stdout/Stderr` row's "no `seq`" is a justified deviation (append-only log lines, dedup via the upstream sender's `_last_sent_seq`); the `stdout_stderr` row's `payload_dict.get("timestamp")` is a justified deviation (no per-row timestamp on the wire).
- Cell is empty / undecidable from the diff — ask the author.

### 3.3 The table is the most reusable artifact in this guide

Every future projection-writer review should rebuild this table. Two reasons:
- The act of filling it forces you to read the writer with the family in mind, which catches "this differs in ways the author didn't notice."
- The completed table goes in your review notes, and over time becomes the reviewer's contract test for the writer family. (See "Gaps" at the end — formalising this is on the wishlist.)

### 3.4 The schema-shape sub-table

Within the family, three concrete schema patterns recur. Place the new writer's design into one of them before grading the rest of the diff:

| Pattern | Exemplar | Use when | Don't use when |
|---|---|---|---|
| Single flat table | `process.py`, `step_memory.py` | One row per sample, all fields known at schema time | The wire payload nests a list of children |
| Parent + child via implicit `(rank, sample_ts_s, seq)` | `system.py` | Per-snapshot child rows (per-GPU, per-device) | Children outlive the parent or need explicit FK |
| Single table + restricted JSON blob | `step_time.py` (`events_json`) | Inner field set is genuinely unbounded by the wire schema | Inner field set is fixed — promote to first-class columns |

Forcing the author to name the pattern in the PR description catches "we picked JSON because it was easy" before the schema lands and ossifies.

---

## 4. Step 3 — Projection-writer-class failure modes

Distilled from `add_sqlite_projection.md` §9 (pitfalls 1–13) plus folklore from the early writer designs. Every projection-writer PR is at risk for these twelve categories. Walk the diff with each one in mind.

### 4.1 Silent payload drop (the registration-omission bug)

Applies to: every new writer.

The bug shape: writer file imports fine, `init_schema` runs at aggregator startup, but no rows ever appear in the projection tables. Cause: the writer is not in `_PROJECTION_WRITERS` in `sqlite_writer.py:52`. The dispatcher only iterates that list; there is no auto-discovery. `raw_messages` will still have rows for the sampler — that's the giveaway.

**What to check:**

- Open `sqlite_writer.py` and confirm both the `from ... import ... as <name>_sql_writer` line and the entry in the `_PROJECTION_WRITERS` list. Both. One without the other is a silent failure.
- Verification gate: see §6.1.

### 4.2 Column-misalignment silent corruption

Applies to: every writer.

The bug shape: the tuple appended in `build_rows` doesn't match the column list in the `INSERT`. SQLite binds positionally and silently inserts the wrong value into the wrong column. No exception. The renderer shows nonsense values (or perfectly reasonable wrong values, which is worse).

The structural cause: copy-pasting from another writer, adding/removing a column in `init_schema`, forgetting to update the corresponding tuple slot or the `INSERT` column list. Three places must agree:

1. `init_schema` table DDL (column order).
2. `build_rows` tuple positional order.
3. `insert_rows` explicit column list (matches the tuple positionally).

**What to check:**

- Side-by-side read of all three. The reviewer's diff viewer should show `init_schema`, `build_rows` tuple, and `insert_rows` column list adjacent.
- Demand an `insert_rows`-round-trip test (see §6.2 example and §8 of the author guide). It catches this immediately.
- Spot-check by counting `?` placeholders against the column list — must match.

### 4.3 Schema-migration not idempotent

Applies to: any PR that modifies an existing writer's schema.

The bug shape: the writer adds a column with `ALTER TABLE foo ADD COLUMN bar REAL`. First aggregator startup succeeds. Second startup raises `sqlite3.OperationalError: duplicate column name: bar`. SQLite's `ALTER TABLE ADD COLUMN` is **not** idempotent the way `CREATE TABLE IF NOT EXISTS` is.

The fix is the `_ensure_column` `PRAGMA table_info` check pattern (author guide §6.4). It is documented but not yet present in the codebase — the first PR to need it will codify it.

**What to check:**

- Any `ALTER TABLE` in the diff is a flag. Stop and read carefully.
- The migration must check `PRAGMA table_info(<table>)` first; only run `ALTER` if the column is absent; tolerate `sqlite3.OperationalError` as a multi-aggregator race.
- Migration test against a synthetic old-schema DB (author guide §8.2).

### 4.4 Renamed/dropped column breaks v0.2.x users

Applies to: any PR that changes an existing column's name, type, or removes it.

The bug shape: a user upgrades from v0.2.3 to the new release. Their existing SQLite DB has the old column name; the new aggregator queries the new column name and gets `sqlite3.OperationalError: no such column`. Or the reverse — the new writer queries a column an old DB doesn't have.

The rule: **schema changes are additive only.** Author guide §6.3 is the exposition.

**What to check:**

- `git diff` on the existing writer's `init_schema` should never show a removed column or renamed column.
- If the PR claims a rename is "just a cleanup," block. Add a new column, deprecate the old one in the docstring, plan its removal for a major version bump.
- Cross-link `principles.md` for the wire-format-stability rule that this is the SQLite expression of.

### 4.5 Missing index → SCAN at scale

Applies to: every writer.

The bug shape: works on the dev DB with 100 rows; the renderer's `EXPLAIN QUERY PLAN` says `SEARCH ... USING INDEX` accidentally because everything fits in one page. In production with 1M rows, the same query falls back to `SCAN <table>` and the dashboard freezes.

The default index for time-series projection tables is `(rank, <step or gpu_idx>, sample_ts_s, id)`. Walk the renderer's exact `WHERE` and `ORDER BY` and confirm the index covers the predicate prefix.

**What to check:**

- Run `EXPLAIN QUERY PLAN <renderer's exact SQL>` against a test DB. Output must contain `SEARCH ... USING INDEX`. `SCAN` anywhere in the plan blocks the PR.
- Verification gate: see §6.3.
- The renderer might not exist yet; if so, the writer's index design must still match the *expected* renderer query shape. Demand the author state the expected query in the PR description.

### 4.6 WAL mode not enabled

Applies to: any PR that touches `sqlite_writer.py::_connect`, even tangentially.

The bug shape: someone removes `conn.execute("PRAGMA journal_mode=WAL;")` from `sqlite_writer.py:248`. Single-writer + many-reader concurrency reverts to rollback-journal mode. Renderer reads start blocking the writer; the dashboard freezes; the writer thread queue fills; messages drop.

This is rare in projection-writer PRs but the cost of missing it is total — the entire aggregator's concurrency story breaks.

**What to check:**

- Sanity-check the diff for any change to `_connect` or `_run` in `sqlite_writer.py`. If the PR is "just" adding a writer, these should be untouched.
- Aggregator-startup verification: `sqlite3 <db> "PRAGMA journal_mode;"` should return `wal`.

### 4.7 Writer raises on malformed payload (fail-open violation)

Applies to: every writer.

The bug shape: `build_rows` does `payload_dict["x"]` instead of `payload_dict.get("x")`, or it omits an `isinstance(...)` check before coercing. A wire payload that's slightly off (an old client, a partially-decoded MessagePack, a sampler with a transient bug) raises `KeyError` or `TypeError`. The exception is caught one frame up in `_collect_flush_rows`, but **the entire payload is dropped** — including any other tables it would have populated. Worse, this happens silently in the aggregator log.

The principle is fail-open: degrade to NULL columns or skip the row, never raise. Cross-link `principles.md`.

**What to check:**

- grep the new module for `payload_dict[`, `row[`, `gpus_raw[` (subscript without `.get`). Each is a flag.
- Compare to `system.py:208` for the house pattern: `int(seq_raw) if isinstance(seq_raw, int) else None`.
- Demand a `test_build_rows_malformed_payload_does_not_raise` test (author guide §8.1).

### 4.8 `SAMPLER_NAME` mismatch with sampler wire-level name

Applies to: every new writer.

The bug shape: the writer's `SAMPLER_NAME = "MyClassName"` doesn't match the sampler's `BaseSampler.__init__(sampler_name=...)` argument. `accepts_sampler` always returns False; payloads route to `raw_messages` only; the projection table is empty. Tests that mock the payload pass because the test passes the matching string.

Easy to miss for samplers with unusual wire names. The most-cited example: `StdoutStderrSampler` ships with `SAMPLER_NAME = "Stdout/Stderr"` — a slash, not the class name.

**What to check:**

- Open the sampler file. Find the `super().__init__(sampler_name=...)` call. Copy the literal string. Compare to the writer's `SAMPLER_NAME`. They must be byte-identical.
- Verification gate: real session smoke test (§6.1) — `SELECT DISTINCT sampler FROM raw_messages` shows the sampler, `SELECT COUNT(*) FROM <projection>` is zero.

### 4.9 Non-deterministic cross-rank ordering

Applies to: every writer whose downstream renderer joins or sorts across ranks.

The bug shape: the renderer does `ORDER BY sample_ts_s` only. Two ranks ticking on the same wall clock produce equal `sample_ts_s` values. SQLite returns them in storage-engine-defined order — visually, the dashboard shows rank 0 first one tick, rank 1 first the next. The user sees flicker and reports "the renderer is broken."

The writer's job: ensure `id INTEGER PRIMARY KEY AUTOINCREMENT`. The renderer's job: include `id` as the tiebreaker (`ORDER BY sample_ts_s, id`). This is a writer/renderer joint contract.

**What to check:**

- The writer's `id` column is `INTEGER PRIMARY KEY AUTOINCREMENT`. Without `AUTOINCREMENT`, SQLite reuses rowids on delete, breaking the monotonicity assumption.
- The downstream renderer's `ORDER BY` includes `id`. If the renderer ships in the same PR, check it. If a future PR, raise the issue in the PR description.

### 4.10 Premature JSON blob column

Applies to: writers using the `step_time.py::events_json` pattern.

The bug shape: the author stuffs nested fields into a JSON blob column "to keep the schema flexible." Six months later, a renderer wants to filter by one of those nested fields and can't — the field is buried in a TEXT blob. The fix is a non-additive schema change (promote to first-class column, deprecate the blob field).

The rule: JSON blobs are for genuinely unbounded fields (event names in `step_time` are not pinned by the wire schema). Anything bounded at schema-design time goes in first-class columns.

**What to check:**

- Any new `<name>_json TEXT` column is a flag. Read the wire schema and ask: is this field set actually unbounded?
- If the answer is "no, but the schema feels cleaner this way," push back. First-class columns are cheap; future migrations are not.

### 4.11 SELECT inside the writer

Applies to: every writer.

The bug shape: `build_rows` or `insert_rows` does `conn.execute("SELECT ...")` to "look up the previous row" or "deduplicate." This is wrong on three axes:
- Reads inside the writer thread compete with renderer reads on the same connection (broken WAL story).
- Transactionally weirder — the read sees uncommitted state from the same flush.
- Performance hit on the hot path; the writer is the bottleneck.

The only allowed `SELECT`-shaped call inside a writer is `PRAGMA table_info(...)` for migration introspection in `init_schema` (author guide §6.4).

**What to check:**

- grep the new module for `SELECT`, `conn.execute("SEL`, `cursor.execute`. Each occurrence is a flag.
- If the author needs prior-state lookup, the architecture is wrong — push the dedup/lookup upstream to the sender (where `_last_sent_seq` lives).

### 4.12 NOT NULL constraint failed

Applies to: any writer with `NOT NULL` columns beyond `id` and `recv_ts_ns`.

The bug shape: the writer declares a column `NOT NULL`. The wire payload occasionally produces `None` for that field (early in a session, or under a transient sampler bug). `executemany` raises `sqlite3.IntegrityError: NOT NULL constraint failed`. The flush fails; **all rows in the batch are lost**, not just the offending one (because the transaction rolls back).

Author-guide §6.2: only `id` and `recv_ts_ns` should be `NOT NULL` by default. `line` is `NOT NULL` in `stdout_stderr_samples` because `build_rows` already filters out null/empty lines (`stdout_stderr.py:140`).

**What to check:**

- Every `NOT NULL` column beyond `id` / `recv_ts_ns` must have a corresponding filter in `build_rows` that skips the row when the field is missing. If both sides aren't there, drop the constraint.

---

## 5. Step 4 — The four meta-questions

Apply each to the PR and write down the answer explicitly. If you can't answer, ask.

### 5.1 Does this PR introduce a new schema axis? What new failure modes?

The five existing writers cover: flat-table-per-sample (`process`, `step_memory`), parent+child via implicit key (`system`), append-only stream (`stdout_stderr`), and JSON-blob-with-stable-columns (`step_time`). If the new writer fits one of these, the failure modes from §4 cover it.

If the new writer introduces a new schema axis — e.g. an explicit foreign-key parent/child relationship, a `BLOB` column that isn't MessagePack-on-`raw_messages`, multi-table joins inside a single `build_rows` call — enumerate the failure modes that axis creates. Examples:

- **Explicit FK (`REFERENCES <parent>(id)`)** — forces row-by-row inserts for children (you need `cursor.lastrowid`); breaks the `executemany` performance assumption; introduces FK-violation failure modes.
- **Multi-table cross-references** — non-trivial transactional ordering; child insert before parent insert raises FK error.
- **`BLOB` storage** — forces every renderer to deserialize, defeats the point of a projection.

**Reviewer move:** when the new writer's row in the §3.2 table fills a column no prior writer fills, enumerate the failure modes that column creates. Block the PR until the author has documented them.

### 5.2 Does it interact with shared infrastructure?

Three pieces of shared infrastructure in the writer family:

- **The `TraceML-SQLiteWriter` thread** — single-threaded, runs every writer's `build_rows` and `insert_rows` in sequence. A 5-ms-per-payload writer competes with the four other writers for ~2 ms of total flush budget (author guide §7.1).
- **The flush queue** (`max_queue=50_000`, `sqlite_writer.py:97`) — fills when the writer can't keep up; messages drop silently.
- **The SQLite connection** — one writer connection, many reader connections; WAL is the contract.

**Reviewer move:** estimate the new writer's `build_rows` cost for an expected payload. For a flat table with ~10 fields, target <100 µs. For a parent+child like `system.py` with 8 GPUs, ~500 µs. If the writer dominates the flush budget, profile with `cProfile` over a 30-second smoke run before merging. Forcing the author to do this estimate catches the case where N writers are individually fine but jointly exhaust the writer thread.

### 5.3 Is the SQLite schema a contract?

Yes — and harder to reverse than the wire format. Once a release ships a table name, column name, column type, or index name, every user with a v0.2.x DB on disk depends on it. Renaming or dropping is a non-additive change that breaks v0.2.x renderers reading the new aggregator's DB layout.

**Reviewer move:** every PR that introduces a new table or column must answer: "if we discover next month this column name describes the wrong thing, what's the migration cost?" A small upfront naming discussion is cheap; a rename in v1.0 is expensive. Cross-link `principles.md` for the cross-cutting wire-compat rule; this is its SQLite specialization.

Three concrete checks:

- Table name follows the family convention: `<domain>_samples` for parents, `<domain>_<child>_samples` or `<domain>_<child>` for children.
- Column unit suffix is consistent: `_bytes`, `_percent`, `_w`, `_c`, `_ms`. Author guide §6.1.
- Index name follows `idx_<table>_<cols joined by _>`.

### 5.4 Which invariants does the PR preserve, and have you verified each one?

The projection-writer family invariants:

1. **`init_schema` is idempotent.** All DDL uses `CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS`; no `DROP`; any `ALTER` guarded by `PRAGMA table_info`.
2. **`accepts_sampler` is on the hot path and cheap.** A string equality. No regex, no compile, no I/O.
3. **`build_rows` is fail-open.** Defensive `.get` and `isinstance` everywhere. Returns `{<table>: []}` on malformed input. Never raises.
4. **`insert_rows` uses explicit column lists.** No `INSERT INTO foo VALUES (?, ?, ...)` without column names — that's how column-misalignment corruption ships.
5. **No `BEGIN`/`COMMIT` inside the writer.** Parent transaction in `_write_flush_rows` only.
6. **No `SELECT` inside the writer.** The only exception is `PRAGMA table_info` in `init_schema` for migrations.
7. **Schema changes are additive only.** No drop, no rename, no type change. New tables, new columns, new indexes only.
8. **Bookkeeping columns are present.** `id`, `recv_ts_ns`, `rank`, `sample_ts_s`, `seq` (or documented exception, e.g. `stdout_stderr` skips `seq`).
9. **Dispatcher registration is complete.** Both the import and the `_PROJECTION_WRITERS` list entry. One without the other is silent failure.

**Reviewer move:** for each invariant, point at the line of the diff that preserves (or could break) it. If you can't, you don't yet understand the PR well enough to approve it.

---

## 6. Step 5 — Verification gates

Every concern in the review must come with a **concrete reproduction recipe**. The shape:

```
1. Setup (1–3 lines): branch checkout, env, fixture.
2. Command (1 line): the invocation.
3. Expected output (specific value, not "should look reasonable").
4. Pass / fail criterion (a single inequality or equality).
```

This is the artifact that turns "I think this is buggy" into "here's the 3-line script that proves it." Without it, a finding is folklore and the author is justified in ignoring it.

### 6.1 Worked example — silent registration omission (§4.1)

```bash
# Setup
git -C /teamspace/studios/this_studio/traceml checkout pr-<N>
pip install -e ".[dev,torch]"

# Command — run any session that exercises the new sampler
traceml run examples/mnist.py --mode summary

# Verification
SESSION=$(ls -t logs/ | head -1)
sqlite3 logs/$SESSION/aggregator/telemetry \
    "SELECT COUNT(*) FROM raw_messages WHERE sampler='<NewSampler>';"
sqlite3 logs/$SESSION/aggregator/telemetry \
    "SELECT COUNT(*) FROM <new_projection_table>;"

# Pass: both > 0 (writer registered, payloads projected)
# Fail: raw > 0 but projection == 0 (writer not in _PROJECTION_WRITERS,
#       OR SAMPLER_NAME does not match wire string)
```

### 6.2 Worked example — writer fail-open on malformed payload (§4.7)

A unit-test-shaped recipe (no GPU needed):

```python
from traceml.aggregator.sqlite_writers import <writer>

# Tables value is not a dict
malformed_a = {"sampler": "<SamplerName>", "tables": "garbage"}
rows_a = <writer>.build_rows(malformed_a, recv_ts_ns=0)
# Pass: returns {table: []} for every declared table
# Fail: raises any exception

# A child list is a string instead of a list
malformed_b = {"sampler": "<SamplerName>", "rank": 0,
               "tables": {"X": [{"seq": 1, "ts": 1.0, "gpus": "garbage"}]}}
rows_b = <writer>.build_rows(malformed_b, recv_ts_ns=0)
# Pass: parent row inserted (degraded, with NULLs); child rows empty
# Fail: raises any exception
```

### 6.3 Worked example — no `SCAN` for the renderer's hot query (§4.5)

```bash
# Setup — populate a test DB with N rows (N >= 10_000) via fixture or smoke run.
SESSION=$(ls -t logs/ | head -1)

# Command
sqlite3 logs/$SESSION/aggregator/telemetry \
    "EXPLAIN QUERY PLAN <renderer's exact SQL>;"

# Pass: every step in the plan starts with SEARCH ... USING INDEX
# Fail: any step starts with SCAN
```

If the renderer doesn't ship in this PR, the recipe runs against the renderer's *expected* SQL — which the author must state in the PR description.

### 6.4 Worked example — column-misalignment round trip (§4.2)

```python
from pathlib import Path
import sqlite3
from traceml.aggregator.sqlite_writers import <writer>

db = Path("/tmp/round_trip.db")
db.unlink(missing_ok=True)
c = sqlite3.connect(str(db), isolation_level=None)
c.execute("PRAGMA journal_mode=WAL;")
<writer>.init_schema(c)

payload = { ... known good payload ... }
rows = <writer>.build_rows(payload, recv_ts_ns=999)
c.execute("BEGIN;")
<writer>.insert_rows(c, rows)
c.execute("COMMIT;")

result = c.execute(
    "SELECT <every column in declaration order> FROM <table>;"
).fetchall()
# Pass: each column holds the value the payload assigned to its semantic field
# Fail: any value lands in the wrong column (e.g. seq shows up under gpu_count)
```

### 6.5 When you can't write a verification gate

If you can't write a recipe — you only have a vague worry — **don't raise the concern in the review yet**. Either escalate it to research (file a follow-up issue, label "investigate"), or hold it back per §7.3. Vague concerns waste author time.

### 6.6 Recipe style rules

- **Specific numbers, not adjectives.** `COUNT > 0` not "should have data."
- **Reproducible from a clean checkout.** No "you also need to apply patch X first" — if the recipe depends on prior fixes, restate them.
- **3–10 lines of actual code or SQL.** Longer means you're testing too much at once; cut to the smallest demonstrating example.
- **State the GPU dependency explicitly.** Most projection-writer recipes are CPU-only — that's the point of testing the writer in isolation. Smoke recipes that boot the aggregator may need GPU; mark them.

---

## 7. Step 6 — Drafting comments

Two granularity choices: line comment vs PR-level comment. They are not interchangeable.

### 7.1 Line comments

Use when: there is a specific code change you're proposing in a specific location. Pin the comment to the line that needs to change.

Pattern: state the issue → propose the fix → reference a verification gate or precedent.

```
This raises KeyError on payloads missing "step". Use
.get("step") + isinstance check, matching system.py:208.

Verification: the §6.2 recipe with payload missing "step" must
return rows=[] for the affected table without raising.
```

Keep it tight. The reviewer's job is to point at the change, not to re-derive the architecture.

### 7.2 PR-level comments

Use when: the concern is **behavioural** or **architectural**, not localised to a single line. The fix may touch multiple files; the discussion is about the PR's intent.

Examples that warrant a PR-level comment:

- "The new schema introduces an explicit FK pattern not used by any existing writer (§5.1). Please document the failure modes and the tradeoff against `executemany`."
- "The `step` column was added without a migration path for v0.2.x DBs. Three options: ship `_ensure_column`, ship a new table name, or drop the column."
- "This new writer is for `LayerMemorySampler`, which is in `_REMOTE_STORE_SAMPLERS`. Per author-guide §11.1, this is the first PR of a two-PR migration. Please state explicitly so the renderer porting follows."

A PR-level comment is also right for cross-cutting concerns: "did you run the smoke test in §6.1?", "does this column name conflict with the wire schema's intent?", "the JSON-blob choice is unstated — please document why this is unbounded enough to warrant it."

### 7.3 What NOT to raise (the holdback discipline)

Two kinds of items belong in your private parking-lot, not in the PR review:

- **Judgement calls about positioning** — "is this column unit-suffix style my preference or the family's?", "should we name the table `*_samples` or `*_records`?" Decide privately by checking the §3.2 table; apply privately.
- **Adjacent improvements** — "while we're here, the test fixture pattern could be hoisted into a shared `conftest.py`." If the improvement isn't required for the PR to ship, file it as a follow-up issue. Don't grow the PR.

The discipline: a PR review delivers a focused set of must-fix items. Bloating the review with parking-lot items dilutes the must-fix signal and trains the author to treat your reviews as discussion threads, not gates.

### 7.4 The maintainer summary

Every review ends with a 2–3 sentence executive summary suitable for the maintainer to read without opening the PR. The shape:

> PR #N adds an SQLite projection writer for [sampler]. Schema follows the [flat | parent+child | JSON-blob] pattern, [registered | NOT registered] in `_PROJECTION_WRITERS`. Review converged on K concrete items: (1) ..., (2) ..., (3) .... All K fixes are localised; each needs one small test. Recommend [verdict].

Maintainer reads three sentences and either agrees with the verdict or opens the PR. This is the artifact your maintainer wants more than the diff comments.

---

## 8. Step 7 — Landing the verdict

Three states. Pick exactly one.

### 8.1 Approve

Conditions:
- Consistency table (§3) is fully matched or has documented justified deviations.
- All four meta-questions (§5) answered explicitly.
- All nine invariants (§5.4) preserved.
- No concerns require a verification gate (§6).
- Tests cover §4.1 (registration), §4.2 (round trip), §4.7 (fail-open), and §4.8 (`SAMPLER_NAME` truth table).
- Smoke test (§6.1) passes locally.

If all six are true, approve cleanly. Don't suggest follow-up work in the approval — file follow-ups separately so the PR can ship.

### 8.2 Approve with changes

Conditions:
- All concerns are minor: docstring fixes, test gaps, naming nits, missing CHANGELOG entry, missing index that the renderer doesn't yet exercise.
- No concern affects metric correctness or schema stability.
- All concerns have a one-line fix or a clear written-down resolution.

This is "accept the PR but require these N small changes." Not "the PR is conceptually broken."

### 8.3 Block (request changes)

Conditions (any one):
- A concern affects **metric correctness** at the projection layer (e.g. §4.2 column-misalignment, §4.10 premature JSON blob hiding a queryable field).
- A concern can **silently drop user data** (e.g. §4.1 not in `_PROJECTION_WRITERS`, §4.8 `SAMPLER_NAME` mismatch, §4.12 `NOT NULL` failure dropping a flush).
- A concern violates a writer-family invariant (§5.4).
- A concern breaks **backward compatibility** with v0.2.x DBs (§4.4).
- The PR introduces a new schema axis without enumerating its failure modes (§5.1).
- Tests don't exist for the four mandatory categories above (§8.1 bullet 5).
- `EXPLAIN QUERY PLAN` shows `SCAN` for the renderer's hot query.

### 8.4 What "block" doesn't mean

It does not mean the architecture is wrong. It does not mean the author has to redesign. It means **these specific items must be resolved before merge.** Frame the verdict that way to keep the relationship healthy with the author.

---

## 9. Reference: the consistency table as a contract test

The §3.2 table is the single most reusable artifact in this guide. Until a `tests/test_writer_family.py` exists (see Gaps), the reviewer's job is to manually rebuild it and add the new writer's column. Pasting the table into your review notes verbatim and adding the new column takes ~3 minutes; it catches a high fraction of the §4 failure modes by construction.

If you're new to reviewing projection-writer PRs, read `add_sqlite_projection.md` §3 (the `system.py` walkthrough) and §9 (the pitfall list) first. That's ~250 lines and gives you the shape. The rest is depth.

---

## 10. Common reviewer mistakes

Numbered, with cause and fix.

1. **Reviewing in isolation.** Cause: opening the diff first, before anchoring to W7/W9. Effect: drowning in mechanical changes. Fix: do §2 before §3 — every time.

2. **Re-deriving the consistency table from scratch.** Cause: the table isn't a formal artifact in source. Effect: each reviewer rebuilds it slightly differently and misses cells. Fix: paste the §3.2 table into your review notes verbatim and add the new column.

3. **Trusting `init_schema` because it has `CREATE TABLE IF NOT EXISTS`.** Cause: the IF-NOT-EXISTS feels idempotent. Effect: missing the case where the PR also adds an `ALTER TABLE` migration that isn't idempotent (§4.3). Fix: grep for `ALTER` separately; idempotency is a per-statement property, not a per-function one.

4. **Approving on schema design without checking dispatcher registration.** Cause: the new writer file is well-written; the reviewer never opens `sqlite_writer.py`. Effect: §4.1 silent payload drop ships. Fix: `_PROJECTION_WRITERS` membership is part of every approval checklist.

5. **Mistaking "tests pass locally" for "the writer works in a real session."** Cause: unit tests use a synthetic payload that matches the test author's mental model of the wire format. Effect: §4.8 `SAMPLER_NAME` mismatch ships because the test passes the writer's own constant. Fix: real-session smoke test (§6.1) is mandatory.

6. **Vague concerns without verification gates.** Cause: time pressure, gut feel. Effect: author can't reproduce, dismisses the concern. Fix: §6 — every concern gets a recipe. The §6.2 fail-open recipe is ~5 lines and runs in CPU CI; there is no excuse for a vague concern about defensive parsing.

7. **Mixing line comments and PR-level comments.** Cause: writing architectural concerns inline next to a specific line. Effect: comment gets resolved by changing one line, the architectural point is lost. Fix: §7.1/§7.2 — pick the granularity deliberately.

8. **Skipping the `EXPLAIN QUERY PLAN` check.** Cause: the index DDL exists in `init_schema`, so it must work. Effect: §4.5 SCAN ships because the index column order doesn't match the renderer's predicate order. Fix: paste the renderer's hot SQL into `EXPLAIN QUERY PLAN`. 30 seconds.

9. **Conflating "matches the family" with "correct."** Cause: §3 consistency check is fully matched, so the reviewer stops. Effect: novel-axis failure modes (§5.1) miss. Fix: every empty cell in the new column is a question, not a free pass.

10. **Skipping the maintainer summary.** Cause: assumes the maintainer will read the diff. Effect: maintainer reads the diff, disagrees on severity, kicks back to the reviewer. Fix: §7.4 — three sentences are the maintainer's reading material; the diff is yours.

11. **Approving a writer for a `_REMOTE_STORE_SAMPLERS` sampler without naming the two-PR migration.** Cause: not aware of the in-memory-store legacy path (author-guide §11.1). Effect: the renderer port never lands; the new SQLite table sits unused. Fix: any PR for a `Layer*Sampler` projection must explicitly state "PR 1 of 2" in the description.

12. **Letting `NOT NULL` slip past on a nullable wire field.** Cause: the column "feels required" to the writer author. Effect: §4.12 `NOT NULL` constraint failures drop entire flushes silently in production. Fix: only `id` and `recv_ts_ns` are `NOT NULL` by default; everything else is nullable unless `build_rows` filters out nulls upstream.

---

## Gaps encountered while writing this guide

Where the reviewer's playbook is currently underspecified or relies on folklore. Flag these in your review process if you hit them.

- **The consistency table (§3) isn't a formal artifact.** It lives in the author guide and now in this guide. If the writer family grows beyond five entries, every reviewer will diverge on the column set. Worth lifting into a contract test in `tests/test_writer_family.py` that introspects each `sqlite_writers/*.py` module and asserts (`SAMPLER_NAME`, `accepts_sampler`, `init_schema`, `build_rows`, `insert_rows`) all present with house signatures, plus that every writer is in `_PROJECTION_WRITERS`. Not yet written.

- **There's no projection-writer test in `tests/`.** Verified via `ls traceml/tests/`: zero `test_*_sql_writer.py` files. The reviewer is currently grading test coverage against a template (author guide §8.1) rather than against a precedent. The first writer PR with tests sets the precedent; until then, reviewer judgement is the precedent.

- **No `_ensure_column` helper in the codebase.** Author guide §6.4 documents the pattern; no writer ships it. The first migration PR will be the test case for both the helper *and* this guide's §4.3 review approach. Until then, treat any `ALTER TABLE` in a writer PR as worthy of extra scrutiny.

- **No central registry of SQLite table names or column units.** A reviewer enforcing §5.3 (schema-as-contract) has to grep across `sqlite_writers/` and trust the result. Worth a constants module `src/traceml/aggregator/sqlite_writers/_schema_meta.py` exporting every table name as a constant; renderers import the constant. Then a test asserts no two tables collide and that every renderer references at least one table constant. Not yet written.

- **`_REMOTE_STORE_SAMPLERS` is a frozenset of strings rather than a derived value.** A new layer-sampler projection writer landing without removing its sampler from the frozenset means rows are double-written (in-memory + SQLite) until someone notices. Author guide §11.1 documents the migration as a two-PR sequence; the reviewer has no automated check that PR 2 actually lands. Worth a follow-up issue tracking the migration of each `Layer*Sampler` in turn.

- **The verdict criteria (§8) are folklore-level.** "Affects metric correctness at the projection layer" is the bright line for blocking, but reasonable people disagree about edge cases — e.g. is a `SCAN` on a query that runs once at session end (final-summary path) a blocker or a follow-up? A formal list of "schema-correctness invariants the project commits to" would resolve these arguments before the PR.

- **No reviewer-side smoke harness.** A reviewer who wants to run the §6.1 recipe needs a working aggregator + sampler stack and a small training script. A `tests/review_harness/sqlite_smoke/` directory with a parametrised fixture (`@pytest.fixture` that boots the aggregator, runs a 10-step minimal training, returns the DB path) would make recipes 5 lines instead of 15. Not yet written.

These gaps are the natural follow-up work after this guide lands. None block the next reviewer from following the workflow above; all would make the next reviewer faster.
