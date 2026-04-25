# How to review a renderer PR

> Internal contributor guide. Audience: one trusted co-founder / new engineer
> reviewing TraceML PRs. Companion to `add_renderer.md`. Not for public docs.

This guide teaches you how to review a PR that adds or modifies a renderer in `src/traceml/renderers/`. It assumes you have already read `add_renderer.md` (the author's guide) and have a working mental model of [W9 (aggregator core)][W9] and [W10 (display drivers + renderers)][W10]. The seven-step meta-workflow is identical to the one in `review_patch.md`; only the consistency table (§3) and the failure-mode catalogue (§4) are renderer-specific.

---
Feature type: renderer
Risk level: medium (read-only by contract; failures are visual, not silent metric corruption — but a slow renderer can stall the dashboard tick and a misregistered renderer ships a broken release)
Cross-cutting impact: aggregator process only (training side untouched)
PyTorch coupling: none
Reference exemplars: `StepCombinedRenderer` (full split), `StepMemoryRenderer` (full split with diagnostics), `StdoutStderrRenderer` (flat), `SystemRenderer` (split + dashboard)
Companion author guide: `add_renderer.md`
Last verified: 2026-04-25
---

## 1. The meta-review-workflow

Same seven steps as `review_patch.md`. Don't reinvent the workflow.

1. **Anchor** the PR diff to [W9][W9] / [W10][W10] and to `add_renderer.md`. Read the PR through the existing renderer family, not line-by-line.
2. **Run the renderer-family consistency check.** Build the table from §3 and grade the new renderer against the four exemplars (`StepCombinedRenderer`, `StepMemoryRenderer`, `StdoutStderrRenderer`, `SystemRenderer`).
3. **Apply the renderer-class failure-mode catalogue** in §4. Each maps to a known bug shape — walk the diff with each in mind.
4. **Apply the four meta-questions** from §5: new axis of variation, shared-infrastructure interaction, layout-section as contract, invariant preservation.
5. **Write a verification gate** for every concern (§6). 3–10 line repro recipe with a clear pass/fail.
6. **Draft comments at the right granularity** (§7). Line comment for code suggestions, PR-level for architectural concerns. Holdback discipline applies.
7. **Land the verdict** (§8). Approve / approve-with-changes / block.

The reviewer ends with a 2–3 sentence executive summary (§7.4) that the maintainer can read without opening the diff.

---

## 2. Step 1 — Anchor the PR to your walkthroughs

Don't open the diff first. Open [`traceml_learning_code_walkthroughs.md`][W10] and re-read W10 §"Renderer compute / schema / render split" and §"Display drivers compose renderers." The renderer family has documented invariants — read-only data source, dataclass payload to dashboard, `BaseRenderer.layout_section_name` as the registration key, stale-cache TTL discipline. Those need to be in cache before you read the diff.

A renderer PR typically touches 4–9 files in stereotyped ways. Map each one before reading deeply.

### How to anchor

| File pattern | W-section | What kind of change should this be? |
|---|---|---|
| `src/traceml/renderers/<name>/schema.py` (NEW) | [W10 §"Compute / schema split"][W10] | Frozen dataclasses. The compute contract. |
| `src/traceml/renderers/<name>/compute.py` (NEW) | [W10 §"SQLite-backed compute"][W10] | Short-lived SQLite connection, bounded read, stale-cache fallback. |
| `src/traceml/renderers/<name>/renderer.py` (NEW) | [W10 §"BaseRenderer subclass"][W10] | `BaseRenderer` subclass. CLI panel + (optional) dashboard payload. |
| `src/traceml/aggregator/display_drivers/layout.py` | [W10 §"Layout constants"][W10] | One-line constant addition. |
| `src/traceml/aggregator/display_drivers/cli.py` | [W10 §"CLI driver"][W10] | Renderer added to `_renderers`; layout node added in `_create_*_layout`; "Waiting for…" placeholder in `_create_initial_layout`. |
| `src/traceml/aggregator/display_drivers/nicegui.py` | [W10 §"NiceGUI driver"][W10] | Renderer appended to `_renderers`. (No section logic here — that's in `nicegui_sections/`.) |
| `src/traceml/aggregator/display_drivers/nicegui_sections/<name>_section.py` (NEW) | [W10 §"NiceGUI sections"][W10] | `build_*_section()` + `update_*_section()` pair. |
| `src/traceml/aggregator/display_drivers/nicegui_sections/pages.py` | [W10 §"pages.py wiring"][W10] | `subscribe_layout(...)` call + (sometimes) `ensure_ui_timer(...)`. |
| `tests/test_<name>_renderer.py` (NEW) | none directly | Compute + panel surface coverage. |

If a file in the diff doesn't fit the table, that's a flag. Renderers that mutate `RemoteDBStore`, write SQLite, or touch sampler internals are red flags — the renderer contract is **read-only**. Ask why before continuing.

After anchoring you should have: one schema file (skim — the contract), one compute file (read deeply — the bug surface), one renderer file (read tightly — the registration surface), plus 3–5 mechanical files. PR_87's collapse trick (one substantive file, the rest mechanical) applies here too.

[W9]: ../deep_dive/code-walkthroughs.md#w9-aggregator-core-tcp-receive-frame-dispatch-sqlite-writes
[W10]: ../deep_dive/code-walkthroughs.md#w10-display-drivers-renderers-terminal-and-web-ui-from-sql

---

## 3. Step 2 — The renderer-family consistency table

Every renderer slots into a small set of axes. Fill in the new column and grade each cell against the existing four.

### 3.1 The columns (these don't change PR-to-PR)

| Axis | What it asks | Where to verify |
|---|---|---|
| **Module structure** | Single-file flat (`stdout_stderr_renderer.py`) or split (`<name>/{schema,compute,renderer}.py`)? | `src/traceml/renderers/`. |
| **Data source** | SQLite `db_path` (preferred, all new) or `RemoteDBStore` (legacy, layer renderers only)? | Constructor signature + `compute.py` imports. |
| **`BaseRenderer.name`** | Set in `super().__init__(name=..., layout_section_name=...)`? | Renderer's `__init__`. |
| **`layout_section_name`** | Matches a constant in `aggregator/display_drivers/layout.py`? | Renderer's `super().__init__`; cross-check `layout.py`. |
| **CLI medium** | `get_panel_renderable()` returning a `Panel`? | Renderer class body. |
| **Dashboard medium** | `get_dashboard_renderable()` returning a typed payload (frozen dataclass)? | Renderer class body. |
| **Notebook medium** | `get_notebook_renderable()` (rare, mostly stubs)? | Renderer class body. |
| **Profile gating** | `watch` / `run` / `deep` — registered in `CLIDisplayDriver._renderers` and/or `NiceGUIDisplayDriver._renderers`? | `aggregator/display_drivers/cli.py` + `nicegui.py`. |
| **Compute connection lifetime** | Short-lived SQLite connection per call (`with sqlite3.connect(self._db_path) as conn:`)? | `compute.py::_compute`. |
| **Bounded read** | `LIMIT window_size * lookback_factor`, never `SELECT * FROM ...` without LIMIT? | The actual SQL string in `compute.py`. |
| **Group in Python, not SQL** | When stale-cache fallback matters, group rows in Python so the empty-DB path is identical to the populated path? | `compute.py` — look for `defaultdict` over fetched rows. |
| **Stale-cache TTL** | `_last_ok` + `_last_ok_ts` + `stale_ttl_s` discipline? Returns last-good when fresh compute fails? | `compute.py::_stale_or_empty` (or equivalent). |
| **Empty-data path** | Explicit `if payload.is_empty: return waiting_panel`? Doesn't index into empty lists? | Renderer's `get_panel_renderable`. |
| **Frozen dataclass payload** | `@dataclass(frozen=True)` on every schema class? Computer constructs new instances rather than mutating? | `schema.py`. |
| **Read-only invariant** | No mutation of `RemoteDBStore`, no SQLite writes, no sampler calls? | Grep `compute.py` and `renderer.py` for `add_record`, `ingest`, `INSERT`, `UPDATE`, `DELETE`. |
| **Threading** | Renderer runs on aggregator thread; doesn't block the tick (< 100 ms)? | Compute body — no network, no `time.sleep`, no `subprocess.run`. |
| **NiceGUI section (if dashboard)** | Builder + updater under `nicegui_sections/`? `subscribe_layout(...)` wired in `pages.py`? Per-client `ui.timer` ≥ 0.5 s? | `nicegui_sections/<name>_section.py` + `pages.py`. |
| **Adaptive width** | `cols, _ = shutil.get_terminal_size(); width = min(max(100, int(cols * 0.75)), 120)` formula? | Renderer's `get_panel_renderable`. |
| **Color semantics** | Domain-aware (high GPU util = green/good; high RAM = red/bad)? Doesn't blind-copy from another renderer? | Helper functions like `_util_color`. |

### 3.2 The current state (April 2026)

| Axis | `StepCombinedRenderer` | `StepMemoryRenderer` | `StdoutStderrRenderer` | `SystemRenderer` |
|---|---|---|---|---|
| Module structure | split (4 files + diagnostics) | split (5 files + diagnostics) | flat (single file) | split (computer + cli/dashboard compute) |
| Data source | SQLite (`step_time_samples`) | SQLite (`step_memory_samples`) | SQLite (`stdout_stderr` table via `StdoutStderrDB`) | SQLite (`system_samples`, `system_gpu_samples`) |
| `BaseRenderer.name` | `"Model Step Summary"` | `"Model Step Memory"` | `"Stdout/Stderr"` | `"System"` |
| `layout_section_name` | `MODEL_COMBINED_LAYOUT` | `MODEL_MEMORY_LAYOUT` | `STDOUT_STDERR_LAYOUT` | `SYSTEM_LAYOUT` |
| CLI medium | YES (Rich `Panel` + `Group` + `Table`) | YES (Rich `Panel` + `Group` + `Table`) | YES (Rich `Panel` + `Text`) | YES (Rich `Panel` + `Table.grid`) |
| Dashboard medium | YES (`StepCombinedTimeResult`) | YES (`StepMemoryCombinedResult`) | NO | YES (dict, not dataclass — legacy) |
| Notebook medium | NO | NO | placeholder HTML | NO |
| Profile gating | run + deep (CLI), all (dashboard) | run + deep (CLI), all (dashboard) | watch + run + deep (CLI only) | all profiles, both drivers |
| Compute connection | per-call (`_connect()` / `with`) | per-call | per-call (via `StdoutStderrDB`) | per-call |
| Bounded read | YES (`LIMIT window_size * lookback_factor`) | YES | YES (`LIMIT display_lines`) | YES (`LIMIT window_n`) |
| Group in Python | YES (after fetch) | YES | N/A (single-shot read) | YES |
| Stale-cache TTL | YES (`stale_ttl_s=30.0`, `_stale_or_empty`) | YES | NO (line buffer is naturally low-noise) | NO (CLI sample is single-row, dashboard windowed) |
| Empty-data path | "Waiting for first fully completed step…" | "Waiting for first fully completed step…" | "Waiting for stdout/stderr…" | "Not available" / "—" cells |
| Frozen dataclass | YES (5 dataclasses, all frozen) | YES (5 dataclasses, all frozen) | YES (`StdoutStderrLine`) | NO (returns dict — pre-existing) |
| Read-only | YES | YES | YES | YES |
| Threading | OK (< 5 ms typical) | OK | OK | OK |
| NiceGUI section | `model_combined_section.py` | `step_memory_section.py` | N/A | `system_section.py` |
| Adaptive width | `min(max(100, int(cols*0.75)), 120)` | `min(max(100, int(cols*0.75)), 120)` | none (no width clamp) | `min(max(100, int(cols*0.75)), 120)` |
| Color semantics | red WAIT share; magenta labels | magenta labels; dim footer | dim "waiting" | green CPU/RAM (good); red unavailable |

When reviewing, **add a column** for the new renderer and walk every row. Three outcomes per cell:

- ✅ Matches the family — note it and move on.
- ❌ Differs from the family — demand a justification in the PR description or an inline comment in the renderer module. Returning a dict instead of a frozen dataclass is the canonical case where "matches `SystemRenderer`" is **not** a free pass; new renderers should follow `StepCombinedRenderer`.
- ⚠ Cell undecidable from the diff — ask the author.

### 3.3 The table is the most reusable artifact in this guide

Every future renderer review should rebuild this table. The act of filling it forces you to read the renderer with the family in mind. Empty cells are questions, not free passes. Long-term, this table should become a `tests/test_renderer_family.py` introspection test (see §11 gaps).

---

## 4. Step 3 — Renderer-class failure modes

Walk the diff with each of these in mind. Each has bitten us at least once or is a plausible bite given the architecture.

### 4.1 Mutation of data source

Applies to: every renderer.

The bug shape: `compute.py` calls `db.add_record(...)`, `remote_store.ingest(...)`, or executes `INSERT` / `UPDATE` against the SQLite projection. Renderers MUST be read-only — anything else corrupts the telemetry view for every other consumer (final summary, other renderers, future replay).

**What to check:**

- Grep the new renderer module for `add_record`, `ingest`, `INSERT`, `UPDATE`, `DELETE`, `executescript`.
- The compute should only `conn.execute("SELECT ...")`. Even `CREATE INDEX` belongs in the SQLite writer's `init_schema()`, never in a renderer.
- If the renderer needs an aggregation that the projection doesn't support, the fix is to extend the projection writer (separate PR), not to recompute in the renderer.

### 4.2 Blocking I/O in `get_*_renderable`

Applies to: every renderer.

The renderer runs on the aggregator's tick thread. The dashboard tick is rate-limited to `render_interval_sec` (default 2.0 s); the per-tick budget is < 100 ms total across all renderers (see [principles.md §5][principles]). Network calls, `time.sleep`, `subprocess.run`, `requests.get`, or `urllib.request.urlopen` stall the whole UI.

**What to check:**

- Grep `compute.py` and `renderer.py` for `requests`, `urllib`, `httpx`, `subprocess`, `time.sleep`, `socket`.
- Any new external dependency in `compute.py` is a flag.
- If the renderer genuinely needs a slow computation (e.g. building a large heatmap), profile it. Verification gate: `python -c "import time; t=time.perf_counter(); r.get_panel_renderable(); print((time.perf_counter()-t)*1000, 'ms')"` — must be < 50 ms typical, < 100 ms p99.

### 4.3 Missing empty-data path

Applies to: every renderer.

Cold start = empty SQLite. The renderer must not crash, must not raise, and must produce something visible. Common bug: `payload.metrics[0].coverage.world_size` without checking `if not payload.metrics`. PR_87-style verification gate: launch `traceml watch examples/mnist.py --mode cli` and confirm the panel reads "Waiting for…" within the first tick, before any sampler payload arrives.

**What to check:**

- Renderer's `get_panel_renderable()` opens with `if payload is None or not payload.metrics: return Panel("Waiting for ...", title=...)` (or equivalent for the renderer's data shape).
- Dashboard's `update_*_section()` opens with `if payload is None or payload.is_empty: return` — leaves the initial UI intact.
- Schema dataclass exposes `is_empty` as a property. Don't make the section guess.
- Test fixture covers the empty case (`test_compute_empty_db`, `test_panel_empty`).

### 4.4 Profile / layout-section registration bugs

Applies to: every new renderer.

The bug shape (from `add_renderer.md` pitfall #6): module exists, tests pass, nothing shows on the dashboard. Five places must agree:

1. `aggregator/display_drivers/layout.py` declares the `*_LAYOUT` constant.
2. The renderer's `super().__init__(layout_section_name=...)` references it.
3. `CLIDisplayDriver._create_*_layout` includes a `Layout(name=*_LAYOUT, ratio=...)` node in the right profile's tree.
4. `CLIDisplayDriver._create_initial_layout` seeds a "Waiting for…" placeholder for the section.
5. `CLIDisplayDriver._renderers` (and/or `NiceGUIDisplayDriver._renderers`) instantiates the renderer in the right profile branch.

`CLIDisplayDriver._register_once` (`cli.py:288`) logs `"CLI layout section not found: ..."` if (1) and (3) disagree, but the failure is silent in the panel. A renderer that lands in `_renderers` but whose layout name doesn't match any `Layout(name=...)` simply doesn't render — no error, no warning.

**What to check:**

- All five places touched in the same PR. If only one is missing, the renderer is dead.
- The `*_LAYOUT` string value is unique. Two renderers with the same `layout_section_name` → last update per tick wins (see "Gaps" in `add_renderer.md`); not exercised today, don't ship it.
- Profile branch is correct: `watch` is intentionally compact (only `SystemRenderer`, `ProcessRenderer`, `StdoutStderrRenderer`); `deep` adds the layer renderers; everything else lives in `run`. Putting a heavy renderer in `watch` is a smell.

### 4.5 NiceGUI threading bugs

Applies to: any renderer with a dashboard section.

The threading model (see `nicegui.py:1-40`):

- Aggregator thread calls `tick()` → `update_display()`. The renderer's `get_dashboard_renderable()` runs here. **No UI calls allowed.**
- UI thread runs `_ui_update_loop()` via `ui.timer(...)`. Section updaters run here. **All UI mutations belong here.**
- `_latest_data_lock` is held only for the snapshot/swap, never across compute or UI work.

The bug shape: section updater touches widgets from the aggregator thread, or the renderer's `get_dashboard_renderable()` calls `ui.label.text = ...`. Symptom: dashboard freezes after first tick or randomly drops updates.

**What to check:**

- Renderer's `get_dashboard_renderable()` returns a frozen dataclass and does nothing else. No `ui.*` references.
- Section updater (`update_*_section`) only mutates passed-in `cards` dict widgets. No `_safe`-style retry loops, no global state.
- `subscribe_layout(...)` is called inside `pages.py::define_pages` (under a NiceGUI client context), not from the aggregator side.
- `ensure_ui_timer(...)` is called once per page (deduped per-client by `_timer_clients`). Forgetting it = renderer freezes after one tick.
- Per-client timer interval ≥ 0.5 s. The author's guide settled on **0.75 s** as load-tested; layer pages use 1.0 s. Below 0.5 s causes browser lag and aggregator backpressure.

### 4.6 Plotly figure churn

Applies to: any NiceGUI section using `ui.plotly`.

The bug shape (`add_renderer.md` pitfall #7): `update_*_section` constructs `go.Figure()` every tick. At 0.75 s cadence, that's 4800 figures per hour. Garbage collection pressure increases, the Plotly DOM remounts, the chart flickers.

**What to check:**

- `build_*_section()` constructs `fig = go.Figure()` once. Multi-trace charts call `fig.add_trace(...)` for each series at build time, with empty `x=[], y=[]`.
- `update_*_section()` mutates `fig.data[i].x = ...; fig.data[i].y = ...` in place, then `panel["plot"].update()`. Or, when traces are dynamic, sets `fig.data = ()` and re-adds (still cheap because the figure object is reused — see `gpu_utilization_section` example in `add_renderer.md` §4 step 6).
- The section dict carries the figure handle (`{"_fig": fig, ...}`) so the updater doesn't have to find it again.

### 4.7 Slow SQLite queries

Applies to: any renderer with a `compute.py`.

The bug shape: unbounded `SELECT * FROM <table>` or a `WHERE` that doesn't hit an index. As the table grows over a multi-hour run, the renderer's compute time grows linearly. Eventually the tick budget blows.

**What to check:**

- Every `conn.execute("SELECT ...")` has a `LIMIT`. The pattern is `LIMIT window_size * lookback_factor` (see `step_time/compute.py::_load_last_steps`).
- The `WHERE` and `ORDER BY` columns are covered by an index in the writer's `init_schema()`. Cross-check `aggregator/sqlite_writers/<sampler>.py`.
- Verification gate (real session, real DB):

```
sqlite3 logs/<session>/aggregator/telemetry "EXPLAIN QUERY PLAN <renderer's actual SQL>"
# Pass: output starts with SEARCH ... USING INDEX
# Fail: output starts with SCAN ...
```

If the renderer's query SCANs, either add the index in the projection writer (separate PR), or restructure the query. SCAN is acceptable only for tiny tables (< 1k rows; `stdout_stderr` is on the edge).

### 4.8 Cross-rank handling

Applies to: any renderer reading per-rank rows.

The bug shape (`add_renderer.md` pitfalls #4 and #5): the renderer reads per-rank rows and dumps them all into the table without aggregation. World size 8 = eight times the rows, all visually duplicated. Or the renderer hard-codes `WHERE rank = 0` without saying why.

The aggregation policies the codebase already uses:

- **Per-rank rows** (with rank as a column) — `LayerCombinedTimeRenderer`. Acceptable when the user wants to see imbalance.
- **"Worst-rank wins" + skew** — `StepCombinedRenderer`, `StepMemoryRenderer`. The window-summed worst row tells the story.
- **Rank 0 only** — `StdoutStderrRenderer`. Acceptable because user logs are typically rank-replicated (DDP all-ranks logging is rare and a different story).

**What to check:**

- The PR description states which policy applies and why.
- If "worst-rank wins," `worst_rank` is in the schema and the renderer surfaces it (column or row). `StepCombinedRenderer.get_panel_renderable` is the model.
- If "rank 0 only," the constructor takes `rank: int = 0` with a docstring; `STDOUT_STDERR (RANK 0)` in the panel title is the convention.
- `rank_filter` semantics consistent with `StepCombinedComputer.rank_filter` (a `set[int]` or `None` for all). The renderer family currently inconsistently uses `rank_filter` / `rank` / hard-coded `0` (see `add_renderer.md` Gaps); pick one and document.

### 4.9 `print()` in a renderer

Applies to: every renderer.

The bug shape (`add_renderer.md` pitfall #12): `print(...)` in a renderer. The CLI driver redirects stdout into the log panel via `StreamCapture`; the print lands inside `stdout_stderr` telemetry, which then gets re-rendered, which then gets re-captured… not actually a loop, but the user sees their own debug noise inside the dashboard.

**What to check:**

- Grep the renderer module for `print(`. Zero allowed.
- All logging via `self._logger = get_error_logger("<Name>Renderer")` from `traceml.loggers.error_log`. See [principles.md §4][principles].
- Stack traces from caught exceptions go to `self._logger.exception(...)`, not to stderr directly.

### 4.10 Frozen dataclass mutation

Applies to: any renderer with `@dataclass(frozen=True)` schemas.

The bug shape: the computer wants to "tag" the result with a status message, writes `result.status_message = "STALE"`, raises `dataclasses.FrozenInstanceError` at runtime. The dashboard goes blank for the next tick.

**What to check:**

- Computers construct **new** instances when they need to alter a field. `StepCombinedComputer._stale_or_empty` is the model: it builds a fresh `StepCombinedTimeResult` carrying the cached metrics + a new status string.
- Same applies to nested dataclasses. `StepCombinedTimeMetric` reconstruction in `step_time/compute.py:260-274` is the worked example (the WAIT-step's `worst_rank` substitution).
- `Optional[T]` fields are fine to leave `None`; the bug only fires on mutation.

### 4.11 Wire-name/event-key drift in the compute

Applies to: renderers consuming `events_json` blobs (currently only `step_time/compute.py`).

The bug shape: the patch family emits `_traceml_internal:forward_time`; the compute's `EVENT_ALIASES` table in `step_time/compute.py:24` maps `"forward" -> "_traceml_internal:forward_time"`. If a new patch ships a new wire-name and the alias table isn't updated, the metric silently reads zero.

**What to check:**

- Whenever a patch PR introduces a new `_traceml_internal:*` event name, the renderer review should confirm `EVENT_ALIASES` (or its analogue) was updated. Cross-link `review_patch.md` §5.3 (wire-name as contract).
- The renderer's `metric_keys` default includes the new name where applicable.
- Future-proofing: the central registry (`utils/event_names.py`) doesn't exist yet (see `review_patch.md` Gaps); reviewer enforces by grep for now.

### 4.12 Bypassing `_safe` wrapping

Applies to: rare, but worth a check.

The bug shape (`add_renderer.md` pitfall #10): a section's `pages.py` block calls a renderer directly (not via `subscribe_layout`/`register_layout_content`), so when the renderer raises, the whole UI thread takes the exception. `CLIDisplayDriver._update_all_sections` and `NiceGUIDisplayDriver._ui_update_loop` both catch per-section exceptions and write a "Render Error" / "Could not update" panel — but only if the renderer goes through them.

**What to check:**

- New renderer is registered through the binding system (`_renderers` list + `_register_once`), not invoked manually anywhere.
- New section in `pages.py` uses `subscribe_layout(...)`; no direct calls to `renderer.get_dashboard_renderable()` from the page builder.

---

## 5. Step 4 — The four meta-questions

Apply each to the PR and write the answer down. If you can't answer, ask.

### 5.1 Does this PR introduce a new axis of variation?

The existing four exemplars cover: SQLite vs RemoteDBStore data source, flat vs split structure, per-rank vs aggregated rendering, CLI-only vs both mediums, single-row vs windowed compute. If the new renderer doesn't fit any cell — for example, the first renderer to read from a non-projected sampler, the first renderer to push to an alert channel, the first renderer to support live mutation of its own state — that's a new axis.

**Reviewer move:** when the new renderer has a column in the consistency table (§3) that no prior renderer fills, enumerate the failure modes that column creates. The table doesn't yet have a "writes to disk" axis; that's not because no one has tried, it's because nobody has been allowed to. If the PR introduces one, the failure-mode enumeration is the gate.

### 5.2 Does it interact with shared infrastructure?

Three pieces of shared infrastructure in the renderer family:

- **SQLite database file** — every renderer's compute opens its own short-lived connection, but they share the same file. SQLite's WAL handles concurrent readers fine; concurrent writers don't exist in this codebase (only the projection writers and the writer thread). Verify the new renderer is reader-only. If the PR adds a new index in `init_schema()`, that's a separate (welcome) change and should go in via the projection writer's PR.
- **`RemoteDBStore`** — single-threaded ingestion on rank 0; safe because ingest and display tick run serially in `TraceMLAggregator._loop`. New renderers should not be added to `_REMOTE_STORE_SAMPLERS`; the long-term direction is SQLite for everything (see `add_renderer.md` §1).
- **Aggregator tick thread budget** — total renderer compute per tick must be < 100 ms (principles §5). The new renderer's contribution adds to this.

**Reviewer move:** estimate the renderer's per-tick compute time. Ask the author to paste a `time.perf_counter()` measurement on a representative session. If it's > 50 ms, ask why.

### 5.3 Is `layout_section_name` a contract?

Yes. Once a renderer ships in a release with `layout_section_name = "foo_section"`, any section name change is a wire-style migration: dashboards in the wild expect that string, and the CLI layout YAML/Python references it. Same logic as `review_patch.md §5.3` for `_traceml_internal:*` names.

**Reviewer move:** every PR introducing a new `*_LAYOUT` constant must:

- Pick a name that **describes what is rendered**, not what was intended (cross-rank consistency: `model_combined_section` describes content; `step_time_v2` is a release artifact, not a description).
- Confirm no collision with existing `*_LAYOUT` constants. Grep `aggregator/display_drivers/layout.py`.
- Avoid suffixing version numbers — there's no story for `model_combined_section_v2` cohabiting with `model_combined_section`.

### 5.4 Which renderer-family invariants does the PR preserve?

The renderer-family invariants:

1. **Read-only.** No writes to any data source.
2. **Fail-open.** `get_panel_renderable()` never raises into the driver; one renderer crash doesn't take down the others. The drivers wrap calls in `_safe` / try-except, but the renderer should be defensive too (catch `Exception` around the SQLite read).
3. **Empty-data tolerant.** Cold start, partial-data, and stale-after-failure all produce a visible non-crashing panel.
4. **Bounded compute.** Every SQLite read has a `LIMIT`; every cross-rank loop is bounded by `len(ranks)` not `len(samples)`.
5. **Frozen dataclass payloads.** Schema classes are `@dataclass(frozen=True)`. New instances on every tick.
6. **Single-shot connection.** Compute opens a SQLite connection per call, closes it on exit. Never holds a long-lived connection on the aggregator thread.
7. **Stale-cache TTL where applicable.** Windowed renderers (anything cross-rank) cache `_last_ok` with a 30 s TTL to absorb transient incompleteness without flicker.
8. **No mutation of widgets from aggregator thread.** Dashboard renderers return data; sections render it.

**Reviewer move:** for each invariant, point at the line of the diff that preserves (or could break) it. If you can't, you don't yet understand the PR well enough to approve it.

---

## 6. Step 5 — Verification gates

Every concern in the review must come with a concrete reproduction recipe. Same shape as `review_patch.md §6`.

### 6.1 Worked example — "no SCAN in renderer query"

```
# Setup
git -C /teamspace/studios/this_studio/traceml checkout pr-XXX
traceml run examples/mnist.py --mode cli   # let it run for ~60 s, then Ctrl-C
SESSION=$(ls -td logs/*/ | head -1)

# Command
sqlite3 "${SESSION}aggregator/telemetry.sqlite" \
    "EXPLAIN QUERY PLAN SELECT step, events_json FROM step_time_samples \
     WHERE rank = 0 ORDER BY step DESC, id DESC LIMIT 400"

# Pass criterion
# Output starts with: SEARCH step_time_samples USING INDEX ...
# Fail criterion:
# Output starts with: SCAN step_time_samples
```

### 6.2 Worked example — "renderer < 50 ms p99 on a real session"

```
# Setup
traceml run examples/mnist.py --mode cli   # let it run for ~60 s
SESSION=$(ls -td logs/*/ | head -1)

# Command — save as bench.py
import time, statistics
from traceml.renderers.<new>.renderer import <NewRenderer>
r = <NewRenderer>(db_path=f"{SESSION}aggregator/telemetry.sqlite")
ts = []
for _ in range(100):
    t0 = time.perf_counter()
    r.get_panel_renderable()
    ts.append((time.perf_counter() - t0) * 1000)
print(f"p50={statistics.median(ts):.1f}ms p99={sorted(ts)[98]:.1f}ms")

# Pass: p99 < 50 ms.
# Fail: p99 >= 100 ms.
```

### 6.3 Worked example — "empty DB cold start doesn't crash"

```
# Setup
rm -f /tmp/empty_telemetry.sqlite
python -c "
import sqlite3, sys
from traceml.aggregator.sqlite_writers.<projection_writer> import init_schema
conn = sqlite3.connect('/tmp/empty_telemetry.sqlite')
init_schema(conn)
conn.close()
"

# Command
python -c "
from rich.panel import Panel
from traceml.renderers.<new>.renderer import <NewRenderer>
p = <NewRenderer>(db_path='/tmp/empty_telemetry.sqlite').get_panel_renderable()
assert isinstance(p, Panel), f'expected Panel, got {type(p)}'
print('OK: cold-start panel is a Panel')
"

# Pass: prints "OK: ...". No traceback, no exception.
# Fail: any traceback.
```

### 6.4 Worked example — "registration parity"

```
# Command
grep -n "<NewRenderer>" \
    src/traceml/aggregator/display_drivers/cli.py \
    src/traceml/aggregator/display_drivers/nicegui.py \
    src/traceml/aggregator/display_drivers/layout.py \
    src/traceml/aggregator/display_drivers/nicegui_sections/pages.py

# Pass: <NewRenderer> appears in cli.py (_renderers + _create_*_layout +
#       _create_initial_layout) AND nicegui.py (_renderers) AND
#       <NEW_LAYOUT> in layout.py AND subscribe_layout in pages.py.
# Fail: missing from any of the four.
```

### 6.5 When you can't write a verification gate

Same rule as `review_patch.md §6.2`. If you only have a vague worry, file a follow-up issue or hold it in your private parking lot. Don't waste author time on folklore.

### 6.6 Recipe style rules

- **Specific numbers, not adjectives.** `p99 < 50 ms` not "should be fast."
- **Reproducible from a clean checkout.** No "you also need fix Y first."
- **3–10 lines of actual code.**
- **State the GPU dependency.** Most renderers are CPU-only by virtue of reading from SQLite; the few that build GPU-shaped payloads (rare) need a CUDA box.

---

## 7. Step 6 — Drafting comments

Two granularity choices: line vs PR-level. They are not interchangeable.

### 7.1 Line comments

Use when there's a specific code change in a specific location.

Pattern: state the issue → propose the fix → reference a verification gate or precedent.

```
This `SELECT * FROM gpu_util_samples ORDER BY id DESC` has no LIMIT;
on a multi-hour session this scans hundreds of MB per tick.

Suggest `LIMIT ?` bound to `window_size * 64` (mirrors
StepCombinedComputer._load_last_steps, step_time/compute.py:402-415).

Verification gate: §6.1 — sqlite3 EXPLAIN QUERY PLAN must show
SEARCH ... USING INDEX, not SCAN.
```

### 7.2 PR-level comments

Use when the concern is **behavioural** or **architectural**, not localised to a single line.

Examples that belong PR-level:

- "This renderer returns a dict, not a frozen dataclass — every other new renderer in the family uses dataclasses (see `StepCombinedTimeResult`). What's the rationale?"
- "Cross-rank policy unstated: the panel shows per-rank rows but doesn't explain how that scales to world_size=8. Either aggregate (worst-rank wins) or document the per-rank choice in the docstring."
- "The new layout_section_name `foo_v2` suggests a future `foo_v3`. Wire-format-style names should describe content, not version."

### 7.3 Holdback discipline

Same as `review_patch.md §7.3`. Two kinds of items belong in your private parking lot, not the PR review:

- **Judgement calls about positioning** — "should this also surface in `watch`?" Decide privately.
- **Adjacent improvements** — "while we're here, the layout.py constants could move to an enum." If the improvement isn't required for the PR to ship, file a follow-up issue. Don't grow the PR.

A renderer review delivers a focused set of must-fix items. Bloating with parking-lot items dilutes the must-fix signal.

### 7.4 The maintainer summary

Every review ends with a 2–3 sentence executive summary:

> PR #N adds [renderer name] reading from [data source]. Architecture matches `[exemplar renderer]`; consistency table is K/L ✅. Review converged on M items: (1) ..., (2) ..., (3) .... All M fixes are localised. Recommend [verdict].

Maintainer reads three sentences and either agrees or opens the PR.

---

## 8. Step 7 — Landing the verdict

Three states. Pick exactly one.

### 8.1 Approve

Conditions:
- Consistency table (§3) is fully ✅ or has documented justified deviations.
- All four meta-questions (§5) answered explicitly.
- All eight invariants (§5.4) preserved.
- No concerns require a verification gate (§6).
- Tests cover empty-data, populated, and stale-fallback paths.
- Registration parity (§6.4) confirmed across all five sites.

### 8.2 Approve with changes

Conditions:
- All concerns are minor: docstring fixes, naming nits, color-palette suggestions, missing `is_empty` property, sub-100 ms perf.
- No concern affects metric correctness or breaks user-visible state.
- All concerns have a one-line fix.

This is the "accept the PR but require these N small changes."

### 8.3 Block (request changes)

Conditions (any one):
- A concern violates the read-only invariant (§4.1) — renderer mutates data.
- A concern blows the per-tick budget (§4.2 or §4.7) under realistic usage.
- A concern means the renderer doesn't show under any profile (§4.4 — registration broken).
- A concern silently drops cross-rank data (§4.8).
- A concern violates a renderer-family invariant (§5.4).
- Tests don't exist for the empty-data path.
- The PR introduces a new axis of variation without enumerating its failure modes (§5.1).

### 8.4 What "block" doesn't mean

Same as `review_patch.md §8.4`. Not "the architecture is wrong"; not "redesign." Means **these specific items must be resolved before merge.**

---

## 9. Reference: a hypothetical worked example

There is no PR_87-equivalent for renderers yet. Use the `add_renderer.md §4` `GpuUtilizationRenderer` walkthrough as the worked example for what a clean renderer PR looks like; map each step there to a §3 row and verify all eight §5.4 invariants hold.

When the first non-trivial renderer PR lands and gets reviewed, document that review here as `Notes/PR_XXX_renderer_review.md` and update this guide to point at it.

---

## 10. Common reviewer mistakes

Numbered, with cause and fix.

1. **Reviewing in isolation.** Cause: opening the diff before anchoring to W10 + `add_renderer.md`. Effect: drowning in 5–9 files. Fix: do §2 before §3 — every time.

2. **Re-deriving the consistency table from scratch.** Cause: the table isn't a formal artifact in source. Effect: each reviewer rebuilds it slightly differently. Fix: paste the §3.2 table into your review notes verbatim and add the new column.

3. **Approving a renderer that returns a dict instead of a frozen dataclass** because `SystemRenderer` does. Cause: "matches an exemplar" treated as a free pass. Effect: type-checking and section-update churn long-term. Fix: flag the deviation; new renderers should follow `StepCombinedRenderer`, not `SystemRenderer` (legacy).

4. **Skipping the registration-parity check.** Cause: the five sites span four files, easy to miss one. Effect: the renderer ships and silently doesn't render. Fix: §6.4 — grep is the gate.

5. **Trusting "tests pass" without running the dashboard locally.** Cause: unit tests cover compute correctness but don't catch registration bugs or layout misfits. Effect: works in tests, missing in production. Fix: §9 of `add_renderer.md` — `traceml watch` and `--mode dashboard` smoke tests are mandatory before approving.

6. **Missing the wire-event-name check on `step_time/compute.py` updates.** Cause: focusing on the renderer's structure, not its consumption of patch events. Effect: silent zero metrics when a new patch lands without an `EVENT_ALIASES` entry. Fix: §4.11 — grep for `EVENT_ALIASES` in any renderer review that lands alongside a patch PR.

7. **Approving on architecture without checking the SQLite query plan.** Cause: code looks clean, query looks short. Effect: SCAN ships; multi-hour session degrades. Fix: §4.7 + §6.1 — `EXPLAIN QUERY PLAN` is non-optional for any new SQL.

8. **Conflating "matches the family" with "correct."** Cause: §3 consistency check is ✅ across the board, so reviewer stops. Effect: novel-axis failure modes (§5.1) miss. Fix: every empty cell in the new column is a question, not a free pass.

9. **Treating dashboard-only and CLI-only as the same review.** Cause: skimming. Effect: missing the NiceGUI threading rules (§4.5) on a CLI-only renderer that stubs `get_dashboard_renderable`, or vice versa. Fix: read §4.5 every time a `nicegui_sections/*.py` file is touched.

10. **Skipping the maintainer summary (§7.4).** Cause: assumes the maintainer will read the diff. Effect: maintainer reads the diff, disagrees on severity, kicks back. Fix: three sentences are the maintainer's reading material.

---

## Gaps encountered while writing this guide

Where the reviewer's playbook is currently underspecified or relies on folklore. Flag these in your review process if you hit them.

- **The consistency table (§3) isn't a formal artifact.** It lives here and in `add_renderer.md §5`. If the renderer family grows beyond the current six, every reviewer will diverge on the column set. Worth lifting into a contract test in `tests/test_renderer_family.py` that introspects each renderer module and asserts (a) `BaseRenderer` subclass, (b) `layout_section_name` is one of the `layout.py` constants, (c) `get_panel_renderable` exists, (d) compute uses a short-lived connection. Not yet written.

- **No central registry of `*_LAYOUT` constants.** `aggregator/display_drivers/layout.py` is a flat list of strings; nothing prevents two renderers from claiming the same constant. Worth a frozen enum or a `pytest` collision check. Not yet written.

- **`SystemRenderer` and `ProcessRenderer` return dicts**, not frozen dataclasses. The reviewer rule "every new renderer returns a dataclass" is stronger than the existing code enforces. A migration PR for these two would close the gap; until then, the rule is folklore.

- **No reviewer-side perf harness.** A reviewer running §6.2 needs a real session with a non-trivial telemetry DB. A `tests/review_harness/seed_telemetry.py` script that pre-populates a SQLite file with realistic shapes (1k rows × 4 ranks × 60 minutes) would make perf gates 5 lines instead of 30. Not yet written.

- **Notebook renderables (`get_notebook_renderable`) are inconsistently implemented.** Some stub with HTML, some return None, most don't override. There's no driver consuming them today. Reviewer should ask the author to leave them unimplemented (defer to base class) rather than ship inconsistent stubs. Not enforced.

- **Cross-rank policy is currently per-renderer choice.** No central doc says "all step-level renderers use worst-rank wins; all log-level renderers use rank 0." `add_renderer.md` Gaps already flags this. Until resolved, the reviewer enforces "the PR description states which policy and why."

- **The "holdback discipline" (§7.3) has no checklist.** Knowing what belongs in the PR vs. a follow-up is currently a judgement call. Worth a short rubric: "must-fix iff the PR-as-merged would (a) violate read-only, (b) blow the tick budget under realistic usage, (c) silently drop a profile/section." Otherwise: follow-up.

These gaps are the natural follow-up work after this guide lands. None block the next reviewer from following the workflow above; all would make the next reviewer faster.

[principles]: principles.md
