# How to add a new renderer

This guide is task-oriented, concrete, and opinionated where the codebase has
an opinion. It assumes a working `pip install -e ".[dev,torch]"` checkout.

---

## 1. Intro and mental model

A renderer is TraceML's **presentation layer**. It reads telemetry the
aggregator has already ingested (SQLite history and/or `RemoteDBStore`),
reduces it to a small typed payload, and formats it for a human-visible
medium. Renderers live in `src/traceml/renderers/` and subclass
`traceml.renderers.base_renderer.BaseRenderer`.

!!! warning "Strictly read-only"
    Renderers MUST NOT mutate any data source. Never write back into
    `RemoteDBStore`, never update SQLite, never call into samplers. Cache on
    `self` (see `StepCombinedRenderer._payload`,
    `SystemCLIComputer._return_stale`). Breaking read-only is how weird
    intermittent bugs enter the telemetry path.

Renderers also MUST NOT block the aggregator tick thread. Target well under
100 ms per compute (the tick is rate-limited to `render_interval_sec`,
default 2.0 s).

### Three output mediums

1. **Rich terminal** — `get_panel_renderable()` → `CLIDisplayDriver`
   (`aggregator/display_drivers/cli.py`) wires it into Rich `Live`.
2. **NiceGUI + Plotly browser dashboard** — `get_dashboard_renderable()` →
   `NiceGUIDisplayDriver` (`aggregator/display_drivers/nicegui.py`) pairs
   the typed payload with a builder/updater under `nicegui_sections/`.
3. **IPython / HTML notebooks** — `get_notebook_renderable()`. Partial
   coverage: most renderers stub or return None. No driver currently
   consumes it in the main CLI path.

### Compute / schema / render split

Mature renderers (`src/traceml/renderers/step_time/`) split concerns:

| File             | Responsibility                                              |
|------------------|-------------------------------------------------------------|
| `schema.py`      | Frozen dataclasses — the compute contract                   |
| `compute.py`     | Reads data source, reduces to schema. No Rich, no UI.       |
| `renderer.py`    | `BaseRenderer` subclass. Formats for CLI and dashboard.     |
| `diagnostics.py` | (Optional) opinionated signals derived from the result      |

Why split? Compute is reused across mediums (CLI + dashboard) and becomes
independently testable (feed a temp SQLite DB, assert on the dataclass,
no Rich / NiceGUI required). Schemas stay stable while visuals churn.

!!! tip "When to skip the split"
    If the renderer does nothing beyond "fetch last N rows, format as
    text", a single `renderer.py` is fine (see `StdoutStderrRenderer`).
    Don't add ceremony for ceremony's sake.

### Data sources

- **SQLite projection tables** — the default. `SQLiteWriterSimple`
  persists every telemetry payload; per-sampler writers in
  `aggregator/sqlite_writers/` project into query-friendly tables
  (e.g. `SystemSampler` → `system_samples` + `system_gpu_samples`).
- **`RemoteDBStore`** — in-memory `{rank: {sampler: Database}}` maintained
  for a small allow-list in `TraceMLAggregator._REMOTE_STORE_SAMPLERS`
  (currently the four layer-time / layer-memory samplers). Used only where
  the sampler has no SQLite projection yet.

Long-term direction: **every new renderer reads from SQLite.**
`RemoteDBStore` remains for legacy layer renderers pending projection.

### How display drivers compose renderers

```python
# aggregator/trace_aggregator.py
_DISPLAY_DRIVERS: Dict[str, Type[BaseDisplayDriver]] = {
    "cli": CLIDisplayDriver,
    "dashboard": NiceGUIDisplayDriver,
    "summary": SummaryDisplayDriver,
}
```

Selected by the `--mode` flag (cli.py):

```
traceml watch my_script.py --mode cli        # Rich (default)
traceml run   my_script.py --mode dashboard  # NiceGUI on :8765
traceml deep  my_script.py --mode summary    # no live UI
```

Each driver owns its own renderer list, layout, and tick.

---

## 2. Before you start: decisions to make

Work through this before touching code.

- [ ] **Data domain** — does an existing sampler emit the rows you need? If
      not, add the sampler first (see `add_sampler.md`).
- [ ] **Live vs windowed** — live-only can skip the schema dataclass;
      windowed views need it.
- [ ] **Data source** — `remote_store` or `db_path`? (See §7 decision tree.)
- [ ] **Output mediums** — CLI / dashboard / both? Most target both;
      `ModelDiagnosticsRenderer` is dashboard-only.
- [ ] **Profile gating** — `watch` / `run` / `deep` / all? See
      `CLIDisplayDriver.__init__`.
- [ ] **Rank behavior** — per-rank, aggregated ("worst-rank wins"), or both?
      `StepCombinedComputer` is the reference for cross-rank joins.
- [ ] **Refresh cadence** — every tick or throttled via stale-TTL cache?
- [ ] **Graceful empty data** — what shows before first data arrives?

---

## 3. Anatomy of a renderer — `StepCombinedRenderer`

Walkthrough of `src/traceml/renderers/step_time/` — the most complete
compute/schema/render split in the codebase.

### `schema.py` — the compute contract

Frozen dataclasses, one shape per concern:
`StepCombinedTimeSeries` (per-step vectors), `StepCombinedTimeSummary`
(window scalars), `StepCombinedTimeCoverage` (rank bookkeeping),
`StepCombinedRankHeatmap` (dashboard-only). All wrapped in:

```python
@dataclass(frozen=True)
class StepCombinedTimeResult:
    metrics: List[StepCombinedTimeMetric]
    status_message: str = "OK"
    rank_heatmap: Optional[StepCombinedRankHeatmap] = None
```

Dataclasses (not dicts) give autocomplete, type checking, and fail-fast
construction. Dashboard-only fields are `Optional` so the CLI path doesn't
pay for heatmap cost.

### `compute.py` — the reduction

```python
def compute_cli(self) -> StepCombinedTimeResult:
    return self._compute(include_series=True, include_rank_heatmap=False)

def compute_dashboard(self) -> StepCombinedTimeResult:
    return self._compute(include_series=False, include_rank_heatmap=True)
```

Implementation highlights in `StepCombinedComputer`:

- **Short-lived SQLite connection per call** (`_connect()`) — sidesteps
  thread-affinity rules.
- **Bounded lookback** in `_load_last_steps`: at most
  `window_size * lookback_factor` rows per rank, `ORDER BY step DESC`.
  Never scan the full table.
- **Stale-cache fallback** (`_stale_or_empty`): keep the last good result
  for `stale_ttl_s` seconds so flaky ranks don't flicker the panel.
- **Pure reduction after the read**: numpy / plain Python, deterministic,
  easy to unit-test.

### `renderer.py` — the `BaseRenderer` subclass

```python
class StepCombinedRenderer(BaseRenderer):
    def __init__(self, db_path):
        super().__init__(name="Model Step Summary",
                         layout_section_name=MODEL_COMBINED_LAYOUT)
        self._computer = StepCombinedComputer(db_path=db_path)
        self._cached = None
```

Two critical things in the constructor:

1. `name` — human label (logs, optional titles).
2. `layout_section_name` — one of the constants in
   `aggregator/display_drivers/layout.py`. Mismatch → renderer silently
   skipped (`CLIDisplayDriver._register_once` logs and continues).

The two methods drivers call:

```python
def get_panel_renderable(self) -> Panel:
    payload = self._payload()
    if payload is None or not payload.metrics:
        return Panel("Waiting for first fully completed step...",
                     title="Model Step Summary")
    # ... build Rich Table ...
    return Panel(Group(diag_text, "", table, footer),
                 title=..., border_style="cyan", width=width)

def get_dashboard_renderable(self) -> StepCombinedTimeResult:
    return self._computer.compute_dashboard()
```

Flow:

```
driver.tick()
  -> renderer.get_*_renderable()
     -> computer.compute_*()
        -> SQLite read + reduction -> frozen dataclass
     -> format as Rich (CLI) | return dataclass (dashboard)
```

**The renderer owns the compute; the NiceGUI section owns the visuals.**

### Contrast: `StdoutStderrRenderer` (no split)

`stdout_stderr_renderer.py` is 103 lines. No `compute.py`, no `schema.py`.
Uses a thin `StdoutStderrDB` helper to fetch the last N lines and formats
them as Rich `Text`. Fine because: single medium, no reduction beyond
"last N rows", no cross-rank aggregation, no dashboard path. Use the split
whenever any of those is false.

---

## 4. Step-by-step: adding a new renderer

Walkthrough target: a hypothetical `GpuUtilizationRenderer` that visualizes
per-GPU utilization and memory pressure. It pairs with a hypothetical
`GpuUtilizationSampler` (not implemented; assume the schema below exists
via a SQLite projection writer).

Assume the projection table is:

```sql
CREATE TABLE gpu_util_samples (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    recv_ts_ns     INTEGER NOT NULL,
    rank           INTEGER,
    sample_ts_s    REAL,
    seq            INTEGER,
    gpu_idx        INTEGER NOT NULL,
    util_percent   REAL,
    mem_used_mb    REAL
);
CREATE INDEX idx_gpu_util_rank_idx_ts
    ON gpu_util_samples(rank, gpu_idx, sample_ts_s, id);
```

(The schema matches the row contract `{timestamp, gpu_idx, util_percent,
mem_used_mb}` — the projection writer would fill the additional bookkeeping
columns.)

### Step 1: Decide structure

This renderer has:

- a windowed reduction (last N samples, per GPU averages and peaks)
- both CLI and dashboard output
- light cross-rank logic (multi-GPU across ranks)

Use the full split. Create:

```
src/traceml/renderers/gpu_utilization/
    __init__.py
    schema.py
    compute.py
    renderer.py
```

### Step 2: Write `schema.py`

```python
# src/traceml/renderers/gpu_utilization/schema.py
"""Renderer-facing schema for GPU utilization. All shapes are frozen."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class GpuUtilizationRow:
    """One (rank, gpu_idx) row over the aggregation window."""
    rank: int
    gpu_idx: int
    util_avg: float
    util_peak: float
    mem_avg_mb: float
    mem_peak_mb: float
    samples: int


@dataclass(frozen=True)
class GpuUtilizationSeries:
    """Optional per-timestep series for dashboard charts."""
    ts: List[float]
    util_by_gpu: List[List[float]]  # [gpu_idx][t]
    mem_by_gpu: List[List[float]]


@dataclass(frozen=True)
class GpuUtilizationResult:
    """Final renderer-ready payload."""
    window_size: int
    samples_used: int
    rows: List[GpuUtilizationRow]
    series: Optional[GpuUtilizationSeries] = None
    status_message: str = "OK"

    @property
    def is_empty(self) -> bool:
        return not self.rows
```

Notes:

- `frozen=True` everywhere. Nobody mutates these in flight.
- `GpuUtilizationSeries` is optional and only populated for the dashboard
  path. The CLI path pays no allocation cost for it.
- `is_empty` is a convenience; dataclass properties are fine.

### Step 3: Write `compute.py`

```python
# src/traceml/renderers/gpu_utilization/compute.py
"""GPU utilization compute. Reads SQLite projection `gpu_util_samples`."""
import sqlite3, time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from traceml.loggers.error_log import get_error_logger
from .schema import (
    GpuUtilizationResult, GpuUtilizationRow, GpuUtilizationSeries,
)

TABLE = "gpu_util_samples"


class GpuUtilizationComputer:
    """Compute a window of GPU utilization statistics."""

    def __init__(self, db_path, window_size=100, stale_ttl_s=30.0):
        self._db_path = str(db_path)
        self._window_size = max(1, int(window_size))
        self._stale_ttl_s = stale_ttl_s
        self._logger = get_error_logger("GpuUtilizationComputer")
        self._last_ok: Optional[GpuUtilizationResult] = None
        self._last_ok_ts: float = 0.0

    def compute_cli(self):       return self._compute(include_series=False)
    def compute_dashboard(self): return self._compute(include_series=True)

    def _compute(self, *, include_series):
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                result = self._compute_impl(conn, include_series)
        except Exception:
            self._logger.exception("GPU utilization compute failed")
            return self._stale_or_empty()
        if result.is_empty:
            return self._stale_or_empty()
        self._last_ok, self._last_ok_ts = result, time.time()
        return result

    def _compute_impl(self, conn, include_series):
        # Bounded read; re-group in Python.
        rows = conn.execute(
            f"SELECT rank, gpu_idx, sample_ts_s, util_percent, mem_used_mb "
            f"FROM (SELECT * FROM {TABLE} ORDER BY id DESC LIMIT ?)",
            (self._window_size * 64,),
        ).fetchall()
        if not rows:
            return GpuUtilizationResult(self._window_size, 0, [], status_message="no data")

        grouped: Dict[Tuple[int, int], List] = defaultdict(list)
        for r in rows:
            if r["rank"] is None or r["gpu_idx"] is None: continue
            grouped[(int(r["rank"]), int(r["gpu_idx"]))].append(r)

        out_rows: List[GpuUtilizationRow] = []
        util_by_gpu: Dict[int, List[float]] = defaultdict(list)
        mem_by_gpu: Dict[int, List[float]] = defaultdict(list)
        ts_axis: List[float] = []
        samples_used = 0

        for (rank, gpu_idx), group in grouped.items():
            group = list(reversed(group))[-self._window_size:]
            if not group: continue
            utils = [float(g["util_percent"] or 0.0) for g in group]
            mems  = [float(g["mem_used_mb"]  or 0.0) for g in group]
            samples_used += len(group)
            out_rows.append(GpuUtilizationRow(
                rank=rank, gpu_idx=gpu_idx,
                util_avg=sum(utils)/len(utils), util_peak=max(utils),
                mem_avg_mb=sum(mems)/len(mems), mem_peak_mb=max(mems),
                samples=len(group),
            ))
            # Only rank 0 series; multi-rank overlays belong in a heatmap.
            if include_series and rank == 0:
                for g in group:
                    ts_axis.append(float(g["sample_ts_s"] or 0.0))
                    util_by_gpu[gpu_idx].append(float(g["util_percent"] or 0.0))
                    mem_by_gpu[gpu_idx].append(float(g["mem_used_mb"] or 0.0))

        out_rows.sort(key=lambda r: (r.rank, r.gpu_idx))
        series = None
        if include_series and util_by_gpu:
            order = sorted(util_by_gpu.keys())
            series = GpuUtilizationSeries(
                ts=sorted(set(ts_axis)),
                util_by_gpu=[util_by_gpu[g] for g in order],
                mem_by_gpu=[mem_by_gpu[g] for g in order],
            )

        return GpuUtilizationResult(
            window_size=self._window_size, samples_used=samples_used,
            rows=out_rows, series=series, status_message="OK",
        )

    def _stale_or_empty(self):
        if (self._last_ok is not None and self._stale_ttl_s is not None
                and (time.time() - self._last_ok_ts) <= self._stale_ttl_s):
            return self._last_ok
        return GpuUtilizationResult(self._window_size, 0, [], status_message="no fresh data")
```

Design notes:

- **Single bounded read.** We scan `window_size * 64` rows at most. On real
  workloads this is microseconds.
- **Group in Python, not SQL.** `GROUP BY` over the raw projection would work
  but makes the stale-handling branch more complex; a straight read keeps
  both paths identical.
- **Series only for rank 0.** Multi-rank overlays are a separate rendering
  decision. Dumping all ranks into the chart by default is the number-one
  way to make a "simple" dashboard unreadable.

### Step 4: Write `renderer.py`

```python
# src/traceml/renderers/gpu_utilization/renderer.py
"""GPU utilization renderer. CLI: Rich table. Dashboard: typed payload."""
import shutil
from typing import Optional

from rich.panel import Panel
from rich.table import Table

from traceml.aggregator.display_drivers.layout import GPU_UTILIZATION_LAYOUT
from traceml.renderers.base_renderer import BaseRenderer
from .compute import GpuUtilizationComputer
from .schema import GpuUtilizationResult


class GpuUtilizationRenderer(BaseRenderer):
    """Rank-aware GPU utilization renderer."""
    NAME = "GpuUtilization"

    def __init__(self, db_path: str, window_size: int = 100) -> None:
        super().__init__(name=self.NAME,
                         layout_section_name=GPU_UTILIZATION_LAYOUT)
        self._computer = GpuUtilizationComputer(db_path, window_size)
        self._cached: Optional[GpuUtilizationResult] = None

    def get_panel_renderable(self) -> Panel:
        payload = self._computer.compute_cli()
        if payload and not payload.is_empty:
            self._cached = payload
        payload = self._cached
        if payload is None or payload.is_empty:
            return Panel("Waiting for GPU utilization samples...",
                         title="GPU Utilization", border_style="cyan")

        table = Table(show_header=True, header_style="bold blue",
                      box=None, expand=False)
        table.add_column("Rank", justify="right", style="magenta")
        table.add_column("GPU",  justify="right", style="magenta")
        for label in ("Util avg", "Util peak", "Mem avg", "Mem peak"):
            table.add_column(label, justify="right")
        table.add_column("N", justify="right", style="dim")

        for r in payload.rows:
            c = _util_color(r.util_peak)
            table.add_row(
                f"r{r.rank}", f"{r.gpu_idx}",
                f"{r.util_avg:.1f}%", f"[{c}]{r.util_peak:.1f}%[/{c}]",
                f"{r.mem_avg_mb:.0f} MB", f"{r.mem_peak_mb:.0f} MB",
                f"{r.samples}",
            )

        cols, _ = shutil.get_terminal_size()
        width = min(max(100, int(cols * 0.75)), 120)
        return Panel(
            table,
            title=f"[bold cyan]GPU Utilization[/bold cyan] "
                  f"(used {payload.samples_used})",
            border_style="cyan", width=width,
        )

    def get_dashboard_renderable(self) -> GpuUtilizationResult:
        return self._computer.compute_dashboard()


def _util_color(peak: float) -> str:
    # High util = GPU busy = good. Inverted vs CPU/mem semantics.
    if peak >= 95.0: return "green"
    if peak >= 70.0: return "yellow"
    return "red"
```

!!! note "Color semantics are domain-specific"
    Unlike CPU/memory where high = bad, for GPU utilization high = good
    (the GPU is busy). The color map inverts accordingly. Don't copy color
    logic from another renderer blindly — think about what you're flagging.

### Step 5: Register with `CLIDisplayDriver`

In `aggregator/display_drivers/cli.py`, import the renderer + layout
constant, then add to the profile-gated list. For GPU util, `run` and
`deep` profiles fit (not `watch`, which is intentionally compact):

```python
from traceml.aggregator.display_drivers.layout import (
    ..., GPU_UTILIZATION_LAYOUT,
)
from traceml.renderers.gpu_utilization.renderer import GpuUtilizationRenderer

# Run profile
if not self._watch_profile:
    self._renderers += [
        StepCombinedRenderer(db_path=self._settings.db_path),
        StepMemoryRenderer(db_path=self._settings.db_path),
        GpuUtilizationRenderer(db_path=self._settings.db_path),  # NEW
    ]
```

`_update_all_sections` wraps renderer calls in try/except — if
`get_panel_renderable()` raises, the section shows a red "Render Error"
panel and other renderers keep working.

### Step 6: Register with `NiceGUIDisplayDriver`

**(a)** Append to `self._renderers` in
`aggregator/display_drivers/nicegui.py`:

```python
from traceml.renderers.gpu_utilization.renderer import GpuUtilizationRenderer
# ... inside __init__ ...
self._renderers += [GpuUtilizationRenderer(db_path=self._settings.db_path)]
```

`_register_once` auto-registers `layout_section_name ->
get_dashboard_renderable` for every entry. Each tick, `update_display`
calls `content_fn()` and stores results in `latest_data` under lock; the
UI thread reads the snapshot and dispatches to subscribers.

**(b)** Create a section builder/updater under
`aggregator/display_drivers/nicegui_sections/`:

```python
# nicegui_sections/gpu_utilization_section.py
"""NiceGUI GPU utilization card. Consumes GpuUtilizationResult."""
from __future__ import annotations
from typing import Any, Dict, Optional

import plotly.graph_objects as go
from nicegui import ui

from traceml.renderers.gpu_utilization.schema import GpuUtilizationResult
from .ui_shell import CARD_STYLE, compact_metric_html


def build_gpu_utilization_section() -> Dict[str, Any]:
    card = ui.card().classes("w-full h-full p-3")
    card.style(CARD_STYLE + "height: 100%; overflow: hidden;")
    with card:
        with ui.row().classes("w-full items-center justify-between mb-1"):
            ui.label("GPU Utilization").classes("text-sm font-bold") \
                .style("color:#d47a00;")
            window_text = ui.html("window: -", sanitize=False) \
                .classes("text-[11px] text-gray-500")
        fig = go.Figure()
        fig.update_layout(height=140, showlegend=True,
                          yaxis=dict(range=[0, 100], title="Util %"))
        plot = ui.plotly(fig).classes("w-full")
        kpis = ui.html("", sanitize=False).classes("mt-2")
    return {"window_text": window_text, "plot": plot, "_fig": fig,
            "kpis": kpis, "_last_kpis": None}


def update_gpu_utilization_section(
    panel: Dict[str, Any], payload: Optional[GpuUtilizationResult],
) -> None:
    if payload is None or payload.is_empty:
        return
    panel["window_text"].content = (
        f"window: {payload.samples_used} / {payload.window_size}"
    )
    fig = panel["_fig"]
    fig.data = ()  # reuse figure; only replace traces
    if payload.series is not None:
        for gpu_idx, y in enumerate(payload.series.util_by_gpu):
            fig.add_trace(go.Scatter(
                x=list(range(len(y))), y=y, mode="lines", name=f"gpu{gpu_idx}",
            ))
    panel["plot"].update()

    items = [
        compact_metric_html(
            f"r{r.rank} / gpu{r.gpu_idx}",
            f"{r.util_peak:.0f}% peak / {r.mem_peak_mb:.0f} MB",
        )
        for r in payload.rows[:8]
    ]
    html = ("<div style='display:grid; grid-template-columns:repeat(2, 1fr);"
            " gap:6px; padding-top:6px; border-top:1px solid #ececec;'>"
            + "".join(items) + "</div>")
    if panel.get("_last_kpis") != html:
        panel["kpis"].content = html
        panel["_last_kpis"] = html
```

**(c)** Wire into `nicegui_sections/pages.py::define_pages`:

```python
# inside main_page(), after step_memory:
with ui.column().classes("h-full flex-1").style("min-width: 0;"):
    cards = build_gpu_utilization_section()
    cls.subscribe_layout(
        GPU_UTILIZATION_LAYOUT, cards, update_gpu_utilization_section,
    )
```

!!! warning "Per-client timers matter"
    `cls.ensure_ui_timer(0.75)` schedules the UI update loop *per client*
    (every browser tab). Forget it and your section renders once and
    freezes. Duplicate calls within a client are deduped by
    `NiceGUIDisplayDriver.ensure_ui_timer`.

### Step 7: Profile gating

Add the layout constant in `aggregator/display_drivers/layout.py`:

```python
GPU_UTILIZATION_LAYOUT = "gpu_utilization_section"
```

Add a `Layout(name=GPU_UTILIZATION_LAYOUT, ratio=...)` node in the right
`CLIDisplayDriver._create_*_layout` method:

```python
dashboard["middle_row"].split_row(
    Layout(name=MODEL_COMBINED_LAYOUT, ratio=3),
    Layout(name=MODEL_MEMORY_LAYOUT, ratio=2),
    Layout(name=GPU_UTILIZATION_LAYOUT, ratio=2),  # NEW
)
```

Seed a "Waiting for..." placeholder in `_create_initial_layout`:

```python
if self._has_section(GPU_UTILIZATION_LAYOUT):
    self._layout[GPU_UTILIZATION_LAYOUT].update(
        Panel(Text("Waiting for GPU Utilization...", justify="center"))
    )
```

Dashboard gating is implicit: presence in `self._renderers` (or under
`self._deep_profile`) plus the page including the section.

!!! tip "Profile quick-reference"
    - `watch` — minimal (CPU/RAM + stdout/stderr). Only add here for
      always-on lightweight views.
    - `run` — default. Most new renderers belong here.
    - `deep` — adds per-layer renderers; heavier runs.
    - `summary` — no live UI; renderers ignored.

### Step 8: Graceful empty data

The pattern is: **return early with a placeholder `Panel` when the compute
returns nothing usable.**

```python
if payload is None or payload.is_empty:
    return Panel(
        "Waiting for GPU utilization samples...",
        title="GPU Utilization",
        border_style="cyan",
    )
```

For the dashboard path, return the typed result even when empty and let the
section's `update_fn` do the right thing:

```python
def update_gpu_utilization_section(panel, payload):
    if payload is None or payload.is_empty:
        return   # leaves the initial "waiting" UI intact
```

The NiceGUI driver additionally guards against `update_fn` exceptions in
`_ui_update_loop` — a raised exception replaces the card with
"Could not update". Don't rely on this; handle empty data explicitly.

### Step 9: (Optional) Notebook support

Most renderers stub this. Implement only with a concrete notebook need — no
driver in the main CLI/dashboard path consumes it today.

```python
from IPython.display import HTML

def get_notebook_renderable(self) -> HTML:
    payload = self._computer.compute_cli()
    if payload.is_empty:
        return HTML("<pre>No GPU utilization data yet.</pre>")
    rows = "".join(
        f"<tr><td>r{r.rank}</td><td>{r.gpu_idx}</td>"
        f"<td>{r.util_peak:.1f}%</td></tr>"
        for r in payload.rows
    )
    return HTML(f"<table>{rows}</table>")
```

---

## 5. Rich renderable patterns

### Standard table construction

```python
table = Table(show_header=True, header_style="bold blue",
              box=None, expand=False)
table.add_column("Metric", style="magenta")
table.add_column("Value", justify="right")
```

Conventions across `StepCombinedRenderer`, `StepMemoryRenderer`,
`LayerCombinedTimeRenderer`:

- `header_style="bold blue"`, `box=None`, `expand=False`
- First (label) column: `style="magenta"`
- Numeric columns: `justify="right"`
- Outer `Panel` provides the border; tables should not draw their own

### Adaptive-width panels

Formula used across `SystemRenderer`, `ProcessRenderer`,
`StepCombinedRenderer`, `StepMemoryRenderer`, `LayerCombinedTimeRenderer`:

```python
cols, _ = shutil.get_terminal_size()
width = min(max(100, int(cols * 0.75)), 120)  # 120 for tables; 100 or 130 also used
```

"75% of terminal width, clamped between 100 and upper bound." Pick the
upper bound by table density: 100 for dense KPI cards, 130 for long layer
names.

### When to use `Panel` / `Group` / `Columns` / `Text`

- **`Panel`** — the frame every section returns. `border_style="cyan"`
  for standard, `"red"` for error, `"blue"` for layer-level views.
- **`Group`** — stack multiple renderables vertically inside one panel.
  `StepCombinedRenderer` uses this to place the diagnosis line above the
  table.
- **`Columns`** — two-column layouts. Not currently used in main renderers.
- **`Text`** — when you need structured styling (spans with different
  styles), prefer `Text` over markup strings.

### Color conventions (no enforced palette, but observed)

| Meaning         | Rich style       | Example                        |
|-----------------|------------------|--------------------------------|
| Metric label    | `magenta`        | First column of most tables    |
| Value neutral   | `bright_white`   | `SystemRenderer` grid values   |
| Emphasis        | `bold green`     | `SystemRenderer` CPU/RAM       |
| Warn / bad      | `red`            | WAIT share, unavailable        |
| Info / headers  | `bold cyan`      | Panel titles                   |
| Deemphasize     | `dim`            | Footers, missing data          |

The dashboard uses hex colors from
`nicegui_sections/ui_shell.py::severity_color` — keep the semantics
consistent between Rich and NiceGUI where possible.

### How panels compose

The CLI driver's tick loop:

```
for binding in self._bindings:
    renderable = binding.render_fn()   # your get_panel_renderable()
    self._layout[binding.section].update(renderable)
self._live.refresh()
```

The `Layout` tree is built once per profile in `_create_*_layout`. Leaves
are named by `*_LAYOUT` constants; your renderer's `layout_section_name`
must match one exactly.

---

## 6. NiceGUI + Plotly patterns

### Plotly figure construction

Initialize the figure once in the builder, mutate `.data` / `.layout` in
the updater. Constructing a new `go.Figure` every tick churns memory.

```python
# build
fig = go.Figure()
fig.add_trace(go.Scatter(x=[], y=[], mode="lines"))
plot = ui.plotly(fig).classes("w-full")

# update — in place
fig.data[0].x = new_x
fig.data[0].y = new_y
plot.update()
```

Multi-trace charts (`system_section.py`, `process_section.py`) add each
trace at build time; the updater only assigns `x/y` and calls
`plot.update()`.

### Typed-payload approach

Renderers return a frozen dataclass — **not** a Plotly `Figure`, not an
HTML string, not a dict of pre-formatted strings. Reasons:

1. Renderer is unit-testable without Plotly imported.
2. The NiceGUI section can swap visualizations (bar ↔ line ↔ heatmap)
   without touching the renderer.
3. Same payload can serve the summary logger and future exporters.

Example: `StepCombinedRenderer.get_dashboard_renderable()` returns
`StepCombinedTimeResult`; `model_combined_section.update_model_combined_section`
does all presentation. The renderer does not know the chart's colors.

### Section registration flow

```
__init__              -> build self._renderers
start()               -> spawn UI thread, call define_pages(self)
define_pages()        -> per page: build_*_section(); subscribe_layout(...)
                      -> ensure_ui_timer(0.75)
tick() [agg thread]   -> _register_once; update_display -> latest_data
_ui_update_loop [UI]  -> snapshot latest_data + subs; update_fn(cards, data)
```

### Threading model (three rules)

1. **Renderers run on the aggregator thread.** Don't touch NiceGUI widgets
   from a renderer — return data.
2. **Locks only around snapshots.** Never hold `_latest_data_lock` across
   compute or UI work.
3. **Updaters must never throw.** The driver catches per-subscriber
   exceptions and writes "Could not update", but still guard explicitly.

### Port and per-client timer

Port is hard-coded to **`8765`** in `NiceGUIDisplayDriver.__init__`. Don't
add per-renderer port arguments.

`ensure_ui_timer(interval_s)` runs once per browser tab (deduped by
`client.id`). Default is `0.75` s; the layer page uses `1.0` s because the
tables are heavier.

!!! warning "Do not set interval below 0.5s"
    NiceGUI + Plotly updates are not free. Below 0.5s you start seeing
    browser lag, dropped frames on long tables, and aggregator backpressure
    if `update_display()` is slow. 0.75 s is a load-tested value.

---

## 7. Data source choice: `remote_store` vs `db_path`

Decision tree:

```
Need cross-run history or window stats?
├── Yes → db_path (SQLite)
└── No → need newest row every tick with rank awareness?
        ├── Yes, sampler is in RemoteDBStore allow-list → remote_store
        └── Otherwise → db_path (preferred), or extend allow-list in
            TraceMLAggregator._REMOTE_STORE_SAMPLERS
```

**`db_path` wins for**: windowed stats (last 100 steps, P95, head/tail
delta), cross-rank aggregations with step alignment, anything that should
survive restart. All step-time, step-memory, system, process, and
stdout/stderr renderers use SQLite.

**`remote_store` wins for**: live rows with no lag, samplers without a
SQLite projection yet (currently the four layer samplers).

Constructor pattern with `remote_store`:

```python
def __init__(self, remote_store: Optional[RemoteDBStore] = None,
             top_n_layers: int = 20):
    super().__init__(name=..., layout_section_name=...)
    self._service = LayerCombinedTimerData(
        top_n_layers=top_n_layers, remote_store=remote_store,
    )
```

The driver injects `store` from the aggregator.

### Thread safety

- **SQLite** — short-lived connection per `_compute` call respects
  `sqlite3`'s thread-affinity check.
- **`RemoteDBStore`** — single-threaded ingestion on rank 0; safe because
  ingest and display tick run serially in `TraceMLAggregator._loop`.
- **Long queries on the tick thread** — avoid; cache with a TTL (copy
  `_last_ok` + `_last_ok_ts` from `StepCombinedComputer`).

### SQLite column schemas

Each writer under `aggregator/sqlite_writers/` declares tables in
`init_schema()`. Copy column names from its `CREATE TABLE`. Current:

- `system.py` — `system_samples`, `system_gpu_samples`
- `process.py` — process CPU/memory samples
- `step_time.py` — `step_time_samples` (events_json blob)
- `step_memory.py` — step-level peak memory
- `stdout_stderr.py` — captured log lines per rank

---

## 8. Profile selection and layout

Profile matrix (from `CLIDisplayDriver.__init__` and
`NiceGUIDisplayDriver.__init__`):

| Renderer                      | watch | run | deep | Dashboard |
|-------------------------------|:-----:|:---:|:----:|:---------:|
| `SystemRenderer`              |   Y   |  Y  |   Y  |     Y     |
| `ProcessRenderer`             |   Y   |  Y  |   Y  |     Y     |
| `StdoutStderrRenderer` (CLI)  |   Y   |  Y  |   Y  |     —     |
| `StepCombinedRenderer`        |   —   |  Y  |   Y  |     Y     |
| `StepMemoryRenderer`          |   —   |  Y  |   Y  |     Y     |
| `ModelDiagnosticsRenderer`    |   —   |  —  |   —  |     Y     |
| `LayerCombinedMemoryRenderer` |   —   |  —  |   Y  |  Y (deep) |
| `LayerCombinedTimeRenderer`   |   —   |  —  |   Y  |  Y (deep) |

**Renderer → layout mapping.** The contract is `layout_section_name` on
each renderer. `_register_once` reads it, checks `_has_section`, binds
`get_panel_renderable`. Missing section → error logged, silently skipped.
Two renderers claiming the same section → last update per tick wins.

**Picking a section.** Read `_create_run_layout` / `_create_deep_layout`.
Reuse an existing layout name when your renderer fits logically into a
row. Otherwise add a `Layout(name=..., ratio=...)` node and a new constant
in `layout.py`. Ratios are heights (3:2 = 60/40).

**Small terminals.** No automatic fallback. The
`min(max(100, int(cols*0.75)), 120)` formula is it. Narrower than 100 cols
wraps. If your renderer looks bad on laptop terminals, lower the clamp or
reduce columns.

---

## 9. Testing

### Existing renderer tests

None. `tests/` currently covers `test_seq_counter.py`, `test_trend_core.py`,
`test_hf_trainer.py`, `test_grad_accum.py`, `test_compare_missing.py`,
`test_msgpack_roundtrip.py`. **Adding renderer unit tests is a net
improvement — don't skip because nobody else does it.**

### Minimum coverage for a new renderer

1. Compute correctness (temp SQLite DB with known rows).
2. Empty DB → empty result, no exceptions.
3. Stale fallback within TTL.
4. `get_panel_renderable()` returns a `Panel` for both empty and populated.
5. `get_dashboard_renderable()` returns the expected dataclass.

### Minimal test template

```python
# tests/test_gpu_utilization_renderer.py
import sqlite3
from pathlib import Path

import pytest
from rich.panel import Panel

from traceml.renderers.gpu_utilization.compute import GpuUtilizationComputer
from traceml.renderers.gpu_utilization.renderer import GpuUtilizationRenderer
from traceml.renderers.gpu_utilization.schema import GpuUtilizationResult


@pytest.fixture
def temp_db(tmp_path: Path) -> str:
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE gpu_util_samples ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, recv_ts_ns INTEGER NOT NULL, "
        "rank INTEGER, sample_ts_s REAL, seq INTEGER, "
        "gpu_idx INTEGER NOT NULL, util_percent REAL, mem_used_mb REAL)"
    )
    rows = [
        (1, 0, 0.0, 0, 0, 10.0, 1000.0),
        (2, 0, 0.1, 1, 0, 50.0, 1200.0),
        (3, 0, 0.2, 2, 0, 90.0, 1400.0),
        (4, 0, 0.1, 1, 1, 20.0, 600.0),
        (5, 0, 0.2, 2, 1, 30.0, 700.0),
    ]
    conn.executemany(
        "INSERT INTO gpu_util_samples VALUES (NULL, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit(); conn.close()
    return db_path


def test_compute_basic(temp_db):
    result = GpuUtilizationComputer(temp_db, window_size=10).compute_cli()
    assert isinstance(result, GpuUtilizationResult)
    assert len(result.rows) == 2
    gpu0 = next(r for r in result.rows if r.gpu_idx == 0)
    assert gpu0.util_peak == 90.0 and gpu0.samples == 3


def test_compute_missing_db(tmp_path):
    result = GpuUtilizationComputer(str(tmp_path / "nope.db")).compute_cli()
    assert result.is_empty  # stale fallback returns empty


def test_panel_empty_and_populated(temp_db, tmp_path):
    assert isinstance(
        GpuUtilizationRenderer(str(tmp_path / "nope.db"))
        .get_panel_renderable(),
        Panel,
    )
    assert isinstance(
        GpuUtilizationRenderer(temp_db).get_panel_renderable(), Panel
    )


def test_dashboard_payload(temp_db):
    result = GpuUtilizationRenderer(temp_db).get_dashboard_renderable()
    assert isinstance(result, GpuUtilizationResult)
    assert result.series is not None  # dashboard path includes series
```

### Manual smoke verification

```bash
pip install -e ".[dev,torch]"
traceml watch examples/mnist.py --mode cli
traceml run   examples/mnist.py --mode dashboard  # http://localhost:8765
```

Expect: CLI panel flips from "Waiting..." to live data within 1–2 ticks;
dashboard card updates roughly every 0.75 s. If nothing shows on CLI, grep
session logs for "CLI layout section not found" or "CLI renderer missing
layout_section_name" — the usual registration bugs.

---

## 10. Common pitfalls

Not theoretical — these have bitten us.

1. **Mutating the data source.** Calling `db.add_record(...)` or
   `remote_store.ingest(...)` from a renderer corrupts the telemetry view
   for every other consumer.
2. **Blocking I/O in `get_*_renderable`.** Network calls, `time.sleep`,
   `subprocess.run` stall the whole dashboard. Precompute in a background
   thread (rare; discuss first).
3. **Assuming data always exists.** Cold start = empty DB. Always check
   `payload.is_empty` / `payload.metrics` before indexing.
4. **Cross-rank rendering without aggregation.** Per-rank rows duplicate
   naively. Group by rank and decide: per-rank view or "worst-rank wins".
5. **Hard-coding rank 0** silently drops rank > 0 data. OK for
   `StdoutStderrRenderer`; aggregate elsewhere, or document the filter.
6. **Forgetting driver registration.** Module exists, tests pass, nothing
   shows. Check `_renderers` list, matching `*_LAYOUT` constant, and
   `Layout(name=...)` in `_create_*_layout`.
7. **Creating `go.Figure()` every tick.** At 0.75 s cadence, 4800 figures
   per hour. Build once in the section builder; mutate `.data` in updater.
8. **`ui.timer` below 0.5 s.** NiceGUI lags on realistic tables. Keep the
   0.75 s default.
9. **Slow SQLite on the tick thread.** Unbounded `SELECT *` is eventual
   disaster. Always `LIMIT window_size * k`; use indexes from the writer.
10. **Bypassing `_safe` wrapping.** One renderer crash can take down the
    whole dashboard if you call renderers outside the binding system.
11. **Mutating a frozen dataclass.** Construct a new instance (see
    `StepCombinedComputer._stale_or_empty`).
12. **`print()` in a renderer.** The CLI driver redirects stdout into the
    log panel — your print lands inside `stdout_stderr`. Use
    `self._logger` (see `get_error_logger(...)`).

---

## 11. Checklist before opening a PR

Copy into the PR description.

- [ ] Renderer module under `src/traceml/renderers/<name>/` (split) or a
      single file (flat)
- [ ] Subclasses `BaseRenderer`; sets `name` + `layout_section_name`
- [ ] Implements `get_panel_renderable()` and/or
      `get_dashboard_renderable()` as targeted
- [ ] Returns `Panel` (CLI) and/or a typed dataclass (dashboard)
- [ ] Empty-data path explicit (waiting panel, stale-TTL fallback)
- [ ] New `*_LAYOUT` constant in
      `aggregator/display_drivers/layout.py`
- [ ] Layout node added in `CLIDisplayDriver._create_*_layout`
- [ ] Initial "Waiting for..." panel in `_create_initial_layout`
- [ ] Registered in `CLIDisplayDriver._renderers` (profile-gated) and/or
      `NiceGUIDisplayDriver._renderers`
- [ ] NiceGUI section builder/updater added; wired via `subscribe_layout`
      in `pages.py`
- [ ] Unit tests: compute correctness, empty data, stale fallback, panel
      non-crash, dashboard payload shape
- [ ] Renderer tick p95 < 10 ms on `watch` profile
- [ ] No new Python deps without PR approval
- [ ] Local smoke: `--mode cli` and `--mode dashboard` both work
- [ ] Docstrings (NumPy style) on computer + renderer class
- [ ] Short commit messages, no `Co-Authored-By` (per root `CLAUDE.md`)

---

## 12. Appendix: adding a whole new display driver

!!! danger "Very rare. talk to maintainers first."
    The existing three drivers (CLI, dashboard, summary) cover current use
    cases. A fourth (e.g. `JsonStreamDisplayDriver` for CI) is a design
    decision, not an implementation detail.

If you're actually doing this:

1. Subclass `BaseDisplayDriver` in
   `aggregator/display_drivers/<name>.py`. Implement `start()` / `tick()` /
   `stop()` — all best-effort, must not raise.
2. Register in `_DISPLAY_DRIVERS` in `trace_aggregator.py`.
3. Add the mode string to `supported_modes` and `--mode choices=[...]` in
   `cli.py`.
4. Instantiate your renderer set in the new driver's `__init__`. If
   existing `get_panel_renderable` / `get_dashboard_renderable` don't fit
   your medium, you may need a new method on `BaseRenderer` — discuss
   before writing code; it affects every renderer.
5. Wire profile gating / layout equivalents if your medium has them.

---

## Gaps and ambiguities

Things not fully resolved in source:

- **`BaseRenderer.get_dashboard_renderable()` is not declared abstract.**
  Only `get_panel_renderable()` and `get_notebook_renderable()` raise
  `NotImplementedError`. The dashboard path is duck-typed in
  `NiceGUIDisplayDriver._register_once` via
  `getattr(rr, "get_dashboard_renderable", None)`. Should probably be
  formalized in the base class.
- **Multi-subscriber composition** is implemented but underdocumented. A
  single layout section can have many subscribers across pages via
  `subscribe_layout(..., replace_for_client=True)`. If two renderers
  claim the same `layout_section_name`, the last `register_layout_content`
  call wins — not exercised today, don't rely on it.
- **Notebook renderables** have no active consumer. Implemented
  inconsistently (`StdoutStderrRenderer` placeholder,
  `LayerCombinedTimeRenderer` `None`, `StepCombinedRenderer` absent).
  Treat as aspirational.
- **`log_summary()`** exists on some renderers as a no-op. The real
  summary path lives in `aggregator/final_summary.py`. Reserved but not
  load-bearing.
- **`RemoteDBStore` allow-list** (`_REMOTE_STORE_SAMPLERS`) is not
  surfaced anywhere except that frozenset. New samplers needing live
  in-memory access must be added there; no warning if forgotten.
- **Rank filtering semantics** differ between renderers
  (`rank_filter`, `rank`, hard-coded rank 0). Pick one and document.
- **`page_layout.py::TRACE_ML_PAGE`** 2D layout is not referenced by the
  NiceGUI pages (which hand-roll layout in `pages.py`). Possibly dead
  code or aspirational — confirm with maintainers before relying on it.

Add new findings to this section in your PR.
