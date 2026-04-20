# Renderers

Renderers are the presentation layer of TraceML. Each renderer owns one data domain — step time, step memory, layer time, layer memory, process, system, or stdout/stderr — and turns the stored telemetry for that domain into a human-readable artifact: a Rich panel for the terminal dashboard, a typed payload for the NiceGUI dashboard, and (where implemented) an HTML fragment for notebooks. Renderers are strictly read-only consumers of the aggregator's data store; they never mutate telemetry and they never block training.

## Role in the architecture

During a `traceml watch|run|deep` session the aggregator process owns a [`RemoteDBStore`](database.md) that ingests telemetry frames from every rank. Samplers push rows; the store fans them into rank-aware bounded tables. A renderer sits on the read side of that store: on every display tick the driver asks each renderer for its latest renderable, the renderer pulls or recomputes a summary over its source tables, and it returns a Rich or dashboard-ready object. Nothing about the store changes — no row is deleted, no counter is advanced, no lock is acquired beyond what the store itself provides.

Renderers never appear on their own in the user interface. They are composed by a [display driver](display-drivers.md) which owns the full terminal layout (for `CLIDisplayDriver`) or the NiceGUI page (for the dashboard driver). The driver decides which renderers to instantiate, which layout section each one occupies (via the renderer's `layout_section_name`), and at what cadence to refresh. The renderer is narrowly scoped: given a database reference, produce a panel or payload. This keeps the rendering logic testable in isolation and allows new output mediums (e.g., a future TUI or HTML exporter) to reuse the same renderers without changes.

The division of labor is:

- **Samplers** collect raw rows into the per-rank `Database`.
- **`RemoteDBStore`** unifies those rows across ranks in the aggregator.
- **Compute services** (one per renderer, living alongside the renderer) turn raw rows into typed result dataclasses — windowed summaries, top-N layer slices, rank-aware peaks.
- **Renderers** format a typed result for a concrete output medium.
- **Display drivers** lay out multiple renderers and drive the refresh loop.

This layered split is what keeps the dashboard code legible: all PyTorch- or DDP-specific knowledge lives below the renderer, and all Rich/NiceGUI-specific knowledge lives at or above it.

## Base class contract

Every renderer subclasses `BaseRenderer` in `src/traceml/renderers/base_renderer.py`. The base class is intentionally small: it captures the renderer's name, the layout section it belongs to, and declares the two abstract methods subclasses are expected to provide.

```python
class BaseRenderer:
    def __init__(self, name: str, layout_section_name: str):
        self.name = name
        self.layout_section_name = layout_section_name
        self._latest_data: Dict[str, Any] = {}

    def get_panel_renderable(self) -> Any: ...
    def get_notebook_renderable(self) -> Any: ...
```

Subclasses must implement:

- **`get_panel_renderable()`** — returns a Rich renderable (typically `rich.panel.Panel`, sometimes `rich.table.Table` or `rich.console.Group`). Called by the CLI display driver on every refresh tick. This method must be safe to call at any point during a run, including before any telemetry has arrived.
- **`get_notebook_renderable()`** — returns an `IPython.display.HTML` block for Jupyter sessions. Several renderers stub this out; see the [catalog](#renderer-catalog) for current coverage.

Most concrete renderers additionally implement `get_dashboard_renderable()` — not declared on the base class, but used by the NiceGUI display driver to fetch a typed dataclass (or plain dict) for the web UI. By convention the dashboard method returns richer information than the CLI method: CLI summaries stay compact and stable, while the dashboard can consume per-step series, rank heatmaps, and per-layer detail.

The `__init__` takes a `name` (used in panel titles and logging) and a `layout_section_name` (a module-level constant imported from `traceml.aggregator.display_drivers.layout`). The layout section is how a renderer advertises *where* in the dashboard it lives — the display driver matches these names to its grid slots.

!!! note "Where does the data live?"
    Most renderers take a `db_path` (the path to the aggregator's SQLite file used by window-based computers) or a `remote_store` (a live `RemoteDBStore` reference). Both resolve to the same underlying telemetry; the difference is whether the renderer works against a persisted file or the in-memory store directly. See [database.md](database.md) for the store model.

The layout constants themselves are string identifiers centralised in `traceml/aggregator/display_drivers/layout.py` — e.g., `MODEL_COMBINED_LAYOUT = "model_combined_section"`, `LAYER_COMBINED_MEMORY_LAYOUT = "layer_combined_memory_section"`, `SYSTEM_LAYOUT`, `PROCESS_LAYOUT`, `STDOUT_STDERR_LAYOUT`, and so on. Keeping them in one file makes it easy to see the full dashboard shape at a glance and prevents typos between renderer and driver.

## Renderer catalog

All concrete renderers live under `src/traceml/renderers/`. Each has a primary CLI output (a Rich panel) and most have a typed dashboard payload; notebook support is partial.

| Renderer | Data domain | CLI output | Dashboard output | Source |
|---|---|---|---|---|
| `StepCombinedRenderer` | Per-step wall time + wait proxy, aligned across ranks over last K completed steps | Rich `Panel` with median/worst/skew table, trend line, WAIT share | Typed `StepCombinedTimeResult` | `renderers/step_time/renderer.py` |
| `StepMemoryRenderer` | Per-step peak allocated/reserved memory, rank-aware over last K steps | Rich `Panel` with median-peak / worst-peak / skew table plus head-tail delta | Typed `StepMemoryCombinedResult` | `renderers/step_memory/renderer.py` |
| `LayerCombinedTimeRenderer` | Per-layer forward + backward timing, top-N with "other" bucket | Rich `Panel` with current/average forward+backward columns and percentage share | Typed `LayerCombinedTimerResult` | `renderers/layer_combined_time/renderer.py` |
| `LayerCombinedMemoryRenderer` | Per-layer param + forward + backward memory (current and peak) | Rich `Panel` with top-N + "other layers" row | Typed `LayerCombinedMemoryResult` | `renderers/layer_combined_memory/renderer.py` |
| `ProcessRenderer` | Per-process CPU usage (worst rank) and GPU memory (least-headroom rank) | Rich `Panel` with CPU cores, GPU used/reserved/total, imbalance | Dict snapshot for dashboard | `renderers/process/renderer.py` |
| `SystemRenderer` | Host CPU %, RAM used/total, GPU utilisation/mem/temp/headroom | Rich `Panel` with a 2x2 grid of system metrics | Dict snapshot, windowed over last N samples | `renderers/system/renderer.py` |
| `ModelDiagnosticsRenderer` | Unified step-time + step-memory diagnosis for the dashboard card | Placeholder panel ("available in dashboard mode") | `ModelDiagnosticsPayload.to_dict()` | `renderers/model_diagnostics/renderer.py` |
| `StdoutStderrRenderer` | Captured rank-0 stdout/stderr lines | Rich `Panel` with the last N log lines | — (notebook stub only) | `renderers/stdout_stderr_renderer.py` |

### Compute / render split

Each domain under `renderers/<domain>/` follows the same three-file pattern:

- **`schema.py`** — frozen dataclasses for the typed payload. For example, step time exposes `StepCombinedTimeSeries` (per-step arrays), `StepCombinedTimeSummary` (median/worst/skew scalars), `StepCombinedTimeCoverage` (ranks present, world size, completed step), and wraps them in a top-level `StepCombinedTimeResult`.
- **`compute.py`** / **`computer.py`** — the reduction logic. A computer holds a reference to the store or DB path and exposes `compute_cli()` (compact summary) and `compute_dashboard()` (richer payload, often including per-step series and rank heatmaps). Computers are the only place where raw rows are touched.
- **`renderer.py`** — the `BaseRenderer` subclass. It instantiates the computer, implements `get_panel_renderable()` by formatting the typed result into a Rich panel, and returns the same (or a richer) typed result from `get_dashboard_renderable()`.

This separation means a new output medium never has to reach into rank reduction logic, and a change to the reduction logic never has to touch Rich styling. It also makes the compute layer directly testable — feed a known `RemoteDBStore` state in, assert on the returned dataclass.

### Shared helpers

A few shared helpers live alongside the renderers:

- **`renderers/utils.py`** — name truncation (`truncate_layer_name`), compact time formatting (`fmt_time_run`), and the `CARD_STYLE` snippet used by NiceGUI cards.
- **`renderers/<domain>/compute.py`** or **`computer.py`** — the compute service for that domain. These contain all rank/window reduction logic and return a typed result. Renderers treat them as black boxes.
- **`renderers/<domain>/schema.py`** — the typed result dataclasses exchanged between compute and render. These are the stable boundary between "numbers" and "display".
- **`renderers/<domain>/diagnostics.py`** (step time and step memory) — builds a short human-readable diagnosis string (e.g., "skew is rising — rank 2 is slow") which the renderer renders above the table.

## Output mediums

Renderers target two output mediums today, plus a partially implemented third:

**Rich / terminal (always implemented).** `get_panel_renderable()` returns a `rich.panel.Panel`, usually containing a `rich.table.Table` or `rich.console.Group`. The CLI display driver places these panels into a `rich.layout.Layout` keyed by `layout_section_name`. Panel width adapts to `shutil.get_terminal_size()` so the dashboard looks sensible on an 80-column shell and on a wide 4K terminal alike — the typical formula is `width = min(max(100, int(cols * 0.75)), 120)`.

**NiceGUI / dashboard (most renderers).** `get_dashboard_renderable()` returns either a typed dataclass (for the step and layer renderers) or a dict (for system, process, and model-diagnostics). The NiceGUI display driver consumes these as data, not as HTML — the actual visual styling lives in the driver's `dashboard_compute.py` and card layout code. This separation is deliberate: the renderer owns *what* to show, the driver owns *how* it looks on a web page.

**IPython / notebook (partial).** `get_notebook_renderable()` returns `IPython.display.HTML`. Several renderers (the layer renderers, stdout/stderr) currently stub this with `pass` or a placeholder string. Notebook support is on the roadmap; for now, the canonical outputs are the CLI panel and the dashboard payload.

The display driver picks the medium, not the renderer. A `CLIDisplayDriver` calls `get_panel_renderable()` on each registered renderer and tiles the result; a `NiceGUIDisplayDriver` calls `get_dashboard_renderable()` and binds the result to reactive UI components. Renderers themselves don't know which driver is active, which is what makes the two output paths interchangeable.

## Design notes

A handful of principles keep the renderer layer small, testable, and robust against the chaos of real training runs. These aren't arbitrary conventions — each one addresses a failure mode observed in practice (flicker, crashes on partial data, panels exploding on wide terminals, rank imbalance going unnoticed).

**Read-only contract.** A renderer must never mutate the `RemoteDBStore`, never delete rows, and never bump any sampler's incremental counter. Compute services are pure functions of the current store snapshot: they read, summarise, and return. If a renderer needs to cache an intermediate result (e.g., to avoid flicker across ticks when ranks are momentarily out of sync — see `StepCombinedRenderer._payload`), that cache lives on the renderer instance, not in the store.

**Rank awareness.** Most domains ship per-rank telemetry; renderers are responsible for presenting it coherently. The step-time and step-memory renderers produce "median / worst / worst-rank / skew" rows so users can spot imbalance at a glance. The layer renderers reduce per-rank data into a single top-N table but show `missing ranks` in the title when a rank hasn't reported yet. The system and process renderers surface the "worst" rank explicitly (worst CPU, least-headroom GPU). Single-rank runs are detected at render time (`world_size <= 1 or ranks_present <= 1`) and collapse the display to a simpler layout — no point showing a skew column when there's one rank.

**Graceful handling of empty and early data.** Training starts producing telemetry asynchronously; the first display tick almost always fires before any step has completed. Renderers handle this without errors. The step renderers return a "Waiting for first fully completed step across all ranks…" panel. The layer renderers emit a "No layers detected" or "No timing data" row. The stdout/stderr renderer shows a dim "Waiting for stdout/stderr..." text block. The stdout renderer additionally wraps its SQLite read in a broad try/except because `"renderer must never crash the display loop"` — the fail-open principle extends from training into the UI.

**Caching for stability.** Display ticks are frequent (default 1 Hz) and telemetry is partial. Several renderers keep a `_cached` last-good payload so that transient `None` returns from the compute service don't cause the panel to blank out. `ModelDiagnosticsRenderer` goes further and wraps its whole compute in a try/except, logging via `get_error_logger` and returning the cached payload on any exception.

**Presentation knowledge lives here, only here.** Number-to-string formatting (`fmt_mem_new`, `fmt_time_ms`, `fmt_percent`, `fmt_time_run`), layer-name truncation, panel borders, column colours, and table box styles all live inside the renderer. A compute service returns numbers and labels; the renderer decides how to render them. This boundary is what makes it practical to change the UI without touching the aggregation logic — and vice versa.

**Diagnostics integration.** The step-time and step-memory renderers pull a short narrative diagnosis from `renderers/<domain>/diagnostics.py` and render it above the summary table — the same diagnostics payload is consumed by `ModelDiagnosticsRenderer` for the unified NiceGUI diagnostics card. Diagnostics live inside the renderer package because their wording and thresholds are part of the user-facing presentation, not part of the raw reduction. A shared trend helper lives at `traceml/diagnostics/trends.py` (`compute_trend_pct`, `format_trend_pct`, `DEFAULT_TREND_CONFIG`) and is used by the step-time renderer to compute the small "Trend" row in the summary table once enough points are available.

**Adaptive width.** Every Rich renderer sizes its panel against the live terminal width, using the common formula:

```python
cols, _ = shutil.get_terminal_size()
width = min(max(100, int(cols * 0.75)), 120)
```

This keeps panels readable on narrow shells (floor at 100 columns) and prevents them from stretching uncomfortably on ultra-wide monitors (ceiling at 120, or 130 for the layer timing panel). No renderer assumes a fixed width.

**Single-rank vs multi-rank dual mode.** The step-time renderer illustrates the pattern well. It detects `single_rank = (world_size <= 1) or (ranks_present <= 1)` at render time and switches row layouts: a single "Sum (Σ K)" row for single-process runs, a four-row "Median / Worst / Worst Rank / Skew" layout for DDP runs. The underlying compute returns the same `StepCombinedTimeResult` in both cases; only the formatting changes. New renderers should follow the same pattern — compute the rank-aware payload once, branch only in the Rich-building code.

**Never crash the display loop.** The stdout/stderr renderer makes the principle explicit in a comment: `"Best-effort only: renderer must never crash the display loop."` It wraps its SQLite read in a broad try/except and returns an empty list on any exception. The model-diagnostics renderer uses the same pattern around its whole compute call. In effect, the fail-open rule TraceML applies to instrumentation (never break the training job) extends into rendering (never break the UI loop). If a renderer cannot produce meaningful content, it must still return *something* — a "Waiting…" panel, a dim placeholder, or the last-good cached payload.

!!! tip "Adding a new renderer"
    1. Pick a data domain and create `renderers/<domain>/` with `schema.py` (typed result), `compute.py` (pure reduction from store to result), and `renderer.py` (Rich panel construction).
    2. Register a layout section in `aggregator/display_drivers/layout.py` and import it in your renderer.
    3. Subclass `BaseRenderer`, call `super().__init__(name=..., layout_section_name=...)`, and implement at least `get_panel_renderable()`. Add `get_dashboard_renderable()` if the NiceGUI driver needs to show this domain.
    4. Handle the empty-store case explicitly — return a "Waiting for data…" panel or a dim placeholder row.
    5. Have the display driver instantiate and register the new renderer. The renderer should never register itself.

## Lifecycle of a render tick

Walking through a single CLI display tick clarifies how the pieces fit together:

1. The display driver's refresh loop fires (default ~1 Hz).
2. For each registered renderer, the driver calls `get_panel_renderable()`.
3. The renderer asks its compute service for the latest payload — `computer.compute_cli()` or an equivalent method.
4. The compute service reads from the `RemoteDBStore` or SQLite file, reduces over ranks/window, and returns a typed result dataclass (or `None` if the data is still warming up).
5. The renderer inspects the result: if empty, returns a "Waiting…" panel; if present, builds a Rich `Table` / `Group` / `Panel` with formatted values.
6. The driver places the returned renderable into the layout slot identified by `layout_section_name` and re-renders the terminal.

No global state changes during a tick. No rows are consumed, no counters advance, no locks are held beyond the store's internal read locks. A failed renderer (exception during compute or formatting) is caught by the driver's `_safe()` wrapper and logged — the next tick retries from scratch.

## Cross-references

- [Database](database.md) — the `RemoteDBStore` and per-rank `Database` classes that renderers read from.
- [Display drivers](display-drivers.md) — the CLI and NiceGUI drivers that own layout and compose renderers.
- [Aggregator](aggregator.md) — the process that owns both the store and the display driver.
- [Samplers](samplers.md) — upstream producers of the telemetry rows that each renderer consumes.
- [Architecture overview](../architecture.md) — end-to-end telemetry flow and the aggregator/runtime split.
