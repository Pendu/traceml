# How to add a new diagnostic verdict

> Internal contributor guide. Audience: one trusted co-founder / new engineer
> being onboarded to TraceML. Not for public docs.

This guide teaches you how to add a new diagnostic verdict to TraceML. It assumes you have read `add_sampler.md`, `add_renderer.md`, and the top-level `CLAUDE.md`, and that `pip install -e ".[dev,torch]"` is working.

---
Feature type: diagnostic verdict
Risk level: medium (verdicts shape user trust; false positives are corrosive)
Cross-cutting impact: multiple subsystems (engine + live renderer + summary adapter + presentation)
PyTorch coupling: none directly (consumes sampler outputs, not PyTorch internals)
Reference PRs: none called out yet — `step_time_diagnosis` and `step_memory_diagnosis` are the only landed exemplars
Companion reviewer guide: none yet
Last verified: 2026-04-25
---

## 1. Intro and mental model

### What is a "diagnostic" in TraceML?

A **diagnostic** (also called a **verdict**) is an opinionated, named class of pathology that TraceML detects from sampler data and reports to the user. Examples already shipped: `INPUT-BOUND`, `COMPUTE-BOUND`, `INPUT-STRAGGLER`, `WAIT-HEAVY`, `MEMORY CREEP`, `HIGH PRESSURE`. Examples enumerated as the design space but not yet implemented: `IDLE-GPU`, `COMM-BOUND`, `HANG`, `EVAL-BOUND` (see `traceml_why.md` §3 for the full taxonomy).

A diagnostic is the **output** of the analytics layer. It takes raw sampler rows (or aggregated metrics built on top of them) and produces a small, named, actionable verdict. Diagnostics are not metrics — they are *interpretations* of metrics. The distinction is the load-bearing claim behind `traceml_why.md` §6.3 ("opinionated verdicts, not raw metrics"). W&B shows you `gpu_util=38%`. TraceML says `INPUT-BOUND: dataloader is 47% of step time. Workers=2, expected ≥ 4*num_gpus.` That difference — between data the engineer has to interpret and a recommendation they can act on — is the entire wedge.

### What a verdict object actually contains

The base contract lives in `src/traceml/diagnostics/common.py` as `BaseDiagnosis`. Every domain-specific verdict subclasses it:

```python
@dataclass(frozen=True)
class BaseDiagnosis:
    severity: Severity        # "info" | "warn" | "crit"
    status: str               # human label, e.g. "INPUT-BOUND"
    reason: str               # one-sentence explanation
    action: str               # what the user should do next
```

Domain subclasses add required identity fields and an optional `confidence` in `[0.0, 1.0]`. Two exist today:

- `StepDiagnosis` — `src/traceml/diagnostics/step_time.py:108` — adds `kind` (the typed enum), `steps_used`, `worst_rank`, `note`.
- `StepMemoryDiagnosis` — `src/traceml/diagnostics/step_memory.py:88` — adds `kind`, `metric` ("peak_allocated" or "peak_reserved"), `steps_used`, `worst_rank`, `note`.

A verdict without `action` is a label, not a diagnostic. The action is the product. If you cannot write one actionable sentence for the worst case, you do not yet have a diagnostic; you have a metric.

### Where diagnostics live in the codebase

There is a **shared engine package** and **two consumers** of it. Read the layout once before writing code; the diagnostic engine actually lives at `src/traceml/diagnostics/`, and the summary path at `aggregator/summaries/` is a thin adapter on top:

```
src/traceml/
├── diagnostics/                          # shared engine (live + summary)
│   ├── common.py                         # BaseDiagnosis, Severity, validate_confidence
│   ├── step_time.py                      # build_step_diagnosis(...)
│   ├── step_time_trend.py                # trend-note enrichment for step time
│   ├── step_time_formatters.py           # format_cli_diagnosis, format_dashboard_diagnosis
│   ├── step_memory.py                    # build_step_memory_diagnosis(...)
│   ├── step_memory_trend.py              # creep score helpers
│   ├── step_memory_formatters.py         # CLI / dashboard formatters
│   ├── trends.py                         # back-compat shim → analytics.trends
│   └── model_diagnostics.py              # composite per-model "card" payload
│
├── renderers/step_time/diagnostics.py    # IMPORT SHIM — re-exports engine
├── renderers/step_memory/diagnostics.py  # IMPORT SHIM — re-exports engine
│
└── aggregator/summaries/
    ├── step_time_diagnosis.py            # SUMMARY ADAPTER — feeds engine post-run
    ├── diagnosis_presentation.py         # rewrites action wording for end-of-run
    ├── step_time.py                      # generate_step_time_summary_card
    └── step_memory.py                    # generate_step_memory_summary_card
```

The two consumers:

1. **Live renderers.** `renderers/step_time/renderer.py:65` and `renderers/step_memory/renderer.py:94` call `build_step_diagnosis(metrics)` / `build_step_memory_diagnosis(metrics)` against the current *windowed* renderer payload, then format the resulting diagnosis with `format_cli_diagnosis` / `format_dashboard_diagnosis`. The diagnosis object refreshes every aggregator tick.
2. **End-of-run summary.** `aggregator/summaries/step_time.py:523` calls `build_summary_step_diagnosis(rank_signals, ...)` — the **summary-mode adapter** in `aggregator/summaries/step_time_diagnosis.py` — which converts SQLite rank aggregates into the same `StepCombinedTimeMetric` shape the live engine consumes, then calls `build_step_diagnosis(...)` underneath (`step_time_diagnosis.py:340`). Same engine, different inputs.

The architectural rule:

!!! tip "One engine, two callers"
    Verdict logic — thresholds, priorities, reason/action text — lives in `src/traceml/diagnostics/`. The live renderer feeds it the live window; the summary adapter feeds it post-run aggregates. **Never duplicate threshold logic.** If the summary needs slightly different defaults (e.g. stricter for higher precision), pass a custom `DiagnosisThresholds` instance — see `SummaryDiagnosisConfig` at `aggregator/summaries/step_time_diagnosis.py:48`.

### Where diagnostics surface today vs. tomorrow

- **Today.** End-of-run summary card (one per domain), driven by `aggregator/final_summary.py:255::generate_summary`. The card is printed to stdout at shutdown and also written to disk as `<db_path>_summary_card.{txt,json}` plus a canonical `final_summary.{txt,json}` if `session_root` is set (`final_summary.py:221::write_summary_artifacts`). Live renderer panels also surface a verdict line via `format_cli_diagnosis`.
- **Tomorrow.** The TraceOpt dashboard regression detector (TraceML → TraceOpt pipeline ingests `final_summary.json` per run, see `traceml_why.md` §6.4) and a future live-UI "verdict band" that promotes the dominant verdict to the top of the panel. **Design new diagnostics so they can run cheaply on a per-tick cadence**, not just post-run, even if the first wired-in caller is summary-only.

### Inputs

Diagnostics consume one of three input shapes:

1. **Renderer-level metric objects** (the most common path). `StepCombinedTimeMetric` and `StepMemoryCombinedMetric` are the typed reductions a renderer's `compute.py` already produces. They carry per-step `series`, per-rank `summary` (median/worst/skew), and `coverage` (steps_used, world_size). The engine functions (`build_step_diagnosis`, `build_step_memory_diagnosis`) take `Sequence[<metric>]` and return one verdict.
2. **Per-rank aggregates** for summary-mode. The summary adapter constructs `RankStepSignals` (`aggregator/summaries/step_time_diagnosis.py:32`) from SQLite rows, then projects them up into the same `StepCombinedTimeMetric` shape.
3. **Raw SQLite tables.** A fresh diagnostic that doesn't fit either metric schema can read SQLite directly via `sqlite3.connect(db_path)` — see `step_memory.py:_gpu_total_bytes` for how the existing summary fetches one auxiliary value. Prefer the metric-object path; reach for raw SQLite only when the metric you need isn't surfaced anywhere else.

### Outputs

A `BaseDiagnosis` subclass instance with at minimum:

- `severity: Severity` (`"info" | "warn" | "crit"`)
- `status: str` — the human label shown in the summary line
- `reason: str` — one-sentence explanation
- `action: str` — what the user should do
- domain-specific identity (`kind`, `metric`, `steps_used`, `worst_rank`, optional `note`, optional `confidence`)

Severity is mapped to color and prominence by the formatters. `confidence` is currently advisory and validated via `validate_confidence()` in `common.py:26` (must be `None` or in `[0.0, 1.0]`). Today only `StepMemoryDiagnosis` populates `confidence` consistently (`step_memory.py` line 217 etc.); the step-time engine omits it. **Set `confidence` explicitly when you have a calibrated value; leave it `None` otherwise.**

### Cross-links (do not duplicate)

- `principles.md` — fail-open contract, overhead budget, wire compat, logging, smoke-test discipline. Not restated here.
- `pipeline_walkthrough.md` — sampler → DB → sender → TCP → aggregator → store → renderer / summary. Not restated here.
- `add_sampler.md` — if your diagnostic needs data the existing samplers don't emit, add the sampler first.
- `add_renderer.md` §3.1 — the `compute.py` layer that produces the metric objects diagnostics consume.
- `traceml_why.md` §3 — the taxonomy of wasted GPU time. Six pathologies, each with frequency and severity. Use it as the design space.
- `traceml_why.md` §6.3 — the "opinionated verdicts, not raw metrics" claim that justifies this whole subsystem.
- [W11](../deep_dive/code-walkthroughs.md#w11-summaries-diagnostics-end-of-run-analysis) — the end-of-run code walkthrough.

### The fail-open contract

!!! danger "A diagnostic must never break the summary or the live UI."
    Verdict construction sits inside `try/except` at every public entry point: `build_step_diagnosis` returns `NO_DATA` rather than raising when inputs are degenerate; `_apply_trend_note` (`diagnostics/step_time.py:172`) returns the original diagnosis unchanged on any trend-note failure. Match that style. If your verdict logic raises, the caller still has to render a summary card. Either return a `NO_DATA`-equivalent verdict or let an empty optional flow through. Never `raise` from a top-level `build_*_diagnosis` function.

---

## 2. Before you start: decisions to make

Write your answers to all of these into the PR description before opening an editor.

- [ ] **Which pathology are you naming?** Map to `traceml_why.md` §3. Is this `IDLE-GPU` (low device utilization on under-saturated workloads)? `COMM-BOUND` (all-reduce dominates backward)? `EVAL-BOUND` (periodic evaluation pauses)? Or a *sub-class* (e.g. `FORWARD-STRAGGLER` inside `STRAGGLER`)? If you can't point at a specific row of the §3 table, reconsider whether this is a verdict or just a metric.
- [ ] **Is this a new domain or an extension to an existing one?** A new `IDLE-GPU` verdict probably belongs as a new `kind` inside `step_time.py::DiagnosisKind` or as a brand-new `gpu_utilization` engine if it consumes different metrics. Adding a kind to an existing engine is cheaper; adding a new engine is the right call when the metric inputs don't overlap.
- [ ] **What sampler data do you need?** Verify it exists. Read the relevant `samplers/schema/*.py` and the SQLite writer in `aggregator/sqlite_writers/`. If the data is not emitted, you need a new sampler first — see `add_sampler.md`. Do not add a diagnostic that depends on hypothetical data.
- [ ] **What shape is your detection rule?** Three patterns exist:
      - **Threshold** — single-value rule (`compute_share >= 0.85`). Exemplar: `COMPUTE_BOUND` branch in `diagnostics/step_time.py:467`.
      - **Ratio** — relative comparison (`dataloader / step_time >= 0.25`). Exemplar: `INPUT_BOUND` branch in `step_time.py:437`.
      - **Trend** — time-series shape (memory rising monotonically over N steps). Exemplar: `CREEP_EARLY` / `CREEP_CONFIRMED` in `diagnostics/step_memory.py:316::_compute_window_creep_evidence`, which delegates to the shared `compute_trend_evidence` engine in `traceml/analytics/trends.py`.
- [ ] **Where does it fire — live, summary, or both?** Live needs per-tick cheapness. Summary can afford full-window scans. The house style ships engine logic that works on a windowed metric object, then has a thin summary adapter feed the same engine from post-run aggregates. Default to that.
- [ ] **What's the priority among existing verdicts?** The step-time engine has a documented priority order (`STRAGGLER` family > `INPUT_BOUND` > `WAIT_HEAVY` > `COMPUTE_BOUND` > `BALANCED`, see `step_time.py:217`). When you add a kind, you must answer "where does it slot?" — and whether it can co-fire with another kind on the same window. The current rule is **single-verdict-per-domain**: one diagnosis per `build_*_diagnosis` call. If two would fire, pick one principled tiebreaker; don't surface both.
- [ ] **Precision vs. recall stance.** A false positive at 3am erodes trust forever (`traceml_why.md` §5.5). Every shipped threshold should have a documented precision/recall stance: "this fires on runs where ≥X% of step time is in the dataloader; we accept a smaller true-positive set to avoid spurious fires." Bias toward precision over coverage. Use `severity="info"` and `confidence < 0.7` for early-warning kinds (e.g. `CREEP_EARLY`).
- [ ] **What's the recommendation?** Write the `action` string before writing the detection logic. If you can't, the diagnostic isn't ready. The action must name a *fix that exists in the user's stack* — see Pitfall §9.7.
- [ ] **Default thresholds.** Numeric, named, in a frozen `*Thresholds` dataclass. Do not inline magic numbers in branch conditions. The pattern is `DiagnosisThresholds` in `step_time.py:62` and `StepMemoryDiagnosisThresholds` in `step_memory.py:53`.
- [ ] **Will the summary use stricter thresholds?** Summary-mode favors precision (one printed line, no live correction). The existing summary adapter does this — see `SummaryDiagnosisConfig` at `aggregator/summaries/step_time_diagnosis.py:48`, where `input_share_warn=0.30` (vs. 0.25 live) and `input_share_crit=0.40` (vs. 0.35 live).

---

## 3. Anatomy of a diagnostic — `step_time.py` end-to-end

This walks `src/traceml/diagnostics/step_time.py` start to finish. It is the cleanest exemplar: it covers thresholds, priority ordering, multi-kind selection, helper functions, and trend-note enrichment.

### 3.1 Module docstring and contract

```python
"""
Step-time diagnosis logic shared by live renderers and post-run summaries.

Semantics
---------
- `step_time` excludes dataloader fetch time.
- `wait_proxy = step_time - (forward + backward + optimizer_step)`.
- Straggler attribution is based on excess local burden on the worst rank,
  normalized by a typical local burden:
      max(0, worst - median) / (median_dataloader + median_compute)
"""
```

**Lesson 1.** The module docstring states the *semantic definitions* of the input metrics, not just the API. `step_time` excludes dataloader fetch — that single sentence prevents anyone debugging the engine from guessing wrong about what `step_time_total` means. Replicate this discipline.

**Lesson 2.** The straggler attribution formula is documented up-front. The reader doesn't have to reverse-engineer it from the helper function. If you introduce a new normalization (e.g. "compute / theoretical_peak for IDLE-GPU"), document it in the module docstring.

### 3.2 Kinds, statuses, and thresholds

```python
DiagnosisKind = Literal[
    "NO_DATA",
    "BALANCED",
    "STRAGGLER",
    "INPUT_STRAGGLER",
    "COMPUTE_STRAGGLER",
    "INPUT_BOUND",
    "COMPUTE_BOUND",
    "WAIT_HEAVY",
]

_STATUS_BY_KIND: dict[DiagnosisKind, str] = {
    "NO_DATA": "NO DATA",
    "INPUT_BOUND": "INPUT-BOUND",
    ...
}
```

**Lesson 3.** `DiagnosisKind` is a `Literal`-typed enum, not a `str` constant. The type system enforces that `StepDiagnosis(kind=...)` only accepts a known kind. New verdicts must extend `DiagnosisKind` and add a row to `_STATUS_BY_KIND`. The machine-friendly kind (`"INPUT_BOUND"`) and the human label (`"INPUT-BOUND"`) are deliberately separated — JSON consumers key off `kind`, terminals show `status`.

```python
@dataclass(frozen=True)
class DiagnosisThresholds:
    input_straggler_score_warn: float = 0.10
    input_straggler_score_crit: float = 0.20
    ...
    input_share_warn: float = 0.25
    input_share_crit: float = 0.35
    ...
    min_steps_for_confident_diag: int = 8

DEFAULT_THRESHOLDS = DiagnosisThresholds()
```

**Lesson 4.** Every threshold has a docstring (the class-level "Design notes" block) that explains the *meaning* of the number, not just its value. `0.10` for `input_straggler_score_warn` is meaningless without "this is the fraction-of-typical-local-burden above which we flag a single rank as a straggler." Replicate the design-notes block on your `*Thresholds` class.

**Lesson 5.** Thresholds are a frozen dataclass with explicit defaults, not module-level constants. Two reasons:

1. The summary adapter constructs a stricter copy (`SummaryDiagnosisConfig.thresholds = DiagnosisThresholds(input_share_warn=0.30, ...)`). Module constants would force a fork.
2. Tests can build a `DiagnosisThresholds(min_steps_for_confident_diag=2)` to drive a small synthetic input through the engine without dropping into NO_DATA.

### 3.3 The diagnosis dataclass

```python
@dataclass(frozen=True)
class StepDiagnosis(BaseDiagnosis):
    """Diagnosis payload used by step-time renderers and summaries."""

    kind: DiagnosisKind
    steps_used: int
    worst_rank: Optional[int] = None
    note: Optional[str] = None
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        validate_confidence(self.confidence)
```

**Lesson 6.** Frozen dataclass, inherits from `BaseDiagnosis`, validates its own `confidence` in `__post_init__`. Domain identity fields (`kind`, `steps_used`, `worst_rank`, `note`) are declared after the base fields. **Always validate optional invariants in `__post_init__`** — the base class only enforces the four required fields.

### 3.4 The dispatcher and priority ordering

`build_step_diagnosis` (`step_time.py:209`) is the engine's single public entry point. Its priority comment is the documentation:

```python
def build_step_diagnosis(
    metrics: Sequence[StepCombinedTimeMetric],
    thresholds: DiagnosisThresholds = DEFAULT_THRESHOLDS,
) -> StepDiagnosis:
    """
    Build one primary diagnosis from step-combined metrics.

    Priority
    --------
    1. INPUT_STRAGGLER / COMPUTE_STRAGGLER / STRAGGLER
    2. INPUT_BOUND
    3. WAIT_HEAVY
    4. COMPUTE_BOUND
    5. BALANCED
    """
```

**Lesson 7.** The priority is documented as part of the function signature contract, not buried in branch comments. When you add a verdict, update this docstring; PR review should refuse a kind that is not slotted into the priority list.

The dispatcher's first job is "is the input usable at all?":

```python
    by_key = {metric.metric: metric for metric in metrics}

    step = by_key.get("step_time")
    if step is None:
        return _mk_diag(kind="NO_DATA", severity="info",
                        reason="step_time metric is missing.",
                        action="Wait for the first complete window.",
                        steps_used=0)

    coverage = step.coverage
    single_rank = (coverage.world_size <= 1) or (coverage.ranks_present <= 1)
    steps_used = int(step.summary.steps_used)
    ...

    if steps_used < thresholds.min_steps_for_confident_diag:
        return _mk_diag(kind="NO_DATA", severity="info",
                        reason=f"Only {steps_used} steps available.",
                        action="Wait for a fuller window.", ...)
```

**Lesson 8.** `NO_DATA` is a *first-class verdict*, not an exception or `None`. The renderer / summary always gets a renderable `StepDiagnosis` back. Reasons for emitting `NO_DATA`: missing required metric, zero total time, fewer than `min_steps_for_confident_diag` steps, duplicate metric keys. Each carries a distinct `reason` so the user knows what's missing.

**Lesson 9.** `single_rank` is detected from coverage, not from environment variables. A run with `WORLD_SIZE=8` but only one rank's data ingested so far is treated as single-rank for this window, which suppresses cross-rank straggler verdicts.

### 3.5 The `_emit` helper and trend-note hook

```python
def _emit(*, kind, severity, reason, action,
          worst_rank=None, note=None, apply_trend=True) -> StepDiagnosis:
    diag = _mk_diag(kind=kind, severity=severity, reason=reason,
                    action=action, steps_used=steps_used,
                    worst_rank=worst_rank, note=note)
    return _finalize(diag) if apply_trend else diag
```

`_finalize` calls `_apply_trend_note` (`step_time.py:172`), which is the optional trend-aware enrichment:

```python
def _apply_trend_note(diagnosis, *, single_rank, step_metric, wait_metric,
                     dataloader_metric, wait_share, dataloader_share,
                     thresholds) -> StepDiagnosis:
    """Best-effort trend annotation. Never raises."""
    try:
        trend_note = build_step_trend_note(...)
        if not trend_note:
            return diagnosis
        return replace(diagnosis, note=_merge_note(diagnosis.note, trend_note))
    except Exception:
        return diagnosis
```

**Lesson 10.** Trend-note enrichment is **best-effort** and wrapped in its own `try/except`. If the trend engine fails on degenerate input, the primary diagnosis is preserved. This is the fail-open law applied to a post-decision enrichment.

### 3.6 The actual rules

The five branches (priority order):

1. **Straggler family** (`step_time.py:351`). Fires only on multi-rank runs. Computes two scores (`_input_straggler_score`, `_compute_straggler_score`); picks the dominant one or `STRAGGLER` if both fire. Severity scales with score against `*_crit` threshold.
2. **INPUT_BOUND** (line 437). `dl_share >= input_share_warn` AND skew is small (so it's not really a straggler). Action: `"Increase workers, prefetch, or storage throughput."`
3. **WAIT_HEAVY** (line 449). `wait_share >= wait_share_warn`. Note carries the `wait_proxy` definition.
4. **COMPUTE_BOUND** (line 467). Compute share dominates AND no other verdict fired. Picks the largest compute phase (`_largest_compute_phase`) for messaging.
5. **BALANCED** (line 491). The fallback. `severity="info"`, recommendation: `"Focus on throughput only if overall speed is still low."`

**Lesson 11.** Each branch returns a fully-formed `StepDiagnosis` with explicit `reason`, `action`, `worst_rank`, and (where relevant) `note`. No verdict slips through to the next branch with partial state. If your new verdict has multiple sub-cases (like the straggler family with three kinds), enumerate them in one branch with shared helpers, not as three top-level branches.

### 3.7 Helper-function discipline

Every numeric primitive is a small named helper (`_share`, `_severity`, `_pct`, `_metric_total`, `_metric_skew`, `_non_negative_finite`, `_finite_float`). They all clamp to non-negative and treat non-finite values as zero (`_non_negative_finite` at line 500). This is the discipline that prevents NaN / Inf from leaking into the verdict text or the JSON-serialized payload.

```python
def _non_negative_finite(value: float) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(out):
        return 0.0
    return max(0.0, out)
```

**Lesson 12.** Validate *every* numeric input at the engine boundary. Sampler bugs, missing data, and arithmetic edge cases (division by zero) will eventually feed `NaN` or `inf` into your branches. Defenses go in helper functions, not in branch conditions.

### 3.8 How the summary adapter feeds it

`aggregator/summaries/step_time_diagnosis.py` is a 360-line adapter, not a separate engine. It does three things:

1. Defines `RankStepSignals` (line 32) — a per-rank tuple of timing averages from SQLite.
2. Builds `StepCombinedTimeMetric` objects from those rank aggregates plus optional per-step metric series (`_build_metric_series`, `_metric_from_rank_values`).
3. Calls `build_step_diagnosis(metrics, thresholds=config.thresholds)` (line 340) with a stricter `SummaryDiagnosisConfig`.

```python
@dataclass(frozen=True)
class SummaryDiagnosisConfig:
    """
    Summary thresholds are slightly stricter than live diagnostics because
    post-run summaries should favor precision over sensitivity.
    """
    thresholds: DiagnosisThresholds = field(
        default_factory=lambda: DiagnosisThresholds(
            input_straggler_score_warn=0.10,
            input_straggler_score_crit=0.18,
            ...
            input_share_warn=0.30,
            input_share_crit=0.40,
            ...
            min_steps_for_confident_diag=20,
        )
    )
    min_steps_for_diag: int = 20
```

**Lesson 13.** The summary adapter does not redefine the verdict semantics. It only changes thresholds and re-projects inputs. **If you find yourself rewriting the priority order or branch logic in the summary adapter, stop — the logic belongs in the engine.**

### 3.9 The presentation layer

`aggregator/summaries/diagnosis_presentation.py` rewrites the engine's `action` text for end-of-run wording. Live actions like `"Wait for a fuller window."` become `"This run did not collect enough step data for a stable timing diagnosis."` in the post-run card. The presentation layer **does not change diagnosis truth** — same status, same reason, only the action string is re-skinned for tense.

**Lesson 14.** When you add a verdict, *always* check whether its `action` reads correctly post-run. If the action says "Watch the next window," add a presentation override. If it's already past-tense ("Inspect the slowest rank"), no override is needed. The override map is the one place you can re-skin tone without forking the engine.

### 3.10 How it gets into `final_summary.py`

`generate_summary` (`final_summary.py:255`) builds four cards (`system`, `process`, `step_time`, `step_memory`) and concatenates them into one printed block. Each card-builder is responsible for its own diagnosis call. **Lesson 15.** `final_summary.py` itself does not know about diagnosis kinds. It only assembles cards. Adding a new diagnostic *to an existing card* requires no edits to `final_summary.py`. Adding a new card (and therefore a new domain engine) requires one entry there: `generate_<domain>_summary_card(...)` plus a call to `_append_wrapped_card_lines`.

---

## 4. Step-by-step: adding `IDLE-GPU`

We'll add `IDLE-GPU` as a new kind to the **step-time engine**. This is the closest pathology to `step_time` data that doesn't already have a verdict. Definition (per `traceml_why.md` §3.4): the model + batch are too small for the GPU; step time is short, GPU utilization is low.

The fact that step-time decomposition alone can't fully verify "GPU utilization is low" — that requires the `system` sampler's GPU util field — is itself the first design decision. We have two choices:

- **Option A.** Add `IDLE-GPU` to the step-time engine but require `compute_share` to be high *and* total step time to be very short (a proxy for under-saturation). Cheap, no cross-engine wiring.
- **Option B.** Build a new domain engine (`src/traceml/diagnostics/gpu_utilization.py`) that consumes a per-tick `gpu_util` series from the system sampler. More accurate but requires a new metric pipeline.

Pick Option A for this walkthrough. It demonstrates "extending an existing engine" — the more common shape of work — without dragging in a new sampler dependency. Note in the PR description that this is a proxy heuristic and that a higher-fidelity `IDLE-GPU` will land later once the `gpu_utilization` engine exists.

### Step 1 — Extend the kind enum and status map

`src/traceml/diagnostics/step_time.py`:

```python
DiagnosisKind = Literal[
    "NO_DATA", "BALANCED", "STRAGGLER", "INPUT_STRAGGLER",
    "COMPUTE_STRAGGLER", "INPUT_BOUND", "COMPUTE_BOUND",
    "WAIT_HEAVY",
    "IDLE_GPU",                    # <-- new
]

_STATUS_BY_KIND: dict[DiagnosisKind, str] = {
    ...,
    "IDLE_GPU": "IDLE-GPU",        # <-- new
}
```

### Step 2 — Add thresholds

```python
@dataclass(frozen=True)
class DiagnosisThresholds:
    ...
    # IDLE-GPU heuristic (step-time-only proxy):
    #   step_total_ms < idle_gpu_step_ms_max_warn
    #   AND compute_share >= idle_gpu_compute_share_min
    # Rationale: a healthy under-saturated job has very short steps where
    # most of the time is genuine compute (no dataloader stall, no wait).
    # We bias hard toward precision: only fire on clearly short, clearly
    # compute-dominated steps.
    idle_gpu_step_ms_max_warn: float = 20.0
    idle_gpu_step_ms_max_crit: float = 10.0
    idle_gpu_compute_share_min: float = 0.80
```

Numeric defaults are deliberately conservative. A step under 20 ms with 80%+ compute share is almost always under-saturated. **Document the precision/recall stance in the docstring**, not in PR review comments.

### Step 3 — Slot into the priority list

The new ordering, with `IDLE_GPU` placed *after* `COMPUTE_BOUND`:

```python
    """
    Priority
    --------
    1. INPUT_STRAGGLER / COMPUTE_STRAGGLER / STRAGGLER
    2. INPUT_BOUND
    3. WAIT_HEAVY
    4. COMPUTE_BOUND
    5. IDLE_GPU              <-- new
    6. BALANCED
    """
```

Multi-verdict resolution policy is currently *first-match-wins by priority list*. Document why `IDLE_GPU` slots after `COMPUTE_BOUND`: a 50-ms step that is 95% compute is `COMPUTE_BOUND`; only a sub-20-ms step that is also compute-dominated is `IDLE_GPU`. The two thresholds are non-overlapping by construction.

### Step 4 — Add the branch

Insert before the `BALANCED` fallback:

```python
    # 5) IDLE-GPU
    if (
        step_total <= thresholds.idle_gpu_step_ms_max_warn
        and compute_share >= thresholds.idle_gpu_compute_share_min
        and dl_share < thresholds.input_share_warn
        and wait_share < thresholds.wait_share_warn
    ):
        return _emit(
            kind="IDLE_GPU",
            severity=_severity(
                thresholds.idle_gpu_step_ms_max_warn - step_total + 0.0,
                thresholds.idle_gpu_step_ms_max_warn
                - thresholds.idle_gpu_step_ms_max_crit,
            ),
            reason=(
                f"Step is only {step_total:.1f}ms with {_pct(compute_share)} "
                f"compute share; the model may be under-saturating the GPU."
            ),
            action=(
                "Increase batch size, enable AMP/bf16, "
                "or train a larger model."
            ),
            worst_rank=None if single_rank else overall_worst_rank,
            note=(
                "Heuristic from step decomposition only; confirm with GPU "
                "utilization metrics before acting."
            ),
        )
```

The `note` is doing real work: it tells the reader the verdict is inferred, not directly measured. When the future `gpu_utilization` engine ships, this note can drop and severity can climb.

### Step 5 — Update summary thresholds (stricter)

`aggregator/summaries/step_time_diagnosis.py`:

```python
        thresholds=DiagnosisThresholds(
            ...
            idle_gpu_step_ms_max_warn=15.0,   # stricter: was 20.0 live
            idle_gpu_step_ms_max_crit=8.0,    # stricter: was 10.0 live
            idle_gpu_compute_share_min=0.85,  # stricter: was 0.80 live
        )
```

### Step 6 — Add a presentation override

`aggregator/summaries/diagnosis_presentation.py::present_step_time_summary_diagnosis`:

The live action `"Increase batch size, enable AMP/bf16, or train a larger model."` reads fine post-run. No override needed — but verify by reading the rendered card. If the live action contains "Watch" or "Monitor", override it.

### Step 7 — Wire into the formatters

The renderer formatters (`diagnostics/step_time_formatters.py`'s `format_cli_diagnosis` and `format_dashboard_diagnosis`) format the verdict from `StepDiagnosis` fields generically — they key off `status` and `severity`, not `kind`. **No change required** for a new kind unless you want a custom color or a special-case layout. If your new kind needs a custom palette, add a kind-specific branch in the formatter.

### Step 8 — Tests

`tests/test_step_time_diagnosis_idle_gpu.py`:

```python
"""
Tests for the IDLE_GPU verdict in build_step_diagnosis.
"""

import pytest

from traceml.diagnostics.step_time import (
    DEFAULT_THRESHOLDS, DiagnosisThresholds, build_step_diagnosis,
)
from traceml.renderers.step_time.schema import (
    StepCombinedTimeCoverage, StepCombinedTimeMetric, StepCombinedTimeSummary,
)


def _metric(key, *, median, worst=None, steps_used=50):
    if worst is None:
        worst = median
    return StepCombinedTimeMetric(
        metric=key, clock="mixed", series=None,
        summary=StepCombinedTimeSummary(
            window_size=100, steps_used=steps_used,
            median_total=float(median), worst_total=float(worst),
            worst_rank=0, skew_ratio=0.0, skew_pct=0.0,
        ),
        coverage=StepCombinedTimeCoverage(
            expected_steps=100, steps_used=steps_used,
            completed_step=steps_used, world_size=1, ranks_present=1,
            incomplete=False,
        ),
    )


class TestIdleGpu:
    def test_fires_on_short_compute_dominated_steps(self):
        # 12ms step, ~92% compute (fwd+bwd+opt = 11ms).
        metrics = [
            _metric("step_time", median=12.0),
            _metric("forward", median=6.0),
            _metric("backward", median=4.0),
            _metric("optimizer_step", median=1.0),
            _metric("dataloader_fetch", median=0.0),
            _metric("wait_proxy", median=1.0),
        ]
        diag = build_step_diagnosis(metrics)
        assert diag.kind == "IDLE_GPU"
        assert diag.status == "IDLE-GPU"
        assert "under-saturating" in diag.reason
        assert "batch size" in diag.action.lower()

    def test_does_not_fire_on_healthy_long_steps(self):
        metrics = [
            _metric("step_time", median=200.0),
            _metric("forward", median=100.0),
            _metric("backward", median=80.0),
            _metric("optimizer_step", median=18.0),
            _metric("dataloader_fetch", median=0.0),
            _metric("wait_proxy", median=2.0),
        ]
        diag = build_step_diagnosis(metrics)
        assert diag.kind != "IDLE_GPU"

    def test_does_not_fire_when_dataloader_dominates(self):
        metrics = [
            _metric("step_time", median=12.0),
            _metric("forward", median=2.0),
            _metric("backward", median=2.0),
            _metric("optimizer_step", median=0.5),
            _metric("dataloader_fetch", median=18.0),
            _metric("wait_proxy", median=0.0),
        ]
        diag = build_step_diagnosis(metrics)
        assert diag.kind == "INPUT_BOUND"

    def test_grey_zone_falls_back_to_balanced(self):
        # 25ms step (above warn threshold of 20ms): not IDLE_GPU.
        metrics = [
            _metric("step_time", median=25.0),
            _metric("forward", median=10.0),
            _metric("backward", median=8.0),
            _metric("optimizer_step", median=2.0),
            _metric("dataloader_fetch", median=2.0),
            _metric("wait_proxy", median=3.0),
        ]
        diag = build_step_diagnosis(metrics)
        assert diag.kind in ("BALANCED", "COMPUTE_BOUND")
```

The four cases in the precision/recall test are non-negotiable: positive case, negative case (healthy), negative case (different verdict wins), grey-zone case. **Any new verdict's PR must include these four.**

### Step 9 — Smoke test the summary

Build a tiny example that's intentionally idle-GPU shaped (200-iteration loop over `nn.Linear(8, 8)` on CUDA), run `traceml run /tmp/idle_example.py --mode summary`, and inspect the printed `Step Time` card. It should contain `- Diagnosis: IDLE-GPU`. If it shows `BALANCED` or `COMPUTE-BOUND`, either your thresholds are too tight or the example is not actually idle. Tune. Document the result of this smoke test in the PR.

---

## 5. Common patterns and exemplars

Reference table mapping diagnostic shape → "copy from this file."

| Pattern                                              | Copy from                                                                                                |
|------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| Single-metric threshold (share-of-step)              | `diagnostics/step_time.py:437` (`INPUT_BOUND`)                                                            |
| Combined threshold + skew gate                       | `diagnostics/step_time.py:467` (`COMPUTE_BOUND`)                                                          |
| Cross-rank attribution score                         | `diagnostics/step_time.py:611` (`_input_straggler_score`)                                                 |
| Composite "both A and B" verdict                     | `diagnostics/step_time.py:351` (`STRAGGLER` family with two scores)                                       |
| Pressure ratio against device capacity               | `diagnostics/step_memory.py:436` (`_pressure_fraction`)                                                   |
| Cross-rank skew threshold                            | `diagnostics/step_memory.py:220` (`IMBALANCE`)                                                            |
| Time-series trend (early vs. confirmed)              | `diagnostics/step_memory.py:316` (`_compute_window_creep_evidence`) → `analytics/trends.compute_trend_evidence` |
| Optional best-effort enrichment note                 | `diagnostics/step_time.py:172` (`_apply_trend_note`)                                                      |
| Stricter summary-mode thresholds via dataclass copy  | `aggregator/summaries/step_time_diagnosis.py:48` (`SummaryDiagnosisConfig`)                               |
| Adapter from SQLite rank aggregates → metric objects | `aggregator/summaries/step_time_diagnosis.py:173` (`_metric_from_rank_values`)                            |
| Multi-metric domain engine (allocated + reserved)    | `diagnostics/step_memory.py:140` (`build_step_memory_diagnosis` evaluates both, picks strongest)          |
| End-of-run action wording override                   | `aggregator/summaries/diagnosis_presentation.py:55` (`present_step_time_summary_diagnosis`)               |
| Composite cross-domain card (model-level)            | `diagnostics/model_diagnostics.py:24-144`                                                                 |

### Reusable helpers

- `validate_confidence(c)` in `diagnostics/common.py` — call from `__post_init__` of any verdict that exposes a `confidence` field.
- `diagnosis_to_dict(d, drop_none=False)` in `diagnostics/common.py` — the canonical JSON encoder for diagnosis dataclasses.
- `compute_trend_evidence(series, config=...)` in `traceml/analytics/trends.py` — use for any trend-shaped detection. Don't write your own linear-regression code.
- `_non_negative_finite`, `_share`, `_severity`, `_pct` in `diagnostics/step_time.py` — small numeric primitives that clamp NaN/Inf and format. Copy them; do not import them across modules (they are deliberately file-local).

---

## 6. Schema and contract rules

### 6.1 Dataclass fields

- **Required fields are inherited** from `BaseDiagnosis`. These are stable forever.
- **Optional fields** can be added to a subclass with a default; consumers must read them with `getattr(diag, "note", None)`.
- **Frozen dataclasses only.** Use `dataclasses.replace(diag, note=...)` to derive a new instance.

### 6.2 `kind` strings (the wire-stable enum)

`kind` is the JSON-keyed identity. It travels into `final_summary.json` and into the future TraceOpt longitudinal store.

- **Never rename a kind.** `INPUT_BOUND` is `INPUT_BOUND` forever.
- **Never change the meaning of a kind.**
- **The `status` string is presentation, the `kind` is contract.**

### 6.3 Severity values

`Severity = Literal["info", "warn", "crit"]`. Three values, stable forever.

### 6.4 The JSON envelope

The shape is `{kind, severity, status, reason, action, steps_used, worst_rank, note}`. This shape is consumed by the future TraceOpt run registry. Adding a field is fine; *never* removing or renaming.

---

## 7. Overhead budget and performance

End-of-run summary diagnostics run **once per run, at shutdown**. The budget is generous — a single SQLite scan over up to 50,000 step rows per rank, plus rank aggregation. Actual cost: low milliseconds.

Live diagnostics run **every aggregator tick** (default `render_interval_sec=2.0`). Engine call is sub-millisecond on top of `compute.py`'s metric build.

Hot-path rules:

1. **No allocations of large arrays.**
2. **No `time.sleep`, no I/O, no logging at INFO+.**
3. **Trend-note enrichment is best-effort.** Wrap in `try/except` and return the unenriched diagnosis on failure.
4. **No `torch.cuda.synchronize()`, no `torch.*` calls at all.** The diagnostics package depends only on numpy and stdlib.

---

## 8. Testing

There are currently no dedicated diagnostic-engine test files. The closest references:

- `tests/test_trend_core.py` — pure-function tests against the trend engine. Excellent shape to copy for any trend-based diagnostic.
- `tests/test_seq_counter.py` — `Database` + `DBIncrementalSender` contract test.

Your new diagnostic should add the file the codebase has been missing: `tests/test_<engine>_diagnosis_<verdict>.py` for a focused verdict test.

### What a new diagnostic's tests must cover

Four cases — non-negotiable:

1. **Positive (high-confidence true positive).** Synthetic input that is unambiguously the pathology. Assert `kind == "<your_kind>"` and verify `reason` and `action` text.
2. **Negative (healthy).** Synthetic input that should NOT fire the verdict. Assert `kind != "<your_kind>"`.
3. **Negative (a different verdict wins).** Input that satisfies one of your thresholds but a higher-priority verdict's branch. Assert the higher-priority kind fires.
4. **Grey zone.** Input that sits just below your `*_warn` threshold. Assert the verdict does NOT fire.

Add a fifth if you have a time-series component:

5. **Trend stability.** Feed a noisy-but-flat series (no drift) and assert the trend signal is `False`.

### Integration smoke

Pathological run shows your verdict; healthy run does not. PR description should include the output of running both synthetic examples through `traceml run --mode summary`.

---

## 9. Common pitfalls

Numbered with symptom and fix.

1. **Symptom:** Verdict fires correctly in a unit test but never appears in the live UI or summary card.
   **Cause:** You added the kind to `DiagnosisKind` but did not add a row to `_STATUS_BY_KIND`. `_mk_diag` raises `KeyError` on the missing entry, the engine's outer `try/except` (in the *caller*, not in the engine) swallows the exception and emits nothing.
   **Fix:** Always update both. Better: add a `_STATUS_BY_KIND` completeness assertion in a test.

2. **Symptom:** False positive at 3am.
   **Cause:** Threshold is too loose.
   **Fix:** Document a precision-first stance up-front. Tune thresholds to <5% false-positive rate against known-healthy examples *before* shipping. Recall: an absent verdict is fine. A wrong one is destructive.

3. **Symptom:** Threshold drift after a PyTorch upgrade.
   **Cause:** Hard-coded numeric defaults that depend on a specific PyTorch release's kernel timing.
   **Fix:** Document the PyTorch version range you tuned against in the threshold class docstring. When the next major PyTorch lands, re-run the synthetic suite.

4. **Symptom:** Magic numbers in branch conditions, not in the thresholds dataclass.
   **Cause:** `if step_total < 20.0 and compute_share > 0.8:` — the number is hidden from the summary adapter and from tests.
   **Fix:** Every numeric default goes in the `*Thresholds` dataclass. Branches read fields.

5. **Symptom:** Verdict surfaces with no actionable recommendation, or one the user can't follow.
   **Cause:** Action text was written generically.
   **Fix:** Action must name a *fix that exists in the user's stack*. Where the recommendation is contextual, hedge it. If the action genuinely is "consult docs," the verdict isn't ready.

6. **Symptom:** Two verdicts overlap; both could fire on the same data. The "first to write" wins.
   **Cause:** Priority order is not principled.
   **Fix:** Document the priority list in the engine docstring. The dispatcher must enumerate branches in priority order with explicit `return` statements.

7. **Symptom:** Verdict depends on data the sampler doesn't yet emit; tests pass on a hand-built metric object, but live runs always show `NO_DATA`.
   **Cause:** The metric key your engine looks up is absent from the renderer's `compute.py` output.
   **Fix:** Add the metric to the renderer's `compute.py` *first*. Don't bolt a `sqlite3.connect()` into the engine to work around a missing metric.

8. **Symptom:** Verdict text is hard-coded in branch bodies; cannot be customized for end-of-run wording, future i18n, or product experimentation.
   **Cause:** No separation between detection truth and presentation.
   **Fix:** The presentation layer (`aggregator/summaries/diagnosis_presentation.py`) exists for exactly this. Engine emits canonical `action` text; presentation rewrites it for medium and tense.

9. **Symptom:** `confidence` field is set but inconsistent across verdicts within the same engine.
   **Cause:** The semantics of confidence are not pinned down.
   **Fix:** `step_memory.py` is the precedent: every kind that fires non-NO_DATA sets a calibrated confidence. If you add confidence, set it on every kind in the engine; otherwise leave it `None` everywhere.

10. **Symptom:** `final_summary.json` shape changed; downstream consumer fails to decode.
    **Cause:** A field was renamed or removed from the diagnosis JSON. Wire compat applies here too.
    **Fix:** Add fields, never remove. Renames go through a deprecation cycle.

11. **Symptom:** Trend-note enrichment occasionally crashes a render tick.
    **Cause:** The trend engine receives a degenerate series and raises.
    **Fix:** Wrap the trend call in `try/except` and return the unenriched diagnosis. Do not let enrichment failures change verdict truth.

12. **Symptom:** A verdict fires on multi-rank data but `worst_rank` is `None` or wrong.
    **Cause:** The engine's `worst_rank` selection used the wrong metric's worst-rank.
    **Fix:** Each branch sets `worst_rank` from the metric that triggered it. When the branch is single-rank-friendly, use `worst_rank=None if single_rank else overall_worst_rank`.

---

## 10. Checklist before opening a PR

Beyond the standard `principles.md` checks:

1. [ ] New `kind` added to `DiagnosisKind` and to `_STATUS_BY_KIND` in the engine module.
2. [ ] Numeric defaults added to the `*Thresholds` dataclass with a docstring explaining the meaning, not just the value.
3. [ ] Priority position documented in the `build_*_diagnosis` docstring; branch ordered correctly in the dispatcher.
4. [ ] Each branch returns a fully-formed verdict via `_emit` / `_mk_diag` with explicit `reason`, `action`, `worst_rank`, and (where relevant) `note`.
5. [ ] **Precision/recall stance documented** in the threshold docstring. Bias toward precision.
6. [ ] **Recommendation is actionable** in the user's stack.
7. [ ] **Smoke test (positive):** the verdict fires on a known-pathological synthetic example. PR description includes the run output.
8. [ ] **Smoke test (negative):** the verdict does NOT fire on a known-healthy synthetic example.
9. [ ] **Smoke test (different verdict):** when a higher-priority verdict's threshold is also crossed, that one wins.
10. [ ] Unit test file `tests/test_<engine>_diagnosis_<verdict>.py` with at least the four cases in §8.
11. [ ] Summary-mode thresholds updated in `aggregator/summaries/step_time_diagnosis.py` (or sibling for a new engine) — stricter than live by default.
12. [ ] `present_*_summary_diagnosis` reviewed for action-text rewording; override added if live action reads wrong post-run.
13. [ ] If your verdict adds a new domain engine: corresponding `aggregator/summaries/<domain>.py` card-builder added and wired into `final_summary.py::generate_summary`.
14. [ ] Sampler dependency check: which sampler emits the data you consume? Is that sampler stable, profiled in `watch` / `run` / `deep`, and tested?
15. [ ] CHANGELOG entry naming the new verdict class.
16. [ ] No new top-level dependencies. The `diagnostics/` package depends only on numpy and stdlib.

---

## 11. Appendix: where the diagnostic library is going

### 11.1 Tuning verdicts against accumulated TraceOpt run data

`traceml_why.md` §6.3 makes the long-running argument: verdicts are opinions, opinions encode product judgment, and product judgment compounds with usage data. The first version of every verdict ships with hand-tuned thresholds. The second version should be tuned against the distribution of runs that actually flowed through the TraceML → TraceOpt pipeline. Today there is no infrastructure for this — runs ship with frozen thresholds. Designing the threshold-tuning loop is its own project; flagging here that any verdict shipped now should anticipate being re-tuned later.

### 11.2 Multi-verdict resolution policy

Today the policy is **single-verdict-per-domain, priority-ordered, first-match-wins**. As it grows, two limitations bite:

1. **Cross-domain co-firing.** A run might be `INPUT_BOUND` AND `MEMORY CREEP` simultaneously. There is no "this run's dominant verdict" promotion.
2. **Within-domain "and also" verdicts.** A run that is `INPUT_BOUND` AND visibly `IDLE_GPU` currently surfaces only `INPUT_BOUND` because of priority.

The right design is a **dominant-verdict resolver** that takes the per-engine outputs and emits a single ordered list with one "primary" and zero or more "also-see" entries. Out of scope for adding a single verdict.

### 11.3 Constants file for verdict text

All `reason` and `action` strings are currently inline in the engine branches. For i18n, A/B testing of recommendations, or even consistent wording across a verdict family, the right factoring is a `diagnostics/text.py` keyed by `(kind, locale)`.

### 11.4 A `gpu_utilization` engine

Several pathologies in `traceml_why.md` §3 — most prominently `IDLE-GPU` (§3.4) — would land cleaner with direct GPU utilization as a first-class metric. The system sampler already emits `util_percent` per GPU. What's missing is a renderer-side `compute.py` that reduces it to a windowed metric object and a `gpu_utilization` engine alongside `step_time` and `step_memory`.

### 11.5 NCCL-aware `COMM-BOUND`

A surface-level `COMM-BOUND` based on backward-phase share is feasible from existing data; a deep `COMM-BOUND` requires NCCL instrumentation that doesn't exist yet. If you ship the surface-level version, document the limitation clearly in the verdict's `note`.

### 11.6 `HANG` detection

`HANG` is detectable from the *absence* of step events. It does not fit the "build_diagnosis(metrics) → verdict" shape because the input is the absence of input. The right home is probably the runtime tick loop itself, with a heartbeat watchdog that fires a special structural verdict when no rank has emitted a step in `T` seconds.

---

## Gaps / ambiguities encountered while writing this guide

These are places where the current source does not fully pin down a contract.

- **No dedicated diagnostic-engine test files.** `traceml/tests/` contains `test_trend_core.py` (which exercises the trend engine underneath `step_memory_diagnosis`) but no `test_step_time_diagnosis.py` or equivalent. The existing engines are tested only indirectly via renderer / summary integration. Adding `tests/test_<engine>_diagnosis.py` with the four-case template is the highest-leverage gap-closing PR.
- **`confidence` semantics are inconsistent across engines.** `StepMemoryDiagnosis` populates `confidence` on every non-NO_DATA kind. `StepDiagnosis` never populates it. We need either a documented "engines opt in" rule or a calibrated value per kind in step-time.
- **Engine-vs-renderer-vs-summary boundary.** The split is mostly clean, but `aggregator/summaries/diagnosis_presentation.py` is in the summary tree and has no peer in the live-renderer tree — meaning live views show the engine's raw `action` text without re-skinning. If we ever want different live-vs-summary wording, the live-side override layer doesn't exist yet.
- **No multi-verdict resolver.** See §11.2.
- **No constants file for verdict text.** All strings inline in the engine. Fine for 8 verdicts, awkward at 30. See §11.3.
- **Threshold tuning is folklore.** The numeric defaults are tuned against a small handful of internal example runs, not against a labeled corpus. Building an offline harness that re-runs synthetic pathological / healthy examples through the engine and reports precision/recall is the right next step before the verdict library grows past a dozen kinds.
- **`final_summary.json` JSON shape is not formally versioned.** The payload includes a `"schema_version": 1` field at the top but no per-section version. A `"diagnosis_schema_version"` field per engine would help.
