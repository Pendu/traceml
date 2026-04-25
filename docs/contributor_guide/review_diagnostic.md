# How to review a diagnostic-verdict PR

> Internal contributor guide. Audience: one trusted co-founder / new engineer
> reviewing TraceML PRs. Companion to `add_diagnostic.md`. Not for public docs.

This guide teaches you how to review a PR that adds or modifies a diagnostic verdict in `src/traceml/diagnostics/` (and its summary adapter under `aggregator/summaries/`). It assumes you have already read `add_diagnostic.md` (the author's guide) and have a working mental model of [W11](../deep_dive/code-walkthroughs.md#w11-summaries--diagnostics--end-of-run-analysis). The seven-step workflow in §1 is the same meta-pattern used by `review_patch.md`; only §3 onward is diagnostic-specific.

---
Feature type: diagnostic verdict
Risk level: medium (verdicts shape user trust; one false positive at 3am erodes the product narrative)
Cross-cutting impact: multiple subsystems (engine + live renderer + summary adapter + presentation + future TraceOpt regression detector)
PyTorch coupling: none (engine consumes sampler-derived metric objects, not PyTorch internals)
Reference reviews: none yet — `step_time` and `step_memory` engines are the only landed exemplars
Companion author guide: `add_diagnostic.md`
Last verified: 2026-04-25
---

## 1. The meta-review-workflow (applies to every TraceML PR)

Every diagnostic review walks the same seven steps in order. Skipping any of them is how a flawed verdict ships:

1. **Anchor** the PR diff to the relevant W-walkthroughs and the `traceml_why.md` taxonomy. Read the PR through your existing mental models, not line-by-line.
2. **Run the diagnostic-family consistency check.** Build the table from §3 of this guide and grade the new verdict against the existing engines on each axis. Discrepancies are either justified deviations (document them) or bugs.
3. **Apply the diagnostic-class failure-mode catalogue** from §4. Each category maps to a known bug shape. Walk the diff with each shape in mind, with §4.1 (false-positive-at-3am) treated as the load-bearing concern.
4. **Apply the four meta-questions** from §5: new pathology vs. existing kind, engine-vs-adapter boundary, wire-format `kind` as contract, invariant preservation.
5. **Write a verification gate** for each concern: a 5–15 line synthetic input that produces a clear pass/fail verdict. "I think this fires too eagerly" becomes "here's the metric set that proves it." See §6.
6. **Draft comments at the right granularity** — line comment for specific code suggestions, PR-level comment for taxonomy / threshold-stance / priority-ordering concerns. Hold parking-lot items back. See §7.
7. **Land the verdict** — approve / approve-with-changes / block. Criteria in §8.

The reviewer's job ends with a 2–3 sentence executive summary the maintainer can read without opening the diff. That goes in the verdict (§8).

This same seven-step shape applies to patch PRs (`review_patch.md`), sampler PRs, renderer PRs — only the consistency table and the failure-mode catalogue change. For diagnostics, **the failure-mode catalogue is bigger and more subjective than for patches**, because verdicts are opinions and reasonable reviewers can disagree about thresholds. The verification-gate discipline (§6) is what keeps that disagreement productive.

---

## 2. Step 1 — Anchor the PR to your walkthroughs

The first thing you do with a diagnostic PR is **not** open the engine diff. Open three things:

- [W11 §"Summaries + diagnostics"](../deep_dive/code-walkthroughs.md#w11-summaries--diagnostics--end-of-run-analysis) — the end-of-run code walkthrough. Diagnostics live inside the summary subsystem's call graph; if you don't have that graph in cache you will misread the adapter layer.
- `traceml_why.md` §3 — the taxonomy of wasted GPU time. Every verdict the codebase will ever ship maps to one row of that table. Identify the row this PR is claiming.
- `add_diagnostic.md` §1.1 — the "what is a diagnostic" definition. The single load-bearing claim is "diagnostics are *interpretations* of metrics, not metrics." If the PR is shipping a metric and labelling it a verdict, anchor on this and push back early.

Two reasons for this anchoring discipline:

- The diagnostic family has documented invariants (one engine, two callers; thresholds in a frozen dataclass; `NO_DATA` as a first-class verdict; fail-open at every public entry point). You'll be checking the diff against those invariants, so they need to be in cache.
- A diagnostic PR will touch four to seven files in stereotyped ways. PR_87-style file mapping collapses the diff into mechanical changes plus one substantive change.

### How to anchor

For each file in the diff, write down (in your review notes, not the PR yet):

| File pattern | What kind of change should this be? |
|---|---|
| `src/traceml/diagnostics/<engine>.py` (new file) OR `src/traceml/diagnostics/<existing_engine>.py` (extension) | The substantive change. New `kind`, new threshold defaults, new branch, possibly new helpers. |
| `src/traceml/diagnostics/<engine>_trend.py` (optional) | Trend-note enrichment. Best-effort, wrapped in try/except. |
| `src/traceml/diagnostics/<engine>_formatters.py` | Mechanical — usually unchanged unless the new kind needs a custom palette. |
| `src/traceml/diagnostics/common.py` | Should be **untouched** for a single-verdict PR. If it changed, ask why. |
| `src/traceml/aggregator/summaries/<engine>_diagnosis.py` | Adapter: summary-mode `*Thresholds` instance with stricter values. |
| `src/traceml/aggregator/summaries/diagnosis_presentation.py` | Action-text override for end-of-run wording, if needed. |
| `src/traceml/aggregator/summaries/<engine>.py` | Card builder. Should NOT contain new threshold logic. |
| `src/traceml/aggregator/final_summary.py` | Should be **untouched** unless this PR adds a brand-new domain engine (new card). |
| `src/traceml/renderers/<engine>/renderer.py` | Should be **untouched** — formatters key off `status`/`severity`, not `kind`. |
| `tests/test_<engine>_diagnosis_<verdict>.py` (NEW) | The four-case template (positive, healthy negative, different-verdict-wins, grey zone). |

If a file in the diff doesn't fit the table, that's a flag — the PR is doing something architecturally novel, and you should ask why before proceeding. Two flags to watch for specifically:

- **Threshold logic added to `aggregator/summaries/<engine>.py`** (the card builder). That's a §4.6 boundary violation.
- **`aggregator/final_summary.py` modified for a single-verdict extension.** New verdicts inside an existing engine require zero `final_summary.py` edits. If the diff touches it, the PR is either adding a new domain engine (verify) or doing something wrong.

The point: **after anchoring, you should have one substantive engine file to read deeply, one summary adapter to skim, and 3–5 mechanical / test files to confirm.**

---

## 3. Step 2 — The diagnostic-family consistency table

Every diagnostic engine slots into a small set of axes. The reviewer's job is to fill in the row for the new verdict (or new engine) and grade each cell against `step_time` and `step_memory`.

### 3.1 The columns (these don't change PR-to-PR)

| Axis | What it asks |
|---|---|
| **Engine location** | Is the engine file in `src/traceml/diagnostics/<engine>.py`? **Not** in `aggregator/summaries/`? |
| **`BaseDiagnosis` subclass** | Is there a frozen dataclass in `diagnostics/<engine>.py` inheriting from `BaseDiagnosis` with the four required fields? |
| **`DiagnosisKind` extension** | Is the new kind a `Literal` member of the engine's kind enum? Is a row added to `_STATUS_BY_KIND`? |
| **Thresholds** | Frozen `*Thresholds` dataclass with explicit defaults? Each threshold has a docstring on meaning, not just value? |
| **Priority position** | Is the position in the priority list documented in the `build_*_diagnosis` docstring? Is the branch placed at the right spot in the dispatcher? |
| **Branch shape** | Does the branch call `_emit(kind=..., severity=..., reason=..., action=..., worst_rank=..., note=...)` with all fields explicitly set? |
| **Action text** | Does the action name a fix that exists in the user's stack? Does the past-tense reading work in the summary card? |
| **Confidence** | Is `confidence` set consistently across all kinds in this engine, or all `None`? Validated in `__post_init__`? |
| **Numeric guards** | Does every numeric input go through a `_non_negative_finite` (or equivalent) helper? NaN/Inf clamped to 0.0? |
| **Summary adapter** | Is there a stricter `*Thresholds` instance in `SummaryDiagnosisConfig` (or sibling) for the summary path? |
| **Presentation override** | If the live `action` contains "Watch" / "Monitor", is there an override in `diagnosis_presentation.py`? |
| **Tests** | Four cases (positive, healthy negative, different-verdict-wins, grey zone)? Plus a fifth for trend-shaped detection? |

### 3.2 The current state (April 2026)

Fill the table for the existing exemplars before adding a column for the new verdict.

| Axis | `StepDiagnosis` (`step_time.py`) | `StepMemoryDiagnosis` (`step_memory.py`) |
|---|---|---|
| Engine location | `src/traceml/diagnostics/step_time.py` | `src/traceml/diagnostics/step_memory.py` |
| Base subclass | `StepDiagnosis(BaseDiagnosis)`, fields `kind, steps_used, worst_rank, note, confidence` | `StepMemoryDiagnosis(BaseDiagnosis)`, fields `kind, metric, steps_used, worst_rank, note, confidence` |
| Kind enum | `DiagnosisKind = Literal["NO_DATA", "BALANCED", "STRAGGLER", "INPUT_STRAGGLER", "COMPUTE_STRAGGLER", "INPUT_BOUND", "COMPUTE_BOUND", "WAIT_HEAVY"]` | `StepMemoryDiagnosisKind = Literal["NO_DATA", "BALANCED", "HIGH_PRESSURE", "IMBALANCE", "CREEP_EARLY", "CREEP_CONFIRMED"]` |
| `_STATUS_BY_KIND` | All 8 kinds mapped | All 6 kinds mapped |
| Thresholds dataclass | `DiagnosisThresholds` (frozen, 11 fields, "Design notes" docstring) | `StepMemoryDiagnosisThresholds` (frozen, 12 fields including `TrendConfig`) |
| Priority docstring | `build_step_diagnosis` lists 5 priority levels in docstring | `build_step_memory_diagnosis` module docstring lists 5 priority levels |
| Priority order | STRAGGLER family > INPUT_BOUND > WAIT_HEAVY > COMPUTE_BOUND > BALANCED | HIGH_PRESSURE > IMBALANCE > CREEP_CONFIRMED > CREEP_EARLY > BALANCED |
| Branch shape | `_emit(kind=..., severity=..., reason=..., action=..., worst_rank=..., note=...)` with `_finalize` trend hook | `_mk_diag(kind=..., severity=..., metric=..., steps_used=..., worst_rank=..., reason=..., action=..., confidence=...)` |
| `confidence` | **None on every kind** (engine omits it) | **Set on every non-NO_DATA kind**: 0.9/0.8 (HIGH_PRESSURE), 0.85/0.75 (IMBALANCE), per-creep score (CREEP_*) |
| Numeric guard | `_non_negative_finite` clamps NaN/Inf to 0.0; used by every total/share/skew helper | Same pattern via internal helpers |
| Summary adapter | `aggregator/summaries/step_time_diagnosis.py::SummaryDiagnosisConfig` — stricter `input_share_warn=0.30` (vs. 0.25 live), `min_steps_for_confident_diag=20` (vs. 8) | sibling pattern in `aggregator/summaries/step_memory_diagnosis.py` (verify in PR) |
| Presentation override | `aggregator/summaries/diagnosis_presentation.py::present_step_time_summary_diagnosis` rewrites "Wait for a fuller window." → past-tense card text | sibling override for memory verdicts |
| Trend enrichment | `_apply_trend_note` wraps `build_step_trend_note` in try/except, returns unenriched diagnosis on failure | `_compute_window_creep_evidence` delegates to `analytics/trends.compute_trend_evidence` |
| Tests today | None dedicated; tested indirectly through renderer/summary integration | None dedicated; `test_trend_core.py` exercises the trend engine |

When reviewing, **add a column** for the new verdict (or new engine) and walk every row. Three outcomes per cell:

- Matches the family — note it and move on.
- Differs from the family — demand a justification in the PR description or a comment in the engine file.
- Cell is empty / undecidable from the diff — ask the author.

### 3.3 The two recurring inconsistencies to call out

These are visible in the table above; reviewer should explicitly ask the PR author to pick a side rather than silently extending the inconsistency:

1. **`confidence` semantics differ between engines.** `StepDiagnosis` never sets it; `StepMemoryDiagnosis` sets it on every kind. A PR that adds a new kind should match its **engine's** existing convention. A PR that adds a new engine should pick one stance and document it in the engine docstring. Don't silently mix.
2. **Tests don't exist yet for either engine.** The author's guide §8 is explicit that the four-case template is non-negotiable. Treat the PR as the opportunity to *establish* the test file. If the PR ships without `tests/test_<engine>_diagnosis_<verdict>.py`, that's a §8.3 block.

### 3.4 The table is the most reusable artifact in this guide

Every future diagnostic review should rebuild this table. Two reasons:
- The act of filling it forces you to read the verdict with the family in mind, which catches "this differs in ways the author didn't notice."
- The completed table goes in your review notes, and over time becomes the reviewer's contract test for the diagnostic family. (See "Gaps" — formalising this is on the wishlist.)

---

## 4. Step 3 — Diagnostic-class failure modes

Twelve categories. Walk the diff with each one in mind. The first one is load-bearing for the entire product.

### 4.1 False-positive at 3am — the single most destructive bug class for diagnostics

Applies to: every threshold in every branch.

The bug shape: threshold is too loose, the verdict fires on a healthy run, the user wakes up to a "INPUT-BOUND" pager and discovers their run was fine. From `traceml_why.md` §5.5 and `add_diagnostic.md` §9.2: a wrong verdict erodes trust forever, an absent verdict is recoverable.

**What to check:**

- For each new threshold, ask: "what's the precision/recall stance?" The PR description must answer this. If the answer is missing, push back before reading code.
- For each branch, demand a synthetic healthy example in the test file that does NOT fire the verdict. This is the **verification gate** (§6) — without it, the threshold value is folklore.
- Is the threshold biased toward precision over coverage? `add_diagnostic.md` §2 is explicit: "bias toward precision." A `_warn` threshold of `0.20` for an input-share rule is suspect; the live default is `0.25` and the summary default is stricter at `0.30`. Numbers below the live default need justification.
- For trend-shaped detection (creep, drift): is there an early/confirmed split? `step_memory.py` uses `CREEP_EARLY` (severity=info, low confidence) for ambiguous cases and `CREEP_CONFIRMED` (severity=warn, higher confidence) for unambiguous ones. A single-tier trend verdict is suspect — the early tier protects against false positives at low evidence.

**Single biggest reviewer move:** before approving, mentally run the new verdict against a typical healthy training run shape (200 ms step, 70 ms forward, 60 ms backward, 20 ms optimizer, 10 ms dataloader, ~40 ms wait_proxy, no skew). Does any branch fire? If so, the threshold is wrong.

### 4.2 `DiagnosisKind` / `_STATUS_BY_KIND` mismatch

Applies to: every PR that extends an engine's kind enum.

The bug shape from `add_diagnostic.md` §9.1: the kind is added to `DiagnosisKind` but no row is added to `_STATUS_BY_KIND`. `_mk_diag` does `status=_STATUS_BY_KIND[kind]`, which raises `KeyError`. The engine's outer `try/except` (in the *caller*, not in the engine) swallows the exception and emits nothing. Unit tests that check `kind == "NEW_KIND"` pass because tests build `StepDiagnosis(...)` directly; live runs silently never fire.

**What to check:**

- Grep the diff for `DiagnosisKind = Literal[`. If a new entry is added, verify `_STATUS_BY_KIND` gains a corresponding row in the same diff.
- Demand a `_STATUS_BY_KIND` completeness test: `for kind in get_args(DiagnosisKind): assert kind in _STATUS_BY_KIND`. If the test isn't in the PR, ask for it as a one-line addition.
- Run the test suite locally; the integration test should drive the engine through the new branch and exercise `_mk_diag`. If only synthetic-dataclass tests exist, the bug is invisible.

### 4.3 Magic numbers in branches

Applies to: every branch with a numeric comparison.

The bug shape from `add_diagnostic.md` §9.4: `if step_total < 20.0 and compute_share > 0.8:` — the numbers are inlined in the branch condition instead of in the `*Thresholds` dataclass. Two harms:

- The summary adapter cannot override them. `SummaryDiagnosisConfig` rebuilds `DiagnosisThresholds(...)` with stricter values; inline numbers fork.
- Tests cannot inject relaxed values. `step_time.py` has `min_steps_for_confident_diag=8` so tests can build `DiagnosisThresholds(min_steps_for_confident_diag=2)` and avoid `NO_DATA` on small synthetic inputs. Inline numbers force tests to construct large fake inputs.

**What to check:**

- Grep the diff for numeric literals in branch conditions: `if .* >= [0-9]\.`, `if .* < [0-9]\.`. Any hit that isn't `0.0` (a degeneracy guard) is a candidate magic number.
- Verify each candidate has a corresponding field in the engine's `*Thresholds` dataclass.
- Verify the `*Thresholds` field has a docstring entry explaining the **meaning** of the value, not just its name. `step_time.py::DiagnosisThresholds` "Design notes" block is the model.

### 4.4 Non-actionable action text

Applies to: every branch's `action=` argument.

The bug shape: action text reads OK but the recommendation doesn't apply to the user's stack. Examples:

- `"Increase batch size"` when the user's batch size is fixed by GPU memory or by experiment design. (Subtle: this might still be correct if the verdict is `IDLE_GPU`, where the whole point is the GPU is under-saturated. But for `INPUT_BOUND` it's wrong.)
- `"Consult docs"` when the project has no public docs page for the verdict.
- `"Use torch.compile"` when the user's PyTorch version doesn't support it.
- `"Tune NCCL_*"` for a single-rank run where NCCL isn't loaded.

The deeper rule from `add_diagnostic.md` §9.5: **action text must name a fix that exists in the user's stack.** Generic recommendations are ignored; specific recommendations build trust.

**What to check:**

- For each new branch, read the `action=` string aloud as if the user just got pinged at 3am. Is the recommendation something they can do *right now* without reading TraceML's source code?
- Check the past-tense reading. Live action `"Wait for a fuller window."` reads wrong in a post-run summary card; that's why `present_step_time_summary_diagnosis` rewrites it. If the new live action contains "Watch", "Monitor", "Wait", "Inspect [in real-time]", a presentation override is required (see §4.11 below and §6.5).
- Does the action mention specific TraceML telemetry (`worst_rank`, the dominant phase label) so the user can act without re-running? Generic actions are weaker.

### 4.5 Priority overlap / non-principled tiebreaks

Applies to: every PR that adds a new kind to an existing engine.

The bug shape from `add_diagnostic.md` §9.6: two verdicts could fire on the same data; the dispatcher's first-match-wins ordering decides the outcome, but no documented rule explains *why* this kind slots before/after the others.

`step_time.py:217` documents the order: `INPUT_STRAGGLER / COMPUTE_STRAGGLER / STRAGGLER` > `INPUT_BOUND` > `WAIT_HEAVY` > `COMPUTE_BOUND` > `BALANCED`. This is principled: cross-rank stragglers are more actionable than overall bound classification, and bound classifications are more actionable than `BALANCED`. A new kind must justify its slot on a similar principle.

**What to check:**

- Is the priority comment in `build_*_diagnosis` updated to include the new kind in the right slot?
- Does the PR description argue *why* the kind slots there? If the argument is "I put it last because the others take priority," that's not a principle.
- Construct a synthetic input that satisfies *both* the new threshold and one of the higher-priority thresholds. Verify the higher-priority verdict wins — this is non-negotiable test case 3 (different-verdict-wins) from §8.
- Construct a synthetic input that satisfies the new threshold and one of the lower-priority thresholds (`BALANCED` for step-time). Verify the new verdict wins.

### 4.6 Engine-vs-adapter boundary violation

Applies to: PRs that touch both `src/traceml/diagnostics/<engine>.py` and `src/traceml/aggregator/summaries/<engine>_diagnosis.py`.

The bug shape from `add_diagnostic.md` §3.8 + §9 (implicit): threshold logic is duplicated in the summary adapter. The engine has one branch shape; the adapter has a parallel branch shape with slightly different numbers. Two harms:

- A bug in one path doesn't fix the other. Live and summary diverge on the same data.
- The single-engine principle (`add_diagnostic.md` §1.1 "One engine, two callers") is broken. Future maintainers can't trust which side is canonical.

`SummaryDiagnosisConfig` at `aggregator/summaries/step_time_diagnosis.py:48` is the correct shape: it holds **only** a stricter `DiagnosisThresholds` instance, no branch logic. The adapter's job is to project rank aggregates into the metric schema and call `build_step_diagnosis(metrics, thresholds=config.thresholds)`.

**What to check:**

- Grep `aggregator/summaries/<engine>_diagnosis.py` for `if .*share .* >=` or any condition that looks like a verdict branch. There should be **none**. If you find one, the logic belongs in the engine.
- Verify the adapter only does: (a) build metric objects from rank aggregates, (b) call `build_*_diagnosis(metrics, thresholds=stricter)`. Anything else is suspect.
- Is `SummaryDiagnosisConfig` a frozen dataclass with a single `thresholds` field (plus an optional `min_steps_for_diag`)? Or did the PR add per-kind override fields like `idle_gpu_share_warn_summary: float`? The latter is a smell — pass a custom `*Thresholds` instance instead.

### 4.7 Wire-format JSON shape break

Applies to: any PR that touches the dataclass field set or the `_STATUS_BY_KIND` map for an existing kind.

The bug shape from `add_diagnostic.md` §6.4 + §9.10: the diagnosis dataclass JSON envelope is `{kind, severity, status, reason, action, steps_used, worst_rank, note}` (plus `metric`/`confidence` for some engines). This shape is consumed by `final_summary.json`, which is the input to the future TraceOpt regression detector. **Adding fields is fine; renaming or removing is a wire break.**

Specific things that break the wire:

- Renaming an existing kind: `INPUT_BOUND` → `INPUT-BOUND` (changing the machine-friendly identifier). Don't.
- Renaming the human label in `_STATUS_BY_KIND` for an existing kind. Less destructive (status is presentation), but the JSON serializes `status` too — TraceOpt may key off it.
- Removing or renaming `severity`, `reason`, `action`, `worst_rank`, or `note`. Each of these is consumed downstream.
- Changing the meaning of `confidence` (from `[0.0, 1.0]` to a Z-score, say) without versioning.

**What to check:**

- For PRs that add a new kind only: this concern doesn't apply. The new kind is additive.
- For PRs that touch `BaseDiagnosis` in `common.py`: this is a **major** concern. Demand a migration plan. The base contract is wire-stable forever.
- For PRs that touch `_STATUS_BY_KIND` rows for *existing* kinds: ask whether the rename is necessary. If yes, demand a deprecation cycle entry in CHANGELOG.

### 4.8 Confidence inconsistency

Applies to: PRs that set `confidence` on a new kind, in an engine where existing kinds don't.

The bug shape from `add_diagnostic.md` §9.9: `step_memory.py` populates `confidence` on every non-NO_DATA kind (0.9 / 0.85 / 0.75 / etc.); `step_time.py` never populates it. A PR that adds `IDLE_GPU` to `step_time.py` and sets `confidence=0.7` is the *only* kind in that engine with a confidence — readers can't tell if `confidence=None` means "low confidence" or "engine doesn't compute it."

**What to check:**

- Look at the existing engine's pattern. If every kind has `confidence`, the new kind must too. If no kind has it, the new kind probably shouldn't either — or the PR must populate `confidence` for every existing kind in the same diff.
- If the PR populates confidence for every kind, verify the values are calibrated. `step_memory.py` uses `0.9 if sev == "crit" else 0.8` — that's a documented mapping. Hand-picked values without a rationale are weaker.
- `validate_confidence` in `__post_init__` is the only structural enforcement. Verify the new dataclass calls it.

### 4.9 Trend-note crashes verdict

Applies to: PRs that add or modify trend-note enrichment.

The bug shape from `add_diagnostic.md` §9.11: trend enrichment receives degenerate input (single-step series, all-zero values, NaN), the trend engine raises, and the whole diagnosis fails. `step_time.py::_apply_trend_note` is the correct shape: try/except around the trend call, return unenriched diagnosis on failure.

**What to check:**

- Every trend-related call must be inside a `try/except Exception` block that returns the original (or a `NO_DATA`) diagnosis on failure.
- The trend engine itself (`analytics/trends.compute_trend_evidence`) is shared infrastructure — don't write a new one. If the PR ships a parallel trend implementation, push back: shared trend logic lives in `analytics/trends.py`.
- Test with degenerate input: empty series, single-element series, all-zero series, NaN series. The diagnosis must still return a valid `*Diagnosis` instance.

### 4.10 Worst-rank attribution bug

Applies to: every multi-rank verdict.

The bug shape: the branch sets `worst_rank` from the wrong metric. For `INPUT_STRAGGLER`, the worst rank should come from the dataloader metric (the metric that triggered the verdict). For `COMPUTE_STRAGGLER`, it should come from the dominant compute phase. For `BALANCED` or single-rank cases, it should be `None`.

`step_time.py:404` (`worst_rank=dl_worst_rank`) and `step_time.py:433` (`worst_rank=compute_rank` from `dominant_compute.worst_rank`) are the correct pattern.

**What to check:**

- For each branch, check that `worst_rank=` is set from the metric that actually triggered the verdict.
- For single-rank-friendly verdicts (`INPUT_BOUND`, `WAIT_HEAVY`, `COMPUTE_BOUND`), verify the pattern is `worst_rank=None if single_rank else <metric>_worst_rank`. A multi-rank verdict that always returns `worst_rank=None` is a bug.
- Test with multi-rank synthetic input: rank 0 has high dataloader, rank 1 is normal. Verify `worst_rank == 0`.

### 4.11 Threshold drift across PyTorch versions

Applies to: thresholds tuned against specific PyTorch / hardware behaviour.

The bug shape from `add_diagnostic.md` §9.3: `compute_bound_share_warn=0.85` is tuned against pre-`torch.compile` runs. Post-`compile`, kernel timing collapses into a single CUDA launch and `compute_share` shifts. The threshold is now miscalibrated.

**What to check:**

- Is the PyTorch version range documented in the threshold class docstring? If the PR introduces a threshold that depends on kernel-level timing behaviour (anything related to compute share, forward/backward decomposition, optimizer-step timing), the version range matters.
- Is the threshold robust to common variations (AMP, bf16, gradient accumulation)? Ask the author to confirm they've tested at least one of these.
- File a follow-up issue to re-tune against the next PyTorch major release. Don't block the PR on this — but capture the tech debt.

### 4.12 Sampler-data dependency drift

Applies to: every verdict that consumes a specific metric key.

The bug shape: the verdict reads `by_key.get("h2d_time")` but the `step_time_sampler` either never emitted that key, or emits it under a different name, or stopped emitting it after a refactor. Unit tests build `StepCombinedTimeMetric` instances by hand and pass — live runs always show `NO_DATA` because the metric is absent.

**What to check:**

- For each metric key the new branch reads, verify the sampler emits it. Read `samplers/step_time_sampler.py` (or sibling) and confirm the key appears in the emitted schema.
- Verify the renderer's `compute.py` produces a `StepCombinedTimeMetric` for that key. The metric flows: sampler emit → SQLite writer → renderer compute → diagnosis input. A break anywhere shows up as `NO_DATA`.
- Demand an integration test, not just a synthetic-dataclass unit test. Run the verdict against a real `traceml run --mode summary` against a synthetic example. PR description must include the printed card.

---

## 5. Step 4 — The four meta-questions

Distilled from the diagnostic engine design. Apply each to the PR and write down the answer explicitly. If you can't answer, ask.

### 5.1 Is this a new pathology, an extension of an existing kind, or a sub-class?

The first decision in `add_diagnostic.md` §2 is "which pathology are you naming?" Map to `traceml_why.md` §3. The PR is one of three shapes:

- **New pathology, new domain.** Needs a new engine file, new card builder, new entry in `final_summary.py`. High blast radius. PR should ship the engine + adapter + tests; the card-builder wire-up can be a separate PR if needed.
- **New pathology, existing domain.** New `kind` in an existing engine's enum, new branch, new threshold fields. Medium blast radius. PR fits in one engine file plus tests plus summary-adapter threshold update.
- **Sub-class of an existing kind.** Refines an existing branch (e.g. `STRAGGLER` becomes `STRAGGLER` + `INPUT_STRAGGLER` + `COMPUTE_STRAGGLER`). Risky because it changes the meaning of the parent kind on data that previously fired it. **Wire-format concern (§4.7).**

**Reviewer move:** make the author state explicitly which of the three shapes this is in the PR description. The shape determines the failure-mode emphasis: a new domain needs §4.6 (engine-vs-adapter discipline) hardest; a new kind in an existing domain needs §4.5 (priority) hardest; a sub-class needs §4.7 (wire format) hardest.

### 5.2 Does the engine-vs-adapter boundary stay clean?

`add_diagnostic.md` §3.8 is the architectural rule: **one engine, two callers**. The engine in `src/traceml/diagnostics/<engine>.py` defines the verdict semantics. The summary adapter in `aggregator/summaries/<engine>_diagnosis.py` only re-projects inputs and tightens thresholds.

**Reviewer move:** for every line the PR adds to the summary adapter, ask "could this go in the engine instead?" If the line is constructing `StepCombinedTimeMetric` from rank aggregates, it belongs in the adapter. If it's deciding whether to emit a verdict, it belongs in the engine. PR_87-style discipline: write your verdict **once**, callers feed it inputs.

### 5.3 Is the wire-format `kind` field a contract?

Yes. Once a kind name lands in `final_summary.json` and a TraceOpt regression detector starts keying off it, renaming is a migration. From `add_diagnostic.md` §6.2: "**Never rename a kind.** `INPUT_BOUND` is `INPUT_BOUND` forever."

**Reviewer move:** every PR that introduces a new kind must answer: "if we discover next month this name describes the wrong thing, what's the migration cost?" A small upfront naming discussion is cheap; a rename in v0.5 is expensive. Push back on names that are too narrow (`IDLE_GPU_SMALL_BATCH` — what about other causes of idleness?), too broad (`SLOW` — which axis?), or framework-specific (`HF_TRAINER_STALL`).

A subtler form: PR adds `STEP_TIME_NO_DATA` to a new engine instead of reusing the existing `NO_DATA` literal. Now there are two kinds-for-no-data across engines. Consistency wins — every engine's enum should have `"NO_DATA"` as its first member.

### 5.4 Which invariants does the PR preserve, and have you verified each one?

The diagnostic engine invariants (paraphrased from `add_diagnostic.md` §1 and §3):

1. **`build_*_diagnosis` never raises.** Top-level entry points return a valid `*Diagnosis` (often `NO_DATA`) on degenerate input.
2. **`NO_DATA` is a first-class verdict.** Not `None`, not an exception. Each `NO_DATA` carries a distinct `reason` so the user knows *what* is missing.
3. **Frozen dataclasses only.** No mutation; use `dataclasses.replace`.
4. **`_STATUS_BY_KIND` is exhaustive over `DiagnosisKind`.**
5. **Numeric inputs are clamped.** `_non_negative_finite` (or sibling) guards every total / share / skew. NaN and Inf become 0.0.
6. **Thresholds are a frozen dataclass with explicit defaults.** No module-level magic numbers.
7. **Trend-note enrichment is best-effort and wrapped in try/except.**
8. **Engine has zero `torch.*` imports.** Diagnostics package depends only on numpy + stdlib.
9. **Wire-format envelope is additive.** Fields can be added; never renamed or removed.

**Reviewer move:** for each invariant, point at the line of the diff that preserves (or could break) it. If you can't, you don't yet understand the PR well enough to approve it. Invariants 1, 2, and 5 catch the most live bugs; invariants 6 and 9 catch the most product debt.

---

## 6. Step 5 — Verification gates

Every concern in the review must come with a **concrete reproduction recipe**. For diagnostics, the recipe is usually a synthetic metric set fed through `build_*_diagnosis`, with an assertion on the resulting `kind`.

The shape:

```
1. Setup (1–3 lines): which engine, which thresholds.
2. Construct metric inputs (5–10 lines): build _metric(...) helpers.
3. Call build_*_diagnosis(metrics).
4. Assert: kind == "<expected>", or kind != "<not_expected>".
```

Pass / fail criterion is the assertion. No "should look reasonable."

### 6.1 Worked example — false-positive gate for a new IDLE_GPU verdict

```python
# Healthy training: 200ms step, balanced compute, no idleness.
# This MUST NOT fire IDLE_GPU.
metrics = [
    _metric("step_time", median=200.0),
    _metric("forward", median=100.0),
    _metric("backward", median=80.0),
    _metric("optimizer_step", median=18.0),
    _metric("dataloader_fetch", median=0.0),
    _metric("wait_proxy", median=2.0),
]
diag = build_step_diagnosis(metrics)
# Pass: diag.kind in {"BALANCED", "COMPUTE_BOUND"}.
# Fail: diag.kind == "IDLE_GPU".
```

That's the negative-case recipe. The positive case is the symmetric one (12 ms step, 92% compute share). Together they're the precision/recall verification gate for the verdict — non-negotiable.

### 6.2 Worked example — different-verdict-wins gate

```python
# Both IDLE_GPU thresholds (short step + high compute share) AND INPUT_BOUND
# threshold (dataloader >= 25%) are crossed. Higher-priority verdict wins.
metrics = [
    _metric("step_time", median=12.0),
    _metric("forward", median=2.0),
    _metric("backward", median=2.0),
    _metric("optimizer_step", median=0.5),
    _metric("dataloader_fetch", median=18.0),  # >> 25% of step
    _metric("wait_proxy", median=0.0),
]
diag = build_step_diagnosis(metrics)
# Pass: diag.kind == "INPUT_BOUND".
# Fail: diag.kind == "IDLE_GPU".
```

This is the §4.5 priority-tiebreak verification. If a reviewer suspects priority is wrong, this is the recipe to construct.

### 6.3 Worked example — wire-format integration gate

For a PR that adds a new kind:

```bash
# Run a synthetic training script under the new build.
git -C /teamspace/studios/this_studio/traceml checkout pr-NN
python /tmp/synthetic_idle_gpu.py  # designed to fire IDLE_GPU
# Inspect:
cat .traceml/<session>/final_summary.json | jq '.cards.step_time.diagnosis'
# Pass: object contains {"kind": "IDLE_GPU", "status": "IDLE-GPU", "severity": ..., "reason": ..., "action": ..., "worst_rank": ..., "note": ...}.
# Fail: any required field missing, or kind absent.
```

This catches §4.2 (status map mismatch) and §4.7 (wire-format break) at the same time.

### 6.4 When you can't write a verification gate

If you only have a vague worry — "this threshold feels too loose but I can't construct the failing case" — **don't raise the concern in the review yet**. File a follow-up issue labelled "investigate-thresholds" and cite the PR. Vague concerns waste author time and dilute must-fix signal.

The exception: §4.1 (false-positive at 3am) is severe enough that "I think this fires too easily" is a legitimate ask for *the author* to construct the gate. If the PR has no synthetic-healthy negative test, the gate is missing — that's a §8.3 block reason.

### 6.5 Gate for action-text past-tense reading

For each new branch, write the recipe:

```
1. Read the new live `action=` string aloud.
2. Re-read it imagining a printed end-of-run card three days after the run.
3. Pass: action reads correctly in both contexts.
4. Fail: action contains "Watch", "Monitor", "Wait" — needs presentation override.
```

This is a less mechanical gate than the others, but it's the right place to catch §4.4. If the action fails the past-tense reading, verify `aggregator/summaries/diagnosis_presentation.py` adds the override in the same PR.

### 6.6 Recipe style rules

- **Specific kinds, not adjectives.** `assert diag.kind == "IDLE_GPU"` not "should fire idle." Adjectives are debate; assertions are tests.
- **Reproducible from a clean checkout.** Tests should run on `pip install -e ".[dev,torch]"` without GPU. The diagnostic engine has zero `torch.*` imports — there is no excuse for tests requiring CUDA.
- **5–15 lines of actual code.** Synthetic inputs longer than that are testing too much at once; cut to the smallest example demonstrating the verdict.
- **State multi-rank vs. single-rank explicitly.** `_metric(...)` defaults to single-rank; multi-rank verdicts need `world_size>=2` and `ranks_present>=2` in the coverage object.

---

## 7. Step 6 — Drafting comments

Two granularity choices: line comment vs PR-level comment. They are not interchangeable.

### 7.1 Line comments

Use when: there is a specific code change you're proposing in a specific location. Pin the comment to the line that needs to change.

Pattern: state the issue → propose the fix → reference a verification gate or precedent.

```
The branch reads `if compute_share > 0.85` directly. That number should be
in DiagnosisThresholds.compute_bound_share_warn (currently 0.85 — same value,
but the inline literal hides it from SummaryDiagnosisConfig).

Suggest: replace with `>= thresholds.compute_bound_share_warn` and confirm
the summary adapter doesn't need a stricter value.
```

Keep it tight. Reviewer's job is to point at the change, not re-derive the architecture.

### 7.2 PR-level comments

Use when: the concern is **behavioural** or **architectural**, not localised to a single line. The fix may touch multiple files; the discussion is about the PR's intent.

Examples that belong in PR-level comments:

- **Threshold-stance disagreements.** "I think `idle_gpu_compute_share_min=0.80` is too loose — a 75% compute share with 25% wait is normal at small batch sizes and could fire IDLE_GPU under your defaults. Recommend tightening to 0.85 or splitting `EARLY_IDLE_GPU` (info, 0.80) from `IDLE_GPU` (warn, 0.92)."
- **Priority-position disagreements.** "Slotting IDLE_GPU after COMPUTE_BOUND means a 15ms step that is 90% compute fires COMPUTE_BOUND, not IDLE_GPU. Is that intended? If a small-batch user gets COMPUTE_BOUND with action='optimize compute,' they will hate us."
- **Engine-vs-adapter boundary concerns.** "The summary adapter in this diff has its own `compute_share` computation. That logic belongs in the engine. Please move it and have the adapter call `build_*_diagnosis(metrics, thresholds=stricter)`."

### 7.3 What NOT to raise (the holdback discipline)

Two kinds of items belong in your private parking-lot, not in the PR review:

- **Judgement calls about taxonomy** — "is `IDLE_GPU` really a step-time verdict, or should it be in a future `gpu_utilization` engine?" Decide privately, apply privately. If the answer materially changes the PR, say so. Otherwise hold.
- **Adjacent improvements** — "while we're here, the existing `DiagnosisThresholds` could use better docstrings on the existing fields." If the improvement isn't required for the PR to ship, file as a follow-up issue. Don't grow the PR.

The discipline: a PR review delivers a focused set of must-fix items. Bloating with parking-lot items dilutes signal and trains the author to treat reviews as discussion threads, not gates.

### 7.4 The maintainer summary

Every review ends with a 2–3 sentence executive summary suitable for the maintainer to read without opening the PR.

> PR #N adds [verdict kind] to the [step_time | step_memory | new domain] engine. Architecture matches the existing pattern; thresholds are [biased-toward-precision | suspect at <threshold>]. Review converged on K concrete items: (1) ..., (2) ..., (3) .... All K fixes are localised; [tests cover all four required cases | tests are missing the different-verdict-wins case]. Recommend [verdict].

Maintainer reads three sentences and either agrees or opens the PR. This is the artifact your maintainer wants more than the diff comments.

---

## 8. Step 7 — Landing the verdict

Three states. Pick exactly one.

### 8.1 Approve

Conditions:
- Consistency table (§3) is fully filled with documented justified deviations.
- All four meta-questions (§5) answered explicitly.
- All nine invariants (§5.4) preserved.
- Failure-mode catalogue (§4) walked; no concerns require a verification gate (§6).
- Tests cover positive, healthy negative, different-verdict-wins, grey zone (§8 of `add_diagnostic.md`).
- Action text reads correctly in both live and post-run contexts; presentation override added if needed.
- PR description includes printed card output from a synthetic pathological run AND a synthetic healthy run.

If all seven are true, approve cleanly. Don't suggest follow-up work in the approval — file follow-ups separately so the PR can ship.

### 8.2 Approve with changes

Conditions:
- All concerns are minor: docstring fixes, threshold-rationale clarifications, naming nits, test-coverage additions, presentation-override polish.
- No concern affects metric correctness, false-positive rate at the documented threshold, or wire-format compatibility.
- All concerns have a one-line fix or a clear written-down resolution.

This is "accept the PR but require these N small changes." Not "the PR is conceptually broken."

### 8.3 Block (request changes)

Conditions (any one):
- The verdict fires on a healthy synthetic example (false-positive verification fails — §4.1).
- The new kind is in `DiagnosisKind` but missing from `_STATUS_BY_KIND` (§4.2).
- A higher-priority verdict's threshold is also crossed in a test, but the new verdict wins (§4.5).
- Threshold logic duplicated in the summary adapter (§4.6).
- Tests don't include all four required cases (§8 of `add_diagnostic.md`).
- Action text is non-actionable in the user's stack (§4.4) and the author can't propose a better one.
- A field in the JSON envelope was renamed or removed (§4.7).
- PR adds a new metric-key dependency that doesn't yet exist in the sampler/renderer pipeline (§4.12).

A "block" verdict is most often triggered by §4.1 or missing tests. Frame it constructively: **these specific items must be resolved before merge.** It does not mean the architecture is wrong or the author has to redesign.

### 8.4 What "block" doesn't mean

Threshold disagreements are **not** automatic blocks. Reasonable reviewers disagree on the right value of `idle_gpu_compute_share_min`. The right move is a PR-level comment with your proposed value and rationale, an "approve with changes" if the author agrees, or a maintainer escalation if not. Blocking on threshold-stance is a power move; reserve it for cases where the threshold is *demonstrably* wrong against a verification gate, not where it's "tighter than I'd choose."

---

## 9. Reference: the diagnostic family seen end-to-end

The two existing engines are the worked examples. Mapping them to the seven steps is a useful exercise even when not reviewing a PR — it builds the consistency table in your head:

| Step | Where to look |
|---|---|
| 1. Anchor | `add_diagnostic.md` §1, [W11](../deep_dive/code-walkthroughs.md#w11-summaries--diagnostics--end-of-run-analysis), `traceml_why.md` §3 |
| 2. Consistency table | §3.2 of this guide (the table itself) |
| 3. Failure modes | §4 of this guide (12 categories) |
| 4. Meta-questions | §5 of this guide (4 angles) |
| 5. Verification gates | §6 of this guide (4 worked recipes) |
| 6. Comments | §7 of this guide |
| 7. Verdict | §8 of this guide |

If you're new to reviewing diagnostic PRs, read `step_time.py` start to finish (~815 lines), then `step_memory.py::build_step_memory_diagnosis` (~250 lines through the dispatcher), then `aggregator/summaries/step_time_diagnosis.py::SummaryDiagnosisConfig`. That gives you the canonical engine, the second engine for cross-reference, and the adapter pattern in one sitting.

---

## 10. Common reviewer mistakes

Numbered, with cause and fix.

1. **Approving on architecture without exercising the false-positive gate.** Cause: the PR follows the family pattern, so it must be fine. Effect: a verdict that fires on healthy runs ships, user trust erodes. Fix: §4.1 is non-optional. Always run the verdict against a typical healthy training shape before approving.

2. **Re-deriving the consistency table from scratch.** Cause: the table isn't a formal artifact in source. Effect: each reviewer rebuilds it slightly differently, catches different deviations. Fix: paste the §3.2 table into your review notes verbatim and add the new column.

3. **Reading the engine diff in isolation, ignoring the summary adapter.** Cause: focus on `diagnostics/<engine>.py` only. Effect: §4.6 boundary violations slip in via the adapter file. Fix: every diagnostic review reads both files in the same sitting.

4. **Vague threshold concerns without verification gates.** Cause: gut feel that a number is too loose. Effect: author can't reproduce, dismisses the concern. Fix: §6 — every threshold concern gets a synthetic input that fails. If you can't construct one, file as investigate-follow-up rather than blocking.

5. **Mixing line comments and PR-level comments.** Cause: writing taxonomy concerns inline next to a code line. Effect: comment gets resolved by changing one line, the architectural point is lost. Fix: §7.1/§7.2 — pick the granularity deliberately.

6. **Missing the wire-format `kind` contract check.** Cause: focusing on the implementation, not the JSON shape. Effect: a kind name lands and becomes immutable before anyone notices it's the wrong name. Fix: §5.3 + read `final_summary.json` shape from a real run before approving.

7. **Reviewing without running the test suite.** Cause: trusting green CI. Effect: missing the `_STATUS_BY_KIND` mismatch (§4.2) when CI doesn't run an integration test. Fix: at minimum run `pytest tests/test_<engine>_diagnosis_<verdict>.py -v` locally; ideally drive through `traceml run --mode summary` against the synthetic example.

8. **Skipping the priority-overlap test.** Cause: §3 consistency check is clean across the board, so the reviewer stops. Effect: a new verdict ships that silently shadows or is shadowed by an existing one. Fix: §4.5 test case 3 (different-verdict-wins) is non-optional.

9. **Conflating "matches the family" with "correct."** Cause: the author copied `step_time.py` patterns. Effect: novel-pathology failure modes (§5.1) miss. Fix: every empty cell in the new column of the consistency table is a question, not a free pass.

10. **Skipping the past-tense reading.** Cause: live action sounds fine, reviewer doesn't picture the post-run card. Effect: cards print "Wait for a fuller window." three days after the run finished. Fix: §6.5 — read each new action aloud in two contexts.

11. **Treating threshold disagreements as blocks.** Cause: reviewer's threshold instinct differs from author's. Effect: PR stalls on debate; author loses trust in review process. Fix: §8.4 — block on demonstrable wrongness, comment on stance disagreement.

12. **Skipping the maintainer summary.** Cause: assumes the maintainer will read the diff. Effect: maintainer reads the diff, disagrees on severity, kicks back. Fix: §7.4 — three sentences are the maintainer's reading material.

---

## Gaps encountered while writing this guide

Where the reviewer's playbook is currently underspecified or relies on folklore.

- **The consistency table (§3) isn't a formal artifact.** It lives in this guide and now in reviewer notes. If the diagnostic family grows beyond two engines, reviewers will diverge on the column set. Worth lifting into a contract test in `tests/test_diagnostic_family.py` that introspects each engine module and asserts `DiagnosisKind`, `_STATUS_BY_KIND`, `*Thresholds`, and `build_*_diagnosis` follow the canonical shape. Not yet written.

- **No registry of `kind` names across engines.** A reviewer enforcing §5.3 (wire-name contract) has to grep across `diagnostics/*.py`. Worth a constants module `src/traceml/diagnostics/kinds.py` with every kind exported as a `Literal` and a registry test asserting no two engines collide on the same kind name. Not yet written. (Currently `NO_DATA` is duplicated across `step_time.py` and `step_memory.py` — fine because the kind-namespace is per-engine, but a registry would force the question.)

- **The false-positive verification gate (§4.1) has no shared corpus.** Each reviewer constructs synthetic-healthy inputs ad hoc. A `tests/diagnostic_corpus/` directory with named healthy/pathological metric fixtures (`healthy_balanced.json`, `pathological_input_bound.json`, `borderline_compute_bound.json`) — replayable through any new engine — would be the reviewer's smoke harness. Not yet written.

- **Threshold-stance arbitration is folklore.** "Bias toward precision" is the rule, but reasonable reviewers disagree about what "precision" means at a specific number. A formal "precision target" doc — "we accept 5% false-positive rate against the corpus in `tests/diagnostic_corpus/healthy/`" — would resolve threshold debates by reference rather than instinct.

- **The `confidence` semantics gap (§4.8) has no resolution path.** `step_time.py` doesn't populate it; `step_memory.py` does. Until the project picks one rule, every new engine has to choose, and reviewers have no precedent to enforce. Worth either (a) adding a "engines opt in" docstring to `BaseDiagnosis.confidence`, or (b) backfilling `confidence` on every kind in `step_time.py` to match `step_memory.py`. Probably (b).

- **The presentation-override layer is summary-only.** `aggregator/summaries/diagnosis_presentation.py` rewrites action text for end-of-run wording. There is no peer in the live-renderer tree — meaning live views show the engine's raw action text without re-skinning. If we ever want different live-vs-summary wording (e.g. live "Watching for X" vs. summary "X was observed"), the live-side override layer doesn't exist yet. Reviewers should be aware.

- **No reviewer-side smoke harness.** A reviewer who wants to run a verification gate against a real `traceml run --mode summary` needs a CUDA box and a manual setup. A `tests/review_harness/diagnostic/` directory with parametrised synthetic training scripts (idle-shaped, input-bound-shaped, compute-bound-shaped, healthy) would make recipes 5 lines instead of 30. Not yet written.

These gaps are the natural follow-up work after this guide lands. None block the next reviewer from following the workflow above; all would make the next reviewer faster.
