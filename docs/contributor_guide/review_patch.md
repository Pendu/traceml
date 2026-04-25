# How to review an instrumentation-patch PR

This guide teaches you how to review a PR that adds or modifies an instrumentation patch in `src/traceml/utils/patches/`. It assumes you have already read `add_patch.md` (the author's guide) and have a working mental model of W3 (user-facing API) and W4 (patches + timing primitives). The seven-step workflow in §1 is the meta-pattern that future reviewer guides (`review_sampler.md`, `review_renderer.md`, ...) will reuse — only §3 onward is patch-specific.

---
Feature type: instrumentation patch
Risk level: high (touches PyTorch internals; one bad patch can corrupt every metric or crash every training run)
Cross-cutting impact: training process only (aggregator side untouched)
PyTorch coupling: deep
Reference reviews: PR #87 (h2d timing) — see `Notes/PR_87_review_through_walkthroughs.md`
Companion author guide: `add_patch.md`
---

## 1. The meta-review-workflow (applies to every TraceML PR)

Every patch review walks the same seven steps in order. Skipping any of them is how a flawed PR ships:

1. **Anchor** the PR diff to the relevant W-walkthroughs and Q/P entries. Read the PR through your existing mental models, not line-by-line.
2. **Run the patch-family consistency check.** Build the table from §3 of this guide and grade the new patch against the existing N patches on each axis. Discrepancies are either justified deviations (document them) or bugs.
3. **Apply the patch-class failure-mode catalogue** from §4. Each category maps to a known bug shape. Walk the diff with each shape in mind.
4. **Apply the four meta-questions** from §5: new axis of variation, shared infrastructure interaction, wire-name as contract, invariant preservation.
5. **Write a verification gate** for each concern: a 3–10 line reproduction recipe with a clear pass/fail criterion. "I think this is buggy" becomes "here's the script that proves it." See §6.
6. **Draft comments at the right granularity** — line comment for specific code suggestions, PR-level comment for behavioural / architectural concerns. Hold parking-lot items back. See §7.
7. **Land the verdict** — approve / approve-with-changes / block. Criteria in §8.

The reviewer's job ends with a 2–3 sentence executive summary the maintainer can read without opening the diff. That goes in the verdict (§8).

This same seven-step shape applies to sampler PRs, renderer PRs, transport PRs — only the consistency table and the failure-mode catalogue change.

---

## 2. Step 1 — Anchor the PR to the walkthroughs

The first thing you do with a patch PR is **not** open the diff. Open [`traceml_learning_code_walkthroughs.md`][W4] and re-read W3 §"Patch policy state machine" and W4 §"5-step patch recipe." Two reasons:

- The patch family has documented invariants (TLS gate, idempotent install, `timed_region` shape, `_STEP_BUFFER` flush). You'll be checking the diff against those invariants, so they need to be in cache.
- Any patch PR will touch four to seven files in stereotyped ways. If you read the PR file-by-file without that map, you'll drown in the diff. PR #87 touched 7 files; mapping them to W-sections collapses the diff into five mechanical changes plus one substantive change.

### How to anchor

For each file in the diff, write down (in your review notes, not the PR yet):

| File pattern | W-section | What kind of change should this be? |
|---|---|---|
| `src/traceml/__init__.py` + `src/traceml/api.py` | [W3 §"Lazy import pattern"][W3] | Mechanical — re-export stub, lazy import. |
| `src/traceml/initialization.py` | [W3 §"Patch policy state machine"][W3] | Frozen-dataclass field + three-mode rule extension. |
| `src/traceml/instrumentation.py` | [W3 §"trace_step nested CMs"][W3] | Add one nested context manager. |
| `src/traceml/utils/patches/<name>_patch.py` (NEW) | [W4 §"5-step patch recipe"][W4] | The substantive change. |
| `src/traceml/wrappers.py` | [W3 §"proxy or in-place mutation"][W3] | Per-instance proxy or method reassignment. |
| `tests/test_<name>_patch.py` (NEW) | none directly | Surface coverage. |

If a file in the diff doesn't fit the table, that's a flag — the PR is doing something architecturally novel, and you should ask why before proceeding.

The point: **after anchoring, you should have one substantive file to read deeply and 4–6 mechanical files to skim.** PR_87 §1 is the worked example of this collapse.

[W3]: ../deep_dive/code-walkthroughs.md#w3-user-facing-api-decorators-instrumentation-wrappers
[W4]: ../deep_dive/code-walkthroughs.md#w4-patches-timing-primitives-how-zero-code-instrumentation-actually-works

---

## 3. Step 2 — The patch-family consistency table

Every patch slots into a small set of axes. The reviewer's job is to fill in the row for the new patch and grade each cell against the existing patches.

### 3.1 The columns (these don't change PR-to-PR)

| Axis | What it asks | Where to verify |
|---|---|---|
| **Target class/method** | Which PyTorch symbol does this patch replace? | Module-level constant in the patch file (e.g. `_ORIG_TENSOR_TO`). |
| **Calls outside training?** | Will the patched method fire during model construction, eval, validation, checkpoint load? | Domain knowledge + W3 lines 545–590. |
| **Needs TLS enable flag?** | Does the patch need to be gated to inside `trace_step()`? Yes if "calls outside training" is YES. | Module-level `_*_TLS` object with an `_enabled` attribute. |
| **Calls nest naturally?** | Does invocation A trigger invocation B of the same patched method? | PyTorch source for the target method. |
| **Needs depth tracking?** | Yes if "calls nest" is YES — only outermost call should time. | Module-level depth counter on the TLS. |
| **Sentinel attribute** | What attribute marks "patch installed"? | `<class>._traceml_<name>_patched`. |
| **`use_gpu` in `timed_region`?** | True for GPU-bound work; False for CPU-bound. | Argument to `timed_region(...)`. |
| **Call-site filter?** | Does only a subset of calls qualify (e.g. CUDA target only)? | Helper like `_is_cuda_target`. |
| **Wrapper class in `wrappers.py`?** | Per-instance proxy companion to the patch? | `_Wrapped<Name>` + `wrap_<name>()`. |
| **Wire-format event name** | The string passed to `timed_region(name=...)`. | Search for `"_traceml_internal:"`. |
| **Sampler routing** | Which sampler's `(name, device, is_gpu)` aggregation key consumes this? | `step_time_sampler.py` — the `name` becomes the row identity. |
| **Error handling** | Try/except around install? Around the wrapper body? Fail-open everywhere? | The patch module + W4. |
| **CUDA event lifecycle** | Are events acquired from / returned to the pool? Only via `timed_region`? | `cuda_event_pool.py` references. |

### 3.2 The current state (April 2026)

| Axis | `forward` | `backward` | `dataloader` | `h2d` (PR #87) |
|---|---|---|---|---|
| Target | `nn.Module.__call__` | `Tensor.backward` + `autograd.backward` | `DataLoader.__iter__` | `Tensor.to` |
| Calls outside training? | YES | YES | NO | YES |
| TLS enable flag? | YES | YES | N/A | YES |
| Calls nest? | YES | YES | NO | NO |
| Depth tracking? | YES | YES | N/A | N/A |
| Sentinel | `nn.Module._traceml_forward_patched` | `torch._traceml_backward_patched` | `DataLoader._traceml_patched` | `torch.Tensor._traceml_h2d_patched` |
| `use_gpu` | True | True | False | True |
| Call-site filter? | NO | NO | NO | YES (`_is_cuda_target`) |
| Wrapper class? | YES (`wrap_forward` — in-place) | YES (`_WrappedBackwardHandle`) | YES (`_WrappedDataLoaderFetch`) | YES (`_WrappedH2D`) |
| Wire name | `_traceml_internal:forward_time` | `_traceml_internal:backward_time` | (different — dataloader sampler) | `_traceml_internal:h2d_time` |

When reviewing, **add a column** for the new patch and walk every row. Three outcomes per cell:

- ✅ Matches the family — note it and move on.
- ❌ Differs from the family — demand a justification in the PR description or a comment in the patch file. PR_87 §2.4 is the example: the new call-site filter axis is novel and warrants discussion.
- ⚠ Cell is empty / undecidable from the diff — ask the author.

### 3.3 The table is the most reusable artifact in this guide

Every future patch review should rebuild this table. Two reasons:
- The act of filling it forces you to read the patch with the family in mind, which catches "this differs in ways the author didn't notice."
- The completed table goes in your review notes, and over time becomes the reviewer's contract test for the patch family. (See "Gaps" at the end — formalising this is on the wishlist.)

---

## 4. Step 3 — Patch-class failure modes

Distilled from PR_87 §3. Every patch PR is at risk for these seven categories. Walk the diff with each one in mind.

### 4.1 Classification bugs (target filter false positives / negatives)

Applies to: any patch with a call-site filter (column "call-site filter?" in §3.2).

The filter introduces **two new failure modes** the older patches don't have: over-timing (filter says yes when it shouldn't) and under-timing (filter says no when it should). PR_87 §3.1 (D2D misclassified as H2D) and §3.9 (substring match on "cuda") are the worked examples.

**What to check:**

- Does the filter check **all** axes that distinguish "interesting" from "uninteresting"? For h2d, that means source device, target device, AND dtype-only.
- Does the filter use defensive parsing? Substring match on "cuda" is brittle; `torch.device(s).type == "cuda"` (in try/except) is robust.
- Is there a test for every classification category? Truth-table exhaustiveness on filter inputs is mandatory.

### 4.2 Scope / timing bugs (overcounting, double-counting)

Applies to: every patch.

The bug shape: one logical user action triggers N patched calls, all timed independently. PR_87 §3.2 (`model.to(device)` calls `tensor.to` once per parameter via `nn.Module._apply`) is the example.

**Critical insight from PR_87 §2.3:** depth tracking does NOT fix overcounting when the calls are sequential rather than nested. Read the PyTorch source for the target method to know which kind you're looking at.

**What to check:**

- For each patched method, ask: "what existing user code calls this method in a way that fans out to many invocations?" Construction-time `_apply`, autograd's recursion, dataloader prefetch — these are the usual suspects.
- Does the PR have a test for the fan-out case? PR_87 had none — the reviewer caught it cold.
- If overcounting is a risk, the fix is usually a **type discriminator** on the receiver (e.g. `isinstance(self, torch.nn.Parameter)`) or a second TLS flag set by the fan-out source. Not depth tracking.

### 4.3 State-machine bugs (initialisation races, wrap-state)

Applies to: every patch with a `wrap_*()` companion.

The bug shape from PR_87 §3.4: `wrap_*()` snapshots global state at wrap time, but the global state is mutable across the wrap → use boundary. Wrappers are per-instance; patches are global. The two need to read state at the same moment to stay consistent.

**What to check:**

- Does `_ensure_<name>_wrapper_allowed()` check `_INIT_CONFIG` at wrap time only? If so, raise the question: what happens if `init(mode=auto)` runs between wrap and use?
- Test for the race: `wrap → init → use`. If the test isn't there, ask for it. A 3-line repro (PR_87 §5.3) is the verification gate.
- The fix is typically to move the check **into the proxy's `.to()`** (or equivalent), so it reads the global at call time.

### 4.4 Dunder forwarding gaps on wrappers

Applies to: every wrapper that uses `__getattr__`-only forwarding.

Python bypasses `__getattr__` for special methods on the class. PR_87 §3.5: `_WrappedH2D` doesn't forward `__len__` / `__getitem__`, so `wrap_h2d({"x": tensor})` followed by `batch["x"]` raises `TypeError`.

**What to check:**

- What objects does the wrapper accept? `wrap_h2d` accepts arbitrary containers (dicts, lists, tensors). If users will plausibly call `len(wrapped)`, `wrapped[k]`, `iter(wrapped)`, `k in wrapped` before the one-shot operation — the wrapper needs explicit dunders.
- Look at peer wrappers in `wrappers.py`. `_WrappedDataLoaderFetch` defines `__len__` for exactly this reason.
- The fix is mechanical: 4–8 lines of explicit dunder forwarding.

### 4.5 Wire-format / `name` collisions

Applies to: every patch that emits `_traceml_internal:*` events.

The `name` argument to `timed_region(...)` is the only wire-format identifier carrying the patch's semantics through to the renderer. From PR_87 §4.4: once a name lands in user dashboards, renaming is a migration.

**What to check:**

- Is the new `name` unique? Grep `src/traceml` for `_traceml_internal:` and verify no collision.
- Does the name **describe what is actually measured**, not what was intended? PR_87 §4.4: if D2D is going to be in scope, the name should reflect that (`to_cuda_time` rather than `h2d_time`); if D2D is out of scope, the filter must enforce it.
- Is the name listed anywhere downstream (renderer constants, summary strings)? Adding a name without claiming it on the renderer side produces a silent metric.

### 4.6 CUDA event lifecycle leaks

Applies to: every patch using `timed_region(use_gpu=True)`.

The CUDA event pool is `deque(maxlen=2000)` ([W4][W4]). Events are acquired in `timed_region` and returned in `try_resolve()` once the GPU has signalled completion. Leaks happen when:

- An event is acquired but the surrounding `timed_region` raises before the `finally` block records `end`.
- The sampler can't keep up with event creation rate (e.g. under the PR_87 §3.2 overcount bug — 300+ events per step), the pool empties, and `get_cuda_event()` falls back to per-call `torch.cuda.Event(...)` allocation. Now there's measurable per-step overhead.

**What to check:**

- Every `timed_region(use_gpu=True)` call site is inside a `with` block, not manually constructed. Manual construction skips the cleanup discipline.
- Is there a test or back-of-envelope calculation for events-per-step under expected use? If the PR could create 100+ events per step under any plausible user pattern, that's a perf concern, not just a metric concern.

### 4.7 Test isolation fragility (the `importlib.reload` pattern)

Applies to: any test file that re-installs / re-imports the patch.

PR_87 §3.6: tests that call `importlib.reload(<patch_module>)` re-execute `_ORIG_<METHOD> = <Class>.<method>` at module scope. If a previous test's teardown didn't restore the true original, the reloaded module captures a **patched** function as its "original," and subsequent reinstalls chain patches.

**What to check:**

- Does the test file use `importlib.reload(...)` or `monkey-patch then re-import` patterns? If so, is there a session-scoped autouse fixture that snapshots `(<Class>.<method>, <Class>.<sentinel>)` at session start and restores at session end?
- Does every test class' teardown reset both the bound method AND the sentinel attribute? Asymmetric teardown is the usual culprit.
- Run the test file with `pytest --random-order` (or repeatedly) to catch ordering-sensitive failures.

---

## 5. Step 4 — The four meta-questions

Distilled from PR_87 §4. Apply each to the PR and write down the answer explicitly. If you can't answer, ask.

### 5.1 Does this PR introduce a new axis of variation? What new failure modes?

The first three patches (`forward`, `backward`, `dataloader`) had two axes: TLS gate (on/off) and depth (counted). PR_87 introduced a third: **call-site filter**. That single new axis bought two new failure modes (false positive / false negative — see §4.1).

**Reviewer move:** when the new patch has a column in the consistency table (§3) that no prior patch fills, enumerate the failure modes that column creates. PR_87 §2.4 and §4.3 are the worked example.

### 5.2 Does it interact with shared infrastructure?

Two pieces of shared infrastructure in the patch family:

- **`_STEP_BUFFER`** — process-global deque drained at `trace_step` exit. More events per step = bigger flush, more memory pressure on the `_STEP_TIME_QUEUE` (`maxsize=2048`).
- **CUDA event pool** — capped deque of 2000 timing events.

**Reviewer move:** estimate the new patch's per-step event count. PR_87 under the §3.2 bug pattern produces ~150 extra events for a small model. That's not catastrophic on its own but compounds with `forward` / `backward` / per-layer hooks. Forcing the author to do this estimate catches the case where N patches are individually fine but jointly exhaust shared infrastructure.

### 5.3 Is the wire-format `name` field a contract?

Yes. Once it lands in a release and a user has a dashboard showing the metric, renaming is a migration. PR_87 §4.4 is the exposition.

**Reviewer move:** every PR that introduces a new `_traceml_internal:*` name must answer: "if we discover next month this name describes the wrong thing, what's the migration cost?" A small upfront naming discussion is cheap; a rename in v0.4 is expensive.

### 5.4 Which invariants does the PR preserve, and have you verified each one?

The patch family invariants (paraphrased from W3 + W4):

1. **Idempotent install.** Calling `patch_<name>()` twice is a no-op.
2. **Fail-open install.** Patching never raises into user code.
3. **Fast path on disable.** Disabled-TLS-flag path is allocation-free.
4. **Fast path on filter reject.** Call-site filter rejection is allocation-free.
5. **No CUDA sync in hot path.** No `torch.cuda.synchronize()` from a patch.
6. **`_STEP_BUFFER` written only from training thread.** Patches run on the user's thread; sampler runs on a background thread. The buffer is not thread-safe.
7. **Original method preserved.** `_ORIG_<method>` captures the true original at first install.

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

### 6.1 Worked example (from PR_87 §5.1, the Module.to overcount)

```
# Setup
git -C <repo> checkout pr-87
# (Lightning Studio with GPU)

# Command — save as repro.py and run with:
#   traceml run repro.py
import traceml, torch
import torch.nn as nn
traceml.init(mode="auto")
model = nn.Sequential(*[nn.Linear(64, 64) for _ in range(10)])
with traceml.trace_step(model):
    model.to("cuda")
    x = torch.randn(8, 64, device="cuda")
    out = model(x)
    out.sum().backward()

# Expected (under the bug)
# StepTimeTable row for h2d_time has n_calls == 20 (or some number > 1).

# Pass criterion (after fix)
# n_calls == 1.
```

That's a 10-line recipe. The author can paste it into their environment; the maintainer can read it without running it. PR_87 §5.1–§5.4 are four such recipes, each focused on one finding.

### 6.2 When you can't write a verification gate

If you can't write a recipe — you only have a vague worry — **don't raise the concern in the review yet**. Either escalate it to research (file a follow-up issue, label "investigate"), or hold it back per §7.3. Vague concerns waste author time.

### 6.3 Recipe style rules

- **Specific numbers, not adjectives.** `n_calls == 1` not "should be small." Adjectives are debate; numbers are tests.
- **Reproducible from a clean checkout.** No "you also need to apply patch X first" — if the recipe depends on prior fixes, restate them.
- **3–10 lines of actual code.** Longer means you're testing too much at once; cut to the smallest demonstrating example.
- **State the GPU dependency explicitly.** "Needs CUDA" / "CPU-only OK." Reviewer running in CPU CI shouldn't try to reproduce GPU-only bugs.

---

## 7. Step 6 — Drafting comments

Two granularity choices: line comment vs PR-level comment. They are not interchangeable.

### 7.1 Line comments

Use when: there is a specific code change you're proposing in a specific location. Pin the comment to the line that needs to change.

Pattern: state the issue → propose the fix → reference a verification gate or precedent. PR_87 Drafts A, C, D, E (§6) are the model.

```
The filter checks the destination device but not the source. If
`self.is_cuda` is already True, this is a D2D copy [...]

Suggest a `if self.is_cuda: return` short-circuit before the filter,
plus a test `test_cuda_to_cuda_not_timed`.
```

Keep it tight. The reviewer's job is to point at the change, not to re-derive the architecture.

### 7.2 PR-level comments

Use when: the concern is **behavioural** or **architectural**, not localised to a single line. The fix may touch multiple files; the discussion is about the PR's intent.

Pattern: state the scenario → walk through what happens under the current diff → propose 2–3 fixes ranked by your preference → invite discussion. PR_87 Draft B (Module.to overcount) is the model — it's a PR-level comment because the fix could be in `_is_cuda_target`, in a new TLS flag, or in documentation. Multiple options, multiple files.

A PR-level comment is also right for cross-cutting concerns: "did you test this on a GPU box?", "does this name conflict with X?", "the depth tracking decision is unstated — please document."

### 7.3 What NOT to raise (the holdback discipline)

PR_87 §8 is the model. Two kinds of items belong in your private parking-lot, not in the PR review:

- **Judgement calls about positioning** — "is this user error or a footgun?", "should we mention the perf compounding angle?" These are about your relationship with the author, not the PR. Decide privately, apply privately.
- **Adjacent improvements** — "while we're here, the `_apply` interaction could use a separate test file." If the improvement isn't required for the PR to ship, file it as a follow-up issue. Don't grow the PR.

The discipline: a PR review delivers a focused set of must-fix items. Bloating the review with parking-lot items dilutes the must-fix signal and trains the author to treat your reviews as discussion threads, not gates.

### 7.4 The maintainer summary

Every review ends with a 2–3 sentence executive summary suitable for the maintainer to read without opening the PR. The shape from PR_87 §10:

> PR #N closes issue #M. Author implemented [feature]; architecture matches the existing [pattern]. Review converged on K concrete items worth fixing before merge: (1) ..., (2) ..., (3) .... All K fixes are localised; each needs one small test. Recommend [verdict].

Maintainer reads three sentences and either agrees with the verdict or opens the PR. This is the artifact your maintainer wants more than the diff comments.

---

## 8. Step 7 — Landing the verdict

Three states. Pick exactly one.

### 8.1 Approve

Conditions:
- Consistency table (§3) is fully ✅ or has documented justified deviations.
- All four meta-questions (§5) answered explicitly.
- All seven invariants (§5.4) preserved.
- No concerns require a verification gate (§6).
- Tests cover every cell in the call-site filter truth table (if any).

If all five are true, approve cleanly. Don't suggest follow-up work in the approval — file follow-ups separately so the PR can ship.

### 8.2 Approve with changes

Conditions:
- All concerns are minor: docstring fixes, test gaps, naming nits, optional dunder forwarding, dataclass default consistency.
- No concern affects metric correctness or training safety.
- All concerns have a one-line fix or a clear written-down resolution.

This is the "accept the PR but require these N small changes." Not "the PR is conceptually broken."

### 8.3 Block (request changes)

Conditions (any one):
- A concern affects the **correctness of a metric** (e.g. PR_87 §3.1 D2D classified as H2D).
- A concern can **inflate user-visible numbers** under realistic usage (e.g. PR_87 §3.2 Module.to overcount).
- A concern violates a patch-family invariant (§5.4).
- The PR introduces a new axis of variation without enumerating its failure modes (§5.1).
- Tests don't exist for a category in §4 that applies.

PR_87's verdict was "request changes" on §3.1 + §3.2 + §3.3 + §3.4 — four items, each with a verification gate, each fixable in <100 lines.

### 8.4 What "block" doesn't mean

It does not mean the architecture is wrong. It does not mean the author has to redesign. It means **these specific items must be resolved before merge.** Frame the verdict that way to keep the relationship healthy with the author.

---

## 9. Reference: PR #87 review as a worked example

The full review is in `Notes/PR_87_review_through_walkthroughs.md` (~1180 lines). Mapping it to the seven steps:

| Step | PR_87 section |
|---|---|
| 1. Anchor | §1 — each of the 7 files mapped to a W-section |
| 2. Consistency table | §2.1 (the table) + §2.2 (analysis) |
| 3. Failure modes | §3 (13 issues, categorised below) |
| 4. Meta-questions | §4 (4 angles surfaced) |
| 5. Verification gates | §5 (4 concrete recipes) |
| 6. Comments | §6 (6 sharpened drafts), §8 (holdback) |
| 7. Verdict | §7 (request changes), §10 (maintainer summary) |

If you're new to reviewing patch PRs, read PR_87 §1, §2.1, and §10 first. That's ~100 lines and gives you the shape. The rest is depth.

---

## 10. Common reviewer mistakes

Numbered, with cause and fix.

1. **Reviewing in isolation.** Cause: opening the diff first, before anchoring to walkthroughs. Effect: drowning in 7 files. Fix: do §2 before §3 — every time.

2. **Re-deriving the consistency table from scratch.** Cause: the table isn't a formal artifact in source. Effect: each reviewer rebuilds it slightly differently. Fix: paste the §3.2 table into your review notes verbatim and add the new column.

3. **Raising parking-lot items in the PR review.** Cause: not distinguishing must-fix from nice-to-have. Effect: review becomes a discussion thread, must-fix items lose signal. Fix: §7.3 holdback. Anything not actionable goes to follow-up issues.

4. **Vague concerns without verification gates.** Cause: time pressure, gut feel. Effect: author can't reproduce, dismisses the concern. Fix: §6 — every concern gets a recipe.

5. **Mixing line comments and PR-level comments.** Cause: writing architectural concerns inline next to a code line. Effect: comment gets resolved by changing one line, the architectural point is lost. Fix: §7.1/§7.2 — pick the granularity deliberately.

6. **Missing the wire-name contract check.** Cause: focusing on the patch implementation, not the event name it emits. Effect: name collisions or mis-named metrics ship. Fix: §5.3 + grep `_traceml_internal:` before approving.

7. **Reviewing without running the tests locally.** Cause: trusting green CI. Effect: missing GPU-only paths, missing ordering-sensitive failures. Fix: at minimum run `pytest tests/test_<new>_patch.py -v` on a CUDA box; if patches touch shared infrastructure, run `pytest tests/ -v` randomised.

8. **Approving on architecture without checking failure modes.** Cause: the patch follows the family pattern, so it must be fine. Effect: classification bugs / overcount bugs ship. Fix: §4 catalogue is non-optional even when the architecture is clean.

9. **Conflating "matches the family" with "correct."** Cause: §3 consistency check is ✅ across the board, so the reviewer stops. Effect: novel-axis failure modes (§5.1) miss. Fix: every empty cell in the new column is a question, not a free pass.

10. **Skipping the maintainer summary.** Cause: assumes the maintainer will read the diff. Effect: maintainer reads the diff, disagrees on severity, kicks back to the reviewer. Fix: §7.4 — three sentences are the maintainer's reading material; the diff is yours.

---

## Gaps encountered while writing this guide

Where the reviewer's playbook is currently underspecified or relies on folklore. Flag these in your review process if you hit them.

- **The consistency table (§3) isn't a formal artifact.** It lives in PR_87's review and now in this guide. If the patch family grows beyond four entries, every reviewer will diverge on the column set. Worth lifting into a contract test in `tests/test_patch_family.py` that introspects each `*_patch.py` module and asserts the `_ORIG_*`, `_*_TLS`, `_traceml_<name>_patched` triple exists. Not yet written.

- **There's no central registry of `_traceml_internal:*` names.** A reviewer enforcing §5.3 (wire-name contract) has to grep and trust their grep. Worth a constants module `src/traceml/utils/event_names.py` with every name exported as a constant; patches import the constant. Then a test asserts no two constants collide and that every renderer references at least one. Not yet written.

- **The verdict criteria (§8) are folklore-level.** "Affects metric correctness" is the bright line, but reasonable people disagree about what "correctness" means under user error (PR_87 §3.2 — is `model.to()` inside `trace_step` user error or a footgun?). A formal list of "metric-correctness invariants the project commits to" would resolve these arguments before the PR.

- **No reviewer-side smoke harness.** A reviewer who wants to run PR_87's §5.1 recipe needs a CUDA box and a manual setup. A `tests/review_harness/` directory with parametrised fixtures (`@pytest.mark.requires_cuda`, mock model factories) would make recipes 5 lines instead of 15. Not yet written.

- **The "holdback discipline" (§7.3) has no checklist.** Knowing what belongs in the PR vs. a follow-up is currently a judgement call. Two reviewers might draw the line differently. Worth a short rubric: "must-fix iff the PR-as-merged would (a) corrupt a metric, (b) inflate overhead by >X%, or (c) break wire-format backward compat." Otherwise: follow-up.

These gaps are the natural follow-up work after this guide lands. None block the next reviewer from following the workflow above; all would make the next reviewer faster.
