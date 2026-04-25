# How to review a CLI-change PR

> Internal contributor guide. Audience: one trusted co-founder / new engineer
> reviewing TraceML PRs. Companion to `add_cli.md`. Not for public docs.

This guide teaches you how to review a PR that adds or modifies a CLI surface in `src/traceml/cli.py` — a new subcommand, a new flag, a new `--mode` choice, a new profile, or a new `TRACEML_*` environment variable. It assumes you have already read `add_cli.md` (the author guide) and have a working mental model of W1 (CLI launcher) and W2 (per-rank runtime). The seven-step workflow in §1 is the meta-pattern shared with `review_patch.md` — only §3 onward is CLI-specific.

---
Feature type: CLI surface (subcommand / flag / mode / profile / env var)
Risk level: medium (anything user-facing breaks v0.2.x users on PyPI)
Cross-cutting impact: multiple subsystems (CLI process → aggregator process → executor → runtime → samplers)
PyTorch coupling: none directly (but profile flags gate sampler choice)
Reference reviews: none called out yet — the failure-mode catalogue here is distilled from `add_cli.md` §9 and the live code-base divergences listed in §11
Companion author guide: `add_cli.md`
Last verified: 2026-04-25
---

## 1. The meta-review-workflow (applies to every TraceML PR)

Every CLI review walks the same seven steps in order. Skipping any of them is how a flawed PR ships:

1. **Anchor** the PR diff to the relevant W-walkthroughs and Q entries. Read the PR through your existing mental models, not line-by-line.
2. **Run the CLI-family consistency check.** Build the table from §3 of this guide and grade the new flag/subcommand against the existing surfaces on each axis. Discrepancies are either justified deviations (document them) or bugs.
3. **Apply the CLI-class failure-mode catalogue** from §4. Each category maps to a known bug shape. Walk the diff with each shape in mind.
4. **Apply the four meta-questions** from §5: new axis of variation, shared-default consistency, wire-name as contract, invariant preservation.
5. **Write a verification gate** for each concern: a 3–10 line reproduction recipe with a clear pass/fail criterion. "I think this is buggy" becomes "here's the script that proves it." See §6.
6. **Draft comments at the right granularity** — line comment for specific code suggestions, PR-level comment for behavioural / architectural concerns. Hold parking-lot items back. See §7.
7. **Land the verdict** — approve / approve-with-changes / block. Criteria in §8.

The reviewer's job ends with a 2–3 sentence executive summary the maintainer can read without opening the diff. That goes in the verdict (§8).

This same seven-step shape applies to patch PRs (`review_patch.md`), sampler PRs, renderer PRs — only the consistency table and the failure-mode catalogue change.

---

## 2. Step 1 — Anchor the PR to your walkthroughs

The first thing you do with a CLI PR is **not** open the diff. Open [`traceml_learning_code_walkthroughs.md`][W1] and re-read W1 §"top-level launcher" and W2 §"per-rank runtime — executor read sites." Two reasons:

- The CLI plumbing has documented invariants (three-defaults rule, env-var wire contract, fail-open at runtime but fail-fast at parse time, no heavy imports in `cli.py`). You'll be checking the diff against those invariants, so they need to be in cache.
- A CLI PR will touch four to six files in stereotyped ways: `cli.py` (parser + assembly), `runtime/settings.py` (dataclass), `runtime/executor.py` (read site), `aggregator/aggregator_main.py` (read site), and possibly `aggregator/trace_aggregator.py` (`_DISPLAY_DRIVERS`) plus the consumer (sampler, renderer, driver). If you read the PR file-by-file without that map, you'll miss a layer.

### How to anchor

For each file in the diff, write down (in your review notes, not the PR yet):

| File pattern | W-section | What kind of change should this be? |
|---|---|---|
| `src/traceml/cli.py` (`_add_launch_args` / `build_parser`) | [W1 §"argparse + subparsers"][W1] | argparse declaration. Mechanical. |
| `src/traceml/cli.py` (`_validate_launch_args`) | [W1 §"cross-arg validation"][W1] | Cross-arg constraint. Mechanical. |
| `src/traceml/cli.py` (`launch_process`) | [W1 §"env-var assembly"][W1] | One new `env["TRACEML_X"] = str(args.x)` line. Mechanical. |
| `src/traceml/cli.py` (`write_run_manifest`) | [W1 §"manifest"][W1] | Field add (signature or `extra=`). Mechanical. |
| `src/traceml/runtime/settings.py` | [W2 §"settings dataclass"][W2] | Frozen-dataclass field. Default agrees with argparse. |
| `src/traceml/runtime/executor.py` (`read_traceml_env`) | [W2 §"env read"][W2] | `.get(name, default)` line + module constant if needed. |
| `src/traceml/aggregator/aggregator_main.py` (`read_traceml_env`) | [W9 §"aggregator entry"][W9] | Mirror of executor read. Default agrees. |
| `src/traceml/aggregator/trace_aggregator.py` (`_DISPLAY_DRIVERS`) | [W10 §"display driver dispatch"][W10] | `--mode` only — registry entry. |
| `src/traceml/aggregator/display_drivers/<name>.py` (NEW) | [W10][W10] | `--mode` only — `BaseDisplayDriver` subclass. |
| `tests/test_<name>_cli.py` (NEW) | none directly | Surface coverage. |

If a file in the diff doesn't fit the table, that's a flag — the PR is doing something architecturally novel, and you should ask why before proceeding. In particular, an `import torch`, `import nicegui`, or `import pandas` at the top of `cli.py` is a hard stop (see §4.7).

The point: **after anchoring, you should have one or two substantive files to read deeply (the consumer + maybe a new driver) and 4–5 mechanical files to skim.**

[W1]: ../deep_dive/code-walkthroughs.md#w1-clipy-top-level-launcher-and-process-orchestrator
[W2]: ../deep_dive/code-walkthroughs.md#w2-per-rank-runtime-executor-runtime-loop-launch-context-session
[W9]: ../deep_dive/code-walkthroughs.md#w9-aggregator-core-tcp-receive-frame-dispatch-sqlite-writes
[W10]: ../deep_dive/code-walkthroughs.md#w10-display-drivers-renderers-terminal-and-web-ui-from-sql

---

## 3. Step 2 — The CLI-family consistency table

Every CLI surface slots into a small set of axes. The reviewer's job is to fill in the row for the new flag/subcommand and grade each cell against the existing surfaces.

### 3.1 The columns (these don't change PR-to-PR)

| Axis | What it asks | Where to verify |
|---|---|---|
| **Category** | Subcommand / flag / `--mode` choice / profile / env var? | The diff's intent. |
| **Argparse declaration** | Added to `_add_launch_args` (shared by `watch`/`run`/`deep`) or to a single subparser direct? `type=`, `default=`, `help=` set? `choices=` for closed-set values? | `cli.py:707` and the subparser block at `cli.py:806`. |
| **Validation layer** | Argparse `type=` (cheap, single-arg)? `_validate_launch_args` (cross-arg)? Both? Neither? | `cli.py:137`. |
| **Env-var assembly** | `cli.py::launch_process` adds `env["TRACEML_X"] = str(args.x)`? Always set, or only when non-default? Cast to string? | `cli.py:455–472`. |
| **Env-var read site (aggregator)** | `aggregator_main.py::read_traceml_env` reads with `.get(name, default)`? Default agrees with argparse? | `aggregator_main.py:93–115`. |
| **Env-var read site (executor)** | `runtime/executor.py::read_traceml_env` reads with `.get(name, default)`? Default agrees with argparse and aggregator? Default sourced from a module constant? | `executor.py:212–246` + module constants `executor.py:42–49`. |
| **Settings field** | `TraceMLSettings` field added (if applicable)? Default agrees with the other two defaults? | `runtime/settings.py:23`. |
| **Manifest record** | `write_run_manifest` extended (signature or `extra=`)? | `cli.py:192`. |
| **Backward compat** | Default invocation `traceml watch script.py` still works after the change? Old env-var name still read with fallback (if renamed)? | `cli.py:707`, `aggregator_main.py:93`. |
| **`--mode` only — driver** | New `BaseDisplayDriver` subclass implemented in `aggregator/display_drivers/`? Registered in `_DISPLAY_DRIVERS` (`trace_aggregator.py:58`)? `supported_modes` set in `launch_process` (`cli.py:535`) updated? | Three sites. |
| **Subcommand only — handler** | Calls `launch_process` (training-launching) or pure-CLI handler (offline)? Decided right? | `cli.py:438` vs. `run_inspect`/`run_compare` shape. |
| **Documentation** | Env var added to `traceml/CLAUDE.md` env-var table? `--help` text useful (not just `"X."`)? | `traceml/CLAUDE.md`, the `help=` strings in `cli.py`. |
| **CHANGELOG** | Entry under "Unreleased" with user-visible change description? | `CHANGELOG.md`. |
| **Tests** | Argparse parse test? `_validate_launch_args` test? Backward-compat test (default invocation)? | `tests/test_<name>_cli.py`. |

### 3.2 The current state (April 2026)

Three reference surfaces. Use them as the column-fill examples for any new flag, mode, or subcommand.

| Axis | `--tcp-port` (flag) | `--mode` (choice flag) | `inspect` (subcommand) |
|---|---|---|---|
| Category | Flag | Closed-set flag with dispatcher | Pure-CLI subcommand |
| Argparse decl | `cli.py:757` `type=int default=29765` | `cli.py:712` `choices=["cli","dashboard","summary"] default="cli"` | `cli.py:849` positional `file` |
| Validation | argparse `type=int` only | argparse `choices=` + `_validate_launch_args:146` (`mode=summary` requires history) | none |
| Env assembly | `cli.py:469` `env["TRACEML_TCP_PORT"] = str(args.tcp_port)` | `cli.py:460` `env["TRACEML_UI_MODE"] = args.mode` | n/a — handler does not call `launch_process` |
| Aggregator read | `aggregator_main.py:108` `int(os.environ.get("TRACEML_TCP_PORT", "29765"))` | `aggregator_main.py:93` reads `TRACEML_UI_MODE` then falls back to `TRACEML_MODE` | n/a |
| Executor read | `executor.py:235` `int(os.environ.get("TRACEML_TCP_PORT", str(DEFAULT_TCP_PORT)))` | `executor.py:212` mirrors aggregator pattern | n/a |
| Settings field | `TraceMLTCPSettings.port: int = 29765` (`settings.py:20`) | `TraceMLSettings.mode: str = "cli"` (`settings.py:38`) | n/a |
| Manifest | `write_run_manifest:235` `"tcp_port": int(tcp_port)` | `write_run_manifest:232` `"ui_mode": str(ui_mode)` | n/a |
| Backward compat | default preserved | old name `TRACEML_MODE` still read with fallback | n/a — new offline command |
| Driver / dispatcher | n/a | `_DISPLAY_DRIVERS` (`trace_aggregator.py:58`) + `supported_modes` set (`cli.py:535`) | n/a |
| Handler shape | n/a | n/a | `run_inspect` reads file, prints, `raise SystemExit` |
| Docs | yes (env-var table) | yes (env-var table + `--help`) | yes (subparser `help=`) |
| Tests | none today (gap) | none today (gap) | none today (gap) |

When reviewing, **add a column** for the new surface and walk every row. Three outcomes per cell:

- ✅ Matches the family — note it and move on.
- ❌ Differs from the family — demand a justification in the PR description or a comment in the diff. Default drift (argparse default ≠ settings default ≠ executor default) is the most common case and is a bug, not a deviation.
- ⚠ Cell is empty / undecidable from the diff — ask the author.

### 3.3 The table is the most reusable artifact in this guide

Every future CLI review should rebuild this table. Two reasons:
- The act of filling it forces you to read the PR with the family in mind, which catches "this differs in ways the author didn't notice."
- The completed table goes in your review notes, and over time becomes the reviewer's contract test for the CLI surface. (See "Gaps" at the end — formalising this is on the wishlist; today there are zero contract tests.)

---

## 4. Step 3 — CLI-class failure modes

Distilled from `add_cli.md` §9 (the 14 author-side pitfalls) and the live code-base divergences. Every CLI PR is at risk for these categories. Walk the diff with each one in mind.

### 4.1 Five-layer plumbing skipped

Applies to: every flag that needs to influence aggregator or executor behaviour.

The bug shape: argparse declaration is added but no env var, env var is added but no read site, read site is added but no consumer. Each missing layer leaves the flag dead in that process — the user changes the value and nothing happens, silently.

**What to check:**

- For every new flag, walk the chain: argparse → `_validate_launch_args` (if cross-arg) → `launch_process` env-var assembly → `aggregator_main.py::read_traceml_env` → `executor.py::read_traceml_env` → `TraceMLSettings` field → consumer call site. Skipping any link is a bug.
- If the flag affects only the CLI process (e.g. manifest layout), argparse + manifest are enough. But state that explicitly in the PR description; it is a deviation from the default five-layer pattern.
- The aggregator-only and executor-only cases (flag affects only one of the two children) are legitimate but require justification. The default is "both children read it" — anything else is a deviation.

### 4.2 Default drift across layers

Applies to: every flag with a default value.

The bug shape: argparse default ≠ env-var read-site default ≠ `TraceMLSettings` field default. Whichever path the user takes (CLI vs. test bypass vs. direct settings construction), they get a different default. Tests pass because each test exercises one path; production breaks when the paths interact.

**Live examples already in the codebase:**

- `--interval` defaults to `2.0` in argparse (`cli.py:725`) but `1.0` in `TraceMLSettings.sampler_interval_sec` (`settings.py:39`) and `executor.py::DEFAULT_INTERVAL_SEC` (`executor.py:47`) and `aggregator_main.py:101`.
- `--num-display-layers` defaults to `5` in argparse (`cli.py:742`) but `20` in `TraceMLSettings.num_display_layers` (`settings.py:41`), `executor.py::DEFAULT_NUM_DISPLAY_LAYERS` (`executor.py:48`), and `aggregator_main.py:105`.

These are pre-existing bugs; the CLI always sets the env var so the argparse default wins in practice. But if a test bypasses the CLI and constructs `TraceMLSettings()` directly, it sees the wrong default. Don't add a third instance.

**What to check:**

- For every new default, grep the value in `cli.py`, `settings.py`, `executor.py`, and `aggregator_main.py`. Three (or four) sites, one value.
- If the new default disagrees with a prior default deliberately, the PR description must document why — and the comment in code must too.

### 4.3 Backward-compat break

Applies to: every CLI change.

The bug shape: a flag is added without a default, an existing default is changed to flip behaviour, or an env var is renamed without a read-both-fallback. v0.2.x users on PyPI run `traceml watch script.py` (no extra args) and CI scripts pin specific env vars. Any of these break, the upgrade breaks them.

**What to check:**

- Run `parser.parse_args(["watch", "train.py"])` mentally (or as a test — see §6). The new attribute should equal the documented default and not raise.
- If the PR renames an env var, both names must be read at the consumer with the new one preferred. The canonical example is `TRACEML_UI_MODE` falling back to `TRACEML_MODE` (`aggregator_main.py:93`, `executor.py:212`). No renames without this pattern.
- If the PR changes a default in a behaviour-flipping way, that is a minor-version bump and a CHANGELOG entry, not a stealth change. Push back if either is missing.

### 4.4 `--mode` choice / `_DISPLAY_DRIVERS` skew

Applies to: any PR adding or removing a `--mode` choice.

The bug shape: argparse `choices=` is updated but `_DISPLAY_DRIVERS` (`trace_aggregator.py:58`) is not (or vice-versa); or both are updated but the redundant `supported_modes` set in `cli.py::launch_process` (`cli.py:535`) is forgotten. The user passes the new mode, argparse accepts it, then the launcher or aggregator raises `ValueError` post-spawn.

**What to check:**

- Three sites, all updated: `cli.py:716` (argparse `choices=`), `cli.py:535` (`supported_modes` set), `trace_aggregator.py:58` (`_DISPLAY_DRIVERS` registry).
- The driver subclass exists in `aggregator/display_drivers/` and implements `start()`, `tick()`, `stop()`. See [`add_renderer.md`](add_renderer.md) §12 for the full contract.
- Any cross-mode constraint (e.g. `--mode=summary` requires history at `_validate_launch_args:146`) belongs in `_validate_launch_args`, not in the driver. Driver-level validation fires post-spawn and the user gets a useless traceback.

### 4.5 Profile flag without `_build_samplers` extension

Applies to: any PR adding a profile.

The bug shape: a new profile string is accepted by argparse and passed via `TRACEML_PROFILE` to the executor, but `runtime.py::_build_samplers` has no branch for it, so the profile is silently equivalent to the default.

**What to check:**

- The new profile name appears in `_build_samplers` (see [`add_sampler.md` §11](add_sampler.md#11-appendix-adding-a-new-profile)).
- The new profile name is the value passed by the matching subparser (e.g. `watch_parser` → `"watch"`); the convention is one subcommand per profile name.
- If this is a flag (`--profile foo`) rather than a subcommand, that is a deviation — TraceML's house style is one subcommand per profile. Push back unless the deviation is justified.

### 4.6 Manifest field missing

Applies to: any flag that affects training data or reproducibility.

The bug shape: the flag's value is consumed at runtime but never recorded in `manifest.json`. Six months later, a user files a bug "my numbers don't reproduce" and the manifest is silent on the cause.

**What to check:**

- Use the rubric from `add_cli.md` §5: record if the value affects data produced or reproducibility; skip if cosmetic; skip for `inspect`/`compare`/offline subcommands.
- The manifest field is added either by extending `write_run_manifest`'s signature (first-class field) or via `extra=` (one-off). Promote first-class for fields touched on every launch.
- The call site in `launch_process` (`cli.py:488`) passes the value through. A signature-extended field that the caller never sets shows as `null` in `manifest.json` — silent miss.

### 4.7 Heavy import in `cli.py`

Applies to: every CLI PR.

The bug shape: a top-level `import torch`, `import nicegui`, `import pandas` at the top of `cli.py`. `traceml --help` slows from ~50 ms to several seconds and fails on a CPU-only box without `torch` installed. This violates the launcher's lean-import contract.

**What to check:**

- `git diff src/traceml/cli.py` for new top-level imports. Anything besides standard library + small first-party modules (`traceml.runtime.session`, `traceml.runtime.launch_context`, `traceml.compare.command`) is a flag.
- Heavy imports belong inside handlers (`run_compare`, `run_inspect`, `run_validate`) where they execute only when the user invokes that subcommand.
- `_DISPLAY_DRIVERS` is not imported by `cli.py` for exactly this reason; doing so would drag NiceGUI and Plotly into `traceml --help`. The hard-coded `supported_modes` set at `cli.py:535` is the workaround. See §11 of `add_cli.md` for the full discussion.

### 4.8 Env var read with `[name]` not `.get(name, default)`

Applies to: any PR adding an env-var read site.

The bug shape: `os.environ["TRACEML_FOO"]` raises `KeyError` when the env var is absent, which happens in tests that bypass the CLI and in any direct invocation of `executor.py` / `aggregator_main.py`.

**What to check:**

- Every new read uses `.get(name, default)`. The default is a string (env-var values are strings).
- The default is not a magic literal — it is a module constant or matches the argparse default. The pattern in `executor.py:222` (`os.environ.get("TRACEML_INTERVAL", str(DEFAULT_INTERVAL_SEC))`) is the model.

### 4.9 `int(os.environ.get("X"))` without default

Applies to: numeric env-var reads.

The bug shape: `int(os.environ.get("TRACEML_FOO"))` raises `TypeError: int() argument must be a string ... not 'NoneType'` when the env var is absent. The default is missing from the `.get(...)` call.

**What to check:**

- `int(os.environ.get("TRACEML_FOO", "0"))` — default is a string inside `.get`, then cast.
- `float(os.environ.get("TRACEML_INTERVAL", "1.0"))` — same shape, never bare `float(None)`.
- Look for any new `int(...)` / `float(...)` / `bool(...)` call wrapping `os.environ.get` in the diff.

### 4.10 Boolean as `type=bool`

Applies to: any new boolean flag.

The bug shape: `parser.add_argument("--enable-foo", type=bool)`. `bool("false")` is `True` (any non-empty string is truthy), so `--enable-foo false` enables the flag. The user is confused; the env var ends up `"True"`, and the consumer reading `os.environ.get("TRACEML_FOO") == "1"` never matches.

**What to check:**

- Boolean flags use `action="store_true"` (off by default, on when present). Existing examples: `--enable-logging`, `--no-history`, `--disable-traceml` (`cli.py:730`, `:782`, `:787`).
- The env-var assembly is `env["TRACEML_FOO"] = "1" if args.foo else "0"`. Never `"true"` / `"false"`.
- The read site is `os.environ.get("TRACEML_FOO", "0") == "1"`.

### 4.11 Subcommand calls `launch_process` when it shouldn't

Applies to: new subcommands.

The bug shape: a pure-CLI subcommand (offline decoder, validator, comparison) calls `launch_process` and ends up spawning the aggregator and torchrun for a command that should read a file and exit.

**What to check:**

- The handler is shaped like `run_inspect` (`cli.py:648`) or `run_compare` (`cli.py:686`): read inputs, do work, print result, `raise SystemExit(0|1)`. No env-var assembly, no subprocess spawning.
- The subparser does **not** call `_add_launch_args(...)`. None of the launch flags (`--tcp-port`, `--mode`, `--profile`) make sense for offline commands.
- Conversely, a subcommand that does need to launch training (a hypothetical `traceml replay`) should use the full `launch_process` shape and `_add_launch_args`. No half-launches.

### 4.12 Validation in argparse vs `_validate_launch_args` confusion

Applies to: any flag with non-trivial validation.

The bug shape: single-argument validation (e.g. "must be non-negative") placed in `_validate_launch_args` instead of argparse `type=`, or cross-argument validation (e.g. "`--mode=summary` requires history") placed in argparse instead of `_validate_launch_args`. The wrong layer fires at the wrong time and the error message is unhelpful.

**What to check:**

- Single-arg type checks (`int`, `float`, `Path`) → argparse `type=`.
- Single-arg closed-set checks → argparse `choices=`.
- Single-arg range checks (e.g. ">= 0") → argparse `type=` with a custom callable, or `_validate_launch_args` if too noisy. Either is fine; pick one and be consistent within the PR.
- Cross-arg constraints (`--mode=summary` and `--no-history`) → `_validate_launch_args` always. Argparse cannot see two args together.
- The validator is called from `cli.py::main:862` after parsing but before dispatch. New validation belongs in this function, not in handlers.

### 4.13 Argparse attribute typo (dashes vs. underscores)

Applies to: any cross-arg validation that uses `getattr(args, "...")`.

The bug shape: argparse converts `--max-step-time-ms` to `args.max_step_time_ms` (dashes become underscores). Code in `_validate_launch_args` that does `getattr(args, "max-step-time-ms", 0)` silently returns the default for every invocation, and the validation is dead.

**What to check:**

- Every `getattr(args, "...")` in `_validate_launch_args` uses the underscored attribute name. Grep the diff for `getattr(args,` and verify each string.
- A unit test for `_validate_launch_args` catches this immediately. If the PR touches the validator, the test goes in.

### 4.14 Test isolation around env vars

Applies to: any test file that sets `TRACEML_*` env vars.

The bug shape: a test sets `os.environ["TRACEML_FOO"] = "..."` without `monkeypatch` or a teardown that unsets, and the env var leaks to subsequent tests in the same process. Unrelated tests now see a non-default value and fail mysteriously.

**What to check:**

- Every test that touches `os.environ` uses `monkeypatch.setenv` (pytest fixture) or has explicit setup/teardown that restores prior state.
- If the test exercises `read_traceml_env`, it pre-deletes the env var via `monkeypatch.delenv(name, raising=False)` before checking the default path.
- Run the test file with `pytest --random-order` or repeatedly to catch ordering-sensitive failures.

---

## 5. Step 4 — The four meta-questions

Apply each to the PR and write down the answer explicitly. If you can't answer, ask.

### 5.1 Does this PR introduce a new axis of variation? What new failure modes?

Most CLI flags fall into existing axes (string flag, int flag, bool flag, choice flag). Some PRs introduce something genuinely new — a config-file flag, a flag that affects only the aggregator, a flag with a non-string serialization. Each new axis carries new failure modes.

**Reviewer move:** when the new flag has a column in the consistency table (§3) that no prior flag fills, enumerate the failure modes that column creates. A config-file flag introduces parse-error handling and a precedence rule (CLI > config > env > default). A flag that affects only the aggregator means the executor read site is unnecessary — but the PR must say so explicitly so the reviewer doesn't expect it.

### 5.2 Are the three (or four) defaults consistent?

Two pieces of shared infrastructure across the CLI family:

- **The defaults triple/quadruple** (argparse / aggregator read / executor read / settings dataclass). All must agree.
- **The env-var name space** (`TRACEML_<UPPER_SNAKE>`). New names must not collide with existing ones; renames must keep the old name readable.

**Reviewer move:** for any new default, grep the value in `cli.py`, `settings.py`, `executor.py`, and `aggregator_main.py`. If the values disagree, the PR has a bug — even if all tests pass, because tests likely don't exercise the bypass path. The pre-existing `--interval` and `--num-display-layers` drift (§4.2) is the cautionary precedent.

### 5.3 Is the env-var name a contract?

Yes. Once it lands in a release and a user has CI scripts setting `TRACEML_FOO=1`, renaming is a migration. From `add_cli.md` §6.4: read both names, prefer the new, deprecate the old over a minor version, then remove.

**Reviewer move:** every PR that introduces a new `TRACEML_*` name must answer: "if we discover next month this name describes the wrong thing, what's the migration cost?" A small upfront naming discussion is cheap; a rename in v0.4 with a fallback-read site is expensive. The canonical good rename is `TRACEML_UI_MODE` falling back to `TRACEML_MODE` at `aggregator_main.py:93` and `executor.py:212`.

### 5.4 Which invariants does the PR preserve, and have you verified each one?

The CLI family invariants (paraphrased from `add_cli.md`):

1. **Lean launcher imports.** `cli.py` does not import `torch`, `nicegui`, `pandas`, or `aggregator/trace_aggregator.py`. `traceml --help` works on a CPU-only box without `torch` installed.
2. **Default invocation works.** `traceml watch script.py` (no extra args) parses, validates, and launches successfully after the change.
3. **Three-defaults-one-value.** Argparse default, env-var read default, settings dataclass default agree.
4. **Env-var values are strings.** Always `str(args.x)` on assembly; always `int(...)` / `float(...)` cast on read with a string default in `.get`.
5. **Booleans are `"1"` / `"0"`.** Never `"true"` / `"false"`, never `type=bool`.
6. **Read with `.get(name, default)`.** Never `os.environ[name]`.
7. **Cross-arg validation in `_validate_launch_args`.** Single-arg validation in argparse.
8. **Backward-compat reads.** Renamed env vars keep the old name readable with a documented fallback.
9. **Manifest captures reproducibility-affecting fields.** Anything that influences training data or reproduction lands in `manifest.json`.
10. **Fail-fast at parse time, fail-open at runtime.** Validation errors `raise SystemExit` with `[TraceML] ERROR:` prefix before any subprocess spawns. Once spawned, the runtime never blocks training.

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

### 6.1 Worked example — backward compat (default invocation still works)

```python
# Setup
git -C /teamspace/studios/this_studio/traceml checkout pr-XX

# Command — save as repro.py and run with: python repro.py
from traceml.cli import build_parser
parser = build_parser()
ns = parser.parse_args(["watch", "train.py"])

# Expected
# ns.<your_new_arg> equals the documented default
print(ns.max_step_time_ms)  # should print 0

# Pass criterion
# AttributeError or different value -> FAIL
# 0 -> PASS
assert hasattr(ns, "max_step_time_ms")
assert ns.max_step_time_ms == 0
```

That's the minimum for any new flag. If this fails, the PR cannot ship — every existing v0.2.x user invocation breaks.

### 6.2 Worked example — env-var read default agrees with argparse

```python
# Setup
import os
import importlib

# Ensure the env var is unset
os.environ.pop("TRACEML_MAX_STEP_TIME_MS", None)

# Command
from traceml.runtime import executor
importlib.reload(executor)  # force re-read of os.environ
read = executor.read_traceml_env()

from traceml.cli import build_parser
ns = build_parser().parse_args(["watch", "x.py"])

# Pass criterion
# Read-site default must agree with argparse default
assert read["max_step_time_ms"] == ns.max_step_time_ms
```

If this fails, you have default drift. Repeat the same recipe with `aggregator_main.read_traceml_env` and with `TraceMLSettings()` (no args) to cover all four sites.

### 6.3 Worked example — `--mode` dispatcher consistency

```python
# Pass criterion
# Every choice in argparse `choices=` must be a key in _DISPLAY_DRIVERS
# AND in the supported_modes set in cli.py:535.

from traceml.cli import build_parser
from traceml.aggregator.trace_aggregator import _DISPLAY_DRIVERS

argparse_choices = None
for action in build_parser()._actions:
    for sa in getattr(action, "choices", {}) or {}:
        if sa in {"watch", "run", "deep"}:
            sub = action.choices[sa]
            for a in sub._actions:
                if a.dest == "mode":
                    argparse_choices = set(a.choices or [])
                    break

assert argparse_choices == set(_DISPLAY_DRIVERS.keys())
```

If this fails, the user gets a `ValueError` post-spawn instead of a clean parse error.

### 6.4 When you can't write a verification gate

If you can't write a recipe — you only have a vague worry — **don't raise the concern in the review yet**. Either escalate it to research (file a follow-up issue, label "investigate"), or hold it back per §7.3. Vague concerns waste author time.

### 6.5 Recipe style rules

- **Specific numbers, not adjectives.** `ns.max_step_time_ms == 0` not "should be sane." Adjectives are debate; numbers are tests.
- **Reproducible from a clean checkout.** No "you also need to apply patch X first" — if the recipe depends on prior fixes, restate them.
- **3–10 lines of actual code.** Longer means you're testing too much at once; cut to the smallest demonstrating example.
- **CLI tests do not need a GPU.** This is one place where TraceML reviewing is cheap — every recipe in this guide runs on a CPU-only laptop. If a CLI recipe needs a GPU, you've left the CLI surface.

---

## 7. Step 6 — Drafting comments

Two granularity choices: line comment vs PR-level comment. They are not interchangeable.

### 7.1 Line comments

Use when: there is a specific code change you're proposing in a specific location. Pin the comment to the line that needs to change.

Pattern: state the issue → propose the fix → reference a verification gate or precedent.

```
The argparse default here is 100 but the executor module constant
DEFAULT_MAX_STEP_TIME_MS is 0 (executor.py:48). Test that bypasses
the CLI and constructs TraceMLSettings() directly will see 0.

Suggest aligning to one value (probably 0 for "disabled by default")
and adding a parse-test + settings-default-test pair (see review_cli.md
§6.2 for the recipe shape).
```

Keep it tight. The reviewer's job is to point at the change, not to re-derive the architecture.

### 7.2 PR-level comments

Use when: the concern is **behavioural** or **architectural**, not localised to a single line. The fix may touch multiple files; the discussion is about the PR's intent.

Pattern: state the scenario → walk through what happens under the current diff → propose 2–3 fixes ranked by your preference → invite discussion. Examples that warrant PR-level treatment: "this flag should be a subcommand, not a flag," "the default change here is a behaviour flip and needs a minor version bump," "we have three defaults disagreeing — should we consolidate to a single source of truth in this PR or file a follow-up?"

A PR-level comment is also right for cross-cutting concerns: "did you smoke-test on a fresh install?", "does this env-var name conflict with `TRACEML_X` from v0.2?", "the manifest field name is awkward — please document the rule."

### 7.3 What NOT to raise (the holdback discipline)

Two kinds of items belong in your private parking-lot, not in the PR review:

- **Judgement calls about positioning** — "should this flag have been a config-file entry instead?", "is the current `supported_modes` duplication worth fixing in this PR?". These are about your relationship with the author, not the PR. Decide privately, apply privately.
- **Adjacent improvements** — "while we're here, the `--interval` default drift could be fixed in the same PR." If the improvement isn't required for the PR to ship, file it as a follow-up issue. Don't grow the PR.

The discipline: a PR review delivers a focused set of must-fix items. Bloating the review with parking-lot items dilutes the must-fix signal and trains the author to treat your reviews as discussion threads, not gates.

### 7.4 The maintainer summary

Every review ends with a 2–3 sentence executive summary suitable for the maintainer to read without opening the PR. The shape:

> PR #N adds [flag/subcommand/mode]; argparse + env-var + read-site + manifest layers all updated; tests cover [parse / validation / backward-compat]. Review converged on K concrete items: (1) ..., (2) ..., (3) .... All K fixes are localised; each needs one small test. Recommend [verdict].

Maintainer reads three sentences and either agrees with the verdict or opens the PR. This is the artifact your maintainer wants more than the diff comments.

---

## 8. Step 7 — Landing the verdict

Three states. Pick exactly one.

### 8.1 Approve

Conditions:
- Consistency table (§3) is fully ✅ or has documented justified deviations.
- All four meta-questions (§5) answered explicitly.
- All ten invariants (§5.4) preserved, especially: lean launcher imports, three-defaults-one-value, default-invocation-still-works.
- No concerns require a verification gate (§6).
- Tests cover argparse parsing, `_validate_launch_args` (if cross-arg constraint added), and backward-compat default invocation.
- New env vars documented in `traceml/CLAUDE.md` env-var table; CHANGELOG entry under "Unreleased."

If all six are true, approve cleanly. Don't suggest follow-up work in the approval — file follow-ups separately so the PR can ship.

### 8.2 Approve with changes

Conditions:
- All concerns are minor: docstring fixes, test gaps for non-critical paths, naming nits, missing `--help` text, missing CHANGELOG entry.
- No concern affects backward compatibility, default agreement, or training reproducibility (manifest field).
- All concerns have a one-line fix or a clear written-down resolution.

This is the "accept the PR but require these N small changes." Not "the PR is conceptually broken."

### 8.3 Block (request changes)

Conditions (any one):
- A concern breaks **backward compatibility** (default invocation fails, default value flips behaviour, env var renamed without fallback read).
- A concern introduces **default drift** across argparse / settings / executor / aggregator.
- A concern leaves a five-layer plumbing chain incomplete (argparse declared but no env var, env var assembled but no read site, read site present but no consumer).
- A concern violates the lean-launcher invariant (`import torch` / `import nicegui` / `import pandas` at the top of `cli.py`).
- A concern affects training reproducibility (a flag changes data but is not in the manifest).
- The PR introduces a new `--mode` choice without updating `_DISPLAY_DRIVERS` or the `supported_modes` set.
- Tests don't exist for a category in §4 that applies (parse, validation, backward-compat).

### 8.4 What "block" doesn't mean

It does not mean the architecture is wrong. It does not mean the author has to redesign. It means **these specific items must be resolved before merge.** Frame the verdict that way to keep the relationship healthy with the author.

---

## 9. Reference: applying the workflow to a hypothetical PR

To make the seven steps concrete, walk a hypothetical PR adding `--max-step-time-ms` (the example from `add_cli.md` §4a) through the workflow:

| Step | What the reviewer does |
|---|---|
| 1. Anchor | 6 files in diff: `cli.py`, `settings.py`, `executor.py`, `aggregator_main.py`, `samplers/step_time_sampler.py` (consumer), `tests/test_max_step_time_cli.py`. Each maps to a row in §2's table. |
| 2. Consistency table | Fill the column for `--max-step-time-ms`. Argparse `type=int default=0`. Env-var assembly `cli.py:472`. Aggregator + executor read with `.get("TRACEML_MAX_STEP_TIME_MS", "0")`. Settings field `max_step_time_ms: int = 0`. Manifest extended via signature. Backward-compat: default `0` means "disabled," no behaviour change. |
| 3. Failure modes | §4.1 (5-layer plumbing — verify all 5 sites). §4.2 (default drift — verify "0" everywhere). §4.9 (`int(None)` — verify `.get(name, "0")`). §4.10 (boolean misuse — n/a, this is `int`). §4.13 (argparse attr typo — verify `args.max_step_time_ms` not `args.max-step-time-ms`). |
| 4. Meta-questions | New axis? No, it's a numeric flag. Defaults consistent? Yes. Wire-name a contract? Yes — `TRACEML_MAX_STEP_TIME_MS` is durable. Invariants preserved? Yes — verify with §6.1 recipe. |
| 5. Verification gates | §6.1 recipe (default invocation still works). §6.2 recipe (env-var read default agrees with argparse). |
| 6. Comments | Line comment on missing `--help` text detail. PR-level comment if the flag should be a sampler-level config instead. |
| 7. Verdict | Approve if all gates pass; approve-with-changes if `--help` is thin; block if default disagrees or backward-compat breaks. |

If the hypothetical PR also added `--mode jsonstream`, you would additionally run §4.4 (dispatcher skew) and §6.3 (dispatcher consistency recipe), and check that the `BaseDisplayDriver` subclass exists.

---

## 10. Common reviewer mistakes

Numbered, with cause and fix.

1. **Reviewing in isolation.** Cause: opening the diff first, before anchoring to walkthroughs. Effect: drowning in 6 files. Fix: do §2 before §3 — every time.

2. **Re-deriving the consistency table from scratch.** Cause: the table isn't a formal artifact in source. Effect: each reviewer rebuilds it slightly differently. Fix: paste the §3.2 table into your review notes verbatim and add the new column.

3. **Raising parking-lot items in the PR review.** Cause: not distinguishing must-fix from nice-to-have. Effect: review becomes a discussion thread, must-fix items lose signal. Fix: §7.3 holdback. Anything not actionable goes to follow-up issues.

4. **Vague concerns without verification gates.** Cause: time pressure, gut feel. Effect: author can't reproduce, dismisses the concern. Fix: §6 — every concern gets a recipe, especially the cheap CPU-only ones for CLI changes.

5. **Mixing line comments and PR-level comments.** Cause: writing architectural concerns inline next to a code line. Effect: comment gets resolved by changing one line, the architectural point is lost. Fix: §7.1/§7.2 — pick the granularity deliberately.

6. **Skipping the backward-compat check.** Cause: the new flag has a default, so the reviewer assumes existing invocations work. Effect: the default disagrees with the old behaviour or the attribute access typos out. Fix: §6.1 recipe is mandatory, not optional.

7. **Approving on argparse correctness without checking the consumer.** Cause: the argparse declaration is clean and the reviewer stops. Effect: env var assembled, never read, flag is dead. Fix: walk all five layers (§4.1) for every flag.

8. **Conflating "matches the family" with "correct."** Cause: §3 consistency check is ✅ across the board, so the reviewer stops. Effect: novel-axis failure modes (§5.1) miss; default drift across layers (§4.2) misses. Fix: the consistency table is a starting point, not a verdict.

9. **Trusting the test suite for backward compat.** Cause: green CI feels like backward-compat coverage. Effect: there are zero tests for argparse parsing of any launch flag (`add_cli.md` §11). The CI passes don't prove anything about CLI surface compatibility. Fix: run §6.1 recipe locally.

10. **Skipping the maintainer summary.** Cause: assumes the maintainer will read the diff. Effect: maintainer reads the diff, disagrees on severity, kicks back to the reviewer. Fix: §7.4 — three sentences are the maintainer's reading material; the diff is yours.

---

## Gaps encountered while writing this guide

Where the reviewer's playbook is currently underspecified or relies on folklore. Flag these in your review process if you hit them.

- **The consistency table (§3) isn't a formal artifact.** It lives in this guide and in `add_cli.md`. If the CLI surface grows another flag or two, every reviewer will diverge on the column set. Worth lifting into a contract test in `tests/test_cli_consistency.py` that introspects `cli.py::_add_launch_args`, `aggregator_main.py::read_traceml_env`, `executor.py::read_traceml_env`, and `TraceMLSettings`, and asserts every flag has all four sites with agreeing defaults. Not yet written.

- **Default drift is a known live bug, not a theoretical risk.** `--interval` (`2.0` in argparse, `1.0` in settings/executor/aggregator) and `--num-display-layers` (`5` in argparse, `20` in settings/executor/aggregator) are pre-existing divergences. The reviewer's failure-mode catalogue (§4.2) tells you to flag a new one; it does not say what to do about the old ones. A dedicated PR consolidating defaults to a single source of truth is the fix; meanwhile, every CLI review is partially adversarial against the existing code-base.

- **There's no central registry of `TRACEML_*` env-var names.** A reviewer enforcing §5.3 (wire-name contract) has to grep `cli.py`, `aggregator_main.py`, `executor.py`, and the consumer module, and trust their grep. Worth a constants module `src/traceml/runtime/env_names.py` with every name exported as a constant; then a test asserts no two constants collide and that every assembly site reads from the constant. Not yet written.

- **Zero tests for the CLI surface.** `tests/test_compare_missing.py` is the only CLI test, and it tests `compare`'s payload shaping rather than the CLI surface itself. There are zero tests for argparse parsing of any launch flag, zero tests for `_validate_launch_args`, zero tests for env-var assembly in `launch_process`, zero tests for the `_DISPLAY_DRIVERS` dispatch path. Reviewers run the §6 recipes locally because there is no automated harness. The §6 recipes should be promoted to fixtures in `tests/test_cli_surface.py`. Not yet written.

- **The verdict criteria (§8) are folklore-level for the medium-risk cases.** "Default drift" is a clean block. "Three defaults disagree but the CLI always sets the env var so production sees argparse" is — what? Today this is a judgement call; reasonable reviewers would split. A formal list of "CLI invariants the project commits to" with severity tagging would resolve these arguments before the PR.

- **The `supported_modes` duplication at `cli.py:535` is a known footgun.** It exists from before the dispatcher was hoisted into `_DISPLAY_DRIVERS`. The right fix lifts `_DISPLAY_DRIVERS.keys()` into a small registry module that can be imported by `cli.py` without dragging NiceGUI/Plotly into `traceml --help`. Until then, every `--mode` PR has to remember the redundant set. The reviewer's job is to remember when the author forgets; that's not a sustainable contract.

- **No reviewer-side smoke harness.** A reviewer who wants to run the §6.1 backward-compat recipe needs to write the script themselves each time. A `tests/review_harness/test_cli_backward_compat.py` parametrised over every flag in `_add_launch_args` would make backward-compat checking automatic. Not yet written.

These gaps are the natural follow-up work after this guide lands. None block the next reviewer from following the workflow above; all would make the next reviewer faster.
