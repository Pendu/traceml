# How to add a new CLI command, flag, or mode

> Internal contributor guide. Audience: one trusted co-founder / new engineer
> being onboarded to TraceML. Not for public docs.

This guide teaches you how to extend the TraceML CLI surface — a new subcommand, a new flag, a new `--mode` choice, a new profile, or a new `TRACEML_*` environment variable. It assumes you have read `add_sampler.md` and `add_renderer.md`, can run the test suite, and have a local checkout of the repository.

---
Feature type: CLI surface (subcommand / flag / mode / profile / env var)
Risk level: medium  (anything user-facing breaks v0.2.x users on PyPI)
Cross-cutting impact: multiple subsystems (CLI process → aggregator process → executor → runtime → samplers)
PyTorch coupling: none directly  (but profile flags gate sampler choice, and samplers couple to PyTorch)
Reference PRs: none called out yet — flag changes have historically gone in with the feature that needed them
Companion reviewer guide: none yet
Last verified: 2026-04-25
---

## 1. Intro and mental model

### What is a "CLI change" in TraceML?

The TraceML CLI lives in a single file: `src/traceml/cli.py`. Everything the user types after the word `traceml` is parsed there. A CLI change falls into exactly one of five categories:

| Category   | Example                            | Blast radius                                                                 |
|------------|------------------------------------|------------------------------------------------------------------------------|
| Subcommand | `traceml compare a.json b.json`    | New subparser, new handler. Either offline-only or spawns aggregator+executor. |
| Flag       | `--tcp-port 29765`                 | Argparse line + env var + consumer. Touches every layer of the launch path. |
| `--mode`   | `--mode summary`                   | New `BaseDisplayDriver` subclass + `_DISPLAY_DRIVERS` entry.                |
| Profile    | `--profile deep` (positional via the `deep` subcommand) | New subparser + `_build_samplers()` branch. See `add_sampler.md` §11.      |
| Env var    | `TRACEML_HISTORY_ENABLED`          | CLI assembly site + read site (`runtime/settings.py` or `aggregator_main.py`). |

Pick the smallest category that fits. Adding a flag is cheaper than adding a subcommand. Adding a `--mode` choice is cheaper than adding a profile. Adding a profile is cheaper than adding a subcommand if the new behaviour is "samplers + a different name on the wire".

### The two-process boundary

The CLI process **does not** run training or telemetry. It is a launcher. After parsing arguments, it spawns two child processes:

```
                     traceml CLI process
                         (cli.py:main)
                              │
              ┌───────────────┴────────────────┐
              ▼                                ▼
   start_aggregator_process              start_training_process
   (cli.py:395)                          (cli.py:422)
              │                                │
              ▼                                ▼
   aggregator_main.py:main              torchrun → executor.py
   (reads TRACEML_* env vars)           (reads TRACEML_* env vars,
                                         constructs TraceMLRuntime)
```

Both children inherit the parent's environment via `subprocess.Popen(env=env, ...)` — see `cli.py::launch_process` lines 452–472 for the exact assembly. **Once a child is spawned, the configuration is locked**: there is no IPC channel for config beyond the TCP telemetry stream, and TCP carries telemetry rows, not knobs.

This means every CLI knob must be plumbed end-to-end:

```
argparse declaration   →  validation        →  env-var assembly      →  consumer
(cli.py:_add_launch_args) (_validate_launch_args)  (cli.py:launch_process)   (executor / aggregator_main)
```

Skip any layer and the flag is dead code in that layer's process.

For a refresher on why fork+exec means children can't read the parent's in-memory state, see [Q7](../deep_dive/learning-qa.md#q7-spawning--fork-exec-and-multiprocessing-start-methods) and [W1](../deep_dive/code-walkthroughs.md#w1-clipy--top-level-launcher-and-process-orchestrator).

### What the CLI process owns

- **Argument parsing** — `argparse` subparsers in `cli.py::build_parser`.
- **Cross-argument validation** — `cli.py::_validate_launch_args`.
- **Session id generation** — via `traceml.runtime.session.get_session_id`.
- **Manifest writing** — `cli.py::write_run_manifest`, `cli.py::update_run_manifest`. The manifest is the forensic record of what the user asked for; if a flag matters for reproduction, it goes in the manifest.
- **Process group lifecycle** — `start_new_session=True`, signal handlers, graceful shutdown (`cli.py::install_shutdown_handlers`).
- **TCP readiness gating** — `cli.py::wait_for_tcp_listen` polls the aggregator's bind port before launching torchrun.

What the CLI process **does not** own:

- It does not import `torch` (anywhere). Importing torch in the launcher would slow startup and would couple `traceml --help` to a heavy dependency. Keep launcher imports lean.
- It does not touch samplers, renderers, the database, or hooks.
- It does not serialize wire payloads (only env vars).

For the end-to-end pipeline (sampler → buffer → DB → sender → TCP → aggregator → store → renderer), see [`pipeline_walkthrough.md`](pipeline_walkthrough.md). The CLI is the launch strap that assembles the inputs to that pipeline.

### The fail-open contract for the CLI

Unlike samplers, the CLI is allowed to refuse to launch. Validation errors should `raise SystemExit(...)` with a `[TraceML] ERROR:` prefix. The fail-open law applies once the runtime is up — not at parse time. Do not catch argparse errors and silently default; surface them so the user sees the problem before training starts.

---

## 2. Before you start: decisions to make

Answer all of these before opening an editor. Write them in the PR description.

- [ ] **Category.** Subcommand / flag / `--mode` choice / profile / env var? If you're adding more than one, split the PR.
- [ ] **Surface scope.** If it's a flag, which subcommands does it apply to? `_add_launch_args` is shared by `watch`, `run`, `deep`. If your flag should appear on all three, add it there. If only on `run`, add it directly to `run_parser`. `compare` and `inspect` have their own argument sets and do not use `_add_launch_args`.
- [ ] **Env var or pure-CLI?** Most flags become a `TRACEML_*` env var because both the aggregator process and the training process need to see the value. A flag that affects only CLI-process behaviour (e.g. manifest layout) does not need an env var.
- [ ] **Backward compat.** Will `traceml watch script.py` (no extra args) still produce the same behaviour after your change? Default invocations of v0.2.x must not break. If your default behaviour changes, document it in `CHANGELOG.md` and bump the minor version.
- [ ] **Manifest impact.** Should the new value be recorded in `manifest.json` for reproducibility? `write_run_manifest` (`cli.py:192`) writes a fixed set of fields; new fields go via the `extra=` parameter or a direct addition to the function signature.
- [ ] **Validation layer.** Argparse `type=` (cheap, per-arg), or `_validate_launch_args` (cross-arg constraints), or runtime startup, or sampler startup? Push validation as far left as possible. Validating in argparse is best because the user gets the error before the aggregator even spawns.
- [ ] **Default value.** What's the right default? Defaults appear in three places (argparse `default=`, env-var read-site default, `TraceMLSettings` dataclass default) and must agree.
- [ ] **Documentation.** Every new env var must land in `traceml/CLAUDE.md` and the relevant `docs/` page. Every new flag should appear in the subparser's `help=`.

---

## 3. Anatomy of three existing surfaces

Walk through one example per major category. The aim is for you to see the full plumbing before you write your own.

### 3.1. An existing flag — `--tcp-port`

The flag declares a TCP port for aggregator ↔ rank communication. End-to-end:

**Step A. Argparse declaration.** `cli.py::_add_launch_args`, line ~757:

```python
    parser.add_argument(
        "--tcp-port", type=int, default=29765, help="Aggregator bind port."
    )
```

`type=int` is the only validation. There is no `_validate_launch_args` hook for it; out-of-range ports surface later as bind failures from the aggregator, which propagate as a non-ready timeout in `wait_for_tcp_listen`.

**Step B. Env-var assembly.** `cli.py::launch_process`, line ~469:

```python
    env["TRACEML_TCP_PORT"] = str(args.tcp_port)
```

Note the `str(...)` — env-var values are always strings.

**Step C. Aggregator-side read.** `aggregator/aggregator_main.py::read_traceml_env`, line ~108:

```python
    "tcp_port": int(os.environ.get("TRACEML_TCP_PORT", "29765")),
```

The default `"29765"` here must match the argparse default. They are both sources of truth — argparse for users running the CLI, env-var default for code paths that bypass the CLI (tests, direct invocation). Keep them in sync.

**Step D. Settings construction.** `aggregator_main.py::main`, line ~163:

```python
    settings = TraceMLSettings(
        ...
        tcp=TraceMLTCPSettings(
            host=str(cfg["tcp_host"]),
            port=int(cfg["tcp_port"]),
        ),
        ...
    )
```

**Step E. Runtime-side read.** `runtime/executor.py::read_traceml_env`, line ~234:

```python
    "tcp_port": int(
        os.environ.get("TRACEML_TCP_PORT", str(DEFAULT_TCP_PORT))
    ),
```

Here `DEFAULT_TCP_PORT = 29765` is a module constant (`executor.py:46`). **Three defaults, one value.** Drift is a bug.

**Step F. Manifest record.** `cli.py::write_run_manifest`, line ~234:

```python
    "launch": {
        ...
        "tcp_port": int(tcp_port),
        ...
    },
```

The flag is captured in `manifest.json` so post-mortem analysis can tell which port a session actually used.

**Step G. CLI readiness probe.** `cli.py::launch_process`, line ~569:

```python
    ready = wait_for_tcp_listen(
        host=args.tcp_host,
        port=int(args.tcp_port),
        proc=agg_proc,
        timeout_sec=DEFAULT_TCP_READY_TIMEOUT_SEC,
    )
```

The CLI uses the same port to verify the aggregator is alive before spawning the training process. Without this gate, torchrun could start before the TCP server is bound and the rank connections would race.

That's seven sites for a single flag. Most flags don't need all seven — the readiness probe is specific to ports — but the pattern of **argparse → env-var → consumer-side read → settings → manifest** is the spine for almost every flag.

### 3.2. An existing subcommand — `inspect`

`traceml inspect <file>` decodes a binary `.msgpack` log file and prints each frame as JSON. It's a **pure-CLI** subcommand: it does **not** spawn an aggregator, does not start torchrun, does not touch any `TRACEML_*` env vars.

**Subparser registration.** `cli.py::build_parser`, line ~849:

```python
    inspect_parser = sub.add_parser(
        "inspect", help="Inspect binary .msgpack logs."
    )
    inspect_parser.add_argument("file", help="Path to a .msgpack file.")
```

Note the absence of `_add_launch_args(inspect_parser)`. None of the launch args make sense for an offline decoder — there's no script to run, no aggregator to spawn. Use the launch helper only for subcommands that genuinely launch training.

**Dispatch.** `cli.py::main`, line ~873:

```python
    elif args.command == "inspect":
        run_inspect(args)
```

**Handler.** `cli.py::run_inspect`, line ~648. The handler reads the file, decodes msgpack frames in a loop, prints JSON, and `raise SystemExit(1)` on errors. No env vars set, no subprocesses spawned.

The same shape applies to `compare` (`cli.py:686`): pure-CLI, no aggregator, no torchrun. It calls into `traceml.compare.command.compare_summaries` and exits.

**The lesson:** if your new subcommand consumes already-written data (manifest, summary JSON, msgpack), it does not need `launch_process`. Reach for `launch_process` only when the user wants to run a training script under TraceML.

### 3.3. An existing `--mode` choice — `summary`

`--mode` selects the display medium. The three current choices are `cli`, `dashboard`, `summary`.

**Argparse declaration.** `cli.py::_add_launch_args`, line ~712:

```python
    parser.add_argument(
        "--mode",
        type=str,
        default="cli",
        choices=["cli", "dashboard", "summary"],
        help=(
            "TraceML display mode to launch. "
            "Use 'summary' for final-summary-only runs. Default: cli."
        ),
    )
```

`choices=` enforces the closed set at parse time — typos surface as `error: argument --mode: invalid choice: 'sumary'` before any process spawns. This is the right place for closed-set validation.

**Cross-argument validation.** `cli.py::_validate_launch_args`, line ~137:

```python
    if getattr(args, "mode", None) == "summary" and getattr(
        args, "no_history", False
    ):
        raise SystemExit(
            "[TraceML] ERROR: --mode=summary requires history. "
            "Remove --no-history to enable final summary generation."
        )
```

Argparse can validate one argument in isolation; cross-argument constraints (`--mode=summary` requires history, etc.) live in `_validate_launch_args`, called from `cli.py::main` line ~862 after parsing but before dispatch.

**Env-var assembly.** `cli.py::launch_process`, line ~460:

```python
    env["TRACEML_UI_MODE"] = args.mode
```

Note the env-var name: `TRACEML_UI_MODE`, not `TRACEML_MODE`. The aggregator (`aggregator_main.py:93`) and the executor (`executor.py:212`) both read `TRACEML_UI_MODE` first and fall back to `TRACEML_MODE` for backward compatibility:

```python
    ui_mode = os.environ.get(
        "TRACEML_UI_MODE",
        os.environ.get("TRACEML_MODE", "cli"),
    )
```

This is the **canonical example of a renamed env var done right**: don't remove the old name; read both, prefer the new. Keep this pattern when you ever rename a `TRACEML_*` variable.

**Dispatcher.** `aggregator/trace_aggregator.py`, line ~58:

```python
_DISPLAY_DRIVERS: Dict[str, Type[BaseDisplayDriver]] = {
    "cli": CLIDisplayDriver,
    "dashboard": NiceGUIDisplayDriver,
    "summary": SummaryDisplayDriver,
}
```

Used at line ~137:

```python
    driver_cls = _DISPLAY_DRIVERS.get(settings.mode)
    if driver_cls is None:
        raise ValueError(
            f"[TraceML] Unknown display mode: {settings.mode!r}. "
            f"Supported: {sorted(_DISPLAY_DRIVERS.keys())}"
        )
```

**Driver implementation.** `aggregator/display_drivers/summary.py`. It subclasses `BaseDisplayDriver` (`base.py`) and implements `start()`, `tick()`, `stop()`. For the renderer side of this story see [`add_renderer.md`](add_renderer.md) §12.

**The full path for one new `--mode` choice:**

```
argparse choices=  →  _validate_launch_args
                   →  TRACEML_UI_MODE env var
                   →  _DISPLAY_DRIVERS dispatcher entry
                   →  BaseDisplayDriver subclass implementation
```

Five sites. Skip the dispatcher and you'll get the `ValueError` above from the aggregator process and the launch will fail post-spawn. Skip the argparse `choices=` and the user gets a confusing aggregator error instead of a clean parse error.

---

## 4. Step-by-step: adding a new CLI surface

Three hypothetical examples, one per major category. Each is concrete enough that you could ship the PR by following the steps verbatim.

### 4a. Adding a new flag — `--max-step-time-ms`

The hypothetical flag: a soft alarm threshold. If a step takes longer than `--max-step-time-ms`, the runtime should log a warning. Default: no threshold (disabled).

**Step 1. Argparse declaration.** Add to `cli.py::_add_launch_args`:

```python
    parser.add_argument(
        "--max-step-time-ms",
        type=int,
        default=0,
        help=(
            "Soft alarm threshold in milliseconds. "
            "If a step exceeds this, a warning is logged. "
            "0 disables the alarm. Default: 0."
        ),
    )
```

`type=int` rejects non-numeric input at parse time. `default=0` means "disabled" — pick a sentinel that obviously means "off" rather than a real-world threshold.

**Step 2. Validation.** Add to `cli.py::_validate_launch_args`:

```python
    max_ms = getattr(args, "max_step_time_ms", 0)
    if max_ms < 0:
        raise SystemExit(
            "[TraceML] ERROR: --max-step-time-ms must be >= 0 "
            f"(got {max_ms})."
        )
```

Argparse's `type=int` won't reject negatives. Add a guard if non-negative matters to your consumer.

**Step 3. Env-var assembly.** Add to `cli.py::launch_process`, alongside the other `env[...] = ...` lines (~line 472):

```python
    env["TRACEML_MAX_STEP_TIME_MS"] = str(args.max_step_time_ms)
```

Always cast to `str(...)`. Always assign unconditionally — do not skip when the value equals the default. Explicit is better than implicit: downstream consumers shouldn't have to know the parent's default.

**Step 4. Read site (runtime).** Add a field to `runtime/settings.py`:

```python
@dataclass(frozen=True)
class TraceMLSettings:
    ...
    max_step_time_ms: int = 0
```

Then read the env var in `runtime/executor.py::read_traceml_env`:

```python
    "max_step_time_ms": int(
        os.environ.get("TRACEML_MAX_STEP_TIME_MS", "0")
    ),
```

And construct it where `TraceMLSettings(...)` is instantiated. Keep the default `"0"` here in sync with the argparse default and the dataclass default. **Three defaults, one value.**

**Step 5. Consumer.** Whatever subsystem actually uses this — likely a sampler or a diagnostic. For a hypothetical use in `StepTimeSampler`:

```python
class StepTimeSampler(BaseSampler):
    def __init__(self):
        super().__init__(...)
        self.max_step_time_ms = int(
            os.environ.get("TRACEML_MAX_STEP_TIME_MS", "0") or 0
        )

    def sample(self):
        ...
        if self.max_step_time_ms > 0 and step_ms > self.max_step_time_ms:
            self.logger.warning(
                f"[TraceML] step {step_idx} took {step_ms:.1f} ms "
                f"(threshold {self.max_step_time_ms} ms)"
            )
```

You could also pass through `TraceMLSettings` instead of reading the env var directly in the sampler — both are precedented. Reading via env var in the sampler keeps the sampler constructor parameter-less (the house style — see `add_sampler.md` §3.1).

**Step 6. Manifest field.** Add to `cli.py::write_run_manifest`. Either extend the function signature:

```python
def write_run_manifest(
    ...
    max_step_time_ms: int = 0,
    ...
):
    manifest = {
        ...
        "launch": {
            ...
            "max_step_time_ms": int(max_step_time_ms),
        },
        ...
    }
```

…or pass via `extra=`:

```python
    manifest_path = write_run_manifest(
        ...,
        extra={"max_step_time_ms": int(args.max_step_time_ms)},
    )
```

Prefer extending the function signature for first-class fields and `extra=` for one-off telemetry.

**Step 7. Tests.** Three:
- Argparse parse test: `parse_args(["watch", "x.py", "--max-step-time-ms", "100"])` produces `args.max_step_time_ms == 100`.
- Validation test: `--max-step-time-ms -1` raises `SystemExit`.
- Backward-compat: `parse_args(["watch", "x.py"])` produces `args.max_step_time_ms == 0`.

See §8 for the test template.

### 4b. Adding a new `--mode` choice — `jsonstream`

Hypothetical: a CI-friendly mode that emits one JSON line per renderer tick to stdout. No Rich, no NiceGUI, no summary card — just structured events for downstream tooling.

**Step 1. Implement the display driver.** Create `src/traceml/aggregator/display_drivers/jsonstream.py`:

```python
"""
JSON-stream display driver for CI / programmatic consumers.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

from traceml.aggregator.display_drivers.base import BaseDisplayDriver
from traceml.database.remote_database_store import RemoteDBStore
from traceml.runtime.settings import TraceMLSettings


class JSONStreamDisplayDriver(BaseDisplayDriver):
    """
    Emits one JSON object per tick to stdout.

    Each line is a complete JSON document. Terminate readers with EOF on
    process exit; do not buffer (every write is followed by flush).
    """

    def __init__(
        self, logger: Any, store: RemoteDBStore, settings: TraceMLSettings
    ) -> None:
        super().__init__(logger=logger, store=store, settings=settings)

    def start(self) -> None:
        self._emit({"event": "start", "ts": time.time()})

    def tick(self) -> None:
        try:
            payload = self._snapshot_store()
            self._emit({"event": "tick", "ts": time.time(), **payload})
        except Exception:
            self._logger.exception("[TraceML] jsonstream tick failed")

    def stop(self) -> None:
        self._emit({"event": "stop", "ts": time.time()})

    def _snapshot_store(self) -> dict:
        # Read-only over RemoteDBStore. See add_renderer.md §3 for
        # the discipline; never mutate the store from a driver.
        return {"ranks": list(self._store.ranks())}

    def _emit(self, obj: dict) -> None:
        sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
        sys.stdout.flush()
```

For the broader display-driver contract — what `start`, `tick`, `stop` should and should not do, how to read SQLite vs `RemoteDBStore`, and the read-only invariant — see [`add_renderer.md`](add_renderer.md) §12.

**Step 2. Register in the dispatcher.** `aggregator/trace_aggregator.py`:

```python
from traceml.aggregator.display_drivers.jsonstream import (
    JSONStreamDisplayDriver,
)

_DISPLAY_DRIVERS: Dict[str, Type[BaseDisplayDriver]] = {
    "cli": CLIDisplayDriver,
    "dashboard": NiceGUIDisplayDriver,
    "summary": SummaryDisplayDriver,
    "jsonstream": JSONStreamDisplayDriver,   # <-- add
}
```

**Step 3. Add the choice to argparse.** `cli.py::_add_launch_args`:

```python
    parser.add_argument(
        "--mode",
        type=str,
        default="cli",
        choices=["cli", "dashboard", "summary", "jsonstream"],
        help=...,
    )
```

**Step 4. Validation rules in `_validate_launch_args`.** Decide whether the new mode imposes constraints on other flags:

```python
    if getattr(args, "mode", None) == "jsonstream" and getattr(
        args, "enable_logging", False
    ):
        raise SystemExit(
            "[TraceML] ERROR: --mode=jsonstream is incompatible with "
            "--enable-logging (both write to stdout). Drop one."
        )
```

Cross-mode validation belongs here, not in the driver. Let the user fail fast at parse time.

**Step 5. The `cli.py::launch_process` supported-modes check.** There's a redundant guard at line ~535:

```python
    supported_modes = {"cli", "dashboard", "summary"}
    if args.mode not in supported_modes:
        raise ValueError(
            f"Invalid display mode '{args.mode}'. "
            f"Supported modes: {sorted(supported_modes)}"
        )
```

**This is a footgun.** The set is hard-coded and disagrees with argparse `choices=` if you forget to update it. Add `"jsonstream"` to this set or — better — refactor it to read from `_DISPLAY_DRIVERS.keys()`. (See §11 "Gaps".)

**Step 6. Tests.** Argparse parse, dispatcher dispatch, and a smoke test that `traceml watch script.py --mode jsonstream` produces valid JSON on stdout.

### 4c. Adding a new subcommand — `traceml validate`

Hypothetical: `traceml validate <session-dir>` validates a session's manifest and SQLite history offline. No aggregator, no torchrun, no env vars.

**Step 1. New subparser.** `cli.py::build_parser`:

```python
    validate_parser = sub.add_parser(
        "validate",
        help="Validate a TraceML session directory's manifest and history.",
    )
    validate_parser.add_argument(
        "session_dir",
        help="Path to the session directory (parent of aggregator/).",
    )
    validate_parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat schema warnings as errors.",
    )
```

Note: no `_add_launch_args(validate_parser)`. This is an offline command; it does not need `--tcp-port`, `--mode`, `--profile`, etc.

**Step 2. Handler.** `cli.py`:

```python
def run_validate(args: argparse.Namespace) -> None:
    """Validate a session directory's manifest and SQLite history."""
    session_dir = Path(args.session_dir)
    if not session_dir.is_dir():
        print(
            f"[TraceML] ERROR: not a directory: {args.session_dir}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    try:
        from traceml.tools.validate import validate_session
        report = validate_session(session_dir, strict=bool(args.strict))
    except Exception as exc:
        print(f"[TraceML] ERROR: validate failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 1)
```

The handler **does not** call `launch_process`. It does not set env vars. It does not spawn anything. It reads, reports, exits. This shape is identical to `run_compare` and `run_inspect`.

**Step 3. Dispatch.** `cli.py::main`:

```python
    elif args.command == "validate":
        run_validate(args)
```

**Step 4. Tests.** `tests/test_validate_cli.py`:
- Construct a temp session directory with a known-good manifest, run the CLI, assert exit 0.
- Construct one with a corrupt manifest, run the CLI, assert exit 1 and a useful error message.

### 4d. Adding a new profile

Out of scope here — see [`add_sampler.md` §11](add_sampler.md#11-appendix-adding-a-new-profile). The profile name lives in argparse subparser registration; the gating lives in `runtime.py::_build_samplers`.

---

## 5. Common patterns and exemplars

When you're writing a new flag/subcommand, find the closest existing one and copy its shape.

| Pattern                                | Copy from                                                      |
|----------------------------------------|----------------------------------------------------------------|
| Boolean flag (off by default)          | `--disable-traceml`, `--enable-logging`, `--no-history` (cli.py:780–789) |
| Numeric flag with default + validation | `--tcp-port`, `--nproc-per-node`, `--num-display-layers`       |
| Float flag                             | `--interval` (cli.py:722)                                      |
| Choice flag (closed set)               | `--mode` (`choices=...`)                                       |
| Path flag                              | `--logs-dir`                                                   |
| Pass-through args to user script       | `--args` with `nargs=argparse.REMAINDER` (cli.py:772)          |
| Subparser with launch args             | `watch`, `run`, `deep` — call `_add_launch_args(subparser)`    |
| Subparser without launch args          | `compare`, `inspect` — define args directly                    |
| Optional output path                   | `--output` on `compare`                                        |
| Manifest-recorded launch field         | `tcp_port`, `nproc_per_node`, `history_enabled` in `write_run_manifest` |
| Manifest-recorded ad-hoc field         | The `extra={"artifacts": {...}}` parameter to `write_run_manifest` |
| Backward-compat env-var alias          | `TRACEML_UI_MODE` falling back to `TRACEML_MODE`               |
| Cross-argument validation              | The `--mode=summary && --no-history` rule in `_validate_launch_args` |

### Manifest-recorded vs. transient

Not every CLI argument needs to land in the manifest. Use this rubric:

- **Record** if the value affects the data produced (profile, interval, history_enabled, num_display_layers, mode).
- **Record** if the value affects reproducibility (tcp_host, tcp_port, nproc_per_node, script path, launch_cwd).
- **Skip** if it's purely cosmetic and run-local (terminal width, color preferences — none currently exist).
- **Skip** if it's an `inspect`/`compare`-style offline command (the manifest doesn't apply).

---

## 6. Schema / contract rules — env vars

Env vars are TraceML's wire between the CLI process and its children. They are not user-facing API; they are an internal contract. Treat them with the same backward-compat discipline as wire schema (see [`principles.md`](principles.md)).

### 6.1. Naming

- Prefix: `TRACEML_` (always, except for the DDP standard vars `RANK`, `LOCAL_RANK`, `WORLD_SIZE`).
- Body: `UPPER_SNAKE_CASE`. `TRACEML_TCP_HOST`, not `TRACEML_TCPHOST` or `TRACEML_tcp_host`.
- Booleans: end in a noun describing what's enabled, not a verb. `TRACEML_HISTORY_ENABLED`, `TRACEML_DISABLED`. Avoid `TRACEML_DO_X`.
- Numerics: include the unit when it's non-obvious. `TRACEML_INTERVAL` (seconds is the documented default), `TRACEML_MAX_STEP_TIME_MS` (milliseconds in the name).

### 6.2. Values

- Env var values are inherently strings. Always assign with `str(...)` on the CLI side.
- Booleans use `"1"` / `"0"`. Never `"true"` / `"false"`. Read as:

  ```python
  os.environ.get("TRACEML_DISABLED", "0") == "1"
  ```

  This is the established pattern in `cli.py:455` and `executor.py:245`.

- Numerics: cast at the read site with `int(...)` or `float(...)`. Wrap with a default first to handle missing keys:

  ```python
  int(os.environ.get("TRACEML_TCP_PORT", "29765"))
  ```

  Do **not** write `int(os.environ["TRACEML_TCP_PORT"])` — this raises `KeyError` if the env var is unset, which can happen in tests.

### 6.3. Defaults

Every env var has up to three defaults: argparse default, env-var read-site default, and (sometimes) a `TraceMLSettings` field default. **All three must agree.** When you change a default, change all three in the same PR.

The CLI assembles env vars unconditionally — even when the user-supplied value matches the default. This is intentional: it keeps `os.environ.get` at the read site from depending on whether the parent set a value or not. Don't switch to "only set when non-default" without thinking through every read site.

### 6.4. Backward compatibility

We have users on `traceml-ai` v0.2.x. Their scripts and CI pipelines may embed env vars or invoke flags. Rules:

- **Never remove a flag without a deprecation cycle.** Print a warning for one minor version, then remove.
- **Never remove an env var without a deprecation cycle.** Read both names; prefer the new. See `TRACEML_UI_MODE` / `TRACEML_MODE` (aggregator_main.py:93).
- **Never change the type of an existing env var.** A boolean stays `"1"`/`"0"`. A port stays an integer. If you genuinely need a richer type, introduce a new env var name.
- **Never change a default in a way that flips behaviour for existing users.** Adding a flag with default-off is fine. Changing an existing flag's default from "watch" to "deep" is a major version bump.

### 6.5. Documentation

Every new env var goes in:

- `traceml/CLAUDE.md` — the canonical list.
- `CLAUDE.md` (repo root) — synced.
- `docs/` — wherever the user-facing flag is documented.
- The relevant subcommand's `--help` text.

If you don't document the env var, future you will reverse-engineer its existence from `aggregator_main.py::read_traceml_env`. Don't make future you do that.

The current canonical env-var list:

| Var                        | Purpose                                            | Default       |
|----------------------------|----------------------------------------------------|---------------|
| `TRACEML_DISABLED`         | Skip all instrumentation, run script natively      | `"0"`         |
| `TRACEML_PROFILE`          | Sampler set: `watch` / `run` / `deep`              | `"watch"` (CLI), `"run"` (settings) |
| `TRACEML_UI_MODE`          | Display driver: `cli` / `dashboard` / `summary`    | `"cli"`       |
| `TRACEML_MODE`             | Legacy alias for `TRACEML_UI_MODE`                 | (read-only fallback) |
| `TRACEML_INTERVAL`         | Sampler tick interval (seconds, float)             | `2.0` (CLI), `1.0` (settings) |
| `TRACEML_LOGS_DIR`         | Session log root                                   | `"./logs"`    |
| `TRACEML_SESSION_ID`       | Explicit session id (else generated)               | (generated)   |
| `TRACEML_SCRIPT_PATH`      | Resolved path to user script                       | (set by CLI)  |
| `TRACEML_ENABLE_LOGGING`   | Enable file-based telemetry log                    | `"0"`         |
| `TRACEML_NUM_DISPLAY_LAYERS` | Live UI layer-row cap                            | `"5"` (CLI), `"20"` (settings) |
| `TRACEML_TCP_HOST`         | Aggregator bind/connect host                       | `"127.0.0.1"` |
| `TRACEML_TCP_PORT`         | Aggregator bind/connect port                       | `"29765"`     |
| `TRACEML_REMOTE_MAX_ROWS`  | `RemoteDBStore` row cap                            | `"200"`       |
| `TRACEML_NPROC_PER_NODE`   | torchrun nproc_per_node                            | `"1"`         |
| `TRACEML_HISTORY_ENABLED`  | SQLite history persistence                         | `"1"`         |
| `TRACEML_LAUNCH_CWD`       | User's original working directory at launch        | (captured)    |
| `RANK` / `LOCAL_RANK` / `WORLD_SIZE` | DDP standard, set by torchrun            | (set by torchrun) |

Note the **interval default mismatch** (`2.0` in argparse, `1.0` in `TraceMLSettings.sampler_interval_sec`) and the **num_display_layers mismatch** (`5` in argparse, `20` in settings). Both are gaps — see §11.

---

## 7. Overhead budget

CLI startup is one-time. Argparse and JSON manifest writes are microseconds. The discipline is not about hot-path overhead but about **not blocking the launch path**:

- No network calls during argument validation.
- No subprocess shell-outs to gather defaults.
- No imports of heavy libraries (`torch`, `pandas`, `nicegui`) from `cli.py`. The CLI must remain importable on a Python install with no GPU drivers and no `torch` package — `traceml --help` should work.
- TCP readiness polling is bounded: `DEFAULT_TCP_READY_TIMEOUT_SEC = 15.0` (cli.py:21). Don't extend this to "wait forever" without a reason.
- `_collect_existing_artifacts` (cli.py:117) walks a fixed set of candidate paths; do not turn it into a directory scan.

The pipeline's hot-path overhead budgets — sampler tick under 1 ms, renderer compute under 100 ms — are out of scope for the CLI. See [`principles.md`](principles.md) for the global rule.

---

## 8. Testing

### Existing test reference

The only direct CLI test today is [`tests/test_compare_missing.py`](https://github.com/Pendu/traceml/blob/main/tests/test_compare_missing.py), which exercises `traceml compare`'s payload-shaping logic by calling the core function directly (not via subprocess). It does not test argparse, does not test env-var assembly, does not test child-process spawning.

This is a thin floor. New CLI changes should add the missing layers.

### What a new CLI change should test

At minimum:

1. **Argparse parses correctly.** `parse_args([...])` produces the expected `Namespace`.
2. **Validation rejects bad combos.** `_validate_launch_args` raises on forbidden combinations.
3. **Backward compat.** Default invocations (no new flag) still parse and produce the same `Namespace` as before.
4. **End-to-end smoke.** Spawn the CLI as a subprocess on a tiny example and confirm it doesn't crash. (Slow; scope to one or two flags.)
5. **(For env vars)** Read-site default survives a missing env var.

### Minimal test template

```python
"""
Tests for the --max-step-time-ms flag.
"""

import pytest

from traceml.cli import _validate_launch_args, build_parser


def _parse(args):
    parser = build_parser()
    return parser.parse_args(args)


class TestMaxStepTimeFlag:
    def test_default_is_zero(self):
        ns = _parse(["watch", "train.py"])
        assert ns.max_step_time_ms == 0

    def test_explicit_value_parsed(self):
        ns = _parse(["watch", "train.py", "--max-step-time-ms", "150"])
        assert ns.max_step_time_ms == 150

    def test_negative_rejected(self):
        ns = _parse(["watch", "train.py", "--max-step-time-ms", "-1"])
        with pytest.raises(SystemExit):
            _validate_launch_args(ns)

    def test_non_numeric_rejected_at_parse_time(self):
        with pytest.raises(SystemExit):
            _parse(["watch", "train.py", "--max-step-time-ms", "fast"])

    def test_omission_does_not_break(self):
        # backward compat: existing v0.2.x invocations still work
        ns = _parse(["watch", "train.py"])
        assert ns.script == "train.py"
        # ...other expected fields unchanged
```

### End-to-end smoke

For the brave: a subprocess test that actually runs the CLI:

```python
import subprocess
import sys

def test_cli_help_does_not_crash():
    result = subprocess.run(
        [sys.executable, "-m", "traceml.cli", "--help"],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert b"watch" in result.stdout
    assert b"run" in result.stdout
    assert b"deep" in result.stdout
```

This catches regressions where someone breaks `cli.py` imports — which have happened. Cheap and worth keeping green.

### What to mock vs. exercise

- **Mock:** `subprocess.Popen`, `socket.create_connection`, `signal.signal`. The CLI spawns child processes; a unit test should not actually torchrun.
- **Exercise:** `argparse`, `_validate_launch_args`, manifest writers (`write_run_manifest`, `update_run_manifest`) using `tmp_path`.

### What's untested

There is currently no test for:

- Env-var assembly correctness in `launch_process` (the `env["TRACEML_*"] = ...` block).
- Manifest field round-trips (write → read).
- The `_DISPLAY_DRIVERS` dispatch path.
- The `--disable-traceml` bypass flow.

If your PR touches any of these, add a test for it.

---

## 9. Common pitfalls

Numbered, with symptom and fix.

1. **Symptom:** New flag accepted by argparse, but the runtime/aggregator never sees the value.
   **Cause:** Forgot to add `env["TRACEML_X"] = str(args.x)` to `cli.py::launch_process`.
   **Fix:** Add the env-var assembly. Verify with `print(os.environ.get("TRACEML_X"))` from inside the executor or aggregator.

2. **Symptom:** Existing v0.2.x users on PyPI suddenly see crashes after upgrade.
   **Cause:** You added a flag without a default, or you changed an existing default in a behaviour-flipping way.
   **Fix:** Always provide a default that preserves previous behaviour. For value changes that matter, bump the minor version and document in `CHANGELOG.md`.

3. **Symptom:** `KeyError: 'TRACEML_FOO'` from the executor or aggregator.
   **Cause:** The read site uses `os.environ["TRACEML_FOO"]` instead of `os.environ.get("TRACEML_FOO", default)`. The CLI sets it, but tests that bypass the CLI don't.
   **Fix:** Always use `.get(name, default)`. The default value goes alongside the read.

4. **Symptom:** `int(os.environ.get("TRACEML_FOO"))` raises `TypeError: int() argument must be a string ... not 'NoneType'`.
   **Cause:** Missing default in the `.get(...)` call. When the env var is absent, `.get()` returns `None`, and `int(None)` blows up.
   **Fix:** `int(os.environ.get("TRACEML_FOO", "0"))`. Default must be a string because env-var values are strings.

5. **Symptom:** `--mode somenewmode` is parsed by argparse but the aggregator raises `ValueError: Unknown display mode 'somenewmode'`.
   **Cause:** You added the choice to argparse `choices=` but not to `_DISPLAY_DRIVERS` in `aggregator/trace_aggregator.py`.
   **Fix:** Add the dispatcher entry. Implement the `BaseDisplayDriver` subclass before adding the argparse choice.

6. **Symptom:** Tests pass, but `traceml watch script.py --mode jsonstream` raises `ValueError: Invalid display mode 'jsonstream'. Supported modes: ['cli', 'dashboard', 'summary']` from `cli.py`.
   **Cause:** You forgot the `supported_modes` set in `cli.py::launch_process` (line ~535) — a redundant guard that duplicates argparse `choices=`.
   **Fix:** Add the new mode to the set. Better: refactor to read from `_DISPLAY_DRIVERS.keys()` (see §11 "Gaps").

7. **Symptom:** Validation in `_validate_launch_args` accepts an invalid combo because the new flag's attribute name doesn't match what `getattr` looks for.
   **Cause:** Argparse converts `--max-step-time-ms` to the attribute `args.max_step_time_ms` (dashes become underscores). Spelling typos in `getattr(args, "max-step-time-ms", 0)` silently return the default.
   **Fix:** Always use the underscored form. Add a unit test.

8. **Symptom:** A new flag affects training behaviour but doesn't appear in `manifest.json`. Post-mortem reproduction is impossible.
   **Cause:** You forgot to extend `write_run_manifest` (or pass via `extra=`).
   **Fix:** Add the field. Extend the function signature for first-class data; use `extra=` for ad-hoc.

9. **Symptom:** New profile flag works in the CLI but samplers don't change.
   **Cause:** You added a new subparser but didn't extend `runtime.py::_build_samplers` to gate samplers on the new profile string.
   **Fix:** See [`add_sampler.md` §11](add_sampler.md#11-appendix-adding-a-new-profile).

10. **Symptom:** Boolean flag accepts `--enable-foo true` instead of just `--enable-foo`, and downstream code reads `os.environ["TRACEML_FOO"] == "1"` and never matches.
    **Cause:** You used `type=bool` instead of `action="store_true"`. `type=bool("true")` is `True` (any non-empty string is truthy), so argparse accepts the value but the conversion produces garbage.
    **Fix:** Use `action="store_true"`. Then assign `env["TRACEML_FOO"] = "1" if args.foo else "0"`.

11. **Symptom:** A pure-CLI subcommand like `traceml inspect` triggers aggregator/torchrun spawning.
    **Cause:** You called `launch_process` from the handler. `inspect`, `compare`, and any new offline command should not.
    **Fix:** Don't call `launch_process`. The handler reads, reports, exits.

12. **Symptom:** Three different defaults for the same env var (argparse, settings dataclass, executor read site) drift apart, and a test for one default passes while production uses another.
    **Cause:** Defaults are duplicated. Today this is observable for `--interval` (CLI default `2.0`, settings default `1.0`) and `--num-display-layers` (CLI default `5`, settings default `20`).
    **Fix:** When you touch a default, grep all three sites and update together. Long-term: see §11.

13. **Symptom:** `traceml --help` is slow (>1 s) or fails on a CPU-only box.
    **Cause:** You added a top-level import of `torch`, `pandas`, or `nicegui` to `cli.py`.
    **Fix:** Defer the import. The CLI must be lean. Do imports inside handlers, not at module top.

14. **Symptom:** Manifest field shows up but with the wrong value (e.g. `"foo": null`).
    **Cause:** You added the field to `write_run_manifest`'s signature but the caller in `launch_process` didn't pass the value.
    **Fix:** Update the call site too. Run a smoke test and `cat manifest.json`.

---

## 10. Checklist before opening a PR

1. [ ] Decided which category the change is (subcommand / flag / mode / profile / env var) and confirmed it's the smallest fit.
2. [ ] All five layers updated where applicable: argparse → validation → env-var assembly → consumer (runtime / aggregator) → manifest.
3. [ ] Argparse declaration includes `type=`, `default=`, and a useful `help=` string.
4. [ ] Cross-argument constraints in `_validate_launch_args`.
5. [ ] Env-var name follows `TRACEML_<UPPER_SNAKE>`.
6. [ ] Env-var value is a string (`str(args.x)`).
7. [ ] Env-var read site uses `.get(name, default)`, never `[name]`.
8. [ ] Defaults agree across argparse, env-var read site, and `TraceMLSettings`.
9. [ ] If `--mode` was added: `BaseDisplayDriver` subclass implemented and registered in `_DISPLAY_DRIVERS`. See [`add_renderer.md` §12](add_renderer.md).
10. [ ] If profile was added: see [`add_sampler.md` §11](add_sampler.md#11-appendix-adding-a-new-profile).
11. [ ] If subcommand was added: decided whether it spawns aggregator (calls `launch_process`) or is offline (does not).
12. [ ] Manifest field added if the value affects training data or reproduction.
13. [ ] Env var documented in `traceml/CLAUDE.md` and the env-var table in §6.5 of this guide.
14. [ ] Backward-compat verified: `traceml watch script.py` (no new flag) still works.
15. [ ] CHANGELOG entry under "Unreleased" with the user-visible behaviour.
16. [ ] Tests cover argparse parsing, validation, and backward-compat. End-to-end subprocess smoke test for non-trivial changes.
17. [ ] Local smoke test run: `traceml <subcommand> examples/<small>.py [<new-flag>]` exits cleanly, manifest contains the new field, no stack traces on stderr.
18. [ ] `pre-commit run --all-files` clean.
19. [ ] Commit message short, single line, no `Co-Authored-By` trailers.

---

## 11. Appendix

### 11.1. How env vars cross fork+exec

`subprocess.Popen(..., env=env)` passes the parent's `env` dict as the new process's initial environment. The child reads via `os.environ.get(...)`. There is **no shared memory**; the child's `os.environ` is a copy made at exec time. Subsequent changes in the parent are invisible to the child, and vice versa.

For the OS-level mechanics see [Q5 (OS fundamentals)](../deep_dive/learning-qa.md#q5-os-fundamentals--kernel-process-internals-pipes-sockets) and [Q7 (fork/exec)](../deep_dive/learning-qa.md#q7-spawning--fork-exec-and-multiprocessing-start-methods).

In TraceML, this means: **changes to `TRACEML_*` env vars after `launch_process` has spawned children have no effect**. If you need to change a value mid-run, you need a new IPC channel — and TraceML doesn't have one.

### 11.2. Why env vars instead of CLI args for child-process config?

Two reasons:

1. **Forensic reproducibility.** The aggregator and executor both need the same view of the configuration. If both processes accept config on argv, the CLI has to remember to pass the right subset to each. Env vars let `launch_process` build the configuration dict once and pass it to both children unmodified.

2. **Simplicity.** TraceML has no IPC config channel. Env vars are the only mechanism that survives the fork+exec boundary without serialization. Adding a JSON-config-file path would mean writing, reading, and validating that file — three more failure modes for a marginal ergonomic gain.

The cost: env vars are flat, untyped strings. Type coercion lives at every read site. We accept this cost.

### 11.3. The redundant `supported_modes` guard

`cli.py::launch_process` line ~535 has:

```python
    supported_modes = {"cli", "dashboard", "summary"}
    if args.mode not in supported_modes:
        raise ValueError(...)
```

This is a defense-in-depth check that duplicates the argparse `choices=` on `--mode`. It exists from before the dispatcher was hoisted into `_DISPLAY_DRIVERS`. The cleanest fix is:

```python
    from traceml.aggregator.trace_aggregator import _DISPLAY_DRIVERS
    if args.mode not in _DISPLAY_DRIVERS:
        raise ValueError(
            f"Invalid display mode '{args.mode}'. "
            f"Supported modes: {sorted(_DISPLAY_DRIVERS)}"
        )
```

…but this would import `aggregator/trace_aggregator.py` from `cli.py`, which transitively imports NiceGUI, Plotly, and SQLite writers. That's a heavy dependency for `traceml --help` to incur. A better fix is to lift `_DISPLAY_DRIVERS.keys()` into a small registry module that can be imported by both `cli.py` and `trace_aggregator.py` without dragging the driver implementations along. Out of scope for a single CLI change PR; flag in code review.

### 11.4. Future direction: a config file

There is currently no `traceml.yaml` / `traceml.toml` config file. All configuration is via CLI flags and env vars. A config file would help:

- CI workflows that want to pin profile + mode + thresholds without long invocations.
- Multi-user environments where defaults vary per project.

Adding one means: a new flag (`--config <path>`), a new loader (`tomllib.load`), and a precedence rule (CLI flag overrides config file overrides env var overrides default — or whatever order makes sense). This is at least a +200-line PR with cross-cutting validation and backward compat to think through. Not yet planned.

### 11.5. A new subcommand that calls `launch_process` (advanced)

If you genuinely need a new subcommand that launches training — say, `traceml replay <session-dir>` that re-runs a script with the same flags as a previous session — the shape is:

1. Subparser with `_add_launch_args(replay_parser)` plus the `session_dir` positional.
2. Handler `run_replay(args)` that:
   - Reads the previous session's manifest.
   - Pre-populates `args` with the manifest's values.
   - Calls `launch_process(script_path, args)` exactly as `run_with_tracing` does.

The non-trivial bit is reconciling the user's CLI overrides with the manifest's recorded values. Define an explicit precedence (CLI > manifest > default), document it, and test it.

---

## Gaps and ambiguities encountered while writing this guide

These are places where the current source does not fully pin down the contract. Flag them in code review if your PR lands near them.

- **Default drift across layers.** `--interval` defaults to `2.0` in argparse (`cli.py:725`) but `1.0` in `TraceMLSettings.sampler_interval_sec` and `executor.py::DEFAULT_INTERVAL_SEC`. Similarly, `--num-display-layers` defaults to `5` in argparse and `20` in `TraceMLSettings.num_display_layers`. The CLI default wins because it always sets the env var, but the dataclass default is misleading. Either consolidate (single source of truth) or document each divergence with a comment. I did not fix this in writing this guide.

- **`supported_modes` duplication.** `cli.py::launch_process` has a hard-coded `{"cli", "dashboard", "summary"}` set that disagrees with argparse `choices=` if you forget to update it. See §11.3. A future refactor should make `_DISPLAY_DRIVERS.keys()` the single registry, imported by a lightweight registry module rather than directly from `aggregator/trace_aggregator.py`.

- **`TRACEML_PROFILE` default mismatch.** `cli.py` always sets it (so the default never fires) but `aggregator_main.py:100` defaults to `"run"` while `TraceMLSettings.profile` defaults to `"run"` and `executor.py::DEFAULT_PROFILE` is `"run"`. The CLI doesn't actually have a default — it's set per-subcommand (`watch` → `"watch"`, `run` → `"run"`, `deep` → `"deep"`). Documenting this in the env-var table required hand-waving.

- **CLI test coverage.** `tests/test_compare_missing.py` is the only CLI test, and it tests `compare`'s payload shaping rather than the CLI surface itself. There are zero tests for argparse parsing of any launch flag, zero tests for `_validate_launch_args`, zero tests for env-var assembly in `launch_process`. New CLI changes should add the missing layer; this guide's §8 template is a starting point.

- **No CLI-level integration test.** There's no end-to-end test that runs `traceml watch examples/<small>.py` from a subprocess and asserts on the manifest, the SQLite history, and the exit code. Smoke testing is currently a human responsibility (see checklist item 17). The hardest part of adding such a test is keeping it fast (<10 s) and hermetic.

- **`extra=` vs. signature extension in `write_run_manifest`.** Both patterns coexist. The `extra={"artifacts": {...}}` parameter is used for the code manifest path; the explicit signature parameters are used for everything else. There is no clear rule for when to promote a field from `extra=` to a first-class parameter. Promoting feels right when the field is touched by every launch; leaving in `extra=` feels right when only some subcommands set it. Document the rule next time someone adds a manifest field.

- **`launch_context.py` env-var serialization.** `LaunchContext.to_env()` serializes a single key (`TRACEML_LAUNCH_CWD`). The plumbing works but doesn't generalize: there's no clean pattern for adding a second piece of structured launch context. If you need to record more (e.g. the original `argv`, the user's `sys.executable`, etc.), decide whether to extend `LaunchContext` or introduce a separate capture object. The current implementation predates the question.
