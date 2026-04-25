# Cross-cutting principles

> Internal contributor reference. Audience: anyone touching TraceML code.
> Every per-feature guide in this folder links here instead of restating these
> rules. Read once; refer back from the per-feature guide you're using.

---
Document type: cross-cutting reference
Applies to: every contributor guide and every PR
Last verified: 2026-04-25
---

## 1. The four load-bearing principles

These are the design principles in [traceml/docs/developer_guide/architecture.md](../developer_guide/architecture.md). Every architectural decision in TraceML traces back to one of them. When something looks over-engineered, ask which principle forced the shape.

**Fail-open** — training must never crash because telemetry broke. Sampler exception, transport failure, hook attachment error, manifest write failure — all logged, all swallowed, training continues. The aggregator can disappear mid-run and training keeps going. The contract is one-way: TraceML degrades; the user's job does not.

**Bounded overhead** — every new sampler, patch, or hook justifies its overhead. The default budget is sub-1% of step time at the high end, sub-microsecond per-row build cost. Bounded deque tables (`deque(maxlen=N)`) evict the oldest record on overflow. Queues are bounded at the producer (`queue.Queue(maxsize=4096)` with drop-on-full). The v0.2.9 benchmark workflow is the artifact that proves overhead claims; reference it in PR descriptions when you've measured.

**Process isolation** — no shared memory between training ranks and the aggregator. TCP + environment variables only. This is what makes "aggregator crashes don't crash training" actually true, and what makes config flow forensically traceable (every env var is in the run manifest).

**Out-of-process UI** — the aggregator is a separate process, spawned by the CLI before training starts. Its crash kills the dashboard, not the model. Training continues in degraded-telemetry mode; the run manifest records the degradation.

---

## 2. The pipeline (short version)

```
[ Training rank ]
  patch / hook / sampler  →  Database (deque-per-table)  →  DBIncrementalSender (only-new-rows)
                                                                       │
                                                                  TCP, msgpack
                                                                       ▼
[ Aggregator ]
  TCPServer  →  RemoteDBStore (rank-aware)  →  SQLite projection writer  →  renderer  →  display driver
```

For the full station-by-station walkthrough — what each box does, what the failure modes are, which file owns which transition — see [pipeline_walkthrough.md](pipeline_walkthrough.md). For depth, see [W6](../deep_dive/code-walkthroughs.md#w6-samplers--schemas--turning-hook-events-into-structured-rows) through [W10](../deep_dive/code-walkthroughs.md#w10-display-drivers--renderers--terminal-and-web-ui-from-sql).

---

## 3. Wire compatibility rules

Users on `traceml-ai` v0.2.x are already in production. The aggregator must continue to decode their wire payloads. The CLI must continue to accept their invocations.

The rules:

- **Never remove or rename an existing key on the wire.** Add new keys instead.
- **Never change the type of an existing key.** A `float` stays a `float` forever.
- **New keys are optional on the consumer side.** Renderers and projection writers use `payload.get("new_field", default)`, not `payload["new_field"]`.
- **Never remove or rename an existing CLI flag or env var.** Add a new one and keep reading the old one for one full release cycle before deprecating with warning.
- **Never change SQLite column types.** `ALTER TABLE ... ADD COLUMN ... DEFAULT NULL` is the only schema change permitted in a minor version.
- **Breaking changes need a major version bump and a migration path.** A new table name (deprecating the old one over a release cycle) is the standard escape hatch.

These rules also keep `traceml compare` working across releases — without them, comparing a v0.2.3 run to a v0.3.0 run silently misaligns metrics.

---

## 4. Logging convention

- All TraceML log lines carry the `[TraceML]` prefix at the start of the message.
- Sampler / patch / writer errors go to the per-component logger (`self.logger.error(...)`), which writes to a rotating file under the session directory and (for critical errors) to stderr.
- Never `print()` in instrumentation code. The CLI driver redirects stdout into the log panel; your `print` lands inside `stdout_stderr` telemetry, not where you expect.
- Use `get_error_logger("ComponentName")` from `loggers/error_log.py` to get a child logger for a component without a `BaseSampler` parent.

---

## 5. Overhead budget

| Layer | Target | Notes |
|---|---|---|
| Periodic sampler `sample()` | < 1 ms / tick | Driven by `TRACEML_INTERVAL` (default 1.0 s); 1 ms is a generous 0.1% budget. |
| Event-driven sampler queue drain | < 1 ms per 1k events | O(queue depth); cheap per-event work. |
| Per-row build (`dataclass.to_wire()`) | sub-microsecond | `deque.append()` is O(1). |
| Patch fast-path (instrumentation off) | sub-microsecond | One TLS check, one branch to original function. No allocation. |
| Patch hot-path (instrumentation on) | < 5 µs | CUDA event allocation from pool, two `event.record()` calls, `TimeEvent` construct, `deque.append`. |
| Renderer compute per tick | < 100 ms total | Display tick is rate-limited to `render_interval_sec` (default 2.0 s). |
| SQLite projection writer per payload | sub-millisecond | Prepared statements, no `SELECT` inside the writer. |

When in doubt: drop a `timeit` block, run on the dev box, confirm. The benchmark workflow at v0.2.9 is the formal harness. None of these targets are test gates today; they're folklore. Hold the line in code review.

---

## 6. Smoke-test discipline

Every PR adds something to the smoke-test list. The author runs it; the reviewer reproduces it. The pattern:

```bash
pip install -e ".[dev,torch]"
traceml watch examples/<small example>.py --mode cli
traceml run   examples/<small example>.py --mode dashboard   # http://localhost:8765
```

Expected: training completes, no stack traces on stderr, the new feature visibly does what it should. For multi-GPU, also `--nproc-per-node 2` and verify per-rank vs. host-level rows behave correctly.

If your feature can't be smoke-tested locally without GPU, say so explicitly in the PR description. Don't fake it with a unit test that mocks CUDA.

---

## 7. Versioning and CHANGELOG

| Change type | Version bump | CHANGELOG |
|---|---|---|
| New sampler / patch / renderer / diagnostic — additive only | Patch | One line: `Added: <feature>` |
| New CLI flag / env var with safe default | Patch | One line |
| Wire schema additions (new optional key) | Minor | Note the new key |
| New `--mode` choice or new profile | Minor | Note the new option |
| Wire schema breaking change (renamed key, type change) | Major | Migration note required |
| CLI deprecation (old flag still works with warning) | Minor | Deprecation notice |
| CLI removal (old flag no longer accepted) | Major | Migration note required |

Every PR adds a CHANGELOG line. None of the existing guides currently call this out — that's a recent rule. The discipline starts now.

---

## 8. Naming conventions

| Concept | Convention | Examples |
|---|---|---|
| Module / file | `snake_case` | `step_time_sampler.py`, `dataloader_patch.py` |
| Class | `PascalCase` with role suffix | `StepTimeSampler`, `GpuUtilizationRenderer` |
| Sampler suffix | `*Sampler` | always |
| Renderer suffix | `*Renderer` | always |
| Patch file suffix | `*_patch.py` (`_auto_timer_patch` for hot-path patches) | `forward_auto_timer_patch.py` |
| Schema dataclass | role-named, frozen | `ProcessSample`, `StepCombinedTimeResult` |
| Wire-format key | short snake_case | `seq`, `ts`, `cpu`, `gpu_mem_used` |
| Python attribute | descriptive snake_case | `cpu_logical_core_count`, `ram_total_bytes` |
| Constant | `SCREAMING_SNAKE_CASE` | `DEFAULT_MAX_ROWS`, `MODEL_COMBINED_LAYOUT` |
| Env var | `TRACEML_<UPPER_SNAKE>` | `TRACEML_DISABLED`, `TRACEML_PROFILE` |
| TimeEvent name (wire identity) | `_traceml_internal:<event>` | `_traceml_internal:h2d_time` |
| SQLite table | `<sampler>_samples` plus child tables | `system_samples`, `system_gpu_samples` |
| Layout section constant | `<NAME>_LAYOUT` | `STEP_TIME_LAYOUT`, `MODEL_COMBINED_LAYOUT` |

Domain-prefix when ambiguous: `gpu_mem_used`, not `mem_used`, when the same row contains CPU and GPU memory. Units in the docstring, not the name.

---

## 9. PyTorch coupling discipline

TraceML reaches into PyTorch internals — `nn.Module._call_impl`, autograd hooks, the DataLoader iterator, CUDA events. This is the maintenance treadmill the project has explicitly accepted in exchange for zero-code instrumentation. The discipline:

- **Patch the smallest stable surface** that gives you what you need. `_call_impl` is more stable than monkey-patching `nn.Module.__init__`.
- **Document the version dependency.** Every patch file should note which PyTorch versions it has been tested against. See [P51](../deep_dive/pytorch-qa.md#p51-which-torchcuda-apis-does-traceml-rely-on-and-how-stable-are-they-across-pytorch-versions-relevant-to-the-pytorch-coupling-constraint-in-claudemd) for the contract.
- **`torch.compile` is the open question.** Compiled regions bypass Python frame execution; Python-level hooks become invisible. The current architecture handles eager-mode training fine; compiled-graph support is a roadmap item, not a present capability.
- **Never call `torch.cuda.synchronize()` from instrumentation code.** Synchronization serializes the GPU and destroys the overhead budget. Use the async `event.query()` resolution path via the CUDA event pool.
- **Reuse CUDA events.** `get_cuda_event()` / `return_cuda_event()` from `utils/cuda_event_pool.py`. Never `torch.cuda.Event(enable_timing=True)` in a hot loop — the allocation is expensive and the events leak if not pooled.

---

## 10. House idioms

| Term | Meaning |
|---|---|
| `rank` / `local_rank` / `world_size` | Standard PyTorch DDP semantics. `RANK` is global; `LOCAL_RANK` is on-host; `WORLD_SIZE` is total. Host metrics filter on `local_rank == 0`, never `rank == 0`. |
| `step` | One optimizer step in the user's training loop. The atomic unit of step-scoped instrumentation. Bounded by `trace_step()`. |
| `tick` | One iteration of `TraceMLRuntime._tick()`. Default 1.0 s. Periodic samplers fire every tick; event-driven samplers drain their queues every tick. Steps and ticks are not the same thing. |
| `profile` | One of `watch` / `run` / `deep`. Gates which samplers get instantiated in `_build_samplers()`. Do not confuse with `--mode`. |
| `mode` | One of `cli` / `dashboard` / `summary`. Selects the display driver. Do not confuse with `profile`. |
| `sampler` | An object that observes one data source and writes rows into its private `Database`. |
| `patch` | A monkey-patch on a PyTorch internal that wraps it with `timed_region(...)`. |
| `hook` | A registered callback on an `nn.Module` (forward / backward) — distinct from a patch. |
| `renderer` | A read-only consumer of telemetry data that produces formatted output for a display medium. |
| `diagnostic` (or `verdict`) | An opinionated class of pathology derived from sampler data — INPUT-BOUND, MEMORY-CREEP, IDLE-GPU, etc. |
| `projection writer` | The aggregator-side bridge between sampler payload and SQLite table. Lives in `aggregator/sqlite_writers/`. |
| `wire format` | The msgpack-encoded payload over TCP. The `name` field on a `TimeEvent` is wire identity. |
| `fail-open` | The discipline of swallowing instrumentation errors so user code never crashes. |
| `fast path` | The "instrumentation off" branch in a patch — sub-microsecond, no allocation. |
| `TLS gate` | Thread-local flag that enables/disables instrumentation. Only `True` inside `trace_step()`. |

Use these terms consistently in code, comments, commit messages, and PR descriptions.

---

## 11. Cross-references

- **Project rules:** [../../../CLAUDE.md](../../../CLAUDE.md) (repo root) and [../../CLAUDE.md](../../CLAUDE.md) (traceml package — line length 79, no `Co-Authored-By` trailers, etc.).
- **Pipeline:** [pipeline_walkthrough.md](pipeline_walkthrough.md) (condensed) and [W6](../deep_dive/code-walkthroughs.md#w6-samplers--schemas--turning-hook-events-into-structured-rows) → [W10](../deep_dive/code-walkthroughs.md#w10-display-drivers--renderers--terminal-and-web-ui-from-sql) (deep).
- **Architecture:** [traceml/docs/developer_guide/architecture.md](../developer_guide/architecture.md).
- **Why these constraints exist:** [traceml_why.md](../deep_dive/why.md) §6 (the four claims) and §7 (limitations).

---

## 12. Gaps and ambiguities

Things this principles document does not yet pin down:

- **Overhead budgets are folklore.** No formal benchmark harness gates them; the v0.2.9 workflow is in design (Item 2 in Abhinav's brief). Numbers in §5 are targets, not test thresholds.
- **CHANGELOG discipline is new.** Existing PRs don't all have entries; backfilling is out of scope. Going forward is the policy.
- **Schema versioning has no first-class table.** The wire-compat rules in §3 are convention, not enforcement. A future `schema_version` table would make breaking changes detectable rather than silent.
- **`torch.compile` strategy** is open-ended. When >50% of training jobs are compiled, the patch architecture needs an answer. Not a 2026 problem.
- **Smoke-test discipline** depends on `examples/` being maintained. As examples drift, the smoke-test path can rot — the smoke target should be re-verified each release.
