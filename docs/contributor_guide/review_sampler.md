# How to review a sampler PR

This guide teaches you how to review a PR that adds or modifies a sampler in `src/traceml/samplers/`. It assumes you have already read `add_sampler.md` (the author's guide), `principles.md` (cross-cutting rules), and have a working mental model of [W6][W6] (samplers + schemas) and [W7][W7] (Database + sender). The seven-step workflow in §1 is the same shape that `review_patch.md` uses; only §3 onward is sampler-specific.

---
Feature type: sampler
Risk level: medium (samplers run on every tick in every rank; a fail-open violation can mask metrics, an unbounded queue can OOM a job, a host-metric duplication can N-multiply telemetry)
Cross-cutting impact: training process (per-rank runtime) plus aggregator-side projection writers and renderers if a new table is introduced
PyTorch coupling: shallow (most samplers are NVML / psutil; CUDA-touching samplers must use `_cuda_safe_to_touch`)
Reference reviews: none yet — this guide is the first reviewer-side rulebook for the family
Companion author guide: `add_sampler.md`
---

## 1. The meta-review-workflow (applies to every TraceML PR)

Every sampler review walks the same seven steps in order. Skipping any of them is how a flawed PR ships:

1. **Anchor** the PR diff to the relevant W-walkthroughs and Q/P entries. Read the PR through your existing mental models, not line-by-line.
2. **Run the sampler-family consistency check.** Build the table from §3 of this guide and grade the new sampler against `SystemSampler`, `ProcessSampler`, `StepTimeSampler`, `LayerForwardTimeSampler` on each axis. Discrepancies are either justified deviations (document them) or bugs.
3. **Apply the sampler-class failure-mode catalogue** from §4. Each category maps to a known bug shape. Walk the diff with each shape in mind.
4. **Apply the four meta-questions** from §5: new axis of variation, shared infrastructure interaction, wire-name as contract, invariant preservation.
5. **Write a verification gate** for each concern: a 3–10 line reproduction recipe with a clear pass/fail criterion. "I think this is buggy" becomes "here's the script that proves it." See §6.
6. **Draft comments at the right granularity** — line comment for specific code suggestions, PR-level comment for behavioural / architectural concerns. Hold parking-lot items back. See §7.
7. **Land the verdict** — approve / approve-with-changes / block. Criteria in §8.

The reviewer's job ends with a 2–3 sentence executive summary the maintainer can read without opening the diff. That goes in the verdict (§8).

This same seven-step shape applies to patch PRs, renderer PRs, transport PRs — only the consistency table and the failure-mode catalogue change.

---

## 2. Step 1 — Anchor the PR to the walkthroughs

The first thing you do with a sampler PR is **not** open the diff. Open [`traceml_learning_code_walkthroughs.md`][W6] and re-read W6 §"sampler tick lifecycle" and W7 §"DBIncrementalSender — only-new-rows shipping." Two reasons:

- The sampler family has documented invariants (fail-open `sample()`, no CUDA sync, bounded producer queue, primitive-only wire payload, `local_rank == 0` for host metrics). You'll be checking the diff against those invariants, so they need to be in cache.
- A sampler PR will touch four to seven files in stereotyped ways. If you read the PR file-by-file without that map, you'll drown in the diff. Mapping each file to a W-section collapses the diff into mechanical changes plus one substantive change.

### How to anchor

For each file in the diff, write down (in your review notes, not the PR yet):

| File pattern | W-section | What kind of change should this be? |
|---|---|---|
| `src/traceml/samplers/<name>_sampler.py` (NEW) | [W6 §"sampler tick lifecycle"][W6] | The substantive change. The body of the review focuses here. |
| `src/traceml/samplers/schema/<name>.py` (NEW) | [W6 §"schema dataclasses"][W6] | Frozen dataclass + `to_wire()` / `from_wire()`. Mechanical against the schema rules. |
| `src/traceml/runtime/runtime.py` (`_build_samplers`) | [W2 §"runtime profile gating"][W2] | One- to three-line registration in the correct profile branch. Mechanical. |
| `src/traceml/aggregator/sqlite_writers/<name>.py` (NEW, if introducing a new table) | [W9 §"projection writers"][W9] | `init_schema()` + write method consuming the sampler payload. |
| `src/traceml/aggregator/aggregator_main.py` (writer registration) | [W9 §"sampler → writer dispatch"][W9] | Wire the new writer into the dispatch map. |
| `src/traceml/renderers/<name>_renderer.py` (NEW, if surfacing in the UI) | [W10][W10] | Read-only consumer of the new SQLite table. |
| `tests/test_<name>_sampler.py` (NEW) | none directly | Construction + happy path + source-failure safety. |

If a file in the diff doesn't fit the table, that's a flag — the PR is doing something architecturally novel, and you should ask why before proceeding. A sampler PR that touches `transport/`, `cli.py`, or `decorators.py` is suspicious by default; samplers don't normally cross those boundaries.

The point: **after anchoring, you should have one substantive file to read deeply (the new sampler) and 3–6 mechanical files to skim.**

[W2]: ../deep_dive/code-walkthroughs.md#w2-per-rank-runtime-executor-runtime-loop-launch-context-session
[W6]: ../deep_dive/code-walkthroughs.md#w6-samplers-schemas-turning-hook-events-into-structured-rows
[W7]: ../deep_dive/code-walkthroughs.md#w7-database-sender-bounded-in-memory-store-and-incremental-tcp-shipping
[W9]: ../deep_dive/code-walkthroughs.md#w9-aggregator-core-tcp-receive-frame-dispatch-sqlite-writes
[W10]: ../deep_dive/code-walkthroughs.md#w10-display-drivers-renderers-terminal-and-web-ui-from-sql

---

## 3. Step 2 — The sampler-family consistency table

Every sampler slots into a small set of axes. The reviewer's job is to fill in the row for the new sampler and grade each cell against the existing family.

### 3.1 The columns (these don't change PR-to-PR)

| Axis | What it asks | Where to verify |
|---|---|---|
| **Driver style** | Periodic (poll every tick) or event-driven (drain a queue)? | Shape of the `sample()` body. Periodic = "build payload directly"; event-driven = "drain queue → resolve → emit". |
| **Rank scope** | All ranks, rank-0 only (gate at `_build_samplers`), or per-rank with sender detached on non-rank-0? | `runtime.py::_build_samplers` branch + sampler `__init__` (look for `self.sender = None`). |
| **Profile gating** | `watch` / `run` / `deep` — which profile(s) include this sampler? | `runtime.py::_build_samplers` `if self.profile == ...` branch. |
| **Sender knobs** | `max_rows_per_flush=` value? `-1` (default) / `1` (latest only) / `5` (rate-cap) / other? | `super().__init__(..., max_rows_per_flush=...)`. |
| **Schema dataclass** | Frozen dataclass in `samplers/schema/`? `to_wire()` and `from_wire()` defined? | `samplers/schema/<name>.py`. |
| **Source initialization** | Defensive try/except in `_init_*` helpers? Sentinel values on failure? | Each `_init_*` method must `try/except Exception` and set a sentinel attribute. |
| **`sample()` shape** | `try/except Exception` wrapping the whole body? `self.logger.error(...)` on failure? `self.sample_idx += 1` somewhere visible? | The `sample()` method body. |
| **CUDA handling (if any)** | Uses `_cuda_safe_to_touch`? Reads `dist.is_initialized()` defensively? Never calls `torch.cuda.synchronize()`? | Search the sampler for `torch.cuda.*`; cross-check against `ProcessSampler._cuda_safe_to_touch`. |
| **Bounded queue (event-driven only)** | Producer queue has `maxsize=`? `queue.Full` handled by drop? | Look at the *producer* (hook / patch), not the sampler. The sampler reads from `get_<name>_queue()`. |
| **Manifest writes (if any)** | Idempotent? Uses `write_json_atomic`? Called once from `__init__`, not from `sample()`? | `system_manifest.write_system_manifest_if_missing` is the precedent. |

### 3.2 The current state (April 2026)

| Axis | `SystemSampler` | `ProcessSampler` | `StepTimeSampler` | `LayerForwardTimeSampler` |
|---|---|---|---|---|
| Driver style | Periodic | Periodic | Event-driven | Event-driven |
| Rank scope | Rank-0 only (gated at `_build_samplers`) | Per-rank (every rank emits) | Per-rank (step-scoped) | Per-rank (step-scoped) |
| Profile gating | `watch`, `run`, `deep` (always on) | `watch`, `run`, `deep` (always on) | `run`, `deep` | `deep` only |
| `max_rows_per_flush` | `1` (latest only) | `-1` (send all) | `-1` (send all) | `5` (rate-cap) |
| Schema dataclass | `SystemSample` (`schema/system.py`) | `ProcessSample` + `ProcessGPUMetrics` (`schema/process.py`) | `StepTimeEventSample` (`schema/step_time_schema.py`) | `LayerForwardBackwardTimeSample` (`schema/layer_forward_backward_time.py`) |
| Source init | `_init_cpu` + `_init_ram` + `_init_gpu`, each with try/except | `_init_process` + `_init_ram` + `_warmup_cpu` + `_init_ddp_context` + `_init_gpu_state`, each with try/except | None (no external source — reads queue) | None (no external source — reads queue) |
| `sample()` outer try/except | YES | YES | YES | YES |
| `sample_idx` increment | YES (top of `sample()`) | YES (top of `sample()`) | YES (top of `sample()`) | YES (only on resolved step — semantics differ; see §4.7) |
| CUDA handling | NVML only — does not touch `torch.cuda` | Uses `_cuda_safe_to_touch` + `_ensure_cuda_device` | Resolves CUDA events via `evt.try_resolve()` (no sync) | Resolves CUDA events via `all_layer_events_resolved` (no sync) |
| Bounded queue | N/A | N/A | `get_step_time_queue()` is bounded at producer | `get_layer_forward_time_queue()` is bounded at producer |
| Manifest write | YES — `write_system_manifest_if_missing` from `__init__` | NO | NO | NO |

When reviewing, **add a column** for the new sampler and walk every row. Three outcomes per cell:

- ✅ Matches a precedent in the family — note which precedent and move on.
- ❌ Differs from every precedent — demand a justification in the PR description or a comment in the sampler file.
- ⚠ Cell is empty / undecidable from the diff — ask the author.

### 3.3 The table is the most reusable artifact in this guide

Every future sampler review should rebuild this table. Two reasons:
- The act of filling it forces you to read the sampler with the family in mind, which catches "this differs in ways the author didn't notice."
- The completed table goes in your review notes, and over time becomes the reviewer's contract test for the sampler family. (See "Gaps" at the end — formalising this is on the wishlist.)

---

## 4. Step 3 — Sampler-class failure modes

Each category below describes a known bug shape that has, will, or could ship in a sampler PR. Walk the diff with each one in mind.

### 4.1 Fail-open violations

Applies to: every sampler.

The bug shape: `sample()` raises into the runtime; or a `_init_*` helper raises from `__init__`; or a sub-source method re-raises after logging. `add_sampler.md` §3.5 spells out the contract: outer `try/except Exception` wraps the entire `sample()` body, sub-sources have their own try/except, init helpers set sentinels on failure rather than raising.

**What to check:**

- Every `_init_*` method wraps its source-attach logic in `try/except Exception` and sets a sentinel attribute (`None`, `0.0`, `-1`, empty list) on failure. No `raise` from `__init__`.
- The body of `sample()` is wrapped in `try/except Exception`. The except branch calls `self.logger.error(f"[TraceML] {sampler_name} failed: {e}")` and returns. No re-raise.
- Sub-sources called from `sample()` (e.g. `self._sample_gpu()`) each have their own try/except so a single sub-source failure does not lose the whole row.
- The error log message carries the `[TraceML]` prefix and names the sampler.

**Common symptom:** training crashes with a stack trace originating in a sampler thread. The runtime's `_safe()` wrapper is a second belt around the first pair of suspenders — don't trust it alone, because exception propagation through Python threading is fragile.

**Verification-gate shape:**

```
Setup: monkeypatch the source (psutil / pynvml / queue.get) to raise.
Command: instantiate the sampler, call sample(), assert no exception escapes.
Pass: sample() returns None; rows table is empty or zero-filled; logger captured an [TraceML] line.
Fail: any uncaught exception out of sample().
```

### 4.2 Rank-awareness bugs

Applies to: any sampler that emits **host-level** metrics (CPU, host RAM, NVML) or that touches CUDA per rank.

Three failure shapes:

1. **`rank == 0` used for host metrics.** `RANK` is the global rank; `LOCAL_RANK` is the on-host rank. On a multi-node run, `local_rank == 0` exists on every host while `rank == 0` exists only on the leader host. Filtering on `rank == 0` produces zero host metrics from non-leader hosts.
2. **Host metrics emitted from every rank without sender detach.** `SystemSampler` is gated at `_build_samplers` so it only constructs on `local_rank == 0`. A new host-level sampler that constructs unconditionally and forgets `self.sender = None` on non-rank-0 produces N-way duplicates on the wire.
3. **Per-rank metrics that swallow ranks.** `ProcessSampler` is per-rank — every rank emits, and the wire envelope carries `rank` so the aggregator's `RemoteDBStore` shards correctly. A sampler that adds an explicit `rank` column to its row payload and then mismatches it against the envelope rank produces ambiguous attribution.

**What to check:**

- For a host-level sampler, the registration in `_build_samplers` is gated `if not (self.is_ddp and self.local_rank != 0): samplers.append(...)`, **OR** the sampler's `__init__` sets `self.sender = None` when `ctx.is_ddp_intended and ctx.local_rank != 0`. Pick one — never both, never neither.
- Any `local_rank == 0` filter uses `local_rank`, not `rank`. Grep the diff for `== 0` and check.
- A per-rank sampler does not stamp `rank` into the row payload. The envelope handles it.

**Common symptom:** aggregator UI shows N copies of CPU% on a multi-rank run, or shows zero CPU% on every rank but rank 0 of the leader host.

**Verification-gate shape:**

```
Setup: TRACEML_SESSION_ID + RANK=1 + LOCAL_RANK=1 + WORLD_SIZE=4 in env.
Command: instantiate the sampler, call sample(), inspect self.sender.
Pass (host-level): self.sender is None OR sampler is not constructed at all.
Pass (per-rank): row payload has no "rank" key, envelope carries rank=1.
```

### 4.3 Wire-format / schema violations

Applies to: every sampler that emits rows.

Three failure shapes:

1. **Non-primitive types in the payload.** `numpy.float64`, `torch.Tensor`, dataclass instances, sets, enums, deeply nested dicts. `msgspec.msgpack` may encode some of these silently and expensively; others crash on encode.
2. **Breaking-rename of an existing key.** A column rename in `SystemSample.to_wire()` breaks every renderer and every SQLite projection writer that reads the old name. v0.2.x users have dashboards on these names.
3. **`NOT NULL`-on-the-wire that shouldn't be.** A new field added without a default; renderers that did `payload["new_field"]` instead of `payload.get("new_field", default)` crash on old payloads.

**What to check:**

- The schema dataclass is `@dataclass(frozen=True)` and exposes `to_wire() -> dict[str, Any]` and `from_wire(data) -> Self`.
- Every value in `to_wire()` is a primitive (`int`, `float`, `str`, `bool`, `None`) or a list/dict of primitives. Tensors call `.item()`; numpy scalars cast via `float(...)` / `int(...)`; dataclass children call `.to_wire()`.
- No existing key on the wire has been renamed or had its type changed. New keys are additive only. (See `principles.md` §3 for the wire-compat contract — link, don't restate.)
- Renderer-side reads use `payload.get("new_field", default)`, not `payload["new_field"]`.
- Timestamp convention: prefer `"ts"` (unix epoch float) for new code; `"timestamp"` is acceptable but legacy. ISO-8601 strings belong in CLI manifests, never in row payloads.

**Common symptom:** `msgspec.EncodeError` on the wire, or aggregator silently shows old metric values because a rename collided with a stale renderer constant.

**Verification-gate shape:**

```
Setup: instantiate sampler + call sample() once.
Command: import msgspec; msgspec.msgpack.encode(rows[0]) — must not raise.
Pass: encode succeeds; decoded payload's keys are a superset of the v0.2.3 keys for this table.
Fail: EncodeError, or any v0.2.3 key is missing or has a different type.
```

### 4.4 Backpressure / unbounded queues

Applies to: every event-driven sampler (Axis A row 2 in `add_sampler.md` §1).

Three failure shapes:

1. **Producer queue created with `Queue()` (no `maxsize`).** Under load, the producer outpaces `sample()`'s drain rate, the queue grows without bound, and the training process OOMs. The `Database` is bounded via `deque(maxlen=N)`; the *upstream* queue is what can explode.
2. **Producer doesn't handle `queue.Full`.** Even with `maxsize=`, if the producer blocks on `queue.put()` instead of using `put_nowait()` + `except queue.Full: drop`, the producer's thread blocks — and the producer is usually the user's training thread.
3. **Misconfigured `max_rows_per_flush`.** A hot event-driven sampler with `max_rows_per_flush=-1` plus a slow aggregator → unbounded payload growth → aggregator log flood. The fix is `max_rows_per_flush=5` (or similar); the sender's cursor advances correctly across deque evictions, so "latest N" semantics are exactly right.

**What to check:**

- The producer hook / patch creates its queue with `queue.Queue(maxsize=4096)` (or similar bound). See `src/traceml/utils/hooks/layer_forward_time_hooks.py` for the precedent.
- The producer uses `put_nowait()` and handles `queue.Full` by dropping (and ideally counting drops in a debug log).
- The sampler uses `drain_queue_nowait(queue)` or `append_queue_nowait_to_deque(queue, deque)` — never a hand-rolled `while not q.empty(): q.get()` loop (it's racy with the producer).
- `max_rows_per_flush` is set explicitly when the sampler can emit more than ~1 row per tick under realistic load.

**Common symptom:** OOM on long runs, or aggregator UI freezes / TCP disconnect under high event rate. Both have shipped on TraceML before.

**Verification-gate shape:**

```
Setup: synthetic producer that calls put_nowait() in a tight loop for 10 seconds.
Command: assert process RSS does not grow unboundedly; queue.qsize() <= maxsize.
Pass: RSS stable; queue capped.
Fail: RSS grows linearly, or producer blocks (visible as a stuck thread).
```

### 4.5 CUDA hazard

Applies to: every sampler that imports `torch.cuda` or `torch.distributed`.

Four failure shapes:

1. **`torch.cuda.synchronize()` called from `sample()`.** Synchronization serializes the GPU and can introduce 10+ ms stalls per tick. Banned by `principles.md` §9. If the sampler needs CUDA event timings, it must use the async resolution path (`event.query()` via the CUDA event pool); see `StepTimeSampler` for the pattern.
2. **CUDA touched before `dist.init_process_group()`.** On a DDP run, distributed init may not have completed when samplers come online. Touching `torch.cuda.set_device(...)` or `torch.cuda.is_available()` before init can hang or raise "CUDA not initialized." Fix: copy `ProcessSampler._cuda_safe_to_touch` exactly.
3. **CUDA event leaked.** Acquired via `get_cuda_event()` but not returned via `return_cuda_event(evt)` after `try_resolve()`. The pool is a `deque(maxlen=2000)`; under leak conditions the pool empties and the sampler falls back to per-call `torch.cuda.Event(enable_timing=True)`, which is expensive.
4. **NVML init not guarded.** `nvmlInit()` raises `NVMLError` on hosts without NVIDIA drivers; if not caught, the sampler `__init__` raises and the runtime fails to construct.

**What to check:**

- Grep the sampler file for `torch.cuda.synchronize` — must not appear. If it does, block the PR.
- Any `torch.cuda.*` access goes through `_cuda_safe_to_touch()` (copy from `ProcessSampler`).
- `nvmlInit()` is wrapped in `try/except NVMLError` (and `except Exception` for the unexpected case); failure sets `self.gpu_available = False` and `self.gpu_count = 0`. See `SystemSampler._init_gpu` for the precedent.
- CUDA events come from `get_cuda_event()` and return to `return_cuda_event()`. No `torch.cuda.Event(enable_timing=True)` constructed in a hot loop.
- `pynvml` is imported inside the sampler module — never at the top of `runtime.py` or `executor.py`. The sampler module is only imported when `_build_samplers` instantiates the sampler.

**Common symptom:** CUDA hang on DDP startup; sampler tick latency spikes from <1 ms to 10+ ms; `pip install traceml-ai` fails on a CPU-only host because pynvml's wrapper misbehaves without drivers.

**Verification-gate shape:**

```
Setup: WORLD_SIZE=2; do NOT call dist.init_process_group.
Command: instantiate sampler; call sample() 5 times.
Pass: no hang, no CUDA error; gpu_available is False or sample emits a degraded row.
Fail: hang, "CUDA not initialized" exception, or any synchronize call traced.
```

### 4.6 Profile-gating bugs

Applies to: every new sampler that needs a profile.

Three failure shapes:

1. **Sampler always-on (ignores profile).** Author appends to `samplers` outside any `if self.profile == ...` branch. A sampler intended for `deep` runs in `watch`, blowing the overhead budget on production training jobs.
2. **Sampler in wrong profile branch.** Author copies the wrong template — e.g. puts a layer-level sampler under `if self.profile in ["run", "deep"]` instead of `if self.profile == "deep"`. The smoke test misses it because it runs in `deep`.
3. **`_build_samplers` not updated.** Author adds the sampler file but forgets the registration. The sampler is dead code; the PR's tests pass; the smoke test produces no rows for the new sampler. (`add_sampler.md` §9 pitfall #1.)

**What to check:**

- Open `src/traceml/runtime/runtime.py` and find `_build_samplers`. The new sampler must appear inside the correct profile branch — confirm with the author which profile they intended.
- The import for the new sampler appears at the top of `runtime.py`. Without the import, the registration line raises `NameError` at construction.
- Run the PR with `traceml watch <example>` and `traceml deep <example>` — the sampler should appear (or not) per its profile gating. Check that rows appear in the aggregator (or in the per-rank database log file) for the right profile and not for the others.

**Common symptom:** PR ships, the next overhead-conscious user reports `watch` mode is slower than v0.2.3, or the new sampler never appears in the dashboard.

**Verification-gate shape:**

```
Setup: traceml watch examples/<small>.py
Command: grep aggregator log for the new sampler's table name.
Pass: row count matches expected profile gating (zero for excluded profiles, >0 for included).
Fail: rows appear in a profile that should exclude them.
```

### 4.7 Test-isolation issues

Applies to: every sampler test file.

Three failure shapes:

1. **Tests touch real GPU / network.** A test that calls `pynvml.nvmlInit()` for real fails in CPU-only CI. A test that constructs a real `DBIncrementalSender` with a real socket binds a port and races other tests.
2. **Tests depend on env vars without setting them.** `resolve_runtime_context()` reads `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `TRACEML_SESSION_ID`, `TRACEML_LOGS_DIR`. A test that doesn't set them either crashes (no session id) or silently uses the developer's environment, causing flakes between dev boxes.
3. **Tests assert on `self.sender` instead of `self.db`.** The sender is wired by the runtime; in unit tests it exists but has no transport attached. The Database is the right surface for assertions about row content. (`add_sampler.md` §8 final-paragraph notice.)

**What to check:**

- `pynvml`, `psutil`, `torch.cuda.*` calls in the sampler are patched at the module level (`from traceml.samplers import <mod> as mod; with patch.object(mod, "nvmlInit", ...)`). The autouse fixture in `add_sampler.md` §8 sets the five env vars `resolve_runtime_context()` reads.
- Assertions look at `s.db.get_table("...")`, not `s.sender.send_*` or any TCP surface.
- `s.sample_idx` is asserted across multiple calls — semantics matter (see Axis B note below).
- The test file does not write to `tests/` or any persistent path; it uses `tmp_path` for any session dir.

**Common symptom:** Tests pass on the author's box, fail in CI, or fail when run after another test that left env vars in a different state.

**Verification-gate shape:**

```
Setup: pytest tests/test_<new>_sampler.py -v on a CPU-only box with no GPU drivers.
Command: pytest --random-order tests/ for the new test file integrated.
Pass: all tests green; no NVMLError, no socket-bind error, no env-var KeyError.
Fail: any of the above.
```

**Note on `sample_idx` semantics:** `LayerForwardTimeSampler` only bumps `sample_idx` on a *resolved* step; periodic samplers bump on every tick. Either is correct, but the test must encode which contract the sampler honors.

---

## 5. Step 4 — The four meta-questions

Apply each to the PR and write down the answer explicitly. If you can't answer, ask.

### 5.1 Does this PR introduce a new axis of variation? What new failure modes?

The existing family has four dominant axes: driver style (periodic / event-driven), rank scope (host / per-rank), profile gating, and CUDA-touching vs. NVML-only. A new sampler that introduces a fifth axis — say, "needs a per-step boundary callback" or "writes to multiple tables from one `sample()`" — buys new failure modes specific to that axis.

**Reviewer move:** when the new sampler has a column in the consistency table (§3) that no prior sampler fills, enumerate the failure modes that column creates. Examples of novelty that has shipped or been proposed:
- Multi-table write from one sampler — needs to ensure both tables advance their senders together.
- Architecture-dedup by signature (`LayerMemorySampler._compute_signature`) — needs a test that the sampler emits exactly once per architecture, not per step.
- One-time manifest write in `__init__` — needs idempotence test (instantiate twice, assert one manifest file).

### 5.2 Does it interact with shared infrastructure?

Two pieces of shared infrastructure in the sampler family:

- **The runtime tick thread.** All samplers run sequentially on a single thread driven by `TraceMLRuntime._tick()` (default cadence 1.0 s via `TRACEML_INTERVAL`). A new sampler that takes >100 ms per `sample()` delays every other sampler.
- **The TCP send batch.** All samplers' senders flush in the same `_tick()` pass. A new sampler with `max_rows_per_flush=-1` and 1k rows per tick produces a fat payload that competes for socket bandwidth with every other sampler.

**Reviewer move:** estimate the new sampler's per-tick cost and per-tick row count. Force the author to answer: "On a 100k-step training run at 1.0 s tick, how many rows does this emit total? What's the deque eviction rate?" If the answer is "rows >> deque maxlen and `max_rows_per_flush=-1`," the sender will fall behind and the deque will evict — fine, but the author should know.

### 5.3 Is the wire-format key set a contract?

Yes. Once a row schema lands in a release and a user has a dashboard / SQLite projection / `traceml compare` baseline reading from it, **every key is a contract**. Renaming or retyping breaks v0.2.x users. (`principles.md` §3 is the cross-cutting rule — link, don't restate.)

**Reviewer move:** for every key the new sampler emits:
- If it's a new key: confirm the consumer (renderer / projection writer) reads it via `payload.get(key, default)`, not `payload[key]`.
- If it's an existing key (the PR modifies an existing sampler): confirm name and type are unchanged. A type change from `int` to `float` is breaking even though both are numeric.
- If the PR introduces a new table name: confirm the projection writer registration in `aggregator/aggregator_main.py` is wired and the renderer (if any) targets the new table.

### 5.4 Which invariants does the PR preserve, and have you verified each one?

The sampler-family invariants (paraphrased from `principles.md` and `add_sampler.md`):

1. **Fail-open.** `sample()` never raises into the runtime; init never raises into `_build_samplers`.
2. **Bounded overhead.** Periodic `sample()` < 1 ms on a healthy host; event-driven drain < 1 ms per 1k events.
3. **Bounded queues at the producer.** Event-driven sampler input queues are `Queue(maxsize=N)` with drop-on-full at the producer side.
4. **No CUDA sync.** No `torch.cuda.synchronize()` from a sampler. Period.
5. **Primitive-only wire payload.** `to_wire()` returns `dict[str, primitive | list[primitive] | dict[str, primitive]]`.
6. **Wire-format additive only.** New keys never replace old ones; types are stable.
7. **Rank-aware host metrics.** `local_rank == 0` filter for host-level data; per-rank data lets the envelope carry the rank.
8. **Sampler is constructed with zero arguments.** Configuration comes from `resolve_runtime_context()`, not from constructor args.

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

### 6.1 Worked example — fail-open violation

```
# Setup
git -C <repo> checkout pr-N
# CPU-only OK.

# Command — save as repro.py:
import os
os.environ.update({
    "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1",
    "TRACEML_SESSION_ID": "test", "TRACEML_LOGS_DIR": "/tmp/traceml-test",
})
from unittest.mock import patch
from traceml.samplers import gpu_utilization_sampler as mod

# Force the source to raise mid-sample.
with patch.object(mod, "nvmlDeviceGetUtilizationRates",
                  side_effect=RuntimeError("simulated")):
    s = mod.GpuUtilizationSampler()
    s.sample()  # must not raise

print(len(list(s.db.get_table("GpuUtilizationTable"))))

# Expected (after fix)
# 1   (zero-filled row, no exception)
#
# Pass criterion: script exits 0; printed value is 0 or 1.
# Fail criterion: any traceback escapes sample().
```

That's a 12-line recipe. The author can paste it; the maintainer can read it without running.

### 6.2 Worked example — host-metric duplication

```
# Setup: Lightning Studio with 2 GPUs.
# Command:
traceml watch examples/small_train.py --nproc-per-node 2 --mode summary

# Expected (under the bug)
# logs/<session>/aggregator/telemetry shows 2 rows per second for SystemTable
# (one per rank), instead of 1 row per second.

# Pass criterion (after fix)
# SystemTable row rate == 1 / TRACEML_INTERVAL on the aggregator side,
# regardless of nproc-per-node.
```

### 6.3 When you can't write a verification gate

If you can't write a recipe — you only have a vague worry — **don't raise the concern in the review yet**. Either escalate it to research (file a follow-up issue, label "investigate"), or hold it back per §7.3. Vague concerns waste author time.

### 6.4 Recipe style rules

- **Specific numbers, not adjectives.** `len(rows) == 1` not "should be small."
- **Reproducible from a clean checkout.** No "you also need to apply patch X first."
- **3–10 lines of actual code.** Longer means you're testing too much at once.
- **State the GPU dependency explicitly.** "Needs CUDA" / "CPU-only OK." Reviewer running in CPU CI shouldn't try to reproduce GPU-only bugs.
- **Use `tmp_path` / `monkeypatch` env vars.** Don't depend on the dev box's environment.

---

## 7. Step 6 — Drafting comments

Two granularity choices: line comment vs PR-level comment. They are not interchangeable.

### 7.1 Line comments

Use when: there is a specific code change you're proposing in a specific location. Pin the comment to the line that needs to change.

Pattern: state the issue → propose the fix → reference a verification gate or precedent.

```
This _init_gpu raises NVMLError on hosts without drivers; the sampler
won't construct.

Wrap nvmlInit() in try/except NVMLError (precedent: SystemSampler._init_gpu
sets gpu_available=False on failure).

Verification: set NVML_NO_DRIVER=1 and instantiate; must not raise.
```

Keep it tight. The reviewer's job is to point at the change, not to re-derive the architecture.

### 7.2 PR-level comments

Use when: the concern is **behavioural** or **architectural**, not localised to a single line. The fix may touch multiple files; the discussion is about the PR's intent.

Pattern: state the scenario → walk through what happens under the current diff → propose 2–3 fixes ranked by your preference → invite discussion.

A PR-level comment is also right for cross-cutting concerns: "did you measure overhead under deep profile?", "does this name conflict with X?", "the rank-scope decision is unstated — please document in the module docstring."

### 7.3 What NOT to raise (the holdback discipline)

Two kinds of items belong in your private parking-lot, not in the PR review:

- **Judgement calls about positioning** — "should this be `watch` or `run`?", "should we expose this via the dashboard immediately?" These are about your strategy with the author, not the PR. Decide privately, apply privately.
- **Adjacent improvements** — "while we're here, the `SystemSampler` GPU init has the same shape as this and could be extracted." If the improvement isn't required for the PR to ship, file it as a follow-up issue. Don't grow the PR.

The discipline: a PR review delivers a focused set of must-fix items. Bloating the review with parking-lot items dilutes the must-fix signal and trains the author to treat your reviews as discussion threads, not gates.

### 7.4 The maintainer summary

Every review ends with a 2–3 sentence executive summary suitable for the maintainer to read without opening the PR. The shape:

> PR #N adds <sampler name> under <profile>. Architecture matches <closest-precedent sampler>. Review converged on K concrete items: (1) ..., (2) ..., (3) .... All K fixes are localised; each needs one small test. Recommend [verdict].

Maintainer reads three sentences and either agrees with the verdict or opens the PR. This is the artifact your maintainer wants more than the diff comments.

---

## 8. Step 7 — Landing the verdict

Three states. Pick exactly one.

### 8.1 Approve

Conditions:
- Consistency table (§3) is fully ✅ or has documented justified deviations.
- All four meta-questions (§5) answered explicitly.
- All eight invariants (§5.4) preserved.
- No concerns require a verification gate (§6).
- Tests cover construction, happy path, source-failure safety. Wire-encode roundtrip asserted.
- Smoke test (`traceml <profile> examples/<small>.py`) passes locally with the new sampler appearing in the right profiles only.

If all six are true, approve cleanly. Don't suggest follow-up work in the approval — file follow-ups separately so the PR can ship.

### 8.2 Approve with changes

Conditions:
- All concerns are minor: docstring fixes, test gaps, naming nits, optional sender-knob tuning, schema field default consistency.
- No concern affects metric correctness or training safety.
- All concerns have a one-line fix or a clear written-down resolution.

This is "accept the PR but require these N small changes." Not "the PR is conceptually broken."

### 8.3 Block (request changes)

Conditions (any one):
- A fail-open violation (§4.1) — `sample()` or init can raise into the runtime.
- A rank-awareness bug (§4.2) — host metrics duplicated, or `rank == 0` used where `local_rank == 0` should be.
- A wire-format break (§4.3) — existing key renamed / retyped.
- An unbounded queue (§4.4) — producer queue without `maxsize`.
- A CUDA hazard (§4.5) — `torch.cuda.synchronize()` from a sampler, or unguarded CUDA touch before distributed init.
- A profile-gating bug (§4.6) — sampler runs in the wrong profile or always-on when it shouldn't be.
- The PR introduces a new axis of variation without enumerating its failure modes (§5.1).
- Tests don't exist for a category in §4 that applies.

### 8.4 What "block" doesn't mean

It does not mean the architecture is wrong. It does not mean the author has to redesign. It means **these specific items must be resolved before merge.** Frame the verdict that way to keep the relationship healthy with the author.

---

## 9. Worked example

There is no canonical sampler-PR review yet. PR #87 (in `Notes/PR_87_review_through_walkthroughs.md`) is the worked example for the patch family; the sampler family will get its first reviewer-side reference review when the next sampler-PR lands. Until then, the consistency table in §3.2 and the failure-mode catalogue in §4 are the corpus.

When you do review the next sampler PR, plan to write up the review in `Notes/PR_<N>_sampler_review.md` mirroring the structure of `PR_87_review_through_walkthroughs.md`. That document will then be the §9 reference for the next reviewer.

---

## 10. Common reviewer mistakes

Numbered, with cause and fix.

1. **Reviewing in isolation.** Cause: opening the diff first, before anchoring to walkthroughs. Effect: drowning in 5–7 files. Fix: do §2 before §3 — every time.

2. **Re-deriving the consistency table from scratch.** Cause: the table isn't a formal artifact in source. Effect: each reviewer rebuilds it slightly differently. Fix: paste the §3.2 table into your review notes verbatim and add the new column.

3. **Trusting `_safe()` to catch fail-open violations.** Cause: assuming the runtime's outer try/except will save the day. Effect: a sampler that raises during construction (in `_init_*` or `__init__`) crashes the runtime — `_safe()` only wraps `sample()`, not `_build_samplers()`. Fix: §4.1 — every `_init_*` must set sentinels and never re-raise.

4. **Approving a host-level sampler without checking the rank gate.** Cause: focusing on `sample()` body, not the registration. Effect: N-way duplication on multi-rank runs ships unnoticed because the dev box is single-rank. Fix: §4.2 — every host-level sampler review must run a 2-rank smoke test.

5. **Vague concerns without verification gates.** Cause: time pressure, gut feel. Effect: author can't reproduce, dismisses the concern. Fix: §6 — every concern gets a recipe.

6. **Mixing line comments and PR-level comments.** Cause: writing architectural concerns inline next to a code line. Effect: comment gets resolved by changing one line, the architectural point is lost. Fix: §7.1/§7.2 — pick the granularity deliberately.

7. **Missing the wire-format additive-only rule.** Cause: focusing on the sampler implementation, not the schema diff. Effect: a quiet column rename breaks v0.2.x consumer dashboards. Fix: §5.3 + grep the schema dataclass against the v0.2.3 release tag for every existing key.

8. **Reviewing without running the tests locally.** Cause: trusting green CI. Effect: missing GPU-only paths, missing ordering-sensitive failures, missing the "test relies on dev-box env vars" trap. Fix: at minimum run `pytest tests/test_<new>_sampler.py -v` on a CPU-only box; if the sampler touches CUDA, also run on a CUDA box. If the test file doesn't have an autouse env fixture, that's a §4.7 finding.

9. **Approving on architecture without checking failure modes.** Cause: the sampler follows the family pattern, so it must be fine. Effect: rank-scope, queue-bound, or CUDA-hazard bugs ship. Fix: §4 catalogue is non-optional even when the architecture is clean.

10. **Conflating "matches the family" with "correct."** Cause: §3 consistency check is ✅ across the board, so the reviewer stops. Effect: novel-axis failure modes (§5.1) miss. Fix: every empty cell in the new column is a question, not a free pass.

11. **Skipping the maintainer summary.** Cause: assumes the maintainer will read the diff. Effect: maintainer reads the diff, disagrees on severity, kicks back to the reviewer. Fix: §7.4 — three sentences are the maintainer's reading material; the diff is yours.

12. **Asserting on `self.sender` in tests.** Cause: thinking the sender is part of the sampler's contract under test. Effect: fragile tests that break when the sender's internal state machine changes. Fix: §4.7 — assertions go on `s.db.get_table(...)`, not on the sender.

---

## Gaps encountered while writing this guide

Where the reviewer's playbook is currently underspecified or relies on folklore. Flag these in your review process if you hit them.

- **The consistency table (§3) isn't a formal artifact.** It lives in this guide. If the sampler family grows to ten entries, every reviewer will diverge on the column set. Worth lifting into a contract test in `tests/test_sampler_family.py` that introspects each `*_sampler.py` module and asserts the `BaseSampler` subclass + `sample()` outer try/except + schema dataclass triple exists. Not yet written.

- **There's no central registry of sampler table names.** A reviewer enforcing §5.3 (wire-format-as-contract) has to grep across renderers and SQLite projection writers and trust their grep. Worth a constants module `src/traceml/samplers/table_names.py` with every table name exported as a constant; samplers import the constant; renderers and projection writers import the same constant. A test would then assert no two constants collide and every table name is referenced by at least one consumer. Not yet written.

- **No reviewer-side overhead harness.** A reviewer who wants to check §5.2 ("estimate per-tick cost and row count") needs to drop a `timeit` block by hand. The benchmark workflow at v0.2.9 is the formal answer; until it lands, overhead claims are folklore. A `tests/sampler_overhead/` directory with parametrised fixtures (autouse env, mocked sources, `pytest-benchmark` integration) would make per-PR overhead claims reproducible in 5 lines instead of 30.

- **Profile gating has no contract test.** A sampler in the wrong profile branch silently runs in `watch` when it should be `deep`-only. The smoke test in `principles.md` §6 catches the egregious case but not subtle gating bugs. A `tests/test_profile_gating.py` that constructs `TraceMLRuntime` for each profile and asserts the sampler set is exactly what `_build_samplers` should produce would make §4.6 a regression test instead of a code-review checklist item.

- **The "fail-open" contract has no enforcement.** A new sampler whose `_init_*` raises into `__init__` will be caught in code review *if the reviewer remembers*. A contract test that imports every sampler, monkeypatches every common source (psutil, pynvml, queue.get) to raise, and asserts construction-then-`sample()` doesn't raise would turn §4.1 from folklore into CI gate. Not yet written.

- **The verdict criteria (§8) are folklore-level.** "Affects metric correctness" is the bright line for blocking, but reasonable people disagree about what "correctness" means under degraded conditions (e.g. "should a degraded zero-filled GPU row count as a metric correctness issue?"). A formal list of "metric-correctness invariants the project commits to" would resolve these arguments before the PR.

- **The "holdback discipline" (§7.3) has no checklist.** Knowing what belongs in the PR vs. a follow-up is currently a judgement call. Two reviewers might draw the line differently. Worth a short rubric: "must-fix iff the PR-as-merged would (a) corrupt a metric, (b) inflate overhead by >X%, (c) break wire-format backward compat, or (d) crash training under any plausible production config." Otherwise: follow-up.

These gaps are the natural follow-up work after this guide lands. None block the next reviewer from following the workflow above; all would make the next reviewer faster.
