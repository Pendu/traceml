# How to land a wire-format / schema-migration change

> Internal contributor guide. Audience: one trusted co-founder / new engineer
> being onboarded to TraceML. Not for public docs.

This is the most cautious guide in the family. You are reading it because you
are about to touch one of the five contracts that survives across processes
and across versions. Every existing user on `traceml-ai==0.2.x` is a
stakeholder in this PR. Read the whole thing before opening an editor.

---
Feature type: wire format / schema migration
Risk level: HIGH (every existing v0.2.x user is a stakeholder)
Cross-cutting impact: every subsystem (sampler, projection writer, renderer, summary, CLI, env var)
PyTorch coupling: indirect
Reference PRs: none — no historical wire-format break to point at
Companion reviewer guide: none yet
Last verified: 2026-04-25
---

## 1. Intro and mental model

### What is "a wire-format change" in TraceML?

A wire-format change touches **the contracts that survive across processes
and across versions**. These contracts are not enforced by the type system.
They are enforced by convention, and by the fact that real users have already
shipped runs that depend on them.

There are five distinct wire surfaces in TraceML, each with backward-compat
obligations:

1. **Sampler payload schema** — the `dict` shape every sampler emits via
   `_add_record(payload)`. Travels via msgpack over TCP. Consumers: the
   aggregator's `RemoteDBStore`, the SQLite projection writers, the renderers,
   `traceml compare`. Producer files: `src/traceml/samplers/schema/*.py`.
2. **`TimeEvent` `name` field** — the string identifying a timing event
   (`_traceml_internal:forward_time`, `_traceml_internal:backward_time`,
   `_traceml_internal:dataloader_next`). Renderers and the step-time
   diagnosis pattern-match on it. Producer call sites: every patch in
   `src/traceml/utils/patches/*.py` invokes `timed_region(name=...)`.
3. **SQLite projection schema** — the table and column shape of the
   persisted database under `logs/<session>/aggregator/telemetry.sqlite`.
   v0.2.x users have on-disk DBs with the old schema; the on-disk file is
   itself a wire surface for any tooling that opens it. Producer files:
   `src/traceml/aggregator/sqlite_writers/*.py` (`init_schema`,
   `build_rows`).
4. **CLI surface** — flag names, env var names, subcommand names. Users on
   PyPI v0.2.3 have shell scripts and CI pipelines that depend on these.
   Producer file: `src/traceml/cli.py` plus every read site for env vars.
5. **Diagnosis JSON** — the `kind` / `severity` / `status` / `reason` /
   `action` shape inside `final_summary.json`. The future TraceOpt
   regression detector will key off this. Producer files:
   `src/traceml/aggregator/summaries/step_time_diagnosis.py` and the
   `final_summary.py` aggregator. Currently uses `schema_version: 1` —
   see §7.

### Two-direction compatibility

The compat obligation goes in **both** directions:

- **Forward (new aggregator decodes old payloads).** A v0.3.0 aggregator
  must still decode payloads produced by a v0.2.5 training process. This
  matters because heterogeneous environments are common — one host updates
  before another, or a developer pins TraceML at a specific version inside
  a Docker image while running the dashboard from a fresher checkout.
- **Backward (old CLI scripts work with new TraceML).** A user's
  `traceml watch ./train.py --mode dashboard --port 8765` script that has
  been running unchanged since v0.2.3 must keep working when they
  `pip install -U traceml-ai`.

Both directions are non-negotiable in a minor bump. Either direction can be
broken in a major bump, with a CHANGELOG migration note.

### Strategic context: TraceOpt as downstream consumer

[`traceml_why.md`](../deep_dive/why.md) §6.4 — longitudinal regression
detection — is the strategic reason wire stability matters even more in
v1.0+. The TraceOpt cloud product is going to ingest historical
`final_summary.json` files. The "your training got 12% slower over the last
month" insight depends on the diagnosis JSON shape being stable across
months of TraceML releases. A `kind` rename now is a downstream-consumer
break later.

When you change one of the five surfaces, you are not just changing TraceML.
You are changing the surface area that future TraceML and TraceOpt code
must keep parsing for as long as anyone has old data they care about.

### Why this guide exists

The per-feature author guides ([add_sampler.md](add_sampler.md),
[add_sqlite_projection.md](add_sqlite_projection.md),
[add_cli.md](add_cli.md), [add_diagnostic.md](add_diagnostic.md)) all
handle the **additive** case: a new field, a new column, a new flag, a new
diagnosis. Their compat sections (each §6) say the same thing in different
words: add, don't rename; add, don't remove; add, don't change types.

This guide is for the **non-additive** case: renames, removals, type
changes, table renames, flag renames, diagnosis-key renames. Every
non-additive wire change in TraceML lands as **at least two PRs**: a
dual-emit / dual-read PR in a minor version, then a drop-old PR in a
major version. There is no clean-one-shot break. Compat is a feature.

For the cross-cutting rules these procedures derive from, see
[principles.md](principles.md) §3 (wire compatibility) and §7
(versioning). This guide refines them into a procedure.

### Where this sits in the pipeline

```
[ Producer side ]                                  [ Consumer side ]
  Sampler payload  ── msgpack TCP  ──▶  RemoteDBStore  ──▶  projection writer
  TimeEvent.name   ── inside payload ──▶ step_time_sampler aggregation
                                         step_time_diagnosis
  CLI flag / env var ── env(7) ────────▶ executor / aggregator
                                                                 │
                                                                 ▼
                                                          SQLite projection
                                                                 │
                                                                 ▼
                                                          renderer / compare
                                                                 │
                                                                 ▼
                                                       final_summary.json
                                                          (diagnosis JSON)
```

Every box is a place where the wire shape is interpreted. A non-additive
change perturbs every box at once.

---

## 2. Decisions to make

### Decision 0 (the most important): should you break compat at all?

Most "wire-format changes" can be done **additively**. Before you proceed
with this guide, work down this list:

- [ ] **Renaming a key?** Don't. Add the new name and keep reading both.
      Mark the old name deprecated in code comments. Remove the old name
      in a future major version. See §6.1.
- [ ] **Changing a type?** Don't change the existing key. Add a new key
      with the new type. Keep the old one populated for one release.
      Migrate consumers. Drop the old key in a major version. See §6.1.
- [ ] **Removing a key?** Stop emitting it from the producer side. Most
      consumers are already using `payload.get(key, default)`, so they
      tolerate `None`. (Audit the consumers; see §3.1.)
- [ ] **Adding a new column to a SQLite table?** That is additive. Use
      the recipe in [add_sqlite_projection.md §6](add_sqlite_projection.md).
      You don't need this guide.
- [ ] **Adding a new CLI flag?** That is additive. Use
      [add_cli.md §6](add_cli.md). You don't need this guide.

If you genuinely need a non-additive change — an existing key really must
be renamed because its current name is misleading, an existing column
really must change type because the wrong type causes user-visible
breakage, an existing flag name conflicts with a feature you cannot ship
otherwise — proceed. Document the reason in the PR description. The bar
is high.

### Decision 1: which surface(s) are you touching?

Pick from the five in §1. If your change touches more than one (e.g.
renaming a sampler payload key *and* the SQLite column that mirrors it),
treat each surface independently in the PR. Each surface has its own
compat strategy in §3.

### Decision 2: dual-emit window length

The dual-emit / dual-read window is the period during which both the old
and new shapes are produced and consumed. Default: **one full minor
version** (e.g. 0.3.x emits both, 0.4.0 drops the old). Longer if you
suspect heavy v0.2.x deployment.

### Decision 3: who is the downstream consumer?

For sampler payloads: aggregator + projection writers + renderers +
`traceml compare`. For TimeEvent names: `step_time_sampler` aggregation
keys, the diagnosis engine, the renderers. For SQLite columns: anyone
with `sqlite3` opening the file (including TraceOpt-side ingest, if it
exists by the time you ship). For CLI flags: shell scripts and CI in
production. For diagnosis JSON: TraceOpt regression detector.

If the answer to "who is the downstream consumer?" is "I don't know," go
find out before continuing. The whole point of compat is that the
consumer doesn't know your change is happening.

### Decision 4: version bump

See §7 for the table. The short form: a non-additive change in
dual-emit mode is a **minor** bump. The drop-old PR that follows is a
**major** bump.

### Decision 5: do you need a migration script?

For sampler payloads and TimeEvent names: no — the deque-based store is
ephemeral; old payloads aren't persisted on the producer side. For
SQLite: usually yes if the change is destructive (renamed table). One-shot
migration scripts go in `src/traceml/aggregator/migrations/` (does not
exist yet — see §11). For diagnosis JSON: no, the file is per-run and
v0.2.x runs already have their JSON locked.

### Decision 6: have you measured the blast radius?

`grep -rn "<old_name>" src/ tests/ examples/ docs/` before writing code.
Every match is a place that needs to either keep working with the old
name (read site) or stop using the old name (write site). Count the
matches; put the count in the PR description. A "small rename" with 47
read sites is not small.

---

## 3. The five wire surfaces (anatomy)

For each surface: what it is, where it's defined, who consumes it, the
compat rule, and the major-version-bump trigger.

### 3.1. Sampler payload schema

**What it is.** Each sampler builds a flat-ish `dict[str, primitive]` and
calls `self._add_record(payload)`. That dict is the wire payload.

**Where it's defined.** Producer side:
`src/traceml/samplers/schema/*.py` — frozen dataclasses with `to_wire()`
methods. The dataclass is a sampler-private convenience; the `dict`
returned by `to_wire()` is the wire surface.

**Who consumes it.**
- `src/traceml/database/database.py::Database.add_record` — appends to a
  `deque`.
- `src/traceml/database/database_sender.py::DBIncrementalSender.collect_payload`
  — wraps in the outer `{"rank", "sampler", "timestamp", "tables"}` envelope.
- `src/traceml/transport/tcp_transport.py` — msgpack-encodes.
- Aggregator side: `RemoteDBStore` ingests, projection writers in
  `src/traceml/aggregator/sqlite_writers/*.py` consume specific keys to
  build SQL rows.
- Renderers in `src/traceml/renderers/` read via the store.
- `src/traceml/compare/` reads `final_summary.json` (a derived projection),
  not the raw payload — but the summary's shape derives from the payload.

**Compat rule.** [principles.md §3](principles.md): never remove or rename
an existing key, never change its type, new keys are optional on the
consumer side.

**Major-bump trigger.** Renaming a key, removing a key that consumers
read positionally, or changing a type. See §6.1 for the dual-emit recipe
that defers the major bump.

### 3.2. `TimeEvent.name`

**What it is.** A short string identifying a timed region. Currently:

| Wire name                            | Producer file                                     |
|--------------------------------------|---------------------------------------------------|
| `_traceml_internal:forward_time`     | `utils/patches/forward_auto_timer_patch.py`       |
| `_traceml_internal:backward_time`    | `utils/patches/backward_auto_timer_patch.py`      |
| `_traceml_internal:dataloader_next`  | `utils/patches/dataloader_patch.py`               |

The convention is `_traceml_internal:<event>` for built-in events. User
code that calls `timed_region("my:thing", ...)` is in a different
namespace and is not a wire concern of this guide.

**Where it's defined.** The string is a literal at the patch call site.
There is no central enum.

**Who consumes it.**
- `src/traceml/utils/timing.py::flush_step_time_buffer` — emits
  `TimeEvent(name=...)` instances.
- `src/traceml/samplers/step_time_sampler.py` — aggregates events by
  `(name, device, is_gpu)` keys.
- `src/traceml/samplers/schema/step_time_schema.py` — defines the wire
  shape that includes `name`.
- `src/traceml/aggregator/sqlite_writers/step_time.py` — stores the name
  in `events_json` (as a JSON object key).
- `src/traceml/aggregator/summaries/step_time_diagnosis.py` — looks up
  events by name to compute the diagnosis.
- Renderers — display human-friendly labels derived from the name.

**Compat rule.** The wire `name` is contract. See
[review_patch.md §5.3](review_patch.md) for the full reasoning. Never
rename without dual-emit. Never repurpose an existing name to mean
something else.

**Major-bump trigger.** Renaming `_traceml_internal:forward_time` to
anything else. The diagnosis engine pattern-matches on this string; any
existing run summary references it.

### 3.3. SQLite projection schema

**What it is.** The on-disk table layout under
`logs/<session>/aggregator/telemetry.sqlite`. Five projection writers
exist: `system`, `process`, `step_time`, `step_memory`, `stdout_stderr`.
Each owns its own tables and exposes `init_schema(conn)` plus a
`build_rows(payload)` (writer-specific name) that consumes the wire
payload.

**Where it's defined.** Producer files:
`src/traceml/aggregator/sqlite_writers/*.py`. Each `init_schema` is
idempotent — uses `CREATE TABLE IF NOT EXISTS` — so adding a new table
in a new release is safe on existing DBs.

**Who consumes it.**
- The aggregator itself reads back via the renderers' SQL queries.
- `traceml compare` reads `final_summary.json` (derived from these
  tables).
- Any user with `sqlite3` opening the file — including the future
  TraceOpt ingest pipeline.

**Compat rule.** [principles.md §3](principles.md):
`ALTER TABLE ... ADD COLUMN ... DEFAULT NULL` is the only schema change
permitted in a minor version. Column type changes, column renames, and
table renames are major bumps.

**Major-bump trigger.** Any non-`ADD COLUMN` schema change. Note: SQLite's
`ALTER TABLE ... DROP COLUMN` exists from SQLite 3.35.0 (March 2021), but
TraceML targets older SQLite for safety — the practical drop-old recipe
is "create new table, copy data, drop old table" inside a one-shot
migration script (§6.3).

**Note on schema versioning.** TraceML does **not** currently use
`PRAGMA user_version` or maintain a `schema_version` table inside the
SQLite file. The `schema_version: 1` strings you'll find via
`grep -rn schema_version src/` (for example `cli.py:179`,
`final_summary.py:206`, `compare/io.py:88`) refer to **JSON file** schema
versions (manifest, summary), not the SQLite database. See §11 for the
gap and the proposed fix.

### 3.4. CLI surface

**What it is.** Subcommand names (`watch`, `run`, `deep`, `compare`,
`inspect`), flag names (`--mode`, `--port`, `--profile`, `--nproc-per-node`),
and environment variable names (`TRACEML_PROFILE`, `TRACEML_UI_MODE`,
`TRACEML_LOGS_DIR`, ...).

**Where it's defined.** `src/traceml/cli.py::build_parser` and
`_add_launch_args`. Env var read sites scatter across the codebase —
search with `grep -rn TRACEML_ src/`.

**Who consumes it.** End users running `traceml ...` from scripts, CI
pipelines, Dockerfile `CMD` directives, internal team runbooks.

**Compat rule.** [principles.md §3](principles.md): never remove or
rename an existing CLI flag or env var. Add a new alias and keep reading
the old one for one full release cycle before deprecating with a
warning.

**Major-bump trigger.** Removing the old flag or env var. Renaming
without dual-read.

**Existing precedent.** `TRACEML_UI_MODE` superseding `TRACEML_MODE`.
Read sites at `aggregator/aggregator_main.py:94-95`,
`runtime/executor.py:213-214`, `final_summary_protocol.py:111-112` all
follow the pattern: prefer the new name, fall back to the old name.
This is the template (§6.4).

### 3.5. Diagnosis JSON

**What it is.** The shape of the per-engine diagnosis dictionary inside
`final_summary.json`:

```python
{
    "kind": diagnosis.kind,         # e.g. "step_time"
    "severity": diagnosis.severity, # e.g. "info" | "warn" | "critical"
    "status": diagnosis.status,     # e.g. "stable" | "degrading"
    "reason": diagnosis.reason,     # human-readable string
    "action": diagnosis.action,     # human-readable string
    ...
}
```

(Source: `aggregator/summaries/step_time_diagnosis.py:353-357`.)

**Where it's defined.** Each engine defines its own diagnosis dataclass;
the wire shape is whatever the engine stamps into the summary JSON via
`final_summary.py`.

**Who consumes it.**
- `traceml compare` reads it across runs.
- Future TraceOpt regression detector (cloud-side) keys off `kind` and
  `severity` for drift detection.
- Users who `cat final_summary.json | jq .diagnosis`.

**Compat rule.** Same as sampler payload: additive only. New diagnosis
fields are optional (`payload.get("new_field", None)`). The set of
`kind` strings is contract — once you ship `kind: "step_time"`, you
cannot rename it to `kind: "step_time_diagnosis"` without a major bump.

**Major-bump trigger.** Renaming a `kind`, removing a field, changing
the meaning of `severity` levels.

**Schema version.** `final_summary.json` already carries
`"schema_version": 1` (see `final_summary.py:206`). Bump it on any
breaking change to the JSON file shape (§7, §10).

---

## 4. Step-by-step: a hypothetical case study

This section walks the **full** migration recipe for a single concrete
change, end-to-end. The change: rename the wire `TimeEvent` name
`_traceml_internal:forward_time` to `_traceml_internal:forward_phase_time`.
Pick this not because it's likely (it isn't), but because it touches the
maximum number of consumers, so the recipe generalizes.

The migration spans **two PRs and at least one full release cycle**.

### PR #1 (minor bump, e.g. 0.3.0): dual-emit

#### Step 1 — Add the new name alongside the old at the producer

`src/traceml/utils/patches/forward_auto_timer_patch.py:35` currently
contains:

```python
        with timed_region(
            "_traceml_internal:forward_time", scope="step", use_gpu=True
        ):
            return original_call(self, *args, **kwargs)
```

Change to dual-emit. Both events fire for the same span — the old name
keeps emitting for one release, the new name starts emitting now:

```python
        # NOTE: dual-emit during deprecation window of forward_time.
        # Drop the old name in 0.4.0 (major bump).
        with timed_region(
            "_traceml_internal:forward_phase_time",
            scope="step",
            use_gpu=True,
        ), timed_region(
            "_traceml_internal:forward_time",  # deprecated; drop in 0.4.0
            scope="step",
            use_gpu=True,
        ):
            return original_call(self, *args, **kwargs)
```

Two `timed_region` context managers nested adds two `event.record()`
pairs per call. The overhead doubles in this hot region during the
dual-emit window — call this out in the PR description, and confirm via
the v0.2.9 benchmark workflow that you're still inside the 5 µs hot-path
budget from [principles.md §5](principles.md).

A leaner alternative if the overhead is unacceptable: `timed_region` can
emit a single span under both names if you patch `timing.py` to accept
`name` as a tuple — but that change is itself a wire-format change to
the buffered event shape, so don't go there casually.

#### Step 2 — Update every read site to accept both

Find them with:

```bash
grep -rn "_traceml_internal:forward_time" src/ tests/
```

The expected read sites (verify against current source):

- **`samplers/step_time_sampler.py`** — aggregation. Either map the old
  name to the new at read time, or accept both as separate keys and emit
  both into the wire payload during dual-emit. The simpler path is to
  collapse on read: the sampler's aggregation key
  `(name, device, is_gpu)` becomes
  `(canonical_name(name), device, is_gpu)` where
  `canonical_name("_traceml_internal:forward_time")` returns
  `"_traceml_internal:forward_phase_time"`.
- **`aggregator/summaries/step_time_diagnosis.py`** — looks up timing
  events by literal name. Update every literal to read the new name with
  fallback to the old, or do the canonicalization once on ingest.
- **`aggregator/sqlite_writers/step_time.py`** — stores the event name
  inside `events_json`. The column is opaque-text from SQLite's
  perspective; canonicalization is the diagnosis engine's job, not the
  writer's.
- **Renderers in `renderers/`** — pattern-match on the name to choose a
  display label. Add the new name to whatever lookup table they use; keep
  the old name pointing at the same label. (Rationale: the user-facing
  label doesn't change just because the wire name did.)

After this step, both old and new names produce identical observable
behavior end-to-end. The aggregator handles a v0.2.5 client (only old
name) and a 0.3.0 client (both names) the same way.

#### Step 3 — Mark the old name deprecated in code

Inline `# DEPRECATED: drop in 0.4.0` comments at every place that still
mentions the old name. The CHANGELOG entry for 0.3.0 has a single line
under "Deprecated":

```
- _traceml_internal:forward_time event name; use
  _traceml_internal:forward_phase_time. Old name removed in 0.4.0.
```

#### Step 4 — Test the dual-emit

Three test categories:

1. **Producer round-trip (new shape).** Patch the forward, run a single
   step, assert two `TimeEvent`s with names
   `_traceml_internal:forward_time` and
   `_traceml_internal:forward_phase_time` both reach the buffer.
2. **Consumer fallback (old shape).** Inject a synthetic
   `step_time_sampler` payload that contains *only* the old name (mimics
   a v0.2.5 client). Run the diagnosis engine. Assert the diagnosis
   computes correctly.
3. **`traceml compare` cross-version.** Two synthetic
   `final_summary.json` files: one v0.2.5-shaped, one 0.3.0-shaped.
   Assert `compare` does not crash and produces a sensible diff.

See §8 for the test templates.

#### Step 5 — Smoke test on real PyTorch

Per [principles.md §6](principles.md):

```bash
pip install -e ".[dev,torch]"
traceml watch examples/<small example>.py --mode cli
# expect: forward timing visible in the live UI; final_summary contains
# the new event name; no stack traces.
```

Then verify the dual-emit explicitly:

```bash
sqlite3 logs/<session>/aggregator/telemetry.sqlite \
  "SELECT events_json FROM step_time_samples LIMIT 1;"
# expect: both _traceml_internal:forward_time and
# _traceml_internal:forward_phase_time present in the JSON.
```

#### Step 6 — Ship 0.3.0

- CHANGELOG: "Deprecated: `_traceml_internal:forward_time` event name."
- Version bump in `pyproject.toml`: 0.2.5 → 0.3.0 (minor).
- PR description: lists every read site updated, the dual-emit overhead
  measurement, the smoke-test output.

### Wait one full release cycle

Real time. One quarter, or one full minor bump. Users who pin TraceML in
their environments need a window to see the deprecation warning and act.
Skipping this window is the most common way to break a downstream user
silently.

### PR #2 (major bump, 0.4.0): drop the old

#### Step 7 — Remove the old name from the producer

```python
        with timed_region(
            "_traceml_internal:forward_phase_time",
            scope="step",
            use_gpu=True,
        ):
            return original_call(self, *args, **kwargs)
```

Back to a single `timed_region`. Hot-path overhead returns to baseline.

#### Step 8 — Remove all fallback paths from consumers

Strip the `canonical_name()` mapping. Strip the dual-key reads from the
diagnosis engine. Strip the old-name entry from the renderer label
lookup table.

#### Step 9 — Document the break

- CHANGELOG: `BREAKING: removed _traceml_internal:forward_time. Use
  _traceml_internal:forward_phase_time. Runs produced by traceml-ai
  <0.3.0 are still parseable by traceml compare; their diagnoses use the
  old name.`
- The 0.4.0 release notes need an explicit migration paragraph.

#### Step 10 — `final_summary.json` schema version bump

If the diagnosis output mentions the event name (it currently does in
the human-readable `reason` and `action` strings), bump
`final_summary.json::schema_version` from 1 to 2. `compare/io.py:88-90`
already raises if `schema_version` is missing — make sure it tolerates
both `1` and `2` for one more release, then drop `1` support.

#### Step 11 — Major-version smoke

Per [principles.md §6](principles.md). Plus:

```bash
pip install traceml-ai==0.2.5
traceml watch examples/<small example>.py
# capture the resulting final_summary.json
pip install -e .  # 0.4.0 from this branch
traceml compare <0.2.5_summary>.json <0.4.0_summary>.json
# expect: no crash, sensible diff. If this crashes you have shipped a
# wire-format break that traceml compare can't handle. Fix before ship.
```

This is the upgrade-path test. It is non-optional for any PR that comes
through this guide. See §10.

---

## 5. Common patterns

Reference table mapping change-type → compat strategy. Find your row,
follow the strategy in §6.

| Wire surface         | Change type        | Strategy                                                                                       |
|----------------------|--------------------|------------------------------------------------------------------------------------------------|
| Sampler payload      | Rename key         | Add new key, keep emitting old, dual-emit for one release; drop old in major bump.             |
| Sampler payload      | Remove key         | Stop emitting; consumer tolerates `None` because consumers already use `.get(key, default)`.   |
| Sampler payload      | Type change        | Add new key with new type, deprecate old; drop old in major bump.                              |
| `TimeEvent` `name`   | Rename             | Patch emits both events for one release; aggregation key becomes `(canonical(name), ...)`.     |
| SQLite column        | Rename             | Add new column, dual-write for one release; drop old in major bump (recipe in §6.3).            |
| SQLite column        | Type change        | Add new column with new type, dual-write, drop old in major bump.                              |
| SQLite table         | Rename             | Create new table, dual-write, deprecate old; drop old in major bump.                           |
| CLI flag             | Rename             | Add new flag as alias, keep old, validate at most one is set, warn on use of old.              |
| Env var              | Rename             | Read both, prefer new (precedent: `TRACEML_UI_MODE` / `TRACEML_MODE`, see §6.4).                |
| Subcommand           | Rename             | Add new subcommand, keep old, warn on use of old; remove old in major bump.                    |
| Diagnosis `kind`     | Rename             | Major bump + `schema_version` bump in `final_summary.json`. TraceOpt ingest must be updated.   |
| Diagnosis field      | Add                | Additive — does not need this guide. Use [add_diagnostic.md](add_diagnostic.md).               |
| Diagnosis field      | Remove             | Stop emitting; consumers use `.get()`. Strip in major bump only.                                |

---

## 6. Migration patterns (the meat)

For each surface, the recipe in detail. Each subsection is the "how" of
the matching row in §5.

### 6.1. Sampler payload — dual-emit, then drop

**Producer side (one release).** Modify the schema dataclass's
`to_wire()` to emit both keys:

```python
@dataclass(frozen=True)
class FooSample:
    sample_idx: int
    timestamp: float
    cpu_pct: float  # was previously named cpu

    def to_wire(self) -> Dict[str, Any]:
        return {
            "seq": self.sample_idx,
            "ts": self.timestamp,
            "cpu_pct": self.cpu_pct,
            "cpu": self.cpu_pct,  # DEPRECATED: drop in 0.4.0
        }
```

Both keys carry the same value. Consumers that read `cpu` keep working.
Consumers that read `cpu_pct` start working.

**Consumer side (same release).** Update every read site to prefer the
new key with fallback to the old:

```python
cpu = row.get("cpu_pct", row.get("cpu"))
```

Order matters: try the new key first. A v0.2.5 producer doesn't emit
`cpu_pct`, so the consumer falls back to `cpu`. A 0.3.0 producer emits
both; the consumer takes `cpu_pct`. A 0.4.0 producer (after drop) emits
only `cpu_pct`; the consumer takes `cpu_pct`.

**Drop-old release (major bump).** Strip the old key from `to_wire()`.
Strip the fallback from every consumer.

**Rule:** never have a phase where producer emits only new and consumer
reads only old. That phase silently breaks v0.2.5 → 0.3.0 upgrades.

### 6.2. `TimeEvent.name` — dual-emit at the patch site

See §4 for the full recipe. The summary:

- Patch wraps two `timed_region` blocks, both spanning the same code.
- Sampler aggregation collapses to a canonical name on ingest, OR carries
  both keys and lets the diagnosis engine pick.
- Drop the old `timed_region` at major bump.

The "two `timed_region` blocks" pattern doubles hot-path overhead during
the dual-emit window. Measure it. If it busts the 5 µs budget, fall back
to a single `timed_region` whose `name=` is changed at major bump only,
and accept that v0.2.5 runs and 0.3.0 runs report different event names
in their summaries — this is OK as long as `traceml compare` knows about
both.

### 6.3. SQLite column / table — dual-write

**For a column rename (additive phase, minor bump).**

`init_schema` is `CREATE TABLE IF NOT EXISTS`, which is a no-op on
existing DBs. Adding a column to an existing DB needs an `ALTER TABLE
... ADD COLUMN`. Wrap it in an `_ensure_column` helper that's idempotent:

```python
def _ensure_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    sql_type: str,
) -> None:
    """
    Add `column` of `sql_type` to `table` if it doesn't already exist.
    Idempotent: safe to call on every aggregator startup.
    """
    cur = conn.execute(f"PRAGMA table_info({table});")
    existing = {row[1] for row in cur.fetchall()}
    if column not in existing:
        conn.execute(
            f"ALTER TABLE {table} ADD COLUMN {column} {sql_type};"
        )
```

In `init_schema`, after the `CREATE TABLE IF NOT EXISTS`, call
`_ensure_column(conn, "step_time_samples", "step_phase", "INTEGER")` (or
whatever the new column is). This handles three cases uniformly:

- Brand-new DB (the column comes from the `CREATE TABLE` itself, the
  `ALTER` is a no-op).
- v0.2.5 DB upgraded to 0.3.0 (the `CREATE TABLE` is a no-op, the
  `ALTER` adds the new column with `NULL` defaults).
- Already-upgraded DB on subsequent aggregator restarts (both no-ops).

`build_rows` populates both old and new columns from the wire payload
during the dual-write window. Read sites prefer the new column with
fallback to the old.

**For a table rename or destructive column drop (major bump).**

SQLite's `ALTER TABLE ... DROP COLUMN` is recent; safer is the
"create new, copy, swap" pattern in a one-shot migration script:

```python
# src/traceml/aggregator/migrations/0001_rename_step_time_table.py
def migrate(conn: sqlite3.Connection) -> None:
    """
    Rename old_step_time → step_time_samples. Idempotent on already-migrated DBs.
    """
    cur = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name IN ('old_step_time', 'step_time_samples');"
    )
    tables = {row[0] for row in cur.fetchall()}
    if "step_time_samples" in tables and "old_step_time" not in tables:
        return  # already migrated
    if "step_time_samples" not in tables:
        # full create from current schema
        init_schema(conn)
    if "old_step_time" in tables:
        conn.execute(
            "INSERT INTO step_time_samples "
            "SELECT id, recv_ts_ns, rank, sample_ts_s, seq, step, events_json "
            "FROM old_step_time;"
        )
        conn.execute("DROP TABLE old_step_time;")
    conn.commit()
```

Run on aggregator startup (idempotent), with the "already-migrated"
check first so you don't wreck a fresh DB. The migrations directory
does **not** exist in the codebase yet — see §11.

### 6.4. CLI flag — dual-read with deprecation warning

Add the new flag as a sibling. Keep the old. Validate that at most one
is set. Warn on use of the old.

```python
parser.add_argument(
    "--profile",
    choices=["watch", "run", "deep"],
    default="watch",
    help="Sampling profile.",
)
parser.add_argument(
    "--mode-profile",
    choices=["watch", "run", "deep"],
    default=None,
    help=argparse.SUPPRESS,  # DEPRECATED alias for --profile, drop in 0.4.0
)

# After parsing:
if args.mode_profile is not None:
    if args.profile != "watch":  # user passed both
        parser.error(
            "Cannot specify both --profile and --mode-profile. "
            "Use --profile (--mode-profile is deprecated)."
        )
    print(
        "[TraceML] WARNING: --mode-profile is deprecated; "
        "use --profile. The old flag will be removed in 0.4.0.",
        file=sys.stderr,
    )
    args.profile = args.mode_profile
```

`argparse.SUPPRESS` keeps the old flag working but hides it from
`--help`. The warning goes to stderr with the standard `[TraceML]`
prefix per [principles.md §4](principles.md).

**Avoid short-flag collisions.** If the old flag was `-m` and the new
flag wants to be `-m` for a different meaning: don't. Pick a different
short flag, or skip short flags during dual-read.

### 6.5. Env var — read both, prefer new

The canonical precedent in the codebase:
`TRACEML_UI_MODE` (new) / `TRACEML_MODE` (old). Read sites at
`aggregator/aggregator_main.py:94-95`, `runtime/executor.py:213-214`,
`final_summary_protocol.py:111-112`. The pattern:

```python
mode = os.environ.get(
    "TRACEML_UI_MODE",
    os.environ.get("TRACEML_MODE", "cli"),
)
```

Two-argument `os.environ.get` makes the precedence explicit: prefer the
new name; if absent, fall back to the old; if both are absent, default.
Do this consistently at every read site — leaving one site reading only
the old name reintroduces the bug.

When you bump major and drop the old name, simplify to:

```python
mode = os.environ.get("TRACEML_UI_MODE", "cli")
```

### 6.6. Diagnosis JSON — additive plus schema_version

Adding a new field is straightforward — see
[add_diagnostic.md §6](add_diagnostic.md). Removing or renaming a field
is a major bump:

1. Add new field with new name, keep old field populated, dual-emit for
   one release.
2. Bump `final_summary.json::schema_version` from 1 to 2 in the
   dual-emit release. Update `compare/io.py` to accept both `1` and `2`.
3. In the major bump, drop the old field. Confirm `compare/io.py` still
   accepts version 1 inputs (read-only — never write version 1) for
   one release after the major bump, then drop version 1 reading.

The double-window (dual-emit minor + major + still-read-old major)
exists because TraceOpt's regression detector ingests historical
summaries. A 12-month-old summary on disk must still be parseable by a
detector that's been updated multiple times since the summary was
written.

---

## 7. Versioning policy

[principles.md §7](principles.md) is the cross-cutting table. This
section refines it for wire-format changes specifically.

| Change type                                                        | Version bump | Notes                                                     |
|---------------------------------------------------------------------|--------------|-----------------------------------------------------------|
| Additive only (new key, new column, new flag, new diagnosis field) | Patch        | Use the per-feature guide; not this one.                  |
| Additive that changes `traceml compare`'s interpretation of adjacent versions | Minor   | E.g. adding a new column that compare relies on.           |
| Non-additive — dual-emit / dual-read phase                          | Minor        | This is the "breaking change in progress" release.        |
| Non-additive — drop-old phase                                      | Major        | This is the actual break; CHANGELOG entry mandatory.       |
| SQLite table rename                                                | Major        | "Create new, copy, drop old" migration required.           |
| Diagnosis `kind` rename                                            | Major        | `schema_version` bump in `final_summary.json` required.    |
| `final_summary.json` shape change (field rename or removal)        | Major        | `schema_version` bump in `final_summary.json` required.    |
| CLI flag removal (after deprecation cycle)                         | Major        | CHANGELOG migration note.                                 |
| Env var removal (after deprecation cycle)                          | Major        | CHANGELOG migration note.                                 |
| `TimeEvent.name` rename — drop-old phase                           | Major        | Diagnosis engine updates landed in same release.          |

The minor-bump-for-dual-emit rule encodes "this is a breaking change in
progress." Users who upgrade across the dual-emit minor see a
deprecation warning. Users who skip the dual-emit minor entirely (jump
from 0.2.5 to 0.4.0) get a hard break — that's the cost of skipping
deprecation cycles, and it's documented.

The major-bump-for-drop-old rule encodes "the old name is gone." This
is the only place TraceML breaks compatibility unannounced from the
producer side; users who didn't notice the deprecation warning find out
now.

---

## 8. Testing migrations

Testing migrations is its own discipline. The standard sampler /
projection / diagnostic tests aren't enough.

### 8.1. The five mandatory tests

For any wire-format PR, write or update the following:

1. **Round-trip test for the new shape.** Producer emits, consumer
   decodes correctly. Standard test pattern; copy from
   `tests/test_msgpack_roundtrip.py`.
2. **Round-trip test for the old shape.** During the dual-emit window:
   producer (with dual-emit on) emits, consumer decodes correctly. After
   the drop-old major bump: synthetic old-shape payload, consumer reads
   correctly via fallback OR raises a clear error (depending on phase).
3. **SQLite migration test (for SQLite changes).** Open a synthetic
   v0.2.5 DB, run `init_schema`, verify the new column exists with NULL
   default, verify old data is intact. Run `init_schema` a second time;
   verify it's a no-op (idempotency).
4. **Cross-version `traceml compare` test.** Two synthetic
   `final_summary.json` files: one in old shape, one in new. Assert
   `compare` does not crash. Spot-check that fields present in both are
   compared correctly.
5. **PyPI install path test.** This one isn't a unit test, it's a
   shell ritual:

   ```bash
   python -m venv /tmp/v025_env
   source /tmp/v025_env/bin/activate
   pip install traceml-ai==0.2.5
   traceml watch examples/<small example>.py --mode summary
   # captures: logs/<session>/aggregator/telemetry.sqlite,
   #           logs/<session>/final_summary.json
   pip install -e ./traceml/  # the new branch
   traceml compare logs/<old_session>/final_summary.json \
                   logs/<new_session>/final_summary.json
   # expect: no crash, diff makes sense
   sqlite3 logs/<old_session>/aggregator/telemetry.sqlite \
     "SELECT COUNT(*) FROM step_time_samples;"
   # expect: existing rows readable
   ```

   Document the output in the PR description.

### 8.2. Test template — SQLite column-add migration

```python
# tests/test_sqlite_migration_step_phase.py
import sqlite3
from pathlib import Path

import pytest


def _build_v025_step_time_table(conn):
    """Reproduce the pre-migration schema by hand."""
    conn.execute(
        """
        CREATE TABLE step_time_samples (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            recv_ts_ns    INTEGER NOT NULL,
            rank          INTEGER,
            sample_ts_s   REAL,
            seq           INTEGER,
            step          INTEGER,
            events_json   TEXT NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO step_time_samples "
        "(recv_ts_ns, rank, sample_ts_s, seq, step, events_json) "
        "VALUES (?, ?, ?, ?, ?, ?);",
        (1_700_000_000_000_000_000, 0, 1.0, 1, 1, "{}"),
    )
    conn.commit()


class TestStepPhaseMigration:
    def test_init_schema_adds_new_column_to_v025_db(self, tmp_path):
        db = tmp_path / "telemetry.sqlite"
        with sqlite3.connect(db) as conn:
            _build_v025_step_time_table(conn)

        from traceml.aggregator.sqlite_writers.step_time import init_schema

        with sqlite3.connect(db) as conn:
            init_schema(conn)
            cur = conn.execute(
                "PRAGMA table_info(step_time_samples);"
            )
            cols = {row[1] for row in cur.fetchall()}
        assert "step_phase" in cols

    def test_init_schema_idempotent(self, tmp_path):
        db = tmp_path / "telemetry.sqlite"
        from traceml.aggregator.sqlite_writers.step_time import init_schema

        with sqlite3.connect(db) as conn:
            init_schema(conn)
            init_schema(conn)  # must not raise
            init_schema(conn)
            cur = conn.execute("PRAGMA table_info(step_time_samples);")
            cols = [row[1] for row in cur.fetchall()]
        # No duplicate columns.
        assert len(cols) == len(set(cols))

    def test_existing_data_preserved(self, tmp_path):
        db = tmp_path / "telemetry.sqlite"
        with sqlite3.connect(db) as conn:
            _build_v025_step_time_table(conn)

        from traceml.aggregator.sqlite_writers.step_time import init_schema

        with sqlite3.connect(db) as conn:
            init_schema(conn)
            cur = conn.execute(
                "SELECT seq, step FROM step_time_samples;"
            )
            rows = cur.fetchall()
        assert rows == [(1, 1)]
```

### 8.3. Test template — sampler payload dual-emit

```python
def test_sampler_emits_both_old_and_new_keys(monkeypatch, tmp_path):
    """During dual-emit, producer emits both names."""
    monkeypatch.setenv("TRACEML_LOGS_DIR", str(tmp_path))
    monkeypatch.setenv("TRACEML_SESSION_ID", "t")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    from traceml.samplers.foo_sampler import FooSampler  # hypothetical

    s = FooSampler()
    s.sample()
    rows = list(s.db.get_table("FooTable"))
    assert len(rows) == 1
    row = rows[0]
    assert "cpu_pct" in row     # new key present
    assert "cpu" in row         # old key still present
    assert row["cpu_pct"] == row["cpu"]  # same value


def test_consumer_reads_old_shape(monkeypatch):
    """Old-only payload (mimics v0.2.5 client) parses via fallback."""
    old_shape = {"seq": 1, "ts": 1.0, "cpu": 42.0}  # no cpu_pct
    from traceml.renderers.foo_renderer import build_label

    label = build_label(old_shape)
    assert "42.0" in label  # consumer used .get() fallback
```

### 8.4. Test template — cross-version compare

```python
def test_compare_handles_old_and_new_summary(tmp_path):
    """traceml compare doesn't crash on a v0.2.5 vs 0.3.0 pair."""
    import json

    old = tmp_path / "v025.json"
    old.write_text(json.dumps({
        "schema_version": 1,
        "diagnosis": {"step_time": {
            "kind": "step_time", "severity": "info", "status": "stable",
            "reason": "...", "action": "...",
        }},
    }))
    new = tmp_path / "v030.json"
    new.write_text(json.dumps({
        "schema_version": 1,
        "diagnosis": {"step_time": {
            "kind": "step_time", "severity": "info", "status": "stable",
            "reason": "...", "action": "...",
            "new_field": 1.23,   # additive
        }},
    }))

    from traceml.compare.command import compare_summaries
    out = tmp_path / "out"
    compare_summaries(left=str(old), right=str(new), output_dir=str(out))
    assert out.exists()
```

---

## 9. Common pitfalls

Numbered, with symptom and fix. If you hit one of these, check here
first.

1. **Symptom:** New release ships, v0.2.5 users report `KeyError` when
   their logs are viewed by the new aggregator.
   **Cause:** A read site dropped the fallback to the old key
   prematurely. The producer side dual-emit was correct, the consumer
   side wasn't.
   **Fix:** Audit every read site. The grep is `grep -rn "<old_key>" src/`.
   Every match must use `.get(old_key)` with a default, never
   `payload[old_key]`.

2. **Symptom:** Dual-emit window shipped, but a v0.2.5 client's payloads
   silently disappear from the dashboard.
   **Cause:** Aggregator reads only the new key, finds it missing,
   skips the row.
   **Fix:** Same as #1 — fallback to old key on the read side.

3. **Symptom:** `_ensure_column` raises on a DB that's already been
   migrated.
   **Cause:** Plain `ALTER TABLE ... ADD COLUMN` without the
   `PRAGMA table_info` check. SQLite returns
   `duplicate column name` if the column exists.
   **Fix:** Use the idempotent helper in §6.3. Test that
   `init_schema(conn); init_schema(conn)` is a no-op the second time.

4. **Symptom:** `traceml compare v0.2.5_summary.json v0.3.0_summary.json`
   crashes with `KeyError`.
   **Cause:** The compare engine reads a key that exists in 0.3.0 but
   not 0.2.5, or vice versa.
   **Fix:** Update `compare/io.py` and `compare/core.py` to use `.get()`
   with defaults for any field whose presence depends on version.

5. **Symptom:** TraceOpt regression detector starts failing to detect
   regressions after a TraceML upgrade.
   **Cause:** A diagnosis `kind` was renamed without a major bump and
   without coordination with TraceOpt-side ingest.
   **Fix:** Diagnosis `kind` renames are major bumps. Coordinate with
   TraceOpt's ingest update before merging the major bump.

6. **Symptom:** A new CLI flag's short form (`-m`) collides with an
   existing flag's short form.
   **Cause:** Two `add_argument` calls registered the same short flag.
   `argparse` raises at parser-build time.
   **Fix:** Pick a different short flag, or skip the short form for the
   new flag. Don't repurpose an existing short flag.

7. **Symptom:** Env var renamed at the source, but a stale read site in
   `final_summary_protocol.py` still reads only the old name.
   **Cause:** Env vars don't have a single read site — every consumer
   reads via `os.environ.get`. One missed site means the new env var is
   ignored from that path.
   **Fix:** `grep -rn TRACEML_<old_name> src/` and update every match
   to the dual-read pattern (§6.5).

8. **Symptom:** SQLite migration runs successfully on the dev box but
   breaks on a user with a slightly older SQLite version.
   **Cause:** You used a SQLite feature (e.g. `ALTER TABLE ... DROP
   COLUMN`) that wasn't in the user's SQLite library. CPython bundles
   SQLite, so this is rare but possible on older Pythons.
   **Fix:** Stick to the "create new, copy, drop old" pattern in §6.3.
   Avoid `DROP COLUMN`. Test against the lowest Python version in
   `pyproject.toml::requires-python`.

9. **Symptom:** `final_summary.json` shape changed but
   `schema_version` wasn't bumped.
   **Cause:** The schema-version bump is an unwritten convention; it's
   easy to forget. `compare/io.py:88-90` will silently mis-parse without
   raising if the bump was skipped.
   **Fix:** Every PR that changes `final_summary.json` shape bumps
   `schema_version`. Add it to your PR checklist (§10). Long-term fix:
   schema-version bumps should be enforced by a test that diffs the
   JSON shape against a frozen golden, but that test doesn't exist yet
   (§11).

10. **Symptom:** Major version bumped on a non-additive change, but
    CHANGELOG doesn't have a `BREAKING:` entry.
    **Cause:** The CHANGELOG discipline from
    [principles.md §7](principles.md) is new; it's easy to forget on a
    quick PR.
    **Fix:** Every wire-format PR adds a CHANGELOG line. Non-additive
    changes use the `BREAKING:` prefix. The reviewer (§10) checks this
    explicitly.

11. **Symptom:** Dual-emit ships, hot-path overhead doubles, training
    runs report 2× the prior overhead during the deprecation window.
    **Cause:** Two `timed_region` blocks where there used to be one
    (§4 step 1). The cost is genuine.
    **Fix:** Either (a) accept the overhead during the deprecation
    window and document it in the CHANGELOG, or (b) modify
    `timing.py::timed_region` to accept a list of names and emit
    multiple `TimeEvent`s from a single timing pair. Option (b) is
    itself a wire-format change to the buffered event shape — apply
    this guide recursively. Don't ship a 2× hot-path regression
    silently.

12. **Symptom:** The drop-old PR (major bump) lands but breaks a
    previously-shipped 0.3.x dual-emit consumer that was reading only
    the old key.
    **Cause:** A consumer site was missed during the dual-emit minor
    bump. Now the producer no longer emits the old key, and the missed
    site is the only one that reads it.
    **Fix:** Re-run the audit grep before the major bump:
    `grep -rn "<old_key>" src/` should return zero hits. The grep is
    the gate.

13. **Symptom:** A new run's `final_summary.json` claims
    `schema_version: 2` but `compare` complains it doesn't know about
    schema 2.
    **Cause:** `compare/io.py` was bumped to write schema 2 but not to
    read it. The reader and writer are in different files; both need
    updating.
    **Fix:** Update `compare/io.py` to accept both schema 1 and 2 in
    the dual-emit minor. Drop schema 1 reading in a later major bump,
    not the same one.

14. **Symptom:** A v0.2.5 user `pip install -U`s to 0.3.0 and the
    dashboard appears empty for the first few seconds.
    **Cause:** The v0.2.5 producer is shipping old-shape payloads; the
    new aggregator's read site has the fallback. This is *correct*
    behavior, not a pitfall — but smoke-test it before assuming.
    **Fix:** Confirm via SQLite query that v0.2.5 producer's data is
    actually landing. If it isn't, you've missed a fallback.

15. **Symptom:** Major bump shipped, downstream TraceOpt regression
    detector throws on every old summary it ingests.
    **Cause:** TraceOpt was not updated before the TraceML major bump.
    **Fix:** Every TraceML major version bump that changes
    `final_summary.json` shape must coordinate with TraceOpt. The
    coordinator-review checkbox in §10 is the gate.

---

## 10. PR checklist

Heavier than the per-feature guides because the blast radius is bigger.
Copy this into the PR description.

### Before opening

- [ ] **Identified which of the five wire surfaces this change touches.**
      One or more of: sampler payload schema, `TimeEvent.name`, SQLite
      projection schema, CLI surface, diagnosis JSON.
- [ ] **Documented in the PR description: is this additive or
      non-additive?** If additive, this guide is not the right one — use
      the per-feature guide.
- [ ] **For non-additive changes:** dual-emit / dual-read phase planned
      for at least one full minor version. The PR you are opening is
      either the dual-emit PR (minor bump) or the drop-old PR (major
      bump) — not both.
- [ ] **Documented the migration path from v0.2.5 to the new version.**
      Step-by-step what a user does. Include in the PR description.
- [ ] **Counted blast radius:** `grep -rn "<old_name>" src/ tests/
      examples/ docs/` — number of matches noted in the PR description.
      If non-zero on a drop-old PR, fix the matches first.

### Code

- [ ] If touching sampler payload: schema dataclass updated, `to_wire()`
      emits both old and new keys (dual-emit) or only new (drop-old).
- [ ] If touching `TimeEvent.name`: every `timed_region(name=...)` call
      site at the producer updated; aggregation key in
      `step_time_sampler.py` updated; diagnosis engine updated;
      renderer label table updated.
- [ ] If touching SQLite: `init_schema` is idempotent (`PRAGMA
      table_info` check, not raw `ALTER`). `build_rows` populates both
      old and new columns during dual-write.
- [ ] If touching CLI flag: new flag added, old flag kept (with
      `argparse.SUPPRESS`), at-most-one validation, deprecation
      warning to stderr with `[TraceML]` prefix.
- [ ] If touching env var: every read site updated to the
      `os.environ.get(NEW, os.environ.get(OLD, default))` precedent.
      Audited via `grep -rn TRACEML_<old> src/`.
- [ ] If touching diagnosis JSON: new field added; if renaming or
      removing, `schema_version` in `final_summary.json` bumped;
      `compare/io.py` accepts both old and new versions.

### Tests

- [ ] Test for the new shape (round-trip).
- [ ] Test for the old shape (round-trip via fallback path).
- [ ] If SQLite: test for `init_schema` idempotency.
- [ ] If SQLite: test for migration of synthetic v0.2.5 DB.
- [ ] Test for `traceml compare` cross-version (old summary vs new).
- [ ] All `pytest` tests pass: `pytest tests/`.

### Smoke

- [ ] Local smoke test per [principles.md §6](principles.md):
      `pip install -e ".[dev,torch]"` then
      `traceml watch examples/<small example>.py --mode cli`. Training
      completes, no stack traces on stderr, the live UI shows the
      change is in effect.
- [ ] **PyPI upgrade-path test:** clean
      `pip install traceml-ai==0.2.5` in a fresh venv, capture a run's
      summary, then `pip install -e ./traceml/`, `traceml compare` the
      old summary against a new run. No crash. Output captured in the
      PR description.
- [ ] Multi-rank smoke if relevant:
      `traceml watch examples/<small>.py --nproc-per-node 2`. No
      duplicate host metrics; per-rank rows still distinct.

### Versioning

- [ ] Version bump in `pyproject.toml` matches §7. Dual-emit phase →
      minor bump. Drop-old phase → major bump.
- [ ] CHANGELOG entry added. Non-additive: `BREAKING:` prefix; lists
      the migration path. Dual-emit: `Deprecated:` prefix; names the
      replacement and the planned drop version.
- [ ] If `final_summary.json` shape changes: `schema_version` bumped
      in `final_summary.py:206` and confirmed in `compare/io.py:88`.

### Coordination

- [ ] **Coordinator review.** This PR is read by both project
      maintainers (Abhinav and Abhijeet). Wire-format breaks need
      two-person sign-off. Flagged at PR-open time, not at merge time.
- [ ] If touching diagnosis JSON: TraceOpt-side ingest impact
      documented (even if TraceOpt doesn't yet exist; future-you will
      thank present-you for the note).

### Hygiene

- [ ] `pre-commit run --all-files` clean (black, ruff, isort,
      codespell).
- [ ] Commit messages: short, single-line, no `Co-Authored-By` trailer
      per repo `CLAUDE.md`.

---

## 11. Appendix

### 11.1. The `schema_version` table that doesn't exist yet

TraceML's SQLite database has no internal version tracking. There is no
`schema_version` table inside the SQLite file, and `PRAGMA user_version`
is unset (defaults to 0). The string `schema_version: 1` you'll find
via grep refers to JSON file schemas (`final_summary.json`,
manifests) — not the on-disk SQLite database.

The proposed fix (out of scope for any single wire-format PR but worth
knowing): a one-row metadata table inserted by `init_schema`:

```sql
CREATE TABLE IF NOT EXISTS traceml_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT OR IGNORE INTO traceml_meta (key, value)
VALUES ('schema_version', '1');
```

Or, lighter-weight, `PRAGMA user_version = 1;` set in `init_schema`.
Either approach gives the aggregator a way to detect on-disk schema
version at startup and trigger version-specific migrations. Until this
ships, the discipline is "all migrations are idempotent and check for
their own work via `PRAGMA table_info`."

### 11.2. The `PRAGMA user_version` pattern

If you want to introduce versioned migrations without adding a metadata
table:

```python
def get_schema_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version;").fetchone()[0]


def set_schema_version(conn: sqlite3.Connection, v: int) -> None:
    conn.execute(f"PRAGMA user_version = {int(v)};")


def init_schema(conn: sqlite3.Connection) -> None:
    current = get_schema_version(conn)
    if current < 1:
        # initial create
        conn.execute("CREATE TABLE IF NOT EXISTS step_time_samples (...)")
        set_schema_version(conn, 1)
    if current < 2:
        _ensure_column(conn, "step_time_samples", "step_phase", "INTEGER")
        set_schema_version(conn, 2)
    # ... etc
```

The advantage over `PRAGMA table_info` checks: explicit migration paths
are code-reviewable as a sequence. The disadvantage: introducing
`user_version` for the first time on already-deployed v0.2.5 DBs sees
`current == 0`, which means "run all migrations." That's the right
behavior, but it has to be the first migration's design assumption.

### 11.3. Why TraceML doesn't have an explicit "v1.0 wire format" yet

The five wire surfaces are convention-stable, not contract-stable. There
is no schema file, no protobuf, no IDL. The agreement that
`_traceml_internal:forward_time` is contract is made in code review, not
in a tracked schema definition.

The cost: every wire-format change is a manual audit of consumers.
The benefit: zero serialization overhead beyond msgpack, full Python
flexibility, no IDL maintenance.

Shipping a "v1.0 wire format" would mean:

- A versioned schema definition for sampler payloads (one per sampler).
- A versioned schema for `TimeEvent`.
- An explicit SQLite schema version.
- An explicit `final_summary.json` schema (already partially present
  via `schema_version: 1`).
- An explicit CLI compatibility manifest documenting which flags and
  env vars are stable.

This is a roadmap item, not a current capability. When >50% of TraceML
deployments are heterogeneous (different versions on producer and
aggregator side), the lack of versioned schemas will hurt — that's
the trigger to invest.

### 11.4. The TraceOpt-as-downstream-consumer constraint

[`traceml_why.md`](../deep_dive/why.md) §6.4 documents the
longitudinal regression detection product TraceOpt is targeting. The
implication for this guide: any wire change that affects
`final_summary.json` is also a TraceOpt-side change.

The discipline:

- Diagnosis `kind` is a hard contract; renames are coordinated
  releases between TraceML and TraceOpt.
- Diagnosis `severity` levels are a contract — adding new levels is
  additive, removing or renaming is a major bump.
- New diagnosis fields are additive and don't need TraceOpt
  coordination, but TraceOpt should be updated to *use* the new field
  in the next release cycle.

When in doubt: if the change affects what `cat final_summary.json |
jq '.diagnosis'` looks like, ask whether TraceOpt is reading that
field. Today the answer is "TraceOpt doesn't exist yet"; tomorrow the
answer is "yes, and it has 6 months of historical data depending on
this shape."

### 11.5. Cross-references

- [principles.md](principles.md) §3 — wire-compatibility rules
  (canonical statement).
- [principles.md](principles.md) §7 — versioning and CHANGELOG.
- [add_sampler.md](add_sampler.md) §6 — sampler-side schema rules
  (additive).
- [add_sqlite_projection.md](add_sqlite_projection.md) §6 —
  SQLite-side additivity rules.
- [add_cli.md](add_cli.md) §6 — env-var compat rules.
- [add_diagnostic.md](add_diagnostic.md) §6 — diagnosis JSON shape.
- [review_patch.md](review_patch.md) §5.3 —
  wire-name-as-contract reasoning.
- [pipeline_walkthrough.md](pipeline_walkthrough.md) — end-to-end
  data flow (where each wire surface lives in the pipeline).
- [`traceml_why.md`](../deep_dive/why.md) §6.4 —
  longitudinal regression detection (strategic context).
- [W6](../deep_dive/code-walkthroughs.md#w6-samplers-schemas-turning-hook-events-into-structured-rows)
  through
  [W11](../deep_dive/code-walkthroughs.md#w11-summaries-diagnostics-end-of-run-analysis)
  — deep walkthroughs of each wire-affecting subsystem.

---

## 12. Gaps and ambiguities encountered while writing this guide

Things this guide does not yet pin down. Flag in code review if your PR
lands near them.

- **No `schema_version` inside SQLite.** §11.1. The discipline is
  idempotent migrations via `PRAGMA table_info`. A first-class version
  tracker (table or `user_version`) would make migrations
  forward-detectable rather than only idempotent. Out of scope for any
  single wire-format PR; in scope for the v1.0 hardening pass.

- **No formal test that locks the wire shape.** A "golden" test that
  freezes the current sampler payload shapes and fails on any
  unintentional change would catch a lot of the pitfalls in §9. Today
  the only enforcement is reviewer attention. The benchmark workflow at
  v0.2.9 is the formal artifact for overhead; nothing analogous exists
  for wire shape.

- **No migrations directory.** §6.3 references
  `src/traceml/aggregator/migrations/` for one-shot scripts. That
  directory does not exist in the codebase as of 2026-04-25. The first
  destructive-migration PR creates it.

- **Dual-emit overhead is folklore.** §6.2 notes that two
  `timed_region` blocks double the hot-path cost. The 5 µs hot-path
  budget in [principles.md §5](principles.md) is itself folklore — no
  test gates it. A wire-format PR that doubles hot-path cost during a
  deprecation window is technically inside the budget by the definition
  of "folklore," but in practice the v0.2.9 benchmark workflow needs to
  prove the doubled cost is still acceptable. Until that workflow gates
  PRs, reviewer judgment is the gate.

- **TraceOpt-side coordination has no tracked process.** §11.4
  describes the constraint; there is no checklist item, no shared
  schema repository, no automated CI between TraceML and TraceOpt. The
  coordinator-review checkbox in §10 is the only enforcement, and it
  works only as long as both projects share both maintainers.

- **CLI deprecation cycle length is not codified.** §2 says "one full
  minor version" by default; in practice some env-var renames have been
  live longer (`TRACEML_UI_MODE` superseded `TRACEML_MODE` and the old
  name is still read at three sites today). There is no rule about when
  the deprecation cycle officially ends — the major bump that drops the
  old name is the de-facto end. A formal deprecation-cycle tracker
  (date introduced, planned removal version) would help.

- **`PRAGMA user_version` migration on an already-deployed DB.**
  §11.2's pattern assumes a virgin v0.2.5 DB has `user_version = 0`,
  which is true (it's the SQLite default). But a user who upgraded
  through some intermediate dev branch could have an arbitrary
  `user_version`. The "idempotent and self-describing" discipline of
  §6.3 is more robust; the `user_version` pattern is the right
  long-term answer once the v1.0 schema is locked.

- **Compare-tool's tolerance for missing fields is not tested
  exhaustively.** `tests/test_compare_missing.py` covers some cases
  but not the full matrix of "old summary × new aggregator," "new
  summary × old aggregator," "summary missing field × renamed field."
  A wire-format PR that touches `final_summary.json` should expand this
  test file; today it's easy to forget because the test file isn't
  named "wire compatibility."
