"""Parse the matrix runner's log -> write `external_wall.json` sidecars.

Why:
    `traceml run` reports `duration_s` in `final_summary.json`, but that
    timer starts AFTER `traceml.init()` — it excludes Python startup,
    imports, and TraceML's own CLI/aggregator boot. The baseline harness
    measures the FULL subprocess wall (Popen -> exit). Comparing those
    two as-is yields negative "overhead %" for short workloads because
    we're subtracting a smaller scope from a larger one.

The fix:
    Use externally-measured subprocess wall for BOTH modes. The matrix
    runner (`run_v2_matrix.py`) prints `[matrix]   rc=N wall=Xs` after
    each trial. This script parses that log and writes an
    `external_wall.json` sidecar in each trial dir. `build_overhead_matrix`
    prefers this sidecar over `duration_s`/`wall_s`.

Usage (after the matrix finishes):
    python bench/inject_external_walls.py [LOG_PATH] [BENCH_ROOT]

Defaults:
    LOG_PATH    = /tmp/matrix_v5.log
    BENCH_ROOT  = bench_output_2xl4
"""

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_LOG = Path("/tmp/matrix_v5.log")
DEFAULT_BENCH_ROOT = REPO / "bench_output_2xl4"


SESSION_RE = re.compile(r"^\[matrix\] === (v2_\S+) ===\s*$")
RC_WALL_RE = re.compile(r"^\[matrix\]\s+rc=(\d+) wall=([\d.]+)s\s*$")
SESSION_PARSE_RE = re.compile(r"^v2_(.+?)_(baseline|traceml_run)_t(\d+)$")


def parse_log(log_path: Path) -> list[tuple[str, int, float]]:
    """Return list of (session, rc, wall_s) in matrix-run order."""
    pairs: list[tuple[str, int, float]] = []
    cur: str | None = None
    for line in log_path.read_text().splitlines():
        m = SESSION_RE.match(line)
        if m:
            cur = m.group(1)
            continue
        m = RC_WALL_RE.match(line)
        if m and cur:
            pairs.append((cur, int(m.group(1)), float(m.group(2))))
            cur = None
    return pairs


def trial_dir(bench_root: Path, session: str) -> Path | None:
    m = SESSION_PARSE_RE.match(session)
    if not m:
        return None
    workload, mode, trial = m.group(1), m.group(2), int(m.group(3))
    return bench_root / workload / mode / f"trial_{trial}"


def main():
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOG
    bench_root = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_BENCH_ROOT

    if not log_path.exists():
        print(f"ERROR: log not found: {log_path}", file=sys.stderr)
        sys.exit(2)
    if not bench_root.exists():
        print(f"ERROR: bench_root not found: {bench_root}", file=sys.stderr)
        sys.exit(2)

    pairs = parse_log(log_path)
    written = 0
    skipped: list[str] = []
    for session, rc, wall in pairs:
        td = trial_dir(bench_root, session)
        if td is None:
            skipped.append(f"unparsable: {session}")
            continue
        if not td.exists():
            skipped.append(f"no trial dir: {td}")
            continue
        sidecar = td / "external_wall.json"
        sidecar.write_text(
            json.dumps(
                {
                    "session": session,
                    "external_wall_s": wall,
                    "subprocess_rc": rc,
                    "source": "run_v2_matrix.py stdout (parsed)",
                },
                indent=2,
            )
            + "\n"
        )
        written += 1

    print(f"Wrote {written} external_wall.json sidecars under {bench_root}")
    if skipped:
        print(f"Skipped {len(skipped)}:")
        for s in skipped:
            print(f"  {s}")


if __name__ == "__main__":
    main()
