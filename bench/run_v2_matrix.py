"""Drive the v2 benchmark matrix: workload x mode x trial.

For each (workload, mode, trial):
    baseline    -> python bench/baseline_harness.py <script> ...
    traceml_run -> traceml run <script> ...

Outputs land in:
    <out_root>/<workload>/<mode>/trial_<n>/{output.json|baseline_metrics.json,
                                            stdout.log}

Workloads come from bench/workloads.py. Modes default to baseline +
traceml_run (the v2 doc's `traceml watch` is out of scope per session
decision).

Usage from repo root:
    python bench/run_v2_matrix.py --trials 3
    python bench/run_v2_matrix.py --trials 5 --workloads gpt2_wikitext,bert_agnews
    python bench/run_v2_matrix.py --modes baseline                # baseline only
    python bench/run_v2_matrix.py --out-root bench_output_a100    # different tier

Re-running is cheap if `--skip-existing` is set: per-trial outputs that
already exist are not re-executed.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
from workloads import WORKLOADS  # noqa: E402


def output_path(out_dir: Path, mode: str) -> Path:
    return out_dir / (
        "baseline_metrics.json" if mode == "baseline" else "output.json"
    )


def run_one(
    name: str,
    spec: dict,
    mode: str,
    trial: int,
    out_root: Path,
    skip_existing: bool,
) -> dict:
    nproc = spec.get("nproc", 1)
    script = spec["script"]
    extra_env = spec.get("extra_env", {})

    session = f"v2_{name}_{mode}_t{trial}"
    out_dir = out_root / name / mode / f"trial_{trial}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_p = output_path(out_dir, mode)
    if skip_existing and out_p.exists():
        return {"session": session, "rc": 0, "skipped": True}

    env = {**os.environ, **{k: str(v) for k, v in extra_env.items()}}

    if mode == "baseline":
        cmd = [
            sys.executable,
            str(REPO / "bench/baseline_harness.py"),
            "--script",
            script,
            "--output",
            str(out_p),
            "--workload",
            name,
            "--trial",
            str(trial),
            "--nproc-per-node",
            str(nproc),
        ]
    elif mode == "traceml_run":
        cmd = [
            "traceml",
            "run",
            script,
            "--mode=summary",
            f"--session-id={session}",
        ]
        if nproc > 1:
            cmd.append(f"--nproc-per-node={nproc}")
    else:
        raise ValueError(f"unknown mode: {mode}")

    print(f"\n[matrix] === {session} ===")
    started = time.time()
    with open(out_dir / "stdout.log", "w") as logf:
        rc = subprocess.run(
            cmd, cwd=REPO, env=env, stdout=logf, stderr=subprocess.STDOUT
        ).returncode
    dur = time.time() - started
    print(f"[matrix]   rc={rc} wall={dur:.1f}s")

    # traceml_run writes its outputs under logs/<session>/ — copy the
    # ones we care about into the bench tree so the analysis layer can
    # read everything from a single place.
    if mode == "traceml_run":
        logs_session = REPO / "logs" / session
        for src_name, dst_name in [
            ("final_summary.json", "output.json"),
            ("final_summary.txt", "output.txt"),
            ("manifest.json", "manifest.json"),
            ("system_manifest.json", "system_manifest.json"),
            ("code_manifest.json", "code_manifest.json"),
        ]:
            src = logs_session / src_name
            if src.exists():
                shutil.copy(src, out_dir / dst_name)

    return {"session": session, "rc": rc, "skipped": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-root",
        default=str(REPO / "bench_output_2xl4"),
        help="per-tier output directory",
    )
    ap.add_argument(
        "--trials",
        type=int,
        default=3,
        help="trials per (workload, mode); v2 doc recommends 5",
    )
    ap.add_argument(
        "--workloads",
        default="",
        help="comma-separated subset (default: all 8)",
    )
    ap.add_argument(
        "--modes",
        default="baseline,traceml_run",
        help="comma-separated modes (default: baseline,traceml_run)",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="don't re-run trials whose output file already exists",
    )
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    keys = list(WORKLOADS.keys())
    if args.workloads:
        wanted = set(args.workloads.split(","))
        keys = [k for k in keys if k in wanted]
        missing = wanted - set(keys)
        if missing:
            print(
                f"[matrix] warning: unknown workloads ignored: "
                f"{sorted(missing)}"
            )
    modes = args.modes.split(",")

    total = len(keys) * len(modes) * args.trials
    print(
        f"[matrix] running {total} trials "
        f"({len(keys)} workloads x {len(modes)} modes x {args.trials} trials)"
    )
    print(f"[matrix] out_root: {out_root}")

    results = []
    matrix_started = time.time()
    for name in keys:
        spec = WORKLOADS[name]
        for mode in modes:
            for trial in range(args.trials):
                results.append(
                    run_one(
                        name, spec, mode, trial, out_root, args.skip_existing
                    )
                )

    matrix_dur = time.time() - matrix_started
    pass_n = sum(1 for r in results if r["rc"] == 0 and not r["skipped"])
    skip_n = sum(1 for r in results if r["skipped"])
    fail_n = sum(1 for r in results if r["rc"] != 0)
    print(
        f"\n[matrix] DONE in {matrix_dur / 60:.1f}min: "
        f"{pass_n} passed, {skip_n} skipped, {fail_n} failed"
    )
    if fail_n:
        print("[matrix] failures:")
        for r in results:
            if r["rc"] != 0:
                print(f"  FAIL {r['session']}  rc={r['rc']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
