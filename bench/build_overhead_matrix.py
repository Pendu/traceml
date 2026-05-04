"""Aggregate per-trial outputs into the v2 overhead matrix.

For each (workload, mode), aggregates wall_s / step_avg_ms /
peak_gpu_mem_gb / peak_rss_gb across trials (mean +/- std). Then joins
baseline and traceml_run rows per workload to compute:

    overhead_pct       = (t_traceml - t_baseline) / t_baseline * 100
    peak_gpu_delta_mb  = peak_gpu_traceml - peak_gpu_baseline
    peak_rss_delta_mb  = peak_rss_traceml - peak_rss_baseline

Outputs:
    bench/overhead_matrix.csv   one row per (workload, mode)
    bench/report.md             v2-doc-shaped overhead table

Stdlib only.
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
from workloads import WORKLOADS  # noqa: E402


def load_traceml_metrics(p: Path) -> dict:
    with open(p) as f:
        d = json.load(f)
    st = d.get("step_time", {})
    sysd = d.get("system", {})
    return {
        "wall_s": d.get("duration_s"),
        "step_avg_ms": (
            st.get("global", {}).get("typical", {}).get("step_avg_ms")
            or st.get("timing_primary", {}).get("step_avg_ms")
        ),
        "training_steps": (
            st.get("overview", {}).get("training_steps")
            or st.get("training_steps", 0)
        ),
        "peak_gpu_mem_gb": (
            sysd.get("global", {}).get("gpu_rollup", {}).get("mem_peak_gb")
        ),
        "peak_rss_gb": (sysd.get("global", {}).get("ram", {}).get("peak_gb")),
    }


def load_baseline_metrics(p: Path) -> dict:
    with open(p) as f:
        d = json.load(f)
    peak_gpu_bytes = d.get("peak_gpu_mem_bytes_per_gpu") or []
    peak_gpu_gb = max(peak_gpu_bytes) / 1e9 if peak_gpu_bytes else None
    peak_rss_gb = (d.get("peak_rss_bytes") or 0) / 1e9 or None
    return {
        "wall_s": d.get("wall_s"),
        "step_avg_ms": None,
        "training_steps": None,
        "peak_gpu_mem_gb": peak_gpu_gb,
        "peak_rss_gb": peak_rss_gb,
    }


def aggregate(per_trial: list[dict]) -> dict:
    out = {}
    keys = ("wall_s", "step_avg_ms", "peak_gpu_mem_gb", "peak_rss_gb")
    for k in keys:
        vals = [t[k] for t in per_trial if t.get(k) is not None]
        if not vals:
            continue
        out[k + "_mean"] = round(statistics.mean(vals), 4)
        out[k + "_std"] = (
            round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
        )
        out[k + "_n"] = len(vals)
    return out


def discover_rows(bench_root: Path) -> dict:
    """Walk bench_root/<workload>/<mode>/trial_<n>/ -> aggregated dict."""
    rows: dict[tuple[str, str], dict] = {}
    for w_dir in sorted(bench_root.iterdir()):
        if not w_dir.is_dir():
            continue
        workload = w_dir.name
        for m_dir in sorted(w_dir.iterdir()):
            if not m_dir.is_dir():
                continue
            mode = m_dir.name
            per_trial = []
            for t_dir in sorted(m_dir.iterdir()):
                if not t_dir.is_dir():
                    continue
                if mode == "traceml_run":
                    p = t_dir / "output.json"
                    if p.exists():
                        per_trial.append(load_traceml_metrics(p))
                else:
                    p = t_dir / "baseline_metrics.json"
                    if p.exists():
                        per_trial.append(load_baseline_metrics(p))
            if per_trial:
                rows[(workload, mode)] = {
                    "n_trials": len(per_trial),
                    **aggregate(per_trial),
                }
    return rows


def write_csv(rows: dict, path: Path) -> None:
    fieldnames = [
        "workload",
        "mode",
        "n_trials",
        "wall_s_mean",
        "wall_s_std",
        "step_avg_ms_mean",
        "step_avg_ms_std",
        "peak_gpu_mem_gb_mean",
        "peak_gpu_mem_gb_std",
        "peak_rss_gb_mean",
        "peak_rss_gb_std",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for (workload, mode), agg in sorted(rows.items()):
            w.writerow({"workload": workload, "mode": mode, **agg})


def fmt(v, suffix=""):
    if v is None:
        return "—"
    return f"{v}{suffix}"


def write_report_md(rows: dict, path: Path) -> None:
    workloads = sorted({wn for wn, _ in rows.keys()})
    lines = [
        "# v2 benchmark matrix",
        "",
        "Per (workload, mode), aggregated across trials. Overhead % and",
        "peak deltas are derived by joining baseline vs traceml_run rows.",
        "",
        "| Workload | Mode | N | Wall (s) | Overhead % | Step (ms) "
        "| Peak GPU (GB) | Peak RSS (GB) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for w in workloads:
        b = rows.get((w, "baseline"))
        r = rows.get((w, "traceml_run"))
        label = WORKLOADS.get(w, {}).get("label", w)
        if b:
            lines.append(
                f"| {label} | baseline | {b['n_trials']} | "
                f"{fmt(b.get('wall_s_mean'))} "
                f"±{fmt(b.get('wall_s_std', 0))} | — | — | "
                f"{fmt(b.get('peak_gpu_mem_gb_mean'))} | "
                f"{fmt(b.get('peak_rss_gb_mean'))} |"
            )
        if r:
            ovh = mem_d = rss_d = "—"
            if b:
                tw, bw = r.get("wall_s_mean"), b.get("wall_s_mean")
                if tw is not None and bw and bw > 0:
                    ovh = f"{100 * (tw - bw) / bw:+.2f}%"
                tm, bm = (
                    r.get("peak_gpu_mem_gb_mean"),
                    b.get("peak_gpu_mem_gb_mean"),
                )
                if tm is not None and bm is not None:
                    mem_d = f"{(tm - bm) * 1000:+.0f} MB"
                tr, br = (
                    r.get("peak_rss_gb_mean"),
                    b.get("peak_rss_gb_mean"),
                )
                if tr is not None and br is not None:
                    rss_d = f"{(tr - br) * 1000:+.0f} MB"
            lines.append(
                f"| {label} | traceml_run | {r['n_trials']} | "
                f"{fmt(r.get('wall_s_mean'))} "
                f"±{fmt(r.get('wall_s_std', 0))} | "
                f"{ovh} | "
                f"{fmt(r.get('step_avg_ms_mean'))} "
                f"±{fmt(r.get('step_avg_ms_std', 0))} | "
                f"{fmt(r.get('peak_gpu_mem_gb_mean'))} ({mem_d}) | "
                f"{fmt(r.get('peak_rss_gb_mean'))} ({rss_d}) |"
            )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root", default=str(REPO / "bench_output_2xl4"))
    ap.add_argument(
        "--out-csv", default=str(REPO / "bench/overhead_matrix.csv")
    )
    ap.add_argument("--out-md", default=str(REPO / "bench/report.md"))
    args = ap.parse_args()

    rows = discover_rows(Path(args.bench_root))
    if not rows:
        print(
            f"No per-trial outputs found under {args.bench_root}.\n"
            "Run `python bench/run_v2_matrix.py` first."
        )
        sys.exit(1)

    write_csv(rows, Path(args.out_csv))
    write_report_md(rows, Path(args.out_md))
    print(f"Wrote {args.out_csv}  ({len(rows)} (workload,mode) rows)")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
