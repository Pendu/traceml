"""Baseline harness: run a fixture WITHOUT TraceML, capture wall-clock,
peak GPU memory (per-GPU), and peak host RSS via background polling.

The v2 doc requires baseline mode to produce comparable metrics so that
overhead % = (t_traceml - t_baseline) / t_baseline can be computed and
peak GPU mem / RSS deltas vs baseline can be reported. TraceML's own
final_summary.json supplies the same metrics in `traceml run` mode, so
this harness is only used for `baseline` mode.

Usage (typically invoked by run_v2_matrix.py, not by hand):
    python bench/baseline_harness.py \\
        --script examples/advanced/gpt2_small_synthetic.py \\
        --output bench_output_2xl4/gpt2_wikitext/baseline/trial_0/baseline_metrics.json \\
        --workload gpt2_wikitext \\
        --trial 0 \\
        [--nproc-per-node 2]

Multi-rank workloads use torch.distributed.run for parity with how
TraceML's CLI launches them.

Per-step median is intentionally NOT captured for baseline — that would
require fixture-side timing instrumentation. Only TraceML mode reports
per-step ms; baseline only reports wall + peak memory metrics.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pynvml

    pynvml.nvmlInit()
    HAS_PYNVML = True
    NUM_GPUS = pynvml.nvmlDeviceGetCount()
except Exception:
    HAS_PYNVML = False
    NUM_GPUS = 0


class MetricsPoller(threading.Thread):
    """Background thread that polls process-tree RSS + per-GPU memory
    every `interval` seconds and tracks peaks until stopped.

    GPU memory is measured system-wide (nvmlDeviceGetMemoryInfo.used)
    rather than filtered to the target PID — simpler and matches what
    TraceML's system sampler reports as `gpu_rollup.mem_peak_gb`.
    """

    def __init__(self, pid, interval=0.5):
        super().__init__(daemon=True)
        self.pid = pid
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak_rss_bytes = 0
        self.peak_gpu_mem_bytes = [0] * max(NUM_GPUS, 1)
        if HAS_PYNVML:
            self._handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(NUM_GPUS)
            ]
        else:
            self._handles = []

    def run(self):
        proc = None
        if HAS_PSUTIL:
            try:
                proc = psutil.Process(self.pid)
            except psutil.NoSuchProcess:
                return
        while not self.stop_event.is_set():
            if proc is not None:
                try:
                    rss = proc.memory_info().rss
                    for child in proc.children(recursive=True):
                        try:
                            rss += child.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
                except psutil.NoSuchProcess:
                    break
                except psutil.AccessDenied:
                    pass
            for i, h in enumerate(self._handles):
                try:
                    info = pynvml.nvmlDeviceGetMemoryInfo(h)
                    self.peak_gpu_mem_bytes[i] = max(
                        self.peak_gpu_mem_bytes[i], info.used
                    )
                except Exception:
                    pass
            self.stop_event.wait(self.interval)

    def stop(self):
        self.stop_event.set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--trial", type=int, required=True)
    ap.add_argument("--nproc-per-node", type=int, default=1)
    args, passthrough = ap.parse_known_args()

    if args.nproc_per_node > 1:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={args.nproc_per_node}",
            args.script,
            *passthrough,
        ]
    else:
        cmd = [sys.executable, args.script, *passthrough]

    started_at = time.time()
    started_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at))

    proc = subprocess.Popen(cmd, env=os.environ.copy())
    poller = MetricsPoller(proc.pid)
    poller.start()

    rc = proc.wait()
    poller.stop()
    poller.join(timeout=2)

    wall_s = time.time() - started_at

    metrics = {
        "workload": args.workload,
        "mode": "baseline",
        "trial": args.trial,
        "script": args.script,
        "nproc_per_node": args.nproc_per_node,
        "started_at": started_iso,
        "wall_s": round(wall_s, 3),
        "peak_rss_bytes": poller.peak_rss_bytes,
        "peak_gpu_mem_bytes_per_gpu": poller.peak_gpu_mem_bytes,
        "exit_code": rc,
        "psutil_available": HAS_PSUTIL,
        "pynvml_available": HAS_PYNVML,
        "num_gpus": NUM_GPUS,
    }

    out_path = args.output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"[baseline_harness] {args.workload} trial={args.trial} "
        f"wall={wall_s:.2f}s rc={rc} "
        f"peak_rss={poller.peak_rss_bytes / 1e9:.2f}GB "
        f"peak_gpu={[round(b / 1e9, 2) for b in poller.peak_gpu_mem_bytes]}GB"
    )

    sys.exit(rc)


if __name__ == "__main__":
    main()
