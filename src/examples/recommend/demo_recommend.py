"""
Demo: traceml recommend

Shows how TraceML's recommendation engine analyzes completed training runs
and suggests optimizations. No GPU required -- uses pre-built summary cards.

Usage (CLI):
    traceml recommend src/examples/recommend/dataloader_bottleneck.json
    traceml recommend src/examples/recommend/compute_bottleneck.json
    traceml recommend src/examples/recommend/rank_skew.json
    traceml recommend src/examples/recommend/optimizer_heavy.json
    traceml recommend src/examples/recommend/gpu_underutilized.json
    traceml recommend src/examples/recommend/healthy_run.json

Usage (this script):
    python src/examples/recommend/demo_recommend.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pathlib import Path

from traceml.optimizer.recommend import (
    format_recommendations,
    generate_recommendations,
    run_recommend,
)

EXAMPLES_DIR = Path(__file__).parent

SCENARIOS = [
    ("dataloader_bottleneck.json", "Slow DataLoader (num_workers=0, no prefetch)"),
    ("gpu_underutilized.json", "GPU underutilized with small batch size"),
    ("compute_bottleneck.json", "Compute-bound (no torch.compile, no AMP)"),
    ("optimizer_heavy.json", "Expensive optimizer step (no fused kernels)"),
    ("rank_skew.json", "4-GPU DDP with straggler rank"),
    ("healthy_run.json", "Well-optimized training run"),
]


def main():
    print("=" * 78)
    print("TraceML Recommend -- Demo")
    print("=" * 78)
    print()
    print("This demo runs `traceml recommend` on sample summary cards that")
    print("represent common training bottleneck scenarios.")
    print()

    for filename, description in SCENARIOS:
        path = EXAMPLES_DIR / filename
        print(f"--- Scenario: {description} ---")
        print(f"    File: {filename}")
        print()
        run_recommend(str(path))
        print()
        print()


if __name__ == "__main__":
    main()
