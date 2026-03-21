"""
Rule-based optimization recommendations from TraceML summary cards.

Reads a completed run's summary_card.json and outputs actionable
recommendations for improving training performance.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Recommendation:
    """A single optimization recommendation."""

    title: str
    priority: int  # 1 = highest
    category: str  # dataloader, compute, memory, optimizer
    description: str
    actions: List[str]
    estimated_impact: str  # e.g. "high", "medium", "low"


def _load_summary_card(path: str) -> Dict[str, Any]:
    """Load summary card JSON from a file path or session directory."""
    p = Path(path)

    # If it's a directory, look for the summary card inside it
    if p.is_dir():
        # Try session directory layout: <session>/aggregator/telemetry_summary_card.json
        candidates = list(p.glob("**/telemetry_summary_card.json"))
        if not candidates:
            candidates = list(p.glob("**/*_summary_card.json"))
        if not candidates:
            # Try via manifest.json
            manifest_path = p / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                db_path = manifest.get("paths", {}).get("db_path")
                if db_path:
                    card_path = Path(db_path + "_summary_card.json")
                    if card_path.exists():
                        candidates = [card_path]
        if not candidates:
            print(
                f"Error: No summary_card.json found in '{path}'.",
                file=sys.stderr,
            )
            sys.exit(1)
        p = candidates[0]

    if not p.exists():
        print(f"Error: File '{p}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _check_dataloader(
    step_time: Dict[str, Any],
) -> Optional[Recommendation]:
    """Check if dataloader is the dominant bottleneck."""
    # Try median split first, fall back to worst split or per-rank data
    split_pct = step_time.get("median_split_pct") or step_time.get(
        "worst_split_pct"
    )
    split_ms = step_time.get("median_split_ms") or step_time.get(
        "worst_split_ms"
    )

    if not split_pct and not split_ms:
        # Try single-rank case from per_rank data
        per_rank = step_time.get("per_rank", {})
        if len(per_rank) == 1:
            rank_data = next(iter(per_rank.values()))
            total = rank_data.get("avg_total_step_ms", 0)
            dl = rank_data.get("avg_dataloader_ms", 0)
            if total > 0:
                split_pct = {"dataloader": 100.0 * dl / total}
                split_ms = {"dataloader": dl}

    if not split_pct:
        return None

    dl_pct = split_pct.get("dataloader")
    if dl_pct is None or dl_pct < 20.0:
        return None

    dl_ms = split_ms.get("dataloader", 0) if split_ms else 0

    if dl_pct >= 40.0:
        priority = 1
        impact = "high"
    elif dl_pct >= 30.0:
        priority = 2
        impact = "medium"
    else:
        priority = 3
        impact = "low"

    return Recommendation(
        title="Dataloader is a bottleneck",
        priority=priority,
        category="dataloader",
        description=(
            f"Dataloader accounts for {dl_pct:.1f}% of step time "
            f"({dl_ms:.1f}ms). The GPU is idle waiting for data."
        ),
        actions=[
            "Increase num_workers in DataLoader (try 4, 8, or os.cpu_count())",
            "Set pin_memory=True for faster CPU-to-GPU transfers",
            "Set persistent_workers=True to avoid worker respawn overhead",
            "Increase prefetch_factor (default is 2; try 4 or 8)",
            "Consider using a faster data format (e.g., WebDataset, FFCV, or pre-tokenized data)",
        ],
        estimated_impact=impact,
    )


def _check_compute_no_compile(
    step_time: Dict[str, Any],
    system: Optional[Dict[str, Any]] = None,
) -> Optional[Recommendation]:
    """Check if forward+backward dominate and torch.compile could help."""
    # If GPU is already well-utilized (>90%), compute dominance is expected
    # and there's limited room for improvement from compile/AMP alone.
    if system:
        gpu_util = system.get("gpu_util_avg_percent")
        if gpu_util is not None and gpu_util > 90.0:
            return None

    split_pct = step_time.get("median_split_pct") or step_time.get(
        "worst_split_pct"
    )
    if not split_pct:
        per_rank = step_time.get("per_rank", {})
        if len(per_rank) == 1:
            rank_data = next(iter(per_rank.values()))
            total = rank_data.get("avg_total_step_ms", 0)
            fwd = rank_data.get("avg_forward_ms", 0)
            bwd = rank_data.get("avg_backward_ms", 0)
            if total > 0:
                split_pct = {
                    "forward": 100.0 * fwd / total,
                    "backward": 100.0 * bwd / total,
                }

    if not split_pct:
        return None

    fwd_pct = split_pct.get("forward", 0) or 0
    bwd_pct = split_pct.get("backward", 0) or 0
    compute_pct = fwd_pct + bwd_pct

    if compute_pct < 70.0:
        return None

    return Recommendation(
        title="Forward+backward dominate step time",
        priority=2,
        category="compute",
        description=(
            f"Forward+backward account for {compute_pct:.1f}% of step time. "
            "Consider torch.compile and mixed precision to reduce compute time."
        ),
        actions=[
            "Wrap the model with torch.compile() for kernel fusion and optimization",
            "Enable mixed precision training with torch.autocast('cuda', dtype=torch.bfloat16)",
            "Consider gradient checkpointing to trade compute for memory if memory-limited",
        ],
        estimated_impact="medium",
    )


def _check_optimizer(
    step_time: Dict[str, Any],
) -> Optional[Recommendation]:
    """Check if optimizer step is disproportionately expensive."""
    split_pct = step_time.get("median_split_pct") or step_time.get(
        "worst_split_pct"
    )
    if not split_pct:
        per_rank = step_time.get("per_rank", {})
        if len(per_rank) == 1:
            rank_data = next(iter(per_rank.values()))
            total = rank_data.get("avg_total_step_ms", 0)
            opt = rank_data.get("avg_optimizer_ms", 0)
            if total > 0:
                split_pct = {"optimizer": 100.0 * opt / total}

    if not split_pct:
        return None

    opt_pct = split_pct.get("optimizer", 0) or 0
    if opt_pct < 15.0:
        return None

    return Recommendation(
        title="Optimizer step is expensive",
        priority=3,
        category="optimizer",
        description=(
            f"Optimizer accounts for {opt_pct:.1f}% of step time. "
            "Fused optimizers can significantly reduce this overhead."
        ),
        actions=[
            "Use fused=True in Adam/AdamW (e.g., torch.optim.AdamW(..., fused=True))",
            "Use foreach=True for multi-tensor optimizer updates",
            "Consider using torch.compile on the optimizer step",
        ],
        estimated_impact="medium" if opt_pct >= 25.0 else "low",
    )


def _check_gpu_underutilization(
    system: Dict[str, Any],
) -> Optional[Recommendation]:
    """Check if GPU utilization is low with memory headroom available."""
    gpu_util = system.get("gpu_util_avg_percent")
    gpu_mem_peak = system.get("gpu_mem_peak_gb")
    gpu_available = system.get("gpu_available")

    if gpu_available is False or gpu_util is None:
        return None

    if gpu_util >= 70.0:
        return None

    actions = [
        "Increase batch size to better saturate GPU compute",
    ]

    # Check memory headroom
    description = f"Average GPU utilization is only {gpu_util:.1f}%."

    if gpu_mem_peak is not None:
        actions.append(
            f"Current peak GPU memory: {gpu_mem_peak:.1f} GB -- "
            "there may be room for a larger batch size"
        )

    actions.extend(
        [
            "Use gradient accumulation to simulate larger effective batch sizes",
            "Ensure the model is not bottlenecked by CPU or data loading",
        ]
    )

    return Recommendation(
        title="GPU is underutilized",
        priority=2,
        category="compute",
        description=description,
        actions=actions,
        estimated_impact="high" if gpu_util < 50.0 else "medium",
    )


def _check_rank_skew(
    step_time: Dict[str, Any],
) -> Optional[Recommendation]:
    """Check for significant rank imbalance in distributed training."""
    ranks_seen = step_time.get("ranks_seen", 1)
    if ranks_seen <= 1:
        return None

    worst_vs_median = step_time.get("worst_vs_median_pct")
    if worst_vs_median is None or worst_vs_median < 10.0:
        return None

    worst_rank = step_time.get("worst_rank", "?")
    median_rank = step_time.get("median_rank", "?")
    worst_ms = step_time.get("worst_avg_step_ms", 0)
    median_ms = step_time.get("median_avg_step_ms", 0)

    return Recommendation(
        title="Significant rank skew detected",
        priority=1,
        category="distributed",
        description=(
            f"Worst rank r{worst_rank} ({worst_ms:.1f}ms) is "
            f"{worst_vs_median:.1f}% slower than median rank r{median_rank} "
            f"({median_ms:.1f}ms). All ranks wait for the slowest."
        ),
        actions=[
            f"Investigate rank r{worst_rank} for hardware issues (thermals, interconnect)",
            "Check for uneven data distribution across ranks",
            "Ensure all GPUs are the same model and not throttling",
            "Consider load balancing strategies for heterogeneous hardware",
        ],
        estimated_impact="high",
    )


def _check_high_gpu_memory(
    system: Dict[str, Any],
) -> Optional[Recommendation]:
    """Check if GPU memory is near capacity."""
    gpu_mem_peak = system.get("gpu_mem_peak_gb")
    gpu_available = system.get("gpu_available")

    if gpu_available is False or gpu_mem_peak is None:
        return None

    # We don't have total GPU memory in system summary, so use a heuristic:
    # if peak memory is very high (>20GB), suggest memory optimization
    # This is a rough heuristic — ideally we'd compare to total GPU memory
    gpu_util = system.get("gpu_util_avg_percent")
    gpu_mem_avg = system.get("gpu_mem_avg_gb")

    if gpu_mem_avg is None:
        return None

    # If peak is much higher than average, there may be memory spikes
    if gpu_mem_peak > 0 and gpu_mem_avg > 0:
        spike_ratio = gpu_mem_peak / gpu_mem_avg
        if spike_ratio > 1.5:
            return Recommendation(
                title="GPU memory spikes detected",
                priority=3,
                category="memory",
                description=(
                    f"Peak GPU memory ({gpu_mem_peak:.1f} GB) is "
                    f"{spike_ratio:.1f}x higher than average ({gpu_mem_avg:.1f} GB). "
                    "Memory spikes can cause OOM errors or limit batch size."
                ),
                actions=[
                    "Enable gradient checkpointing to reduce activation memory",
                    "Use mixed precision (bfloat16/float16) to halve memory per parameter",
                    "Consider gradient accumulation with smaller micro-batches",
                ],
                estimated_impact="medium",
            )

    return None


def generate_recommendations(
    summary: Dict[str, Any],
) -> List[Recommendation]:
    """
    Analyze a summary card and generate optimization recommendations.

    Parameters
    ----------
    summary:
        Parsed summary_card.json contents.

    Returns
    -------
    List of Recommendation objects, sorted by priority (1 = highest).
    """
    recommendations: List[Recommendation] = []

    step_time = summary.get("step_time", {})
    # System fields are at the top level of the summary
    system = {
        k: v for k, v in summary.items() if k != "step_time" and k != "card"
    }

    checks = [
        lambda: _check_dataloader(step_time),
        lambda: _check_compute_no_compile(step_time, system),
        lambda: _check_optimizer(step_time),
        lambda: _check_gpu_underutilization(system),
        lambda: _check_rank_skew(step_time),
        lambda: _check_high_gpu_memory(system),
    ]

    for check in checks:
        rec = check()
        if rec is not None:
            recommendations.append(rec)

    recommendations.sort(key=lambda r: r.priority)
    return recommendations


def format_recommendations(recommendations: List[Recommendation]) -> str:
    """Format recommendations as readable text output."""
    if not recommendations:
        return (
            "+----------------------------------------------------------------------------+\n"
            "|  TraceML Recommendations                                                   |\n"
            "+----------------------------------------------------------------------------+\n"
            "|  No optimization issues detected. Your training looks healthy!              |\n"
            "+----------------------------------------------------------------------------+"
        )

    width = 78
    lines = [
        "+" + "-" * (width - 2) + "+",
        f"|  {'TraceML Optimization Recommendations':<{width - 4}}|",
        f"|  {f'{len(recommendations)} issue(s) found':<{width - 4}}|",
        "+" + "-" * (width - 2) + "+",
    ]

    for i, rec in enumerate(recommendations, 1):
        impact_label = f"[{rec.estimated_impact.upper()} IMPACT]"
        header = f"#{i} {rec.title} {impact_label}"
        lines.append(f"|  {'':<{width - 4}}|")
        lines.append(f"|  {header:<{width - 4}}|")
        lines.append(f"|  {'Category: ' + rec.category:<{width - 4}}|")
        lines.append(f"|  {'':<{width - 4}}|")

        # Word-wrap description
        desc_lines = _wrap_text(rec.description, width - 6)
        for dl in desc_lines:
            lines.append(f"|  {dl:<{width - 4}}|")

        lines.append(f"|  {'':<{width - 4}}|")
        lines.append(f"|  {'Suggested actions:':<{width - 4}}|")
        for action in rec.actions:
            action_lines = _wrap_text(f"  - {action}", width - 6)
            for al in action_lines:
                lines.append(f"|  {al:<{width - 4}}|")

        if i < len(recommendations):
            lines.append(f"|  {'-' * (width - 4)}|")

    lines.append("+" + "-" * (width - 2) + "+")
    return "\n".join(lines)


def _wrap_text(text: str, max_width: int) -> List[str]:
    """Simple word wrapping."""
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > max_width:
            lines.append(current)
            current = word
        elif current:
            current += " " + word
        else:
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def run_recommend(path: str) -> None:
    """
    Main entry point for `traceml recommend`.

    Parameters
    ----------
    path:
        Path to a summary_card.json file or a session directory.
    """
    summary = _load_summary_card(path)
    recommendations = generate_recommendations(summary)
    output = format_recommendations(recommendations)
    print(output)
