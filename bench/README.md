# `bench/` — v2 overhead benchmark pipeline

Sibling to `study/`. Different question.

| Pipeline | Question it answers |
|---|---|
| `study/` | Where does the wall time go inside one TraceML run? (per-run wall-clock breakdown across CPU/T4/L4 tiers) |
| `bench/` | What is TraceML's overhead vs running the same fixture without it? (overhead %, peak GPU mem delta, peak RSS delta — averaged across N trials, per the v2 doc) |

Spec: [`design/TraceML_Benchmarking_Workflow_v2.md`](../../design/TraceML_Benchmarking_Workflow_v2.md).

## Layout

```
bench/
├── README.md                      ← (this file)
├── workloads.py                   ← single-source registry of the 8 v2 workloads
├── baseline_harness.py            ← runs a fixture WITHOUT traceml; captures wall + peak GPU + peak RSS
├── run_v2_matrix.py               ← outer loop: workload × mode × trial
├── build_overhead_matrix.py       ← aggregates per-trial JSONs → overhead_matrix.csv + report.md
└── (after build_overhead_matrix.py:)
    ├── overhead_matrix.csv        ← one row per (workload, mode) with mean ± std
    └── report.md                  ← v2-doc-shaped overhead table

bench_output_<tier>/                ← raw per-trial outputs (separate per hardware tier)
└── <workload>/<mode>/trial_<n>/
    ├── output.json                ← traceml_run mode (copied from logs/<session>/final_summary.json)
    ├── output.txt                 ← traceml_run mode
    ├── baseline_metrics.json      ← baseline mode (from baseline_harness.py)
    └── stdout.log                 ← captured stdout/stderr
```

## Workloads (from v2 doc)

| # | Key | Script | Hardware |
|---|---|---|---|
| 1 | `bert_agnews`         | `examples/advanced/bert_gradient_accum.py`     | 1×A100 |
| 2 | `vit_cifar10`         | `examples/advanced/huggingface_vision_vit.py`  | 1×A100 |
| 3 | `gpt2_wikitext`       | `examples/advanced/gpt2_small_synthetic.py`    | 1×A100 |
| 4 | `resnet50_cifar100`   | `examples/advanced/resnet50_cifar.py`          | 1×A100 |
| 5 | `bert_fsdp_agnews`    | `examples/advanced/fsdp_bert_minimal.py`       | 2×A100 SXM |
| 6 | `tiny_mlp_ddp`        | `examples/ddp_minimal.py`                      | 2×A100 SXM |
| 7 | `tiny_mlp`            | `examples/pytorch_minimal.py`                  | 1×L4 |
| 8 | `hf_trainer`          | `examples/huggingface_trainer_minimal.py`      | 1×A100 |

Single source of truth for these mappings: [`bench/workloads.py`](workloads.py).

## Modes

- **`baseline`** — `python <script>` (or `torchrun --nproc_per_node=N <script>`) with no TraceML.
  Wall-clock + per-GPU peak memory + peak host RSS captured by `baseline_harness.py` via a
  background polling thread (psutil + pynvml).
- **`traceml_run`** — `traceml run <script>` with full instrumentation. Metrics come from
  TraceML's own `final_summary.json` (`duration_s`, `step_time.global.typical.step_avg_ms`,
  `system.global.gpu_rollup.mem_peak_gb`, `system.global.ram.peak_gb`).
- **`traceml watch`** — out of scope per session decision. Skipped.

Per-step median in `baseline` mode is intentionally not captured: that would require
fixture-side timing instrumentation. Only TraceML mode reports per-step ms.

## Workflow

From the repo root (`/teamspace/studios/this_studio/traceml`):

```bash
# 1. Run the matrix. Defaults: bench_output_2xl4/, 3 trials, all 8 workloads,
#    both modes (baseline + traceml_run).
python bench/run_v2_matrix.py --trials 3

# Variations:
python bench/run_v2_matrix.py --trials 5                        # v2-canonical N
python bench/run_v2_matrix.py --workloads gpt2_wikitext         # one workload
python bench/run_v2_matrix.py --workloads bert_agnews,vit_cifar10
python bench/run_v2_matrix.py --modes baseline                  # baseline only
python bench/run_v2_matrix.py --skip-existing                   # resume after a partial
python bench/run_v2_matrix.py --out-root bench_output_a100      # different tier

# 2. Build the overhead matrix once trials are populated.
python bench/build_overhead_matrix.py
# -> bench/overhead_matrix.csv, bench/report.md
```

## Cost notes

| Trials | Workloads | Modes | Approx wall-time on 2×L4 |
|---|---|---|---|
| 3 | all 8 | both | ~2–3 hours (depends on cache state for first-run downloads) |
| 5 | all 8 | both | ~4–5 hours |
| 5 | 1, 2, 3, 4 (single-GPU heavy) | both | ~2 hours |

A100 is faster: budget ~half the time per the v2 doc's $35–50 estimate.

Everything is idempotent with `--skip-existing` — safe to re-invoke after a crash.

## Adding a new tier

To run on a different hardware tier (e.g. 1×A100):

1. Provision the host, install TraceML.
2. `python bench/run_v2_matrix.py --out-root bench_output_a100_1x --trials 5`
3. `python bench/build_overhead_matrix.py --bench-root bench_output_a100_1x \
        --out-csv bench/overhead_matrix_a100_1x.csv \
        --out-md bench/report_a100_1x.md`

Per-tier directories stay separate; no shared-state munging.

## Known issues

- **Workload 5 (BERT FSDP)** runs with `BERT_FSDP_USE_SYNTHETIC=1` (synthetic random tokens
  instead of real ag_news) because the real-data path SIGSEGVs at the first forward step
  under FSDP. Fixture docstring documents it. Pre-A100 production run must investigate
  the FSDP+dynamic-padding interaction.
- **Per-step ms in baseline mode**: not measured. Only TraceML's `traceml_run` reports it.
- **Trial count default = 3**, not v2's recommended 5. Bump with `--trials 5` for the
  release-grade campaign once the pipeline is validated end-to-end on this 2×L4 host.
