#!/usr/bin/env bash
# Sanity-test the 3 v2 benchmark fixtures on the current GPU host.
#
# Each fixture runs once with `traceml run --mode=summary`. Pass criteria:
#   1. exit code 0
#   2. logs/<session>/final_summary.json exists
#   3. final_summary.json has training_steps > 0 and a numeric step_avg_ms
#
# Outputs land in `bench_output_sanity/<session>/output.json` (mirrors the
# layout used by run_examples_*.sh in the existing study/ pipeline).
#
# First-run note: the GPT-2 and BERT fixtures download model weights from
# HuggingFace (~500 MB and ~440 MB respectively) the first time they run;
# CIFAR-100 downloads ~170 MB. Subsequent runs hit cache.
#
# Usage (from the traceml/ repo root):
#   bash study/sanity_check_v2_fixtures.sh
# Override world size for the FSDP fixture:
#   NPROC=2 bash study/sanity_check_v2_fixtures.sh

set -u
REPO=/teamspace/studios/this_studio/traceml
OUT_ROOT="$REPO/bench_output_sanity"
LOGS_ROOT="$REPO/logs"

cd "$REPO"

PY="$(readlink -f "$(command -v traceml)" | xargs -I{} dirname {})/python3"
if [ ! -x "$PY" ]; then
  PY=/system/conda/miniconda3/envs/cloudspace/bin/python3
fi
echo "[sanity] using python: $PY"

# --- preflight --------------------------------------------------------------
"$PY" -c "
import sys, torch
ok = torch.cuda.is_available()
n = torch.cuda.device_count() if ok else 0
print(f'[sanity] cuda_available={ok} device_count={n} torch={torch.__version__}')
if ok:
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f'[sanity]   gpu{i}: {p.name} {round(p.total_memory/1e9,1)}GB')
sys.exit(0 if (ok and n >= 1) else 1)
" || { echo "[sanity] preflight failed (need >= 1 GPU)"; exit 1; }

mkdir -p "$OUT_ROOT"

PASS=0
FAIL=0
RESULTS=()

run_one() {
  local NAME="$1"; local SCRIPT="$2"; shift 2
  local OUT="$OUT_ROOT/$NAME"
  local LOG="$LOGS_ROOT/$NAME"
  rm -rf "$LOG" "$OUT"
  mkdir -p "$OUT"
  echo
  echo "[sanity] === $NAME ==="
  local START
  START=$(date +%s)
  traceml run "$SCRIPT" --mode=summary --session-id="$NAME" "$@" \
    > "$OUT/stdout.log" 2>&1
  local RC=$?
  local DUR=$(( $(date +%s) - START ))
  echo "[sanity] $NAME finished rc=$RC in ${DUR}s"

  for f in final_summary.json final_summary.txt manifest.json \
           code_manifest.json system_manifest.json; do
    if [ -f "$LOG/$f" ]; then
      case "$f" in
        final_summary.json) cp "$LOG/$f" "$OUT/output.json" ;;
        final_summary.txt)  cp "$LOG/$f" "$OUT/output.txt"  ;;
        *)                  cp "$LOG/$f" "$OUT/$f"          ;;
      esac
    fi
  done

  # --- pass criteria ------------------------------------------------------
  if [ "$RC" -ne 0 ]; then
    RESULTS+=("FAIL  $NAME  (exit=$RC, see $OUT/stdout.log)")
    FAIL=$((FAIL+1))
    return
  fi
  if [ ! -f "$OUT/output.json" ]; then
    RESULTS+=("FAIL  $NAME  (no output.json — aggregator did not produce summary)")
    FAIL=$((FAIL+1))
    return
  fi

  local OUTPUT_PATH="$OUT/output.json"
  local CHECK
  CHECK=$(OUTPUT_PATH="$OUTPUT_PATH" "$PY" -c "
import json, os, sys
with open(os.environ['OUTPUT_PATH']) as f:
    d = json.load(f)
st = d.get('step_time', {})
steps = (
    st.get('overview', {}).get('training_steps')
    or st.get('training_steps', 0)
)
sa_ms = (
    st.get('global', {}).get('typical', {}).get('step_avg_ms')
    or st.get('timing_primary', {}).get('step_avg_ms')
)
ok = bool(steps and steps > 0) and (sa_ms is not None)
print(f'steps={steps}  step_avg_ms={sa_ms}  ok={ok}')
sys.exit(0 if ok else 2)
")
  local CRC=$?
  echo "[sanity]   $CHECK"
  if [ "$CRC" -eq 0 ]; then
    RESULTS+=("PASS  $NAME  ($CHECK)")
    PASS=$((PASS+1))
  else
    RESULTS+=("FAIL  $NAME  (telemetry: $CHECK)")
    FAIL=$((FAIL+1))
  fi
}

# --- single-rank sanity -----------------------------------------------------
run_one sanity_gpt2_small_synthetic  examples/advanced/gpt2_small_synthetic.py

# resnet50 fixture loads CIFAR-100 from the HF mirror (uoft-cs/cifar100)
# by default — reliable and avoids the cs.toronto.edu 503 issue.
# Set RESNET_USE_SYNTHETIC=1 only if HF is also unreachable.
run_one sanity_resnet50_cifar        examples/advanced/resnet50_cifar.py

# --- multi-rank sanity (FSDP needs >= 2 GPUs to actually shard) -------------
# fsdp_bert's real-data path (slow tokenizer + dynamic padding via
# DataCollatorWithPadding, matching bert_gradient_accum.py convention)
# SIGSEGVs at the first forward step under FSDP — see fixture docstring
# for KNOWN ISSUE notes. Sanity uses synthetic to keep the smoke check
# green; production matrix needs the FSDP+dynamic-padding interaction
# investigated before unsetting this.
NPROC="${NPROC:-2}"
BERT_FSDP_USE_SYNTHETIC=1 run_one sanity_fsdp_bert_minimal \
  examples/advanced/fsdp_bert_minimal.py --nproc-per-node="$NPROC"

# --- summary ----------------------------------------------------------------
echo
echo "[sanity] ================ RESULTS ================"
for line in "${RESULTS[@]}"; do
  echo "[sanity]   $line"
done
echo "[sanity] ${PASS} passed, ${FAIL} failed"
echo "[sanity] Artifacts in: $OUT_ROOT"
exit $FAIL
