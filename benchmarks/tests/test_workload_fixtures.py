import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKLOADS = REPO / "benchmarks" / "workloads"

TINY_TRANSFORMER = [
    "--layers",
    "2",
    "--d-model",
    "64",
    "--nhead",
    "2",
    "--ffn-mult",
    "2",
    "--seq-len",
    "16",
    "--vocab-size",
    "100",
    "--num-classes",
    "10",
    "--batch-size",
    "4",
    "--num-samples",
    "16",
    "--steps",
    "2",
]

TINY_CNN = [
    "--img-size",
    "32",
    "--num-classes",
    "10",
    "--batch-size",
    "4",
    "--num-samples",
    "16",
    "--steps",
    "2",
]


def _run(script, args):
    proc = subprocess.run(
        [sys.executable, str(WORKLOADS / script), *args],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return proc


def _run_ddp(script, args):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--nnodes=1",
            str(WORKLOADS / script),
            *args,
            "--dist",
            "ddp",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return proc


def test_transformer_single_cpu_smoke():
    proc = _run("transformer_e2e.py", [*TINY_TRANSFORMER, "--dist", "single"])
    assert proc.returncode == 0, proc.stderr
    assert "Done. steps=2" in proc.stdout
    assert "dist=single" in proc.stdout
    assert "world_size=1" in proc.stdout


def test_transformer_ddp_gloo_cpu_smoke():
    proc = _run_ddp("transformer_e2e.py", TINY_TRANSFORMER)
    assert proc.returncode == 0, proc.stderr
    assert "Done. steps=2" in proc.stdout
    assert "dist=ddp" in proc.stdout
    assert "world_size=2" in proc.stdout


def test_cnn_single_cpu_smoke():
    proc = _run("cnn_e2e.py", [*TINY_CNN, "--dist", "single"])
    assert proc.returncode == 0, proc.stderr
    assert "Done. steps=2" in proc.stdout
    assert "dist=single" in proc.stdout
    assert "world_size=1" in proc.stdout


def test_cnn_ddp_gloo_cpu_smoke():
    proc = _run_ddp("cnn_e2e.py", TINY_CNN)
    assert proc.returncode == 0, proc.stderr
    assert "Done. steps=2" in proc.stdout
    assert "dist=ddp" in proc.stdout
    assert "world_size=2" in proc.stdout
