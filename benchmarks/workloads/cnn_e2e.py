"""Heavy CNN (ResNet-50) benchmark workload for end-to-end TraceML timing.

Standalone fixture (mirrors benchmarks/workloads/ddp_mlp_e2e.py), parameterized
for the overhead matrix. Runs native and under ``traceml run``:

    torchrun --nproc_per_node=1 benchmarks/workloads/cnn_e2e.py \
        --dist ddp --steps 200
    traceml run benchmarks/workloads/cnn_e2e.py --dist ddp --steps 200

TraceML is enabled only under ``traceml run``. Single-vs-2-node is torchrun.
"""

from __future__ import annotations

import argparse
import functools
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

import traceml_ai as traceml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ResNet-50 benchmark workload.")
    p.add_argument("--dist", choices=["single", "ddp", "fsdp"], default="ddp")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--num-samples", type=int, default=16384)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.num_samples < args.batch_size:
        raise SystemExit("--num-samples must be >= --batch-size")
    if args.img_size < 32:
        raise SystemExit("--img-size must be >= 32")
    if args.num_classes <= 1:
        raise SystemExit("--num-classes must be > 1")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def traceml_active() -> bool:
    if os.environ.get("TRACEML_DISABLED", "0") == "1":
        return False
    return bool(os.environ.get("TRACEML_SESSION_ID"))


def setup_distributed(dist_mode: str) -> tuple[int, int, int]:
    if dist_mode == "single":
        return 0, 0, 1
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, local_rank, world_size


def prepare_data(args, rank, world_size):
    g = torch.Generator().manual_seed(args.seed)
    images = torch.randn(
        args.num_samples, 3, args.img_size, args.img_size, generator=g
    )
    labels = torch.randint(
        0, args.num_classes, (args.num_samples,), generator=g
    )
    dataset = TensorDataset(images, labels)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    return loader, sampler


def wrap_model(base_model, dist_mode, use_cuda, local_rank, trace_enabled):
    if dist_mode == "single":
        return base_model
    if dist_mode == "ddp":
        if use_cuda:
            return nn.parallel.DistributedDataParallel(
                base_model, device_ids=[local_rank], output_device=local_rank
            )
        return nn.parallel.DistributedDataParallel(base_model)
    # fsdp -- attach TraceML hooks BEFORE wrapping (mirror
    # examples/advanced/fsdp_minimal_cuda.py for the exact API).
    if trace_enabled:
        traceml.trace_model_instance(base_model)
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=int(1e7)
    )
    kwargs = {"auto_wrap_policy": policy}
    if use_cuda:
        kwargs["device_id"] = torch.cuda.current_device()
    return FSDP(base_model, **kwargs)


def main() -> None:
    args = parse_args()
    validate_args(args)

    rank, local_rank, world_size = setup_distributed(args.dist)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        amp_dtype = torch.float16
    else:
        device = torch.device("cpu")
        amp_dtype = torch.float32

    trace_enabled = traceml_active()
    if trace_enabled:
        traceml.init(mode="auto")

    set_seed(args.seed + rank)
    loader, sampler = prepare_data(args, rank, world_size)

    base_model = torchvision.models.resnet50(
        weights=None, num_classes=args.num_classes
    ).to(device)

    model = wrap_model(
        base_model, args.dist, use_cuda, local_rank, trace_enabled
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(
        enabled=use_cuda, device="cuda" if use_cuda else "cpu"
    )

    model.train()
    start_s = time.perf_counter()
    global_step = 0
    epoch = 0

    while global_step < args.steps:
        sampler.set_epoch(epoch)
        epoch += 1
        for batch_x, batch_y in loader:
            if global_step >= args.steps:
                break
            context = (
                traceml.trace_step(base_model)
                if trace_enabled
                else nullcontext()
            )
            with context:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type="cuda" if use_cuda else "cpu",
                    enabled=use_cuda,
                    dtype=amp_dtype,
                ):
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            global_step += 1

    if use_cuda:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_s
    if rank == 0:
        print(
            f"Done. steps={global_step} elapsed_s={elapsed:.2f} "
            f"steps_per_s={global_step / elapsed:.3f} "
            f"traceml_active={trace_enabled} dist={args.dist} "
            f"world_size={world_size}"
        )
    if args.dist != "single":
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
