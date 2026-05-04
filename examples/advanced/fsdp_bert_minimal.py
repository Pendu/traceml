"""BERT-base FSDP on ag_news.

TraceML benchmark fixture — workload 5 of the v2 matrix
(see design/TraceML_Benchmarking_Workflow_v2.md).

Mirrors fsdp_minimal_cuda.py's structure (init -> trace_model_instance
-> FSDP wrap -> trace_step inside the loop) but runs real BERT-base on
real ag_news text. Pairs with bert_gradient_accum.py (workload 1) on
the same dataset, with the same data-prep conventions, so the FSDP
overhead row is directly comparable to the single-GPU BERT row.

Conventions follow bert_gradient_accum.py:
    * AutoModelForSequenceClassification + "bert-base-uncased" with
      4-class head matching ag_news.
    * AutoTokenizer with use_fast=False (slow Python tokenizer).
    * DataCollatorWithPadding(padding=True) — dynamic padding to the
      longest sequence in each batch.
    * load_batch_to_device helper to move all batch tensors to GPU.

FSDP-specific bits:
    * transformer_auto_wrap_policy keyed on BertLayer so FSDP shards
      per encoder block (without this, FSDP wraps the whole BERT body
      as one unit and effectively gets no sharding).

KNOWN ISSUE: as of v0.2.13 + torch 2.8.0+cu128, this fixture's
real-data path SIGSEGVs at the first forward step under FSDP — likely a
combination of dynamic padding (variable per-batch shape) and FSDP
all-gather. Sanity checks therefore set BERT_FSDP_USE_SYNTHETIC=1 to
swap in random token IDs (same shape contract, no tokenizer/dataset
download). Production benchmark runs must investigate the FSDP+dynamic-
padding interaction (likely fixes: padding="max_length", switch to fast
tokenizer, drop token_type_ids — each was a working workaround in
isolation but was reverted to keep convention parity with workload 1).

Run with at least 2 GPUs:
    torchrun --nproc-per-node=2 examples/advanced/fsdp_bert_minimal.py
or via the traceml CLI:
    traceml run examples/advanced/fsdp_bert_minimal.py --nproc-per-node=2

Env vars (defaults sized for 2xL4 24 GB sanity):
    SEED                     (42)
    BATCH_SIZE               (16)    per-rank
    SEQ_LEN                  (128)   tokenizer max length
    NUM_SAMPLES              (4000)  total ag_news rows used
    EPOCHS                   (2)
    BERT_FSDP_USE_SYNTHETIC  (0)     1 = bypass dataset / tokenizer
"""

import functools
import os
from collections.abc import Mapping

import torch
import torch.distributed as dist
import torch.optim as optim
from datasets import load_dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from transformers.models.bert.modeling_bert import BertLayer

import traceml

SEED = int(os.getenv("SEED", "42"))
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 4

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
SEQ_LEN = int(os.getenv("SEQ_LEN", "128"))
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", "4000"))
EPOCHS = int(os.getenv("EPOCHS", "2"))
LR = 2e-5
USE_SYNTHETIC = bool(int(os.getenv("BERT_FSDP_USE_SYNTHETIC", "0")))


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_data(rank: int, world_size: int, vocab_size: int):
    if USE_SYNTHETIC:
        input_ids = torch.randint(
            0, vocab_size, (NUM_SAMPLES, SEQ_LEN), dtype=torch.long
        )
        attention_mask = torch.ones_like(input_ids)
        labels = torch.randint(0, NUM_LABELS, (NUM_SAMPLES,), dtype=torch.long)
        dataset = TensorDataset(input_ids, attention_mask, labels)
        collator = None
    else:
        # Convention parity with bert_gradient_accum.py: slow tokenizer,
        # truncation only (no padding here), DataCollator handles dynamic
        # padding to the longest sequence in each batch.
        raw = load_dataset("ag_news")
        n = min(NUM_SAMPLES, len(raw["train"]))
        train_raw = raw["train"].select(range(n))
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

        def tok(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=SEQ_LEN,
            )

        train_ds = train_raw.map(tok, batched=True, remove_columns=["text"])
        train_ds = train_ds.rename_column("label", "labels")
        dataset = train_ds
        collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
        num_workers=0,
        collate_fn=collator,
    )
    return loader, sampler


def load_batch_to_device(batch, device):
    # HF DataCollatorWithPadding returns BatchEncoding (UserDict, not a
    # built-in dict) — Mapping check covers both that and the synthetic
    # tuple from TensorDataset.
    if isinstance(batch, Mapping):
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    input_ids, attention_mask, labels = batch
    return {
        "input_ids": input_ids.to(device, non_blocking=True),
        "attention_mask": attention_mask.to(device, non_blocking=True),
        "labels": labels.to(device, non_blocking=True),
    }


def main():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if not torch.cuda.is_available():
        raise RuntimeError("fsdp_bert_minimal expects CUDA GPUs.")

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    traceml.init(mode="auto")

    set_seed(SEED + rank)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    ).to(device)

    train_loader, train_sampler = prepare_data(
        rank, world_size, vocab_size=base_model.config.vocab_size
    )

    # Attach TraceML hooks BEFORE FSDP wrapping
    traceml.trace_model_instance(base_model)

    # FSDP with per-BertLayer wrap so sharding actually happens
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={BertLayer},
    )
    model = FSDP(
        base_model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.cuda.current_device(),
    )

    optimizer = optim.AdamW(model.parameters(), lr=LR)

    model.train()
    global_step = 0
    running_loss = 0.0

    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)

        for batch in train_loader:
            with traceml.trace_step(base_model):
                # H2D inside trace_step per v0.2.13 convention
                # (study/README.md): attributes transfer to the
                # dataloader phase instead of residual.
                batch = load_batch_to_device(batch, device)

                optimizer.zero_grad(set_to_none=True)

                out = model(**batch)
                loss = out.loss

                loss.backward()
                optimizer.step()

                running_loss += loss.detach()
                global_step += 1

                if rank == 0 and global_step % 50 == 0:
                    avg_loss = (running_loss / 50).item()
                    print(
                        f"[Train] epoch {epoch+1} step {global_step} "
                        f"| loss {avg_loss:.4f}"
                    )
                    running_loss.zero_()

    if rank == 0:
        print("Done.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
