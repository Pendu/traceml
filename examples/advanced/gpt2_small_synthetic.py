"""GPT-2 small on wikitext-2 (real text by default).

TraceML benchmark fixture — workload 3 of the v2 matrix
(see design/TraceML_Benchmarking_Workflow_v2.md).

Filename retained from the v2 doc draft; the default data path is now
real text from `wikitext` config `wikitext-2-raw-v1` (~12 MB on first
download, cached after). Tokenize with the GPT-2 tokenizer, concatenate,
chunk to fixed SEQ_LEN blocks — the standard HF causal-LM prep pattern.

Set GPT2_USE_SYNTHETIC=1 to bypass the dataset download and use random
token IDs in [0, vocab_size) instead — sanity / smoke tests when HF is
unreachable.

Causal LM: labels == input_ids; GPT2LMHeadModel handles label shifting
internally.

Env vars (defaults sized for 2xL4 24 GB sanity):
    SEED                 (42)
    BATCH_SIZE           (4)
    SEQ_LEN              (512)
    MAX_TRAIN_EXAMPLES   (512)   -> ~128 steps with batch=4
    EPOCHS               (1)
    GPT2_USE_SYNTHETIC   (0)     1 = bypass dataset download
"""

import os
import random
from collections.abc import Mapping

import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

import traceml

SEED = int(os.getenv("SEED", "42"))
MODEL_NAME = "gpt2"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
SEQ_LEN = int(os.getenv("SEQ_LEN", "512"))
MAX_TRAIN_EXAMPLES = int(os.getenv("MAX_TRAIN_EXAMPLES", "512"))
EPOCHS = int(os.getenv("EPOCHS", "1"))
LR = 5e-5
USE_SYNTHETIC = bool(int(os.getenv("GPT2_USE_SYNTHETIC", "0")))


def set_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_data(vocab_size: int):
    if USE_SYNTHETIC:
        input_ids = torch.randint(
            0, vocab_size, (MAX_TRAIN_EXAMPLES, SEQ_LEN), dtype=torch.long
        )
        labels = input_ids.clone()
        dataset = TensorDataset(input_ids, labels)
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    raw = load_dataset("wikitext", WIKITEXT_CONFIG, split="train")
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)

    def tok(examples):
        return tokenizer(examples["text"])

    tokenized = raw.map(tok, batched=True, remove_columns=raw.column_names)

    # Standard HF causal-LM prep: concat all token streams, chunk to
    # SEQ_LEN blocks, drop the trailing partial block.
    def group(examples):
        all_ids = sum(examples["input_ids"], [])
        total = (len(all_ids) // SEQ_LEN) * SEQ_LEN
        chunks = [all_ids[i : i + SEQ_LEN] for i in range(0, total, SEQ_LEN)]
        return {"input_ids": chunks, "labels": [c[:] for c in chunks]}

    chunked = tokenized.map(
        group, batched=True, remove_columns=tokenized.column_names
    )

    n = min(MAX_TRAIN_EXAMPLES, len(chunked))
    chunked = chunked.select(range(n))
    chunked.set_format(type="torch", columns=["input_ids", "labels"])

    return DataLoader(
        chunked,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def to_device_dict(batch, device):
    if isinstance(batch, Mapping):
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    # synthetic path: TensorDataset yields (input_ids, labels) tuple
    input_ids, labels = batch
    return {
        "input_ids": input_ids.to(device, non_blocking=True),
        "labels": labels.to(device, non_blocking=True),
    }


def main():
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    dtype = torch.float16 if use_amp else torch.float32

    traceml.init(mode="auto")

    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
    model.train()

    train_loader = prepare_data(vocab_size=model.config.vocab_size)

    optimizer = AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    global_step = 0
    running_loss = 0.0

    for epoch in range(EPOCHS):
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            with traceml.trace_step(model):
                # H2D inside trace_step per v0.2.13 convention
                # (study/README.md): attributes transfer to the
                # dataloader phase instead of residual.
                batch = to_device_dict(batch, device)

                with torch.amp.autocast(
                    device_type="cuda", enabled=use_amp, dtype=dtype
                ):
                    out = model(**batch)
                    loss = out.loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.detach().item()
                global_step += 1

                if global_step % 50 == 0:
                    avg_loss = running_loss / 50
                    print(
                        f"[Train] epoch {epoch+1} step {global_step} "
                        f"| loss {avg_loss:.4f}"
                    )
                    running_loss = 0.0

        print(f"Finished epoch {epoch + 1}")

    print("Done.")


if __name__ == "__main__":
    main()
