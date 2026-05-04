"""ResNet-50 on CIFAR-100 (subset).

TraceML benchmark fixture — workload 4 of the v2 matrix
(see design/TraceML_Benchmarking_Workflow_v2.md).

torchvision ResNet-50 (random init — this is a benchmark fixture, not a
real training) with the final FC swapped to 100 classes. CIFAR-100 is
loaded from the HuggingFace mirror (`uoft-cs/cifar100`) because the
upstream torchvision mirror (cs.toronto.edu) has been intermittently
returning HTTP 503. Images are resized 32 -> 224 to match ResNet's
first-conv stride.

If even HF is unreachable, set RESNET_USE_SYNTHETIC=1 to swap the
dataset for random (3, 224, 224) tensors with random labels — same
shape and dtype as the real path; lets sanity / smoke tests run when
network is fully down.

Env vars (defaults sized for 2xL4 24 GB sanity):
    SEED                  (42)
    BATCH_SIZE            (64)
    MAX_TRAIN_EXAMPLES    (2000)   -> ~31 steps with batch=64
    EPOCHS                (1)
    RESNET_USE_SYNTHETIC  (0)      -> 1 = bypass dataset download
"""

import os
import random

import torch
import torch.nn as nn
import torchvision
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

import traceml

SEED = int(os.getenv("SEED", "42"))
NUM_CLASSES = 100
HF_DATASET = "uoft-cs/cifar100"

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
MAX_TRAIN_EXAMPLES = int(os.getenv("MAX_TRAIN_EXAMPLES", "2000"))
EPOCHS = int(os.getenv("EPOCHS", "1"))
LR = 1e-3
USE_SYNTHETIC = bool(int(os.getenv("RESNET_USE_SYNTHETIC", "0")))


def set_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_data():
    if USE_SYNTHETIC:
        images = torch.randn(MAX_TRAIN_EXAMPLES, 3, 224, 224)
        labels = torch.randint(
            0, NUM_CLASSES, (MAX_TRAIN_EXAMPLES,), dtype=torch.long
        )
        dataset = TensorDataset(images, labels)
    else:
        # HF returns PIL images + integer labels. Slice notation downloads
        # only the requested rows on first access (subsequent runs hit
        # local arrow cache).
        n = max(1, MAX_TRAIN_EXAMPLES)
        ds = load_dataset(HF_DATASET, split=f"train[:{n}]")
        tfm = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408],
                    std=[0.2675, 0.2565, 0.2761],
                ),
            ]
        )

        def to_tensor(batch):
            batch["pixel_values"] = [
                tfm(img.convert("RGB")) for img in batch["img"]
            ]
            return batch

        dataset = ds.with_transform(to_tensor)

    def collate(rows):
        if USE_SYNTHETIC:
            xs = torch.stack([r[0] for r in rows])
            ys = torch.stack([r[1] for r in rows])
        else:
            xs = torch.stack([r["pixel_values"] for r in rows])
            ys = torch.tensor(
                [r["fine_label"] for r in rows], dtype=torch.long
            )
        return xs, ys

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collate,
    )


def build_model() -> nn.Module:
    model = torchvision.models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def main():
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    dtype = torch.float16 if use_amp else torch.float32

    traceml.init(mode="auto")

    train_loader = prepare_data()

    model = build_model().to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    global_step = 0
    running_loss = 0.0

    for epoch in range(EPOCHS):
        for images, labels in train_loader:
            # Canonical trace_step boundary per src/dev/scenarios/bert_ddp.py:
            # dataloading is OUTSIDE; H2D, zero_grad, fwd, bwd, opt all INSIDE.
            with traceml.trace_step(model):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(
                    device_type="cuda", enabled=use_amp, dtype=dtype
                ):
                    logits = model(images)
                    loss = criterion(logits, labels)

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
