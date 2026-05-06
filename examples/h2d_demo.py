"""
H2D (host-to-device) timing demo.

Mirrors the structure of ``src/dev/scenarios/bert_ddp.py`` (HF model + AdamW
+ AMP-style training loop) but trimmed to single-process, single-GPU/CPU,
synthetic data — runs out of the box with only ``torch`` + ``transformers``
installed, no dataset download, no torchrun.

Demonstrates two ways to surface ``_traceml_internal:h2d_time`` events:

1. **Auto mode** (default in this script) — ``traceml.init(mode="auto")``
   installs the global ``Tensor.to`` patch. Every ``batch.to(device)`` call
   inside ``trace_step`` is timed automatically with zero user code change.

2. **Manual mode** — set ``USE_MANUAL_MODE = True``. ``traceml.init(mode="manual")``
   skips the global patch; ``traceml.wrap_h2d(t).to(device)`` times exactly
   the tensors the user wraps. Useful when you want surgical control over
   which transfers count toward the H2D budget.

After training, the script drains ``_STEP_TIME_QUEUE`` and prints a summary
of the event names that landed in the buffer, so you can confirm
``_traceml_internal:h2d_time`` reached the queue.

Usage::

    python examples/h2d_demo.py
    # or:
    USE_MANUAL_MODE=1 python examples/h2d_demo.py

Note on coverage: the auto patch fires on ``Tensor.to(...)`` only.
Convenience shortcuts (``.cuda()``, ``.cpu()``, ``.float()``, ``.half()``)
bypass the patch at the C++ level and will NOT produce h2d events. Prefer
``.to(device, non_blocking=True)`` in training-loop code.
"""

from __future__ import annotations

import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import traceml

SEED = 42
INPUT_DIM = 128
HIDDEN_DIM = 256
NUM_CLASSES = 4
BATCH_SIZE = 16
NUM_SAMPLES = 256
EPOCHS = 2

USE_MANUAL_MODE = os.environ.get("USE_MANUAL_MODE", "0") == "1"


class TinyClassifier(nn.Module):
    """A small classifier — just enough to exercise forward + backward."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        logits = self.net(input_ids)
        loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}


def make_loader() -> DataLoader:
    """Build a deterministic synthetic loader with pinned memory.

    Pinning is what lets ``non_blocking=True`` actually overlap the H2D copy
    with compute on the GPU path. On CPU it has no effect but is harmless.
    """
    inputs = torch.randn(NUM_SAMPLES, INPUT_DIM)
    labels = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    dataset = TensorDataset(inputs, labels)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
    )


def load_batch_to_device(
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move a (input, label) pair to ``device``.

    In auto mode each ``.to(...)`` call below is timed automatically. In
    manual mode we wrap each tensor with ``traceml.wrap_h2d(...)`` so only
    the wrapped transfers count.
    """
    inputs, labels = batch
    if USE_MANUAL_MODE:
        inputs = traceml.wrap_h2d(inputs).to(device, non_blocking=True)
        labels = traceml.wrap_h2d(labels).to(device, non_blocking=True)
    else:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
    return {"input_ids": inputs, "labels": labels}


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "manual" if USE_MANUAL_MODE else "auto"
    print(f"Running on {device} | TraceML mode: {mode}")

    traceml.init(mode=mode)

    loader = make_loader()
    model = TinyClassifier().to(device)  # h2d gate is OFF here — silent
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    if USE_MANUAL_MODE:
        # In manual mode trace_step still tracks step boundaries, but the
        # forward / backward / optimizer hooks aren't installed globally —
        # use the wrap_* helpers to instrument what you care about.
        optimizer = traceml.wrap_optimizer(optimizer)

    model.train()
    global_step = 0

    for epoch in range(EPOCHS):
        for raw_batch in loader:

            with traceml.trace_step(model):
                # H2D transfers happen here. Auto mode: timed via patch.
                # Manual mode: timed via wrap_h2d proxy in load_batch_to_device.
                batch = load_batch_to_device(raw_batch, device)

                optimizer.zero_grad(set_to_none=True)

                if USE_MANUAL_MODE:
                    out = model(**batch)
                    loss = out["loss"]
                    traceml.wrap_backward(loss).backward()
                    optimizer.step()
                else:
                    out = model(**batch)
                    loss = out["loss"]
                    loss.backward()
                    optimizer.step()

                global_step += 1

                if global_step % 8 == 0:
                    print(
                        f"  epoch {epoch + 1} step {global_step} "
                        f"| loss {loss.item():.4f}"
                    )

    # ----------------------------------------------------------------
    # Drain the step-time queue and summarize observed event names so a
    # user can confirm h2d_time landed alongside step_time / forward_time
    # / backward_time / dataloader_next.
    # ----------------------------------------------------------------
    from traceml.utils.timing import _STEP_TIME_QUEUE

    counts: dict[str, int] = {}
    while not _STEP_TIME_QUEUE.empty():
        batch_obj = _STEP_TIME_QUEUE.get_nowait()
        for evt in batch_obj.events:
            counts[evt.name] = counts.get(evt.name, 0) + 1

    print("\nObserved event counts (after draining _STEP_TIME_QUEUE):")
    for name in sorted(counts):
        print(f"  {name:42s} {counts[name]}")

    h2d_count = counts.get("_traceml_internal:h2d_time", 0)
    if h2d_count == 0:
        print(
            "\n[!] No h2d_time events observed. In manual mode, ensure "
            "wrap_h2d(...) is called for the tensors you want timed. In "
            "auto mode, ensure ``traceml.init(mode='auto')`` ran before "
            "the first ``trace_step``."
        )
    else:
        print(f"\n[OK] {h2d_count} h2d_time event(s) recorded.")


if __name__ == "__main__":
    main()
