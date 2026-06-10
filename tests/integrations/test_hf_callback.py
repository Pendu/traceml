"""Tests for the callback-based Hugging Face integration (issue #88).

Mirrors test_hf_trainer.py: a tiny local BERT, no downloads, CPU-friendly.
Asserts the global-step alignment contract (1 TraceML step == 1 HF global
step), single optimizer event per step (no synthetic dummies), the
forward/backward split in run mode, shim equivalence, and the disable switch.
"""

import tempfile

import pytest

try:
    import torch
    from transformers import (
        BertConfig,
        BertForSequenceClassification,
        Trainer,
        TrainingArguments,
    )

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from traceml_ai.integrations.huggingface import (
    TraceMLTrainer,
    TraceMLTrainerCallback,
)
from traceml_ai.sdk.decorators_compat import TraceState
from traceml_ai.utils.timing import _STEP_BUFFER, get_step_time_queue

pytestmark = pytest.mark.skipif(
    not HAS_TRANSFORMERS, reason="transformers not installed"
)

FWD = "_traceml_internal:forward_time"
BWD = "_traceml_internal:backward_time"
OPT = "_traceml_internal:optimizer_step"
STEP = "_traceml_internal:step_time"


def _reset_timing():
    """Isolate a test from leftover buffered/queued events."""
    _STEP_BUFFER.clear()
    q = get_step_time_queue()
    while True:
        try:
            q.get_nowait()
        except Exception:
            break


def _drain():
    q = get_step_time_queue()
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except Exception:
            break
    return out


class _TinyTokenizedDataset(
    torch.utils.data.Dataset if HAS_TRANSFORMERS else object
):
    def __init__(self, num_rows=64, seq_len=16, vocab_size=128, num_labels=4):
        self._rows = []
        for idx in range(num_rows):
            token_ids = torch.arange(seq_len, dtype=torch.long) % vocab_size
            token_ids = token_ids + (idx % 7)
            self._rows.append(
                {
                    "input_ids": token_ids.clone(),
                    "attention_mask": torch.ones(seq_len, dtype=torch.long),
                    "labels": torch.tensor(idx % num_labels, dtype=torch.long),
                }
            )

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        return self._rows[index]


def _tiny_model():
    return BertForSequenceClassification(
        BertConfig(
            vocab_size=128,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=32,
            num_labels=4,
        )
    )


def _args(tmp, max_steps=5, gas=1):
    return TrainingArguments(
        output_dir=tmp,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=gas,
        max_steps=max_steps,
        logging_steps=1,
        use_cpu=not torch.cuda.is_available(),
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
    )


def test_callback_runs():
    _reset_timing()
    before = TraceState.step
    with tempfile.TemporaryDirectory() as tmp:
        Trainer(
            model=_tiny_model(),
            args=_args(tmp, max_steps=5),
            train_dataset=_TinyTokenizedDataset(),
            callbacks=[TraceMLTrainerCallback()],
        ).train()
    assert TraceState.step - before == 5


def test_global_step_alignment():
    """gas=2: one TraceML step per OPTIMIZER update (not per micro-batch),
    exactly one optimizer event per step (no dummies), fwd/bwd once per
    micro-batch (summed downstream)."""
    _reset_timing()
    before = TraceState.step
    gas, max_steps = 2, 4
    with tempfile.TemporaryDirectory() as tmp:
        Trainer(
            model=_tiny_model(),
            args=_args(tmp, max_steps=max_steps, gas=gas),
            train_dataset=_TinyTokenizedDataset(num_rows=64),
            callbacks=[TraceMLTrainerCallback()],
        ).train()

    # 1:1 with HF global steps, NOT micro-batches (would be max_steps*gas):
    assert TraceState.step - before == max_steps

    batches = _drain()
    assert len(batches) == max_steps
    for b in batches:
        names = [e.name for e in b.events]
        assert names.count(OPT) == 1, "exactly one (real) optimizer event/step"
        assert names.count(FWD) == gas, "one forward per micro-batch"
        assert names.count(BWD) == gas, "one backward per micro-batch"


def test_forward_backward_split_run_mode():
    _reset_timing()
    with tempfile.TemporaryDirectory() as tmp:
        Trainer(
            model=_tiny_model(),
            args=_args(tmp, max_steps=3),
            train_dataset=_TinyTokenizedDataset(),
            callbacks=[TraceMLTrainerCallback()],
        ).train()
    batches = _drain()
    assert batches
    for b in batches:
        names = {e.name for e in b.events}
        assert FWD in names and BWD in names
    fwd = [e for b in batches for e in b.events if e.name == FWD]
    assert any(e.cpu_end > e.cpu_start for e in fwd), "real fwd CPU duration"


def test_shim_equivalence():
    """TraceMLTrainer (shim) behaves like Trainer(callbacks=[...])."""
    _reset_timing()
    b0 = TraceState.step
    with tempfile.TemporaryDirectory() as tmp:
        TraceMLTrainer(
            model=_tiny_model(),
            args=_args(tmp, max_steps=5),
            train_dataset=_TinyTokenizedDataset(),
        ).train()
    shim_delta = TraceState.step - b0
    shim_names = {e.name for b in _drain() for e in b.events}

    b1 = TraceState.step
    with tempfile.TemporaryDirectory() as tmp:
        Trainer(
            model=_tiny_model(),
            args=_args(tmp, max_steps=5),
            train_dataset=_TinyTokenizedDataset(),
            callbacks=[TraceMLTrainerCallback()],
        ).train()
    cb_delta = TraceState.step - b1
    cb_names = {e.name for b in _drain() for e in b.events}

    assert shim_delta == cb_delta == 5
    assert shim_names == cb_names


def test_disabled(monkeypatch):
    monkeypatch.setenv("TRACEML_DISABLED", "1")
    _reset_timing()
    before = TraceState.step
    with tempfile.TemporaryDirectory() as tmp:
        Trainer(
            model=_tiny_model(),
            args=_args(tmp, max_steps=3),
            train_dataset=_TinyTokenizedDataset(),
            callbacks=[TraceMLTrainerCallback()],
        ).train()
    assert TraceState.step == before
    assert _drain() == []


@pytest.mark.parametrize("use_class", [False, True])
def test_shim_dedups_user_callback(use_class):
    """TraceMLTrainer must not double-install when the user also passes the
    callback in callbacks=[...] -- whether as an INSTANCE or the CLASS form.
    HF instantiates the class form during super().__init__(), so a pre-init
    isinstance check would miss it and double-instrument (advancing the step
    counter twice per optimizer step)."""
    _reset_timing()
    cb = TraceMLTrainerCallback if use_class else TraceMLTrainerCallback()
    with tempfile.TemporaryDirectory() as tmp:
        trainer = TraceMLTrainer(
            model=_tiny_model(),
            args=_args(tmp, max_steps=2),
            train_dataset=_TinyTokenizedDataset(),
            callbacks=[cb],
        )
        installed = [
            c
            for c in trainer.callback_handler.callbacks
            if isinstance(c, TraceMLTrainerCallback)
        ]
        assert len(installed) == 1, (
            f"expected exactly one TraceMLTrainerCallback after dedup, "
            f"found {len(installed)}"
        )
