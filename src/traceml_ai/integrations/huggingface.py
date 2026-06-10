"""Hugging Face integration for TraceML.

Preferred entry point is :class:`TraceMLTrainerCallback`, a standard
``transformers.TrainerCallback`` that instruments any ``Trainer`` without
subclassing::

    from traceml_ai.integrations.huggingface import TraceMLTrainerCallback
    trainer = Trainer(..., callbacks=[TraceMLTrainerCallback()])

It drives the SDK ``trace_step`` context manager at the optimizer-step
boundaries (``on_step_begin`` / ``on_step_end``), so one TraceML step maps 1:1
to one Hugging Face *global* step. ``trace_step`` owns the rest: step timing,
forward/backward/optimizer/H2D auto-timers, peak-memory tracking, step
advancement and event flushing.

:class:`TraceMLTrainer` is kept as a thin, backward-compatible shim that simply
installs the callback (single source of truth, no double-counting).
"""

import logging
import os
from typing import Any, Dict, Optional

from traceml_ai.sdk.decorators_compat import trace_model_instance, trace_step

logger = logging.getLogger(__name__)

try:
    from transformers import Trainer, TrainerCallback

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    Trainer = object  # Fallback for type hinting
    TrainerCallback = object


def _disabled() -> bool:
    """Read the disable flag dynamically.

    Read at call time (not captured at import) so the integration responds to
    ``TRACEML_DISABLED`` being set after the module is imported (e.g. in tests
    or interactive sessions).
    """
    return os.environ.get("TRACEML_DISABLED") == "1"


class TraceMLTrainerCallback(TrainerCallback if HAS_TRANSFORMERS else object):
    """``transformers.TrainerCallback`` that instruments a Trainer with TraceML.

    One TraceML step == one Hugging Face global step (i.e. one optimizer
    update). Under gradient accumulation the per-micro-batch forward/backward
    times are summed into the global step by the step-time sampler.

    Parameters
    ----------
    traceml_kwargs:
        Optional kwargs forwarded to ``trace_model_instance`` for deep-profile
        per-layer hook attachment (only active when ``TRACEML_PROFILE='deep'``).
    """

    def __init__(self, traceml_kwargs: Optional[Dict[str, Any]] = None):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "TraceMLTrainerCallback requires 'transformers' to be "
                "installed. Please run `pip install transformers`."
            )
        super().__init__()
        self._traceml_kwargs = traceml_kwargs
        self._ctx = None

    def _close_ctx(self) -> None:
        """Best-effort close of the active ``trace_step`` context."""
        if self._ctx is not None:
            try:
                self._ctx.__exit__(None, None, None)
            except Exception as exc:
                logger.error("[TraceML] step context close failed: %s", exc)
            finally:
                self._ctx = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if _disabled():
            return
        # Self-heal: a reused callback whose previous run crashed mid-step may
        # have left a context open; closing it restores the auto-timer
        # thread-local flags before this run starts.
        self._close_ctx()
        if model is not None:
            # No-op unless TRACEML_PROFILE='deep'; attaches per-layer hooks.
            try:
                trace_model_instance(model, **(self._traceml_kwargs or {}))
            except Exception as exc:
                logger.error("[TraceML] trace_model_instance failed: %s", exc)

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if _disabled() or model is None:
            return
        try:
            self._ctx = trace_step(model)
            self._ctx.__enter__()
        except Exception as exc:
            logger.error("[TraceML] step begin failed: %s", exc)
            self._ctx = None

    def on_step_end(self, args, state, control, **kwargs):
        # Closing trace_step advances the step counter, records peak memory and
        # flushes the buffered events for this global step.
        self._close_ctx()

    def on_train_end(self, args, state, control, **kwargs):
        self._close_ctx()


class TraceMLTrainer(Trainer if HAS_TRANSFORMERS else object):
    """Backward-compatible thin shim that installs ``TraceMLTrainerCallback``.

    Preserves the v0.2.x import path and constructor signature
    (``traceml_enabled`` / ``traceml_kwargs``). The previous implementation
    overrode ``training_step``; that is no longer needed -- all instrumentation
    now flows through the callback (single source of truth, no double-counting).
    """

    def __init__(
        self,
        *args,
        traceml_enabled: bool = True,
        traceml_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "TraceMLTrainer requires 'transformers' to be installed. "
                "Please run `pip install transformers`."
            )

        super().__init__(*args, **kwargs)
        self.traceml_enabled = traceml_enabled
        self.traceml_kwargs = traceml_kwargs

        if not traceml_enabled or _disabled():
            return

        # Dedup guard: inspect the POST-init handler so we catch a
        # TraceMLTrainerCallback passed as either an instance OR the class form
        # (HF instantiates class-form callbacks during super().__init__()).
        existing = getattr(self.callback_handler, "callbacks", [])
        if any(isinstance(cb, TraceMLTrainerCallback) for cb in existing):
            return

        self.add_callback(
            TraceMLTrainerCallback(traceml_kwargs=traceml_kwargs)
        )
