from queue import Queue
from typing import Dict, Optional

import torch.nn as nn


def get_model_queue(
    model_id: Optional[int] = None,
) -> Queue:
    """Return the model queue for parameter-memory sampling.

    Parameters
    ----------
    model_id : int, optional
        If given, returns the session-scoped queue.
        If None, returns the first active session's queue.
    """
    from traceml.session_registry import (
        _registry,
        get_session,
    )

    if model_id is not None:
        return get_session(model_id).model_queue
    for sess in _registry.values():
        return sess.model_queue
    return Queue()


def collect_layer_parameter_memory(model: nn.Module) -> Dict[str, float]:
    """
    Collect per-layer parameter memory for a PyTorch model.

    This function performs a one-time, synchronous inspection of the model
    and computes parameter memory per leaf module. It is safe to call from
    training code and does not retain references to the model.

    Notes
    -----
    - Only leaf modules are included (containers are skipped)
    - Only parameters directly owned by the module are counted
    - Returned values are in bytes
    - Intended to be called once per model instance or architecture

    Parameters
    ----------
    model : nn.Module
        PyTorch model instance.

    Returns
    -------
    Dict[str, float]
        Mapping from module name to parameter memory in bytes.
    """
    layer_memory: Dict[str, float] = {}

    for name, module in model.named_modules():
        # Skip container modules
        if any(module.children()):
            continue

        total_bytes = 0.0
        for p in module.parameters(recurse=False):
            total_bytes += p.element_size() * p.nelement()

        if total_bytes > 0:
            layer_memory[name] = total_bytes

    return layer_memory
