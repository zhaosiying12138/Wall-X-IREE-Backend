"""Shared helpers for IREE flow backends."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _artifact(path, description):
    if path is None or not str(path):
        raise ValueError(f"IREE {description} artifact must be configured explicitly")
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"IREE {description} artifact not found: {resolved}")
    return resolved


def _host(tensor, dtype=None):
    if dtype is not None:
        numpy_dtype = np.dtype(dtype)
        if numpy_dtype == np.dtype(np.float32):
            tensor = tensor.float()
        elif numpy_dtype == np.dtype(np.float64):
            tensor = tensor.double()
        elif numpy_dtype == np.dtype(np.int64):
            tensor = tensor.long()
        elif numpy_dtype == np.dtype(np.bool_):
            tensor = tensor.bool()
    array = tensor.detach().cpu().numpy()
    return array.astype(dtype, copy=False) if dtype is not None else array
