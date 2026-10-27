"""Independent serving backends for Wall-X.

Importing this package does not import IREE or compile a model.  Use
:func:`create_backend` to select an implementation explicitly.
"""
from wall_x.serving.backend.base import FlowActionBackend, ServingBackend
from wall_x.serving.backend.registry import (
    available_backends,
    backend_class,
    create_backend,
    get_backend,
)

__all__ = [
    "FlowActionBackend",
    "ServingBackend",
    "available_backends",
    "backend_class",
    "create_backend",
    "get_backend",
]
