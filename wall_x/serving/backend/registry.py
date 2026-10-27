"""Lazy serving-backend registry and explicit environment configuration."""
from __future__ import annotations

import os
from importlib import import_module
from typing import Any

DEFAULT_BACKEND = "eager"
BACKEND_ENV = "WALLX_BACKEND"

# Values are import paths, not classes: selecting eager must not import IREE.
_BACKENDS = {
    "eager": ("wall_x.serving.backend.eager", "EagerFlowBackend"),
    "torch-compile": (
        "wall_x.serving.backend.torch_compile",
        "TorchCompileFlowBackend",
    ),
    "iree": ("wall_x.serving.backend.iree.model", "IreeFlowBackend"),
    "iree-expanded": (
        "wall_x.serving.backend.iree.model",
        "IreeExpandedBackend",
    ),
}
_ALIASES = {
    "official": "eager",
    "torch_compile": "torch-compile",
    "torch.compile": "torch-compile",
    "staged": "iree",
    "iree-staged": "iree",
    "expanded": "iree-expanded",
    "iree_expanded": "iree-expanded",
}

_ARTIFACT_ENV = {
    "iree": {
        "prefill_vmfb": "WALLX_IREE_PREFILL_VMFB",
        "prefill_irpa": "WALLX_IREE_PREFILL_IRPA",
        "step_vmfb": "WALLX_IREE_STEP_VMFB",
        "step_irpa": "WALLX_IREE_STEP_IRPA",
    },
    "iree-expanded": {
        "full_flow_vmfb": "WALLX_IREE_FULL_FLOW_VMFB",
        "full_flow_irpa": "WALLX_IREE_FULL_FLOW_IRPA",
    },
}


def normalize_backend_name(name: str) -> str:
    """Resolve a documented alias, rejecting every unregistered name."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"unknown Wall-X serving backend: {name!r}")
    normalized = _ALIASES.get(name, name)
    if normalized not in _BACKENDS:
        choices = ", ".join(available_backends())
        raise ValueError(
            f"unknown Wall-X serving backend {name!r}; choose one of: {choices}"
        )
    return normalized


def available_backends() -> tuple[str, ...]:
    return tuple(_BACKENDS)


def backend_class(name: str):
    """Load one backend class on demand."""
    normalized = normalize_backend_name(name)
    module_name, class_name = _BACKENDS[normalized]
    return getattr(import_module(module_name), class_name)


def _environment_options(name: str) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if name == "torch-compile":
        mode = os.environ.get("WALLX_TORCH_COMPILE_MODE")
        if mode:
            options["mode"] = None if mode == "default" else mode
        compile_backend = os.environ.get("WALLX_TORCH_COMPILE_BACKEND")
        if compile_backend:
            options["backend"] = compile_backend
    if name in _ARTIFACT_ENV:
        for option, env_name in _ARTIFACT_ENV[name].items():
            value = os.environ.get(env_name)
            if value:
                options[option] = value
    if name.startswith("iree"):
        for option, env_name in (
            ("device", "WALLX_IREE_DEVICE"),
            ("cudalib", "WALLX_IREE_CUDALIB"),
        ):
            value = os.environ.get(env_name)
            if value:
                options[option] = value
    return options


def create_backend(model, name: str | None = None, **options):
    """Construct a backend without initializing its runtime.

    ``name=None`` reads ``WALLX_BACKEND`` and otherwise selects eager.  Artifact
    paths for IREE are mandatory and come from keywords or their explicit
    environment variables; no repository-local artifact is inferred.
    """
    selected = name if name is not None else os.environ.get(
        BACKEND_ENV, DEFAULT_BACKEND
    )
    normalized = normalize_backend_name(selected)
    configured = _environment_options(normalized)
    configured.update(options)
    if normalized in _ARTIFACT_ENV:
        missing = [
            env_name
            for option, env_name in _ARTIFACT_ENV[normalized].items()
            if not configured.get(option)
        ]
        if missing:
            raise ValueError(
                f"backend {normalized!r} requires explicit artifact "
                f"configuration; set {', '.join(missing)} or pass the "
                "corresponding keyword arguments to create_backend"
            )
    return backend_class(normalized)(model, **configured)


def get_backend(model, name: str | None = None, **options):
    """Compatibility spelling for :func:`create_backend`."""
    return create_backend(model, name=name, **options)
