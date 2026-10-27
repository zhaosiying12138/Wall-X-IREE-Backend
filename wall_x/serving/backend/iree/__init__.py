"""IREE serving backend; upstream runtime is imported only on initialization."""

__all__ = ["IreeExpandedBackend", "IreeFlowBackend"]


def __getattr__(name):
    if name in __all__:
        from wall_x.serving.backend.iree import model
        return getattr(model, name)
    raise AttributeError(name)
