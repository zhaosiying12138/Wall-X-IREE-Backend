"""Lazy compatibility exports for the split IREE flow backends."""

__all__ = ["IreeExpandedBackend", "IreeFlowBackend"]


def __getattr__(name):
    if name == "IreeFlowBackend":
        from wall_x.serving.backend.iree.staged import IreeFlowBackend
        return IreeFlowBackend
    if name == "IreeExpandedBackend":
        from wall_x.serving.backend.iree.expanded import IreeExpandedBackend
        return IreeExpandedBackend
    raise AttributeError(name)
