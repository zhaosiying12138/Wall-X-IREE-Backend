"""Common interface for Wall-X flow-action serving backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ServingBackend(ABC):
    """A delegated implementation of the model's flow-action call boundary.

    Backends own execution only.  The vendored serving wrapper still owns input
    construction and robot-action serialization.  Implementations must either
    return a finalized model output or raise; falling back to another backend is
    deliberately not part of this interface.
    """

    name = "abstract"

    def __init__(self, model: Any):
        self.model = model

    def initialize(self) -> None:
        """Initialize optional runtime resources before the first request."""

    @abstractmethod
    def generate_flow_action(self, **kwargs):
        """Execute and finalize one flow-action request."""


FlowActionBackend = ServingBackend
