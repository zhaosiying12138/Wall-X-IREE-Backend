"""Default Wall-X eager backend."""
from __future__ import annotations

from wall_x.serving.backend.base import ServingBackend


class EagerFlowBackend(ServingBackend):
    """Delegate to the official model without changing its execution path."""

    name = "eager"

    def generate_flow_action(self, **kwargs):
        # In particular, do not consume or regenerate an explicitly supplied
        # noise tensor: the official model owns that contract.  Omitting an
        # unset optional value also preserves older Wall-X model signatures.
        if kwargs.get("noise") is None:
            kwargs.pop("noise", None)
        return self.model.generate_flow_action(**kwargs)
