"""Unrolled WALL-OSS suffix solver with every Euler update inside IREE.

``StaticJointPrefill`` already computes the official first update
``x0 = noise + (times[1] - times[0]) * v0``.  This module consumes that x0 and
the complete official time grid (11 entries for num_inference_timesteps=10),
then statically unrolls the remaining nine calls exactly as
``torchdiffeq.odeint(..., times[1:], method="euler")``:

    for i in 0..8:
        v_i = denoise(x_i, times[1 + i])
        x_(i+1) = x_i + (times[2 + i] - times[1 + i]) * v_i

K/V stay in their full 638-row device buffers and are cropped by the existing
``StaticDenoiseStep`` in-graph.  ``v_padding = padding_action - initial_noise``
is fixed for all steps and the dof mask semantics remain in the existing step.
The normalized final action is returned; official unnormalization remains an
explicit host/model operation.  Velocity and trajectory outputs are small and
make every solver step independently verifiable with the same noise.
"""
from __future__ import annotations

import torch

from wall_x.serving.backend.static_flow.suffix import StaticDenoiseStep


class StaticDenoiseEulerLoop(torch.nn.Module):
    """Fixed 9-step compiled Euler schedule for B=1/A=32/D=26."""

    def __init__(self, model, dof_mask: torch.Tensor,
                 num_inference_timesteps: int = 10):
        super().__init__()
        if num_inference_timesteps != 10:
            raise ValueError(
                "StaticDenoiseEulerLoop is compiled for the official fixed "
                f"1+9 schedule (num_inference_timesteps=10); got "
                f"{num_inference_timesteps}. Export a separate unrolled "
                "module for another schedule.")
        if getattr(model.config, "use_x_pred", False):
            raise ValueError(
                "StaticDenoiseStep implements velocity prediction only; "
                "use_x_pred=True requires a distinct verified export")
        self.num_inference_timesteps = num_inference_timesteps
        self.step = StaticDenoiseStep(
            model, len(model.model.layers), dof_mask=dof_mask)

    def forward_final(self, x0: torch.Tensor, times: torch.Tensor,
                      postfix_pos: torch.Tensor, suffix_mask: torch.Tensor,
                      k_past_f32: torch.Tensor, v_past_f32: torch.Tensor,
                      v_padding: torch.Tensor,
                      dof_mask: torch.Tensor | None = None):
        """Production solver: identical Euler math without any trace stacks."""
        x = x0
        for i in range(self.num_inference_timesteps - 1):
            timestep = times[1 + i].reshape(1)
            dt = times[2 + i] - times[1 + i]
            velocity = self.step(
                x, timestep, postfix_pos, suffix_mask,
                k_past_f32, v_past_f32, v_padding=v_padding,
                dof_mask=dof_mask)
            x = x + dt * velocity
        return x

    def forward(self, x0: torch.Tensor, times: torch.Tensor,
                postfix_pos: torch.Tensor, suffix_mask: torch.Tensor,
                k_past_f32: torch.Tensor, v_past_f32: torch.Tensor,
                v_padding: torch.Tensor,
                dof_mask: torch.Tensor | None = None):
        # ``times`` is the official [0..10] grid, not a recomputed linspace.
        # Static slicing preserves its actual shifted/scaled values and the
        # exact fp32 subtraction order used by the official Euler solver.
        x = x0
        velocities = []
        trajectory = [x0]
        for i in range(self.num_inference_timesteps - 1):
            timestep = times[1 + i].reshape(1)
            dt = times[2 + i] - times[1 + i]
            velocity = self.step(
                x, timestep, postfix_pos, suffix_mask,
                k_past_f32, v_past_f32, v_padding=v_padding,
                dof_mask=dof_mask)
            x = x + dt * velocity
            velocities.append(velocity)
            trajectory.append(x)
        return x, torch.stack(velocities), torch.stack(trajectory)
