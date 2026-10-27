"""One exportable WALL-OSS full-flow tensor graph.

``StaticFullFlow`` is the production module: its sole output is the final
normalized action.  ``StaticFullFlowTrace`` is a separate diagnostic module with
the same weights and computation plus context/KV/solver trace outputs.  Do not
deploy the trace VMFB as the serving artifact.

The shared forward computation directly chains:

  tokenized text/state + ViT context
  -> t=0 action embedding and ordered scatter
  -> joint prefill (K/V, v0, first Euler state x0)
  -> nine suffix steps consuming those K/V tensors
  -> all nine in-graph Euler updates
  -> final normalized action.

K/V and every intermediate action state remain tensor values inside one main
invocation.  Explicit noise, the actual official time grid, runtime dof_mask,
and v_padding=padding_action-initial_noise are inputs/computations, not
regenerated constants.

Host-only boundary: JPEG decode, image resize/normalization, tokenizer and
state-string quantization, mRoPE position construction, joint/suffix masks,
noise generation, action-normalizer lookup/final unnormalization, msgpack and
serialization.  The graph begins at processor tensors and ends at the final
*normalized* action; it is not an end-to-end WebSocket graph.
"""
from __future__ import annotations

import torch

from wall_x.serving.backend.static_flow.euler import StaticDenoiseEulerLoop
from wall_x.serving.backend.static_flow.prefill import StaticJointPrefill
from wall_x.serving.backend.static_flow.vision import (
    StaticVisionTextContext,
    StaticVisionTextStateContext,
    _ordered_scatter,
)


class StaticFullFlow(torch.nn.Module):
    """Production unified graph for state-string WALL-OSS-0.5."""

    TRACE = False

    def __init__(self, model, grid_thw: torch.Tensor,
                 dof_mask: torch.Tensor | None = None,
                 vision_attention: str = "eager-bf16"):
        super().__init__()
        if getattr(model.config, "use_adarms", False):
            raise ValueError(
                "StaticJointPrefill currently requires use_adarms=False; "
                "conditional prefill norms need a distinct verified export")
        if dof_mask is None:
            dof_mask = torch.ones(1, 32, 26, dtype=torch.float32)
        self.vision_attention = vision_attention
        self.context = StaticVisionTextContext(
            model, grid_thw, attention_mode=vision_attention)
        self.prefill = StaticJointPrefill(model, dof_mask=dof_mask)
        self.solver = StaticDenoiseEulerLoop(model, dof_mask=dof_mask)
        self.ap = model.action_preprocessor
        # One official action processor is shared by t=0, prefill projection,
        # and all suffix steps in the unified module.
        self.prefill.ap = self.ap
        self.solver.step.ap = self.ap
        self.action_token_id = int(
            model.action_token_id_set["action_token_id"])

    def _execute(self, context_features, input_ids, position_ids, mask3d,
                 noise, padding_action, times, postfix_pos, suffix_mask,
                 dof_mask):
        action_embed, _adarms_cond = self.ap.step(
            timestep=times[0].reshape(1), noisy_action=noise,
            dof_mask=dof_mask)
        action_embed = action_embed.reshape(
            -1, context_features.shape[-1]).to(context_features.dtype)
        action_mask = input_ids == self.action_token_id
        embeds = _ordered_scatter(context_features, action_mask, action_embed)

        k, v, v0, x0 = self.prefill(
            embeds.float(), position_ids, mask3d, noise, padding_action,
            (times[1] - times[0]).reshape(1), dof_mask=dof_mask)
        v_padding = padding_action - noise
        if self.TRACE:
            final, velocities, trajectory = self.solver(
                x0, times, postfix_pos, suffix_mask, k, v, v_padding,
                dof_mask=dof_mask)
            return (final, embeds.float(), k, v, v0, x0, velocities, trajectory)
        final = self.solver.forward_final(
            x0, times, postfix_pos, suffix_mask, k, v, v_padding,
            dof_mask=dof_mask)
        return (final,)

    def _forward_all(self, input_ids, pixel_values, position_ids, mask3d,
                     noise, padding_action, times, postfix_pos, suffix_mask,
                     dof_mask):
        features = self.context.forward_features(input_ids, pixel_values)
        return self._execute(
            features, input_ids, position_ids, mask3d, noise, padding_action,
            times, postfix_pos, suffix_mask, dof_mask)

    def forward(self, input_ids, pixel_values, position_ids, mask3d,
                noise, padding_action, times, postfix_pos, suffix_mask,
                dof_mask):
        # Production ABI: exactly one output.  Diagnostic tensors are not
        # returned and cannot be copied to host by the serving session.
        return self._forward_all(
            input_ids, pixel_values, position_ids, mask3d, noise,
            padding_action, times, postfix_pos, suffix_mask, dof_mask)[0]


class StaticFullFlowTrace(StaticFullFlow):
    """Diagnostic-only unified graph with eight inspectable outputs."""

    TRACE = True

    def forward(self, input_ids, pixel_values, position_ids, mask3d,
                noise, padding_action, times, postfix_pos, suffix_mask,
                dof_mask):
        # final, embeds, K, V, v0, x0, nine velocities, ten states
        return self._forward_all(
            input_ids, pixel_values, position_ids, mask3d, noise,
            padding_action, times, postfix_pos, suffix_mask, dof_mask)


class StaticFullFlowTensorState(StaticFullFlow):
    """Production unified ABI with two additional tensor-state inputs."""

    def __init__(self, model, grid_thw: torch.Tensor,
                 dof_mask: torch.Tensor | None = None,
                 vision_attention: str = "eager-bf16"):
        super().__init__(model, grid_thw, dof_mask=dof_mask,
                         vision_attention=vision_attention)
        self.context = StaticVisionTextStateContext(
            model, grid_thw, attention_mode=vision_attention)

    def _forward_all_tensor_state(
            self, input_ids, pixel_values, proprioception, agent_pos_mask,
            position_ids, mask3d, noise, padding_action, times, postfix_pos,
            suffix_mask, dof_mask):
        features = self.context.forward_features(
            input_ids, pixel_values, proprioception, agent_pos_mask)
        return self._execute(
            features, input_ids, position_ids, mask3d, noise, padding_action,
            times, postfix_pos, suffix_mask, dof_mask)

    def forward(self, input_ids, pixel_values, proprioception, agent_pos_mask,
                position_ids, mask3d, noise, padding_action, times,
                postfix_pos, suffix_mask, dof_mask):
        return self._forward_all_tensor_state(
            input_ids, pixel_values, proprioception, agent_pos_mask,
            position_ids, mask3d, noise, padding_action, times, postfix_pos,
            suffix_mask, dof_mask)[0]


class StaticFullFlowTensorStateTrace(StaticFullFlowTensorState):
    """Diagnostic-only tensor-state unified graph."""

    TRACE = True

    def forward(self, input_ids, pixel_values, proprioception, agent_pos_mask,
                position_ids, mask3d, noise, padding_action, times,
                postfix_pos, suffix_mask, dof_mask):
        return self._forward_all_tensor_state(
            input_ids, pixel_values, proprioception, agent_pos_mask,
            position_ids, mask3d, noise, padding_action, times, postfix_pos,
            suffix_mask, dof_mask)
