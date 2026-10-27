"""Torch-compile peer using the shared static Wall-X prefill and suffix."""
from __future__ import annotations

import torch

from wall_x.serving.backend.static_flow import StaticDenoiseStep, StaticJointPrefill
from wall_x.serving.backend.wallx_flow import WallxFlowBackend


def execute_static_flow(
    prefill,
    suffix,
    *,
    inputs_embeds,
    position_ids,
    mask3d,
    noise,
    padding_action,
    times,
    postfix_positions,
    postfix_mask,
    dof_mask,
):
    """Run the validated static 1+(N-1) Euler flow on device tensors.

    K/V and x0 cross separately compiled regions.  They must be cloned before a
    reduce-overhead suffix replay can invalidate or reuse prefill graph buffers.
    """
    k, v, _v0, x0 = prefill(
        inputs_embeds,
        position_ids,
        mask3d,
        noise,
        padding_action,
        (times[1] - times[0]).reshape(1),
        dof_mask=dof_mask,
    )
    k = k.clone()
    v = v.clone()
    x = x0.clone()
    v_padding = padding_action - noise
    for index in range(len(times) - 2):
        timestep = times[1 + index].reshape(1)
        velocity = suffix(
            x,
            timestep,
            postfix_positions,
            postfix_mask,
            k,
            v,
            v_padding=v_padding,
            dof_mask=dof_mask,
        )
        dt = times[2 + index] - times[1 + index]
        x = x + dt * velocity
    return x


class TorchCompileFlowBackend(WallxFlowBackend):
    """Independent torch.compile backend over static prefill/suffix modules."""

    name = "torch-compile"

    def __init__(
        self,
        model,
        *,
        mode="reduce-overhead",
        backend=None,
        compile_fn=None,
        prefill=None,
        suffix=None,
    ):
        super().__init__(model)
        self.mode = mode
        self.compile_backend = backend
        if prefill is None:
            prefill = StaticJointPrefill(model)
        if suffix is None:
            suffix = StaticDenoiseStep(model, len(model.model.layers))
        try:
            device = next(model.parameters()).device
        except (AttributeError, StopIteration):
            device = torch.device("cpu")
        prefill = prefill.to(device).eval()
        suffix = suffix.to(device).eval()
        compile_fn = torch.compile if compile_fn is None else compile_fn
        compile_options = {"dynamic": False}
        if mode is not None:
            compile_options["mode"] = mode
        if backend is not None:
            compile_options["backend"] = backend
        self.prefill = compile_fn(prefill, **compile_options)
        self.suffix = compile_fn(suffix, **compile_options)

    def generate_flow_action(
        self,
        *,
        action_horizon,
        action_dim,
        num_inference_timesteps,
        padding_action,
        noise,
        **batch_inputs,
    ):
        with torch.no_grad():
            batch_inputs = self._pad_to_bucket(batch_inputs)
            ctx = self.prepare_ctx(
                action_horizon,
                action_dim,
                num_inference_timesteps,
                padding_action,
                noise,
                batch_inputs,
            )
            self._validate_static_context(ctx, action_horizon, action_dim)
            mask3d = self._joint_attention_mask(ctx)
            postfix_positions, postfix_mask = self._postfix(ctx)
            final = execute_static_flow(
                self.prefill,
                self.suffix,
                inputs_embeds=ctx["inputs_embeds"].detach().float(),
                position_ids=ctx["position_ids"],
                mask3d=mask3d,
                noise=ctx["noise"],
                padding_action=self._padding_action(ctx),
                times=ctx["times"],
                postfix_positions=postfix_positions,
                postfix_mask=postfix_mask,
                dof_mask=ctx["dof_mask"],
            )
            return self._finalize(ctx, final)
