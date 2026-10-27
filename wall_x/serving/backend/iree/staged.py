"""Staged IREE implementation of the common flow interface."""
from __future__ import annotations

import numpy as np
import torch

from wall_x.serving.backend.iree._utils import _artifact, _host
from wall_x.serving.backend.wallx_flow import WallxFlowBackend


class IreeFlowBackend(WallxFlowBackend):
    """Torch context preparation plus staged IREE prefill/suffix execution."""

    name = "iree"
    _host = staticmethod(_host)

    def __init__(
        self,
        model,
        prefill_vmfb,
        prefill_irpa,
        step_vmfb,
        step_irpa,
        *,
        device="cuda",
        cudalib=None,
    ):
        super().__init__(model)
        self.prefill_vmfb = _artifact(prefill_vmfb, "joint prefill VMFB")
        self.prefill_irpa = _artifact(prefill_irpa, "joint prefill IRPA")
        self.step_vmfb = _artifact(step_vmfb, "denoise step VMFB")
        self.step_irpa = _artifact(step_irpa, "denoise step IRPA")
        self.device = device
        self.cudalib = cudalib
        self._session = None

    def _get_session(self):
        if self._session is None:
            from wall_x.serving.backend.iree.runtime import IreeResidentSession

            self._session = IreeResidentSession(
                self.prefill_vmfb,
                self.prefill_irpa,
                self.step_vmfb,
                self.step_irpa,
                device=self.device,
                cudalib=self.cudalib,
            )
        return self._session

    def initialize(self):
        self._get_session()

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
            times = self._host(ctx["times"], np.float32)
            padding = self._padding_action(ctx)
            prefill_inputs = (
                self._host(ctx["inputs_embeds"], np.float32),
                self._host(ctx["position_ids"]),
                self._host(mask3d, np.bool_),
                self._host(ctx["noise"], np.float32),
                self._host(padding, np.float32),
                np.asarray([times[1] - times[0]], dtype=np.float32),
            )
            session = self._get_session()
            k_past, v_past, _v0, x = session.prefill(*prefill_inputs)
            x = np.asarray(x, dtype=np.float32)
            postfix_positions = self._host(postfix_positions)
            postfix_mask = self._host(postfix_mask, np.bool_)
            v_padding = self._host(padding - ctx["noise"], np.float32)
            for index in range(len(times) - 2):
                velocity = session.step(
                    x,
                    np.asarray([times[1 + index]], dtype=np.float32),
                    postfix_positions,
                    postfix_mask,
                    k_past,
                    v_past,
                    v_padding,
                )
                x = x + (times[2 + index] - times[1 + index]) * velocity
            predict = torch.from_numpy(x).to(ctx["inputs_embeds"].device)
            return self._finalize(ctx, predict)
