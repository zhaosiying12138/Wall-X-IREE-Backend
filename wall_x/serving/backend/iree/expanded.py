"""Expanded single-graph IREE implementation of the common flow interface."""
from __future__ import annotations

import numpy as np
import torch

from wall_x.serving.backend.iree._utils import _artifact, _host
from wall_x.serving.backend.wallx_flow import WallxFlowBackend


class IreeExpandedBackend(WallxFlowBackend):
    """Single-invocation IREE graph for context, prefill, and unrolled Euler."""

    name = "iree-expanded"
    EXPECTED_GRID = ((1, 32, 32), (1, 32, 32))
    EXPECTED_PIXELS = (2048, 1176)
    EXPECTED_IMAGE_FEATURES = 512
    _host = staticmethod(_host)

    def __init__(
        self,
        model,
        full_flow_vmfb,
        full_flow_irpa,
        *,
        device="cuda",
        cudalib=None,
        trace=False,
    ):
        super().__init__(model)
        self.full_flow_vmfb = _artifact(full_flow_vmfb, "full-flow VMFB")
        self.full_flow_irpa = _artifact(full_flow_irpa, "full-flow IRPA")
        self.device = device
        self.cudalib = cudalib
        self.trace = bool(trace)
        self._session = None

    def _get_session(self):
        if self._session is None:
            from wall_x.serving.backend.iree.runtime import IreeExpandedSession

            self._session = IreeExpandedSession(
                self.full_flow_vmfb,
                self.full_flow_irpa,
                trace=self.trace,
                device=self.device,
                cudalib=self.cudalib,
            )
        return self._session

    @property
    def _state_string(self):
        return bool(
            getattr(self.model.config, "use_state_string_representation", False)
        )

    def _validate_context_tensors(self, batch_inputs):
        if batch_inputs.get("inputs_embeds") is not None:
            raise RuntimeError(
                "iree-expanded owns context construction; precomputed "
                "inputs_embeds are unsupported"
            )
        if batch_inputs.get("pixel_values_videos") is not None:
            raise RuntimeError("iree-expanded has no static video tower export")
        ids = batch_inputs["input_ids"]
        pixels = batch_inputs.get("pixel_values")
        grid = batch_inputs.get("image_grid_thw")
        if tuple(ids.shape) != (1, self.sequence_length):
            raise RuntimeError(
                f"iree-expanded requires input_ids [1,{self.sequence_length}]; "
                f"got {tuple(ids.shape)}"
            )
        if pixels is None or grid is None:
            raise RuntimeError(
                "iree-expanded requires the static two-image pixel/grid profile"
            )
        if tuple(pixels.shape) != self.EXPECTED_PIXELS:
            raise RuntimeError(
                f"iree-expanded pixel_values must be {self.EXPECTED_PIXELS}; "
                f"got {tuple(pixels.shape)}"
            )
        actual_grid = tuple(
            tuple(int(value) for value in row)
            for row in grid.detach().cpu().tolist()
        )
        if actual_grid != self.EXPECTED_GRID:
            raise RuntimeError(
                f"iree-expanded image_grid_thw must be {self.EXPECTED_GRID}; "
                f"got {actual_grid}"
            )
        image_count = int((ids == self.model.config.image_token_id).sum().item())
        if image_count != self.EXPECTED_IMAGE_FEATURES:
            raise RuntimeError(
                f"image token count {image_count} != compiled feature count "
                f"{self.EXPECTED_IMAGE_FEATURES}"
            )
        action_id = self.model.action_token_id_set["action_token_id"]
        action_count = int((ids == action_id).sum().item())
        if action_count != self.action_horizon:
            raise RuntimeError(
                f"action token count {action_count} != compiled horizon "
                f"{self.action_horizon}"
            )
        if self._state_string:
            return

        propri = batch_inputs.get("proprioception")
        agent_mask = batch_inputs.get("agent_pos_mask")
        if propri is None or agent_mask is None:
            raise RuntimeError(
                "tensor-state iree-expanded requires proprioception and agent_pos_mask"
            )
        if tuple(propri.shape[:2]) != (1, 1):
            raise RuntimeError(
                f"iree-expanded proprioception must be [1,1,D]; got {tuple(propri.shape)}"
            )
        if propri.shape != agent_mask.shape:
            raise RuntimeError("proprioception/agent_pos_mask shapes differ")
        propri_dim = int(self.model.action_preprocessor.propri_dim)
        if int(propri.shape[-1]) != propri_dim:
            raise RuntimeError(
                f"proprioception last dim {propri.shape[-1]} != {propri_dim}"
            )
        propri_id = self.model.action_token_id_set["propri_token_id"]
        state_count = int((ids == propri_id).sum().item())
        if state_count != 1:
            raise RuntimeError(
                f"proprio token count {state_count} != static feature count 1"
            )

    def prepare_ctx(
        self,
        action_horizon,
        action_dim,
        num_inference_timesteps,
        padding_action,
        noise,
        batch_inputs,
    ):
        self._validate_context_tensors(batch_inputs)
        ids = batch_inputs["input_ids"]
        weight = self.model.model.embed_tokens.weight
        # Metadata-only placeholder: the unified graph computes real ViT/text/
        # state features and t=0 action embeddings; this tensor is never run.
        placeholder = torch.zeros(
            1,
            self.sequence_length,
            int(self.model.config.hidden_size),
            device=ids.device,
            dtype=weight.dtype,
        )
        return super().prepare_ctx(
            action_horizon,
            action_dim,
            num_inference_timesteps,
            padding_action,
            noise,
            batch_inputs,
            inputs_embeds=placeholder,
            pixel_values=None,
            pixel_values_videos=None,
            proprioception=None,
            agent_pos_mask=None,
            skip_action_embedding=True,
        )

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
        if num_inference_timesteps != 10:
            raise RuntimeError(
                "iree-expanded is compiled for num_inference_timesteps=10; "
                f"got {num_inference_timesteps}"
            )
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
            graph_args = [
                self._host(batch_inputs["input_ids"], np.int64),
                self._host(batch_inputs["pixel_values"], np.float32),
            ]
            if not self._state_string:
                graph_args.extend(
                    [
                        self._host(batch_inputs["proprioception"], np.float32),
                        self._host(batch_inputs["agent_pos_mask"], np.float32),
                    ]
                )
            graph_args.extend(
                [
                    self._host(ctx["position_ids"]),
                    self._host(mask3d, np.bool_),
                    self._host(ctx["noise"], np.float32),
                    self._host(self._padding_action(ctx), np.float32),
                    self._host(ctx["times"], np.float32),
                    self._host(postfix_positions),
                    self._host(postfix_mask, np.bool_),
                    self._host(ctx["dof_mask"], np.float32),
                ]
            )
            session = self._get_session()
            if self._state_string:
                final = session.run(*graph_args)
            else:
                final = session.run_tensor_state(*graph_args)
            if final.shape != (1, self.action_horizon, self.action_dim):
                raise RuntimeError(
                    f"full_flow returned {final.shape}, expected "
                    f"{(1, self.action_horizon, self.action_dim)}"
                )
            predict = torch.from_numpy(final).to(batch_inputs["input_ids"].device)
            return self._finalize(ctx, predict)
