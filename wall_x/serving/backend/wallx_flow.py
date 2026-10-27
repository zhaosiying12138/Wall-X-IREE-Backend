"""Shared official Wall-X flow preparation and finalization semantics."""
from __future__ import annotations

import torch

from wall_x.serving.backend.base import ServingBackend


class WallxFlowBackend(ServingBackend):
    """Base for backends specializing the validated static Wall-X profile.

    The profile is the active 606-token prefix + 32-token action suffix used by
    the staged IREE, expanded IREE, and torch.compile implementations.  Host
    preparation and final action normalization remain official model calls.
    """

    sequence_length = 638
    prefix_length = 606
    action_horizon = 32
    action_dim = 26

    def _pad_to_bucket(self, batch_inputs):
        """Left-pad the prefix while keeping action tokens at the right edge."""
        ids = batch_inputs.get("input_ids")
        if ids is None:
            raise RuntimeError("static flow backend requires input_ids")
        sequence = int(ids.shape[1])
        if ids.shape[0] != 1:
            raise RuntimeError("static flow backend currently requires batch size 1")
        if sequence > self.sequence_length:
            raise RuntimeError(
                f"static {self.sequence_length}-token bucket cannot hold sequence "
                f"{sequence}; export a matching bucket explicitly"
            )
        if batch_inputs.get("positional_masks") is not None:
            raise RuntimeError(
                "static flow backend does not accept custom positional_masks"
            )
        if sequence == self.sequence_length:
            return batch_inputs

        output = dict(batch_inputs)
        count = self.sequence_length - sequence
        tokenizer = self.model.processor.tokenizer
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            raise RuntimeError("bucket padding requires tokenizer.pad_token_id")
        pad_ids = torch.full(
            (ids.shape[0], count), pad_id, dtype=ids.dtype, device=ids.device
        )
        output["input_ids"] = torch.cat([pad_ids, ids], dim=1)

        for name in ("attention_mask", "moe_token_types"):
            value = batch_inputs.get(name)
            if value is None:
                if name == "attention_mask":
                    value = torch.ones_like(ids)
                else:
                    raise RuntimeError("static flow backend requires moe_token_types")
            fill = 0 if name == "attention_mask" else False
            padding = torch.full(
                (value.shape[0], count),
                fill,
                dtype=value.dtype,
                device=value.device,
            )
            output[name] = torch.cat([padding, value], dim=1)

        positions = batch_inputs.get("position_ids")
        if positions is not None:
            pad_shape = list(positions.shape)
            pad_shape[-1] = count
            padding = torch.zeros(
                pad_shape, dtype=positions.dtype, device=positions.device
            )
            output["position_ids"] = torch.cat([padding, positions], dim=-1)
        return output

    def prepare_ctx(
        self,
        action_horizon,
        action_dim,
        num_inference_timesteps,
        padding_action,
        noise,
        batch_inputs,
        **overrides,
    ):
        """Call the official preparation helper with explicit noise preserved."""
        del padding_action  # the official helper derives it from DOF/normalizer
        values = dict(
            input_ids=batch_inputs["input_ids"],
            action_horizon=action_horizon,
            action_dim=action_dim,
            num_inference_timesteps=num_inference_timesteps,
            prefix_length=None,
            attention_mask=batch_inputs.get("attention_mask"),
            position_ids=batch_inputs.get("position_ids"),
            past_key_values=None,
            inputs_embeds=batch_inputs.get("inputs_embeds"),
            moe_token_types=batch_inputs["moe_token_types"],
            start_indices=None,
            end_indices=None,
            positional_masks=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            pixel_values=batch_inputs.get("pixel_values"),
            pixel_values_videos=batch_inputs.get("pixel_values_videos"),
            image_grid_thw=batch_inputs.get("image_grid_thw"),
            video_grid_thw=batch_inputs.get("video_grid_thw"),
            action_chunk=batch_inputs.get("action_chunk"),
            proprioception=batch_inputs.get("proprioception"),
            unnorm_proprioception=batch_inputs.get("unnorm_proprioception"),
            rope_deltas=batch_inputs.get("rope_deltas"),
            cache_position=None,
            second_per_grid_ts=None,
            dataset_names=batch_inputs["dataset_names"],
            dof_mask=batch_inputs["dof_mask"],
            agent_pos_mask=batch_inputs.get("agent_pos_mask"),
            noise=noise,
        )
        values.update(overrides)
        return self.model._prepare_flow_action_inputs(**values)

    def _joint_attention_mask(self, ctx):
        from wall_x.model.core.attention.mask import update_joint_attention_mask_2d

        if ctx["attention_mask"] is None:
            raise RuntimeError("static flow backend requires an attention mask")
        valid = ctx["attention_mask"].to(torch.bool)
        positional_masks = dict(ctx["positional_masks"] or {})
        positional_masks["padding_positions"] = ~valid
        mask = update_joint_attention_mask_2d(
            ctx["attention_mask"],
            ctx["moe_token_types"],
            positional_masks,
            causal_action_attention_mask=(
                self.model.config.causal_action_attention_mask
            ),
        ).to(torch.bool)
        # Avoid an empty softmax for fully masked left-pad query rows.  Real
        # queries still cannot attend to padded key columns.
        pad_rows = ~valid
        eye = torch.eye(
            mask.shape[-1], dtype=torch.bool, device=mask.device
        ).unsqueeze(0)
        return torch.where(pad_rows.unsqueeze(-1), eye & pad_rows.unsqueeze(-1), mask)

    def _validate_static_context(self, ctx, action_horizon, action_dim):
        if (action_horizon, action_dim) != (self.action_horizon, self.action_dim):
            raise RuntimeError(
                f"static flow profile requires action_horizon={self.action_horizon} "
                f"and action_dim={self.action_dim}; got {action_horizon}, {action_dim}"
            )
        if int(ctx["prefix_length"]) != self.prefix_length:
            raise RuntimeError(
                f"static flow profile requires prefix {self.prefix_length}; "
                f"got {ctx['prefix_length']}"
            )
        if tuple(ctx["inputs_embeds"].shape[:2]) != (1, self.sequence_length):
            raise RuntimeError(
                "static flow preparation returned an unexpected token bucket: "
                f"{tuple(ctx['inputs_embeds'].shape)}"
            )

    def _postfix(self, ctx):
        positions, _embeds, mask, _types, _ids, _start, _end, _padding = (
            self.model._prepare_flow_postfix(ctx, None)
        )
        return positions, mask

    def _padding_action(self, ctx):
        padding = ctx["padding_action"]
        if padding is None:
            # With every DOF active, padding velocity is multiplied by zero.
            padding = torch.zeros(
                1,
                1,
                self.action_dim,
                dtype=torch.float32,
                device=ctx["inputs_embeds"].device,
            )
        return padding.reshape(1, 1, -1)

    def _finalize(self, ctx, final_normalized_action):
        """Use the model's official finalizer, including virtual-tail DOFs/GT."""
        return self.model._finalize_flow_action_output(
            [final_normalized_action], ctx
        )
