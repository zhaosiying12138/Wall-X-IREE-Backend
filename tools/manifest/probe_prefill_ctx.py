#!/usr/bin/env python3
"""Probe prefill ctx shapes: attention_mask, positions, segments (P4 prep)."""
import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "iree_wallx"))

from bench_model_api import (  # noqa: E402
    _build_fake_observation, _right_gripper_dim_from_config, build_adapter)

ns = argparse.Namespace(
    checkpoint="/home/zhaosiying/codebase/weights/wall-oss-0.5",
    train_config=None, norm_key="x2_normal", architecture="qwen2_5",
    cam_names=["face_view", "right_wrist_view"], action_horizon=None,
    image_size=448, seed=0)
adapter = build_adapter(ns)
model = adapter._model
from wall_x.model.core import ops as wall_ops
for name in ("rmsnorm", "m_rope", "rot_pos_emb", "permute", "unpermute",
             "get_rope_index", "get_window_index"):
    op = getattr(wall_ops, name, None)
    if op is not None:
        op._resolved_fn = op._pytorch_fallback

bank = torch.load(REPO_ROOT / "artifacts/bank/sample_000.pt", weights_only=False)
gd = _right_gripper_dim_from_config(getattr(adapter, "_train_config", {}))
from wall_x._vendor.harrix.envs.libero_common import encode_proprio
obs = _build_fake_observation(0, 448, gd)
ob = encode_proprio(obs, adapter._train_config, adapter._action_horizon)
pre_t, post_t = adapter._get_flow_prompt(bank["meta"]["instruction"])
bi = adapter._construct_model_input([ob], [pre_t], [post_t])
ctx = model._prepare_flow_action_inputs(
    input_ids=bi["input_ids"], action_horizon=adapter._action_horizon,
    action_dim=adapter._action_dim,
    num_inference_timesteps=adapter._num_inference_timesteps,
    prefix_length=None, attention_mask=bi.get("attention_mask"),
    position_ids=bi.get("position_ids"), past_key_values=None,
    inputs_embeds=None, moe_token_types=bi["moe_token_types"],
    start_indices=None, end_indices=None, positional_masks=None,
    labels=None, use_cache=None, output_attentions=None,
    output_hidden_states=None, return_dict=None,
    pixel_values=bi.get("pixel_values"),
    pixel_values_videos=bi.get("pixel_values_videos"),
    image_grid_thw=bi.get("image_grid_thw"),
    video_grid_thw=bi.get("video_grid_thw"), action_chunk=None,
    proprioception=bi.get("proprioception"),
    unnorm_proprioception=bi.get("unnorm_proprioception"),
    rope_deltas=bi.get("rope_deltas"), cache_position=None,
    second_per_grid_ts=None, dataset_names=bi["dataset_names"],
    dof_mask=bi["dof_mask"], agent_pos_mask=bi.get("agent_pos_mask"),
    noise=bank["noise"].cuda())

am = ctx["attention_mask"]
print("ctx attention_mask:", tuple(am.shape), am.dtype)
if am.dim() == 3:
    b_, q, k = am.shape
    print("  row0 (q0 sees):", am[0, 0].sum().item(), "/", k)
    print("  action q31 sees:", am[0, -1].sum().item(), "/", k)
    print("  prefix q0 sees:", am[0, 0].sum().item(), "/", k)
print("ctx position_ids:", tuple(ctx["position_ids"].shape))
print("ctx inputs_embeds:", tuple(ctx["inputs_embeds"].shape),
      ctx["inputs_embeds"].dtype)
print("ctx adarms_cond:", None if ctx["adarms_cond"] is None
      else tuple(ctx["adarms_cond"].shape))
print("pixel_values:", tuple(bi["pixel_values"].shape),
      bi["pixel_values"].dtype)
print("grid_thw:", bi["image_grid_thw"].tolist())
print("start/end:", ctx["start_indices"].tolist(),
      ctx["end_indices"].tolist())
print("flow_action_mask:", int(ctx["flow_action_mask"].sum()))
