#!/usr/bin/env python3
"""Runtime probes for the model manifest (P0-2 runtime section).

Collects: image grid_thw, vision/prefix/action token counts, prefix_length,
moe_token_types distribution, attention backend per tower, load_state_dict
missing/unexpected keys, dtype policy of loaded weights.

Merges into artifacts/manifests/model.json under runtime_probes.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "iree_wallx"))

from bench_model_api import (  # noqa: E402
    _build_fake_observation,
    _right_gripper_dim_from_config,
    build_adapter,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--manifest", type=Path,
                    default=REPO_ROOT / "artifacts" / "manifests" / "model.json")
    args = ap.parse_args()

    ns = argparse.Namespace(
        checkpoint=args.checkpoint, train_config=None, norm_key="x2_normal",
        architecture="qwen2_5", cam_names=["face_view", "right_wrist_view"],
        action_horizon=None, image_size=args.image_size, seed=0,
    )
    adapter = build_adapter(ns)
    model = adapter._model
    gripper_dim = _right_gripper_dim_from_config(
        getattr(adapter, "_train_config", {}))

    from wall_x._vendor.harrix.envs.libero_common import encode_proprio

    obs = _build_fake_observation(0, args.image_size, gripper_dim)
    observation = encode_proprio(obs, adapter._train_config,
                                 adapter._action_horizon)
    prefix, postfix = adapter._get_flow_prompt("pick up the object")
    bi = adapter._construct_model_input([observation], [prefix], [postfix])

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
    )

    input_ids = bi["input_ids"]
    types = ctx["moe_token_types"].flatten().tolist()
    probe = {
        "probed_utc": datetime.now(timezone.utc).isoformat(),
        "image_size_requested": args.image_size,
        "image_grid_thw": bi.get("image_grid_thw", torch.zeros(0)).tolist(),
        "seq_len": int(input_ids.shape[1]),
        "prefix_length": int(ctx["prefix_length"]),
        "vision_tokens": int((input_ids == model.config.image_token_id).sum()),
        "action_tokens": int(ctx["flow_action_mask"].sum()),
        "moe_token_types_hist": dict(Counter(types)),
        "attention_implementation_text": getattr(
            model.config, "_attn_implementation", None),
        "attention_implementation_vision": getattr(
            model.config.vision_config, "_attn_implementation", None),
        "ops_backends": {},
    }

    from wall_x.model.core import ops as wall_ops
    for name in ("rmsnorm", "m_rope", "rot_pos_emb", "permute", "unpermute",
                 "get_rope_index", "get_window_index"):
        op = getattr(wall_ops, name, None)
        if op is not None:
            probe["ops_backends"][name] = op.backend

    dtypes = Counter(str(p.dtype).replace("torch.", "")
                     for p in model.parameters())
    probe["param_dtypes"] = dict(dtypes)
    probe["param_count"] = sum(p.numel() for p in model.parameters())

    m = json.loads(args.manifest.read_text())
    m["runtime_probes"] = probe
    args.manifest.write_text(json.dumps(m, indent=2, sort_keys=False))
    print(json.dumps(probe, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
