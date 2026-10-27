#!/usr/bin/env python3
"""Verify left-bucket padding against the official short-sequence path."""
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks" / "iree_wallx"))

from bench_model_api import (build_adapter, _build_fake_observation,
                             _right_gripper_dim_from_config)
from wall_x.serving.backend import create_backend


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("staged", "expanded"),
                    default="staged")
    args = ap.parse_args()
    ckpt = os.environ.get(
        "WALLX_CKPT", "/home/zhaosiying/codebase/weights/wall-oss-0.5")
    ns = type("NS", (), dict(
        checkpoint=ckpt, train_config=None, norm_key="x2_normal",
        architecture="qwen2_5", cam_names=["face_view", "right_wrist_view"],
        action_horizon=None, image_size=448, seed=0))()
    adapter = build_adapter(ns)
    model = adapter._model
    from wall_x.model.core import ops as wall_ops
    for name in ("rmsnorm", "m_rope", "rot_pos_emb", "permute", "unpermute",
                 "get_rope_index", "get_window_index"):
        op = getattr(wall_ops, name, None)
        if op is not None:
            op._resolved_fn = op._pytorch_fallback

    bank = torch.load(REPO / "artifacts/bank/sample_000.pt",
                      weights_only=False)
    gd = _right_gripper_dim_from_config(getattr(adapter, "_train_config", {}))
    from wall_x._vendor.harrix.envs.libero_common import encode_proprio
    obs = _build_fake_observation(0, 448, gd)
    ob = encode_proprio(obs, adapter._train_config, adapter._action_horizon)
    pre_t, post_t = adapter._get_flow_prompt("pick")
    bi = adapter._construct_model_input([ob], [pre_t], [post_t])
    original_seq = int(bi["input_ids"].shape[1])
    if original_seq >= 638:
        print("short-bucket setup failed: seq", original_seq)
        return 1

    noise = bank["noise"].cuda()
    be = create_backend(model, name=args.backend)
    padded_seq = int(be._pad_to_bucket(bi)["input_ids"].shape[1])

    with torch.no_grad():
        official = model.generate_flow_action(
            action_horizon=adapter._action_horizon,
            action_dim=adapter._action_dim,
            num_inference_timesteps=adapter._num_inference_timesteps,
            noise=noise, **bi)
        got = be.generate_flow_action(
            action_horizon=adapter._action_horizon,
            action_dim=adapter._action_dim,
            num_inference_timesteps=adapter._num_inference_timesteps,
            padding_action=None, noise=noise, **bi)

    a = official["predict_action"].detach().float().cpu()
    b = got["predict_action"].detach().float().cpu()
    max_abs = float((a - b).abs().max())
    rel = max_abs / float(a.abs().max())
    print("backend:", args.backend)
    print("original_seq:", original_seq)
    print("padded_seq:", padded_seq)
    print("short_official_shape:", tuple(a.shape))
    print("short_iree_shape:", tuple(b.shape))
    print("short_physical_action_maxabs:", max_abs)
    print("short_physical_action_relmax:", rel)
    if a.shape != b.shape or rel >= 0.02:
        print("SHORT-BUCKET FAIL")
        return 1
    print("SHORT-BUCKET PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
