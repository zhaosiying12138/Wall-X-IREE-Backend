#!/usr/bin/env python3
"""P1: explicit-noise injection point verification.

The official generate_flow_action used to silently drop the caller's noise
(fell into **kwargs). After the injection-point change, passing the same
noise tensor twice must reproduce the action bitwise; omitting it keeps the
original randomized behavior.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "iree_wallx"))

from bench_model_api import (  # noqa: E402
    _build_fake_observation, _right_gripper_dim_from_config, build_adapter)


def main() -> int:
    ns = argparse.Namespace(
        checkpoint=os.environ.get(
            "WALLX_CKPT", "/home/zhaosiying/codebase/weights/wall-oss-0.5"),
        train_config=None, norm_key="x2_normal", architecture="qwen2_5",
        cam_names=["face_view", "right_wrist_view"], action_horizon=None,
        image_size=448, seed=0)
    adapter = build_adapter(ns)
    model = adapter._model
    from wall_x.serving.backend import create_backend
    backend = create_backend(model, name="eager")
    gd = _right_gripper_dim_from_config(getattr(adapter, "_train_config", {}))

    from wall_x._vendor.harrix.envs.libero_common import encode_proprio

    def run(noise):
        obs = _build_fake_observation(0, 448, gd)
        ob = encode_proprio(obs, adapter._train_config,
                            adapter._action_horizon)
        pre, post = adapter._get_flow_prompt("pick up the object")
        bi = adapter._construct_model_input([ob], [pre], [post])
        out = backend.generate_flow_action(
            action_horizon=adapter._action_horizon,
            action_dim=adapter._action_dim,
            num_inference_timesteps=adapter._num_inference_timesteps,
            noise=noise, **bi)
        return out["predict_action"].detach().cpu().numpy()

    rng = np.random.default_rng(20261023)
    noise = torch.from_numpy(
        rng.standard_normal((1, adapter._action_horizon,
                             adapter._action_dim)).astype(np.float32))

    a1 = run(noise)
    a2 = run(noise)
    a3 = run(noise.clone())
    b1 = run(None)
    b2 = run(None)

    ok_same = np.array_equal(a1, a2) and np.array_equal(a1, a3)
    ok_diff = not np.array_equal(b1, b2)
    ok_noise_effect = not np.array_equal(a1, b1)
    print(f"same-noise bitwise repro: {ok_same}")
    print(f"none-noise randomizes:    {ok_diff}")
    print(f"noise changes output:     {ok_noise_effect}")
    if not (ok_same and ok_diff and ok_noise_effect):
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
