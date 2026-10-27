#!/usr/bin/env python3
"""P7 smoke: build the official policy and validate one IREE inference."""
import base64
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "iree_wallx"))

CKPT = os.environ.get(
    "WALLX_CKPT", "/home/zhaosiying/codebase/weights/wall-oss-0.5")


def main() -> int:
    backend_name = os.environ.get("WALLX_BACKEND")
    if backend_name not in ("iree", "iree-expanded"):
        print("POLICY-SMOKE FAIL: set WALLX_BACKEND=iree or iree-expanded")
        return 1
    from wall_x._vendor.harrix.serving.launch_serving import (
        ServerModelConfig, get_wallx_policy)
    cfg = ServerModelConfig(
        checkpoint_path=CKPT, train_config_path=CKPT + "/config.yml",
        robot_type="desktop", action_horizon=32, action_dim=None,
        model_device="cuda", num_inference_timesteps=10,
        cam_names=["face_view", "right_wrist_view"],
        norm_key="x2_normal")
    cfg.env_mode = "LIBERO"
    policy = get_wallx_policy(cfg, "base64", False)

    rng = np.random.default_rng(0)
    import cv2
    imgs = {}
    for name in ("face_view", "right_wrist_view", "left_wrist_view"):
        arr = rng.integers(0, 255, (448, 448, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", arr)
        if not ok:
            print("POLICY-SMOKE FAIL: JPEG encoding failed")
            return 1
        imgs[name] = base64.b64encode(buf.tobytes()).decode()
    obs = {
        "instruction": "pick up the object",
        "state": {
            "eef_pos": rng.normal(size=(3,)),
            "eef_axisangle": rng.normal(size=(3,)),
            "gripper": rng.normal(size=(1,)),
        },
        "views": imgs,
    }
    out = policy.infer(obs)
    print("infer output type:", type(out))
    if not isinstance(out, dict) or "predict_action" not in out:
        print("POLICY-SMOKE FAIL: predict_action is missing")
        return 1
    action = np.asarray(out["predict_action"], dtype=np.float32)
    backend = getattr(policy.model_wrapper, "_serving_backend", None)
    print("predict_action:", f"shape={action.shape}",
          f"min={action.min():.6f}", f"max={action.max():.6f}")
    print("serving_backend:", type(backend).__name__ if backend else None)
    expected_backend = {"iree": "IreeFlowBackend",
                        "iree-expanded": "IreeExpandedBackend"}[backend_name]
    if backend is None or type(backend).__name__ != expected_backend:
        print(f"POLICY-SMOKE FAIL: expected {expected_backend}, got {type(backend).__name__}")
        return 1
    if action.shape != (1, 32, 26) or not np.isfinite(action).all():
        print("POLICY-SMOKE FAIL: invalid action tensor")
        return 1
    print("POLICY-SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
