#!/usr/bin/env python3
"""P2: export the static denoise_step to MLIR via turbine with externalized
parameters, then compile for sm_120.

Inputs ABI v2 (all static for B=1/A=32/S=638):
  noisy_action [1,32,26] f32 | timestep [1] f32
  postfix_pos [3,1,32] i64 | suffix_mask [1,32,638] i1
  k_past / v_past [36,1,2,638,128] f32 | v_padding [1,32,26] f32
  The module crops the KV buffers to the first 606 prefix rows in-graph.
Output: velocity [1,32,26] f32 (pre-Euler; caller applies the update).

Usage:
  PYTHONPATH=.:benchmarks/... .venv-iree/bin/python tools/iree/export_denoise_step.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "iree_wallx"))

CKPT = os.environ.get(
    "WALLX_CKPT", "/home/zhaosiying/codebase/weights/wall-oss-0.5")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir-out", type=Path,
                    default=REPO_ROOT / "artifacts/ir/denoise_step.mlir")
    ap.add_argument("--params-out", type=Path,
                    default=REPO_ROOT / "artifacts/bin/denoise_step.weights")
    args = ap.parse_args()

    from bench_model_api import build_adapter
    from wall_x.serving.backend.static_flow import StaticDenoiseStep

    ns = argparse.Namespace(
        checkpoint=CKPT, train_config=None, norm_key="x2_normal",
        architecture="qwen2_5", cam_names=["face_view", "right_wrist_view"],
        action_horizon=None, image_size=64, seed=0)
    adapter = build_adapter(ns)
    model = adapter._model

    bank = torch.load(REPO_ROOT / "artifacts/bank/sample_000.pt",
                      weights_only=False)
    dof = bank["ctx"]["dof_mask"]

    m = StaticDenoiseStep(model, len(model.model.layers),
                          dof_mask=dof).cpu().eval()

    from iree.turbine.aot.params import (externalize_module_parameters,
                                         save_module_parameters)
    externalize_module_parameters(m)

    A, P, L = 32, 638, len(model.model.layers)
    ex = (
        torch.zeros(1, A, 26, dtype=torch.float32),
        torch.zeros(1, dtype=torch.float32),
        torch.zeros(3, 1, A, dtype=torch.int64),
        torch.zeros(1, A, P, dtype=torch.bool),
        torch.zeros(L, 1, 2, P, 128, dtype=torch.float32),
        torch.zeros(L, 1, 2, P, 128, dtype=torch.float32),
        torch.zeros(1, A, 26, dtype=torch.float32),
    )

    from iree.turbine.aot import export
    exp = export(m, *ex)
    args.ir_out.parent.mkdir(parents=True, exist_ok=True)
    exp.save_mlir(args.ir_out)
    print("mlir:", args.ir_out, args.ir_out.stat().st_size, "bytes")

    args.params_out.parent.mkdir(parents=True, exist_ok=True)
    save_module_parameters(args.params_out, m)
    print("params:", args.params_out, args.params_out.with_suffix(".weights.irpa").stat().st_size if args.params_out.with_suffix(".weights.irpa").exists() else "(see .irpa)", "bytes")


if __name__ == "__main__":
    main()
