#!/usr/bin/env python3
"""P4: export the static joint prefill to MLIR via turbine with externalized
parameters, then compile for sm_120.

ABI (static, B=1/S=638/A=32):
  inputs_embeds_f32 [1,638,2048] | position_ids [3,1,638] i64
  mask3d [1,638,638] i1 | noise [1,32,26] f32 | padding_action [1,1,26] f32
  dt0 [1] f32
Outputs: k/v [36,1,2,638,128] f32, v0 [1,32,26] f32, x0 [1,32,26] f32.
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
                    default=REPO_ROOT / "artifacts/ir/joint_prefill.mlir")
    ap.add_argument("--params-out", type=Path,
                    default=REPO_ROOT / "artifacts/bin/joint_prefill.weights")
    ap.add_argument("--n-layers", type=int, default=None,
                    help="export only the first N layers (differential "
                         "profiling; None = all)")
    ap.add_argument("--attn", choices=("fp16", "sdpa"), default="fp16",
                    help="attention variant: fp16-score math (deployed, "
                         "1.60pct) or sdpa masked (quality anchor)" )
    args = ap.parse_args()

    from bench_model_api import build_adapter
    from wall_x.serving.backend.static_flow import StaticJointPrefill

    ns = argparse.Namespace(
        checkpoint=CKPT, train_config=None, norm_key="x2_normal",
        architecture="qwen2_5", cam_names=["face_view", "right_wrist_view"],
        action_horizon=None, image_size=64, seed=0)
    adapter = build_adapter(ns)
    model = adapter._model

    bank = torch.load(REPO_ROOT / "artifacts/bank/sample_000.pt",
                      weights_only=False)
    n_layers = (args.n_layers if args.n_layers else
                len(model.model.layers))
    if args.attn == "sdpa":
        from wall_x.serving.backend.static_flow import prefill as _sjp
        _sjp.StaticPrefillLayer.math_attention = False
        m = StaticJointPrefill(model, dof_mask=bank["ctx"]["dof_mask"],
                               n_layers=n_layers,
                               assume_causal=False).cpu().eval()
    else:
        m = StaticJointPrefill(model, dof_mask=bank["ctx"]["dof_mask"],
                               n_layers=n_layers).cpu().eval()

    from iree.turbine.aot.params import (externalize_module_parameters,
                                         save_module_parameters)
    externalize_module_parameters(m)

    S, L, A = 638, len(model.model.layers), 32
    ex = (
        torch.zeros(1, S, 2048, dtype=torch.float32),
        torch.zeros(3, 1, S, dtype=torch.int64),
        torch.zeros(1, S, S, dtype=torch.bool),
        torch.zeros(1, A, 26, dtype=torch.float32),
        torch.zeros(1, 1, 26, dtype=torch.float32),
        torch.zeros(1, dtype=torch.float32),
    )
    from iree.turbine.aot import export
    exp = export(m, *ex)
    args.ir_out.parent.mkdir(parents=True, exist_ok=True)
    exp.save_mlir(args.ir_out)
    print("mlir:", args.ir_out, args.ir_out.stat().st_size, "bytes")

    args.params_out.parent.mkdir(parents=True, exist_ok=True)
    save_module_parameters(args.params_out, m)
    irpa = args.params_out.with_suffix(".weights.irpa")
    print("params:", irpa, irpa.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
