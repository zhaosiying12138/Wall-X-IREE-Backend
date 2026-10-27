#!/usr/bin/env python3
"""Export the single WALL-OSS full_flow graph with external parameters.

Deployable artifacts (local-only):
  artifacts/ir/full_flow.mlir
  artifacts/bin/full_flow.weights.irpa
  artifacts/bin/full_flow.vmfb  (iree-compile output)

One main entry, state-string WALL-OSS-0.5 ABI:
   0 input_ids      [1,638] i64
   1 pixel_values   [2048,1176] f32
   2 position_ids   [3,1,638] i64
   3 mask3d         [1,638,638] i1
   4 noise          [1,32,26] f32
   5 padding_action [1,1,26] f32
   6 times          [11] f32
   7 postfix_pos    [3,1,32] i64
   8 suffix_mask    [1,32,638] i1
   9 dof_mask       [1,32,26] f32

Production output: final normalized action [1,32,26] f32 only.  With
``--trace``, a separate diagnostic full_flow_trace.mlir exposes final,
prepared inputs_embeds, full K/V, v0, x0, nine velocities, and ten states.
The diagnostic and production modules share the same external parameter names.

Compile using IREE's built-in pipeline (no custom passes):
  iree-compile artifacts/ir/full_flow.mlir \
    --iree-hal-target-device=cuda --iree-cuda-target=sm_120 \
    -o artifacts/bin/full_flow.vmfb
"""
from __future__ import annotations

import argparse
import gc
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
    ap.add_argument("--ir-out", type=Path, default=None,
                    help="default: full_flow.mlir, or full_flow_trace.mlir "
                         "with --trace")
    ap.add_argument("--params-out", type=Path,
                    default=REPO_ROOT / "artifacts/bin/full_flow.weights")
    ap.add_argument("--skip-params", action="store_true",
                    help="export MLIR only when the matching IRPA already exists")
    ap.add_argument("--trace", action="store_true",
                    help="export the separate eight-output diagnostic module")
    ap.add_argument("--vision-attn",
                    choices=("sdpa", "eager-bf16", "math-fp16", "math-fp32"),
                    default="eager-bf16", help="static ViT attention implementation")
    args = ap.parse_args()
    artifact_stem = "full_flow_trace" if args.trace else "full_flow"
    if args.vision_attn != "eager-bf16":
        artifact_stem += "_" + args.vision_attn.replace("-", "_")
    ir_out = args.ir_out or (REPO_ROOT / "artifacts/ir" /
                             f"{artifact_stem}.mlir")

    from bench_model_api import build_adapter
    from wall_x.serving.backend.static_flow import (
        StaticFullFlow,
        StaticFullFlowTensorState,
        StaticFullFlowTensorStateTrace,
        StaticFullFlowTrace,
    )

    ns = argparse.Namespace(
        checkpoint=CKPT, train_config=None, norm_key="x2_normal",
        architecture="qwen2_5", cam_names=["face_view", "right_wrist_view"],
        action_horizon=None, image_size=448, seed=0)
    model = build_adapter(ns)._model
    bank = torch.load(REPO_ROOT / "artifacts/bank/sample_000.pt",
                      map_location="cpu", weights_only=False)
    state_string = bool(getattr(
        model.config, "use_state_string_representation", False))
    if state_string:
        module_cls = StaticFullFlowTrace if args.trace else StaticFullFlow
    else:
        module_cls = (StaticFullFlowTensorStateTrace if args.trace
                      else StaticFullFlowTensorState)
    module = module_cls(
        model, bank["image_grid_thw"], dof_mask=bank["ctx"]["dof_mask"],
        vision_attention=args.vision_attn).cpu().eval()
    print("full-flow profile:", module.context.vision.profile,
          "state_string:", state_string, "trace:", args.trace,
          "vision_attention:", args.vision_attn, flush=True)

    causal_action_attention = bool(model.config.causal_action_attention_mask)
    propri_dim = (None if state_string else
                  int(model.action_preprocessor.propri_dim))
    # The static module owns every weight needed below.  Release the unused
    # official wrapper (including lm_head) before high-memory Turbine export.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    from wall_x.model.core.attention.mask import (
        update_joint_attention_mask_2d)
    valid = bank["attention_mask"].bool()
    mask3d = update_joint_attention_mask_2d(
        bank["attention_mask"], bank["moe_token_types"],
        {"padding_positions": ~valid},
        causal_action_attention_mask=causal_action_attention).bool()
    pad_rows = ~valid
    eye = torch.eye(638, dtype=torch.bool).unsqueeze(0)
    mask3d = torch.where(pad_rows.unsqueeze(-1),
                         eye & pad_rows.unsqueeze(-1), mask3d)

    from iree.turbine.aot.params import (
        externalize_module_parameters, save_module_parameters)
    externalize_module_parameters(module)
    dof = bank["ctx"]["dof_mask"].cpu().float()
    example = [
        bank["input_ids"].cpu(),
        bank["pixel_values"].cpu().float(),
    ]
    if not state_string:
        example.extend([
            torch.zeros(1, 1, propri_dim, dtype=torch.float32),
            torch.zeros(1, 1, propri_dim, dtype=torch.float32),
        ])
    example.extend([
        bank["ctx"]["position_ids"].cpu(),
        mask3d,
        bank["noise"].cpu().float(),
        bank["ctx"]["padding_action"].cpu().float().reshape(1, 1, -1),
        bank["ctx"]["times"].cpu().float(),
        bank["postfix"]["position_ids"].cpu(),
        bank["postfix"]["attention_mask"].cpu().bool(),
        dof,
    ])
    from iree.turbine.aot import export
    exported = export(module, *example)
    ir_out.parent.mkdir(parents=True, exist_ok=True)
    exported.save_mlir(ir_out)
    print("mlir:", ir_out, ir_out.stat().st_size, "bytes")

    if not args.skip_params:
        args.params_out.parent.mkdir(parents=True, exist_ok=True)
        save_module_parameters(args.params_out, module)
        irpa = args.params_out.with_suffix(".weights.irpa")
        print("params:", irpa, irpa.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
