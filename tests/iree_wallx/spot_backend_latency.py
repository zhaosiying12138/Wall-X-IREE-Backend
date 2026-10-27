#!/usr/bin/env python3
"""Same-input latency spot check at the common generate-call boundary.

Run each backend in a separate process.  Session creation and model-input
construction are outside the timed region; every timed call includes context,
flow, and official finalization work.  This is a GPU/model benchmark and is not
part of the CPU-safe test suite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks" / "iree_wallx"))
CKPT = os.environ.get(
    "WALLX_CKPT", "/home/zhaosiying/codebase/weights/wall-oss-0.5"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("official", "staged", "expanded", "torch-compile"),
        required=True,
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--array-out", type=Path, default=None)
    parser.add_argument("--expanded-vmfb", type=Path, default=None)
    parser.add_argument("--expanded-irpa", type=Path, default=None)
    parser.add_argument("--prefill-vmfb", type=Path, default=None)
    parser.add_argument("--prefill-irpa", type=Path, default=None)
    parser.add_argument("--step-vmfb", type=Path, default=None)
    parser.add_argument("--step-irpa", type=Path, default=None)
    parser.add_argument("--torch-compile-mode", default="reduce-overhead")
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmup < 0:
        raise ValueError("repetitions must be >=1 and warmup >=0")
    if args.repetitions + args.warmup > 20:
        raise ValueError("safety limit: warmup + repetitions must be <=20")
    out_path = args.out or Path(f"/tmp/wallx_spot_{args.backend}.json")

    from bench_model_api import (
        _build_fake_observation,
        _right_gripper_dim_from_config,
        build_adapter,
    )

    namespace = argparse.Namespace(
        checkpoint=CKPT,
        train_config=None,
        norm_key="x2_normal",
        architecture="qwen2_5",
        cam_names=["face_view", "right_wrist_view"],
        action_horizon=None,
        image_size=448,
        seed=0,
    )
    adapter = build_adapter(namespace)
    model = adapter._model
    from wall_x.model.core import ops as wall_ops

    for name in (
        "rmsnorm",
        "m_rope",
        "rot_pos_emb",
        "permute",
        "unpermute",
        "get_rope_index",
        "get_window_index",
    ):
        op = getattr(wall_ops, name, None)
        if op is not None:
            op._resolved_fn = op._pytorch_fallback
    bank = torch.load(
        REPO / "artifacts/bank/sample_000.pt",
        map_location="cpu",
        weights_only=False,
    )
    gripper_dim = _right_gripper_dim_from_config(
        getattr(adapter, "_train_config", {})
    )
    from wall_x._vendor.harrix.envs.libero_common import encode_proprio

    observation = _build_fake_observation(0, 448, gripper_dim)
    encoded = encode_proprio(
        observation, adapter._train_config, adapter._action_horizon
    )
    prefix, postfix = adapter._get_flow_prompt(bank["meta"]["instruction"])
    batch_inputs = adapter._construct_model_input([encoded], [prefix], [postfix])
    noise = bank["noise"].cuda()

    options = {}
    if args.backend == "expanded":
        options.update(
            full_flow_vmfb=args.expanded_vmfb,
            full_flow_irpa=args.expanded_irpa,
        )
    elif args.backend == "staged":
        options.update(
            prefill_vmfb=args.prefill_vmfb,
            prefill_irpa=args.prefill_irpa,
            step_vmfb=args.step_vmfb,
            step_irpa=args.step_irpa,
        )
    elif args.backend == "torch-compile":
        options["mode"] = (
            None if args.torch_compile_mode == "default" else args.torch_compile_mode
        )

    from wall_x.serving.backend import create_backend

    backend = create_backend(model, name=args.backend, **options)
    backend.initialize()  # exclude artifact mapping/loading from timed calls

    rows, outputs = [], []
    with torch.no_grad():
        for index in range(args.warmup + args.repetitions):
            torch.cuda.synchronize()
            start = time.perf_counter()
            # Identical API, explicit noise, and finalization for all four arms.
            output = backend.generate_flow_action(
                action_horizon=32,
                action_dim=26,
                num_inference_timesteps=10,
                padding_action=None,
                noise=noise,
                **batch_inputs,
            )
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            value = output["predict_action"].detach().cpu().numpy()
            if index >= args.warmup:
                rows.append(elapsed_ms)
                outputs.append(value)
            print(
                f"{args.backend} iter={index} wall_ms={elapsed_ms:.3f}", flush=True
            )
    equal = all(np.array_equal(outputs[0], value) for value in outputs[1:])
    result = {
        "backend": args.backend,
        "resolved_backend": backend.name,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "wall_ms": rows,
        "mean_ms": statistics.fmean(rows),
        "median_ms": statistics.median(rows),
        "same_noise_outputs_bitwise_equal": equal,
        "predict_shape": list(outputs[0].shape),
        "peak_torch_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_torch_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    if args.backend == "torch-compile":
        result["torch_compile_mode"] = args.torch_compile_mode
    if args.array_out is not None:
        args.array_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.array_out, predict_action=value)
        result["array_out"] = str(args.array_out)
        result["physical_action_sha256"] = hashlib.sha256(
            value.tobytes()).hexdigest()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
