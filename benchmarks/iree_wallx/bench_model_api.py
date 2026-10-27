#!/usr/bin/env python3
"""Model-API latency benchmark for the official Wall-X flow path (P0: B0/B1).

Replicates scripts/fake_inference.py's invocation of the harrix adapter and
times ``generate_flow_action`` end to end (tensor call start -> output
available on host-synced clock = Model API boundary in the plan's terms).

Discipline:
- 3 pre-generated explicit noise tensors alternate across iterations (same
  shape, different content) to defeat any accidental content caching.
- torch.cuda.synchronize before/after every timed iteration.
- Per-iteration JSONL + summary JSON with p50/p95/p99.
- --profile N: run N extra iterations under torch.profiler, save chrome trace,
  and count CUDA graph runtime API events (capture / instantiate / launch) as
  the runtime evidence for "flag-on, capture-or-not" (B1 vs B0).

Usage (repo root, PYTHONPATH=.):
  .venv-serve/bin/python benchmarks/iree_wallx/bench_model_api.py \
      --checkpoint <ckpt> --label B0 --iters 100 --out artifacts/logs
  ENABLE_CUDA_GRAPH=true ... --label B1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _build_fake_observation(seed: int, image_size: int, gripper_dim: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "eef_pos": rng.normal(size=(3,)).astype(np.float32),
        "eef_axisangle": rng.normal(size=(3,)).astype(np.float32),
        "gripper": rng.normal(size=(gripper_dim,)).astype(np.float32),
        "face_view": rng.integers(0, 256, (image_size, image_size, 3), dtype=np.uint8),
        "wrist_view": rng.integers(0, 256, (image_size, image_size, 3), dtype=np.uint8),
    }


def _right_gripper_dim_from_config(train_config: dict) -> int:
    layout = train_config.get("agent_pos_config") or train_config.get("task", {}).get(
        "agent_pos_config", {}
    )
    if not isinstance(layout, dict):
        return 1
    gripper_dim = 1
    for key, dim in layout.items():
        bare = key.replace("follow_", "").replace("master_", "")
        if bare == "right_gripper":
            gripper_dim = int(dim)
            break
    norm_dim = int(train_config.get("_libero_proprio_norm_dim") or 0)
    real_dim = sum(int(v) for k, v in layout.items() if k != "action_padding")
    if norm_dim == real_dim + 1:
        gripper_dim += 1
    return max(1, gripper_dim)


def build_adapter(args):
    import wall_x._vendor.harrix.adapters  # noqa: F401  register model adapters
    from wall_x._vendor.harrix.adapters.registry import build_adapter
    from wall_x._vendor.harrix.eval_config import (
        EvalConfig,
        LiberoEnvParams,
        autofill_from_checkpoint,
    )

    cfg = EvalConfig()
    cfg.model.checkpoint_path = args.checkpoint
    cfg.model.train_config_path = args.train_config
    cfg.model.norm_key = args.norm_key
    cfg.model.cam_names = list(args.cam_names)
    cfg.model.action_horizon = args.action_horizon
    cfg.model.architecture = args.architecture
    cfg.model.action_mode = "flow"
    cfg.env.libero = LiberoEnvParams(num_trials_per_task=1, task_indices=[0])
    cfg = autofill_from_checkpoint(cfg)
    adapter = build_adapter(cfg)
    return adapter


def predict(adapter, payload, noise):
    from wall_x._vendor.harrix.envs.libero_common import encode_proprio

    observation = encode_proprio(
        payload["observation"], adapter._train_config, adapter._action_horizon
    )
    prefix, postfix = adapter._get_flow_prompt(payload["instruction"])
    batch_inputs = adapter._construct_model_input([observation], [prefix], [postfix])

    dataset_names = batch_inputs["dataset_names"]
    padding = torch.zeros(
        (len(dataset_names), 1, adapter._action_dim), dtype=torch.float32
    )
    padding_action = adapter._normalizer_action.normalize_data(
        padding, dataset_names
    ).to(batch_inputs["input_ids"].device)

    model_output = adapter._model.generate_flow_action(
        action_horizon=adapter._action_horizon,
        action_dim=adapter._action_dim,
        num_inference_timesteps=adapter._num_inference_timesteps,
        padding_action=padding_action,
        noise=noise,
        **batch_inputs,
    )
    predict_action = model_output["predict_action"]
    if isinstance(predict_action, torch.Tensor):
        predict_action = predict_action.detach().cpu().numpy()
    return np.asarray(predict_action[0])


def graph_api_summary(trace_path: Path) -> dict:
    """Count CUDA graph runtime API events in an exported chrome trace."""
    events = json.loads(trace_path.read_text()).get("traceEvents", [])
    counts: dict[str, int] = {}
    kernel_names = set()
    for ev in events:
        name = ev.get("name", "")
        cat = ev.get("cat", "")
        if cat == "cuda_runtime" or cat == "cuda_driver":
            low = name.lower()
            if any(tag in low for tag in ("graph", "capture")):
                counts[name] = counts.get(name, 0) + 1
        if cat == "kernel":
            kernel_names.add(name)
    return {"graph_api_events": counts, "distinct_kernels": len(kernel_names)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train-config", default=None)
    ap.add_argument("--norm-key", default="x2_normal")
    ap.add_argument("--architecture", default="qwen2_5")
    ap.add_argument("--cam-names", nargs="+",
                    default=["face_view", "right_wrist_view"])
    ap.add_argument("--action-horizon", type=int, default=None)
    ap.add_argument("--instruction", default="pick up the object")
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="B0")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "logs")
    ap.add_argument("--profile", type=int, default=0,
                    help="extra iterations to run under torch.profiler")
    args = ap.parse_args()

    run_id = f"{args.label}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out_dir = args.out / f"model_api_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter = build_adapter(args)
    gripper_dim = _right_gripper_dim_from_config(
        getattr(adapter, "_train_config", {}))
    payload = {"observation": _build_fake_observation(
        args.seed, args.image_size, gripper_dim),
        "instruction": args.instruction}

    # explicit noise corpus: 3 fixed tensors, alternating
    noise_rng = np.random.default_rng(1234)
    noises = [
        torch.from_numpy(noise_rng.standard_normal(
            (1, adapter._action_horizon, adapter._action_dim)).astype(np.float32))
        for _ in range(3)
    ]

    # first call: correctness anchor + shapes for the manifest
    t0 = time.perf_counter()
    action = predict(adapter, payload, noises[0])
    torch.cuda.synchronize()
    first_s = time.perf_counter() - t0
    action_digest = hashlib.sha256(
        np.ascontiguousarray(action, dtype=np.float64).tobytes()).hexdigest()

    for i in range(args.warmup):
        predict(adapter, payload, noises[i % 3])
    torch.cuda.synchronize()

    rows = []
    for i in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        predict(adapter, payload, noises[i % 3])
        torch.cuda.synchronize()
        rows.append(time.perf_counter() - t0)

    ms = sorted(r * 1000 for r in rows)

    # write raw iterations first so a later failure never loses measurements
    jsonl = out_dir / "iterations.jsonl"
    with jsonl.open("w") as f:
        for i, v in enumerate(ms):
            f.write(json.dumps({"run_id": run_id, "iter": i,
                                "wall_ms": v}) + "\n")

    def pct(p):
        k = min(len(ms) - 1, max(0, int(round(p / 100 * (len(ms) - 1)))))
        return ms[k]

    summary = {
        "run_id": run_id,
        "label": args.label,
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "ENABLE_CUDA_GRAPH": os.environ.get("ENABLE_CUDA_GRAPH", "<unset>"),
            "ENABLE_EXPERIMENTAL_INFERENCE_ENGINE": os.environ.get(
                "ENABLE_EXPERIMENTAL_INFERENCE_ENGINE", "<unset>"),
        },
        "workload": {
            "image_size": args.image_size,
            "instruction": args.instruction,
            "action_horizon": adapter._action_horizon,
            "action_dim": adapter._action_dim,
            "num_inference_timesteps": adapter._num_inference_timesteps,
            "noise_shape": list(noises[0].shape),
            "noise_seed": 1234,
        },
        "first_call_s": first_s,
        "action_shape": list(action.shape),
        "action_sha256_float64": action_digest,
        "iters": args.iters,
        "p50_ms": pct(50), "p95_ms": pct(95), "p99_ms": pct(99),
        "mean_ms": statistics.fmean(ms),
        "min_ms": ms[0], "max_ms": ms[-1],
        "vram": {
            "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        },
    }

    if args.profile > 0:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for i in range(args.profile):
                predict(adapter, payload, noises[i % 3])
            torch.cuda.synchronize()
        trace_path = out_dir / "trace.json"
        prof.export_chrome_trace(str(trace_path))
        summary["graph_audit"] = graph_api_summary(trace_path)
        # keep the trace out of git; record name only
        summary["graph_audit"]["trace_file"] = trace_path.name

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in
                      ("run_id", "label", "p50_ms", "p95_ms", "p99_ms",
                       "mean_ms", "first_call_s")}, indent=2))
    if "graph_audit" in summary:
        print("graph_audit:", json.dumps(summary["graph_audit"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
