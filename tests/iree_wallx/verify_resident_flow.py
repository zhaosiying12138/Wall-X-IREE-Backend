#!/usr/bin/env python3
"""Verify that resident and CLI execution produce identical flow trajectories.

Runs the same staged IREE prefill and suffix inputs through ``iree-run-module``
and the in-process serving runtime, then compares the saved normalized final
action arrays elementwise.  Artifact paths come from the common explicit
``WALLX_IREE_*`` environment configuration.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks" / "iree_wallx"))
CKPT = os.environ.get(
    "WALLX_CKPT", "/home/zhaosiying/codebase/weights/wall-oss-0.5")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-module", type=Path,
        default=Path(os.environ.get(
            "IREE_RUN_MODULE",
            str(Path(sys.executable).with_name("iree-run-module")))),
        help="iree-run-module executable for the CLI comparison arm")
    args = parser.parse_args()

    from bench_model_api import (
        _build_fake_observation, _right_gripper_dim_from_config, build_adapter)
    from wall_x.serving.backend import create_backend

    ns = argparse.Namespace(
        checkpoint=CKPT, train_config=None, norm_key="x2_normal",
        architecture="qwen2_5", cam_names=["face_view", "right_wrist_view"],
        action_horizon=None, image_size=448, seed=0)
    adapter = build_adapter(ns)
    model = adapter._model
    backend = create_backend(model, name="staged")
    from wall_x.model.core import ops as wall_ops
    for name in ("rmsnorm", "m_rope", "rot_pos_emb", "permute", "unpermute",
                 "get_rope_index", "get_window_index"):
        op = getattr(wall_ops, name, None)
        if op is not None:
            op._resolved_fn = op._pytorch_fallback

    bank = torch.load(REPO / "artifacts/bank/sample_000.pt",
                      map_location="cpu", weights_only=False)
    noise = bank["noise"].cuda()
    gripper_dim = _right_gripper_dim_from_config(
        getattr(adapter, "_train_config", {}))
    from wall_x._vendor.harrix.envs.libero_common import encode_proprio
    obs = _build_fake_observation(0, 448, gripper_dim)
    encoded = encode_proprio(obs, adapter._train_config, adapter._action_horizon)
    prefix, postfix = adapter._get_flow_prompt(bank["meta"]["instruction"])
    batch_inputs = adapter._construct_model_input([encoded], [prefix], [postfix])
    batch_inputs = backend._pad_to_bucket(batch_inputs)
    ctx = backend.prepare_ctx(
        adapter._action_horizon, adapter._action_dim,
        adapter._num_inference_timesteps, None, noise, batch_inputs)
    backend._validate_static_context(
        ctx, adapter._action_horizon, adapter._action_dim)
    mask3d = backend._joint_attention_mask(ctx)
    postfix_pos, postfix_mask = backend._postfix(ctx)
    padding = backend._padding_action(ctx)

    times = backend._host(ctx["times"], np.float32)
    prefill_inputs = (
        backend._host(ctx["inputs_embeds"], np.float32),
        backend._host(ctx["position_ids"]),
        backend._host(mask3d, np.bool_),
        backend._host(ctx["noise"], np.float32),
        backend._host(padding, np.float32),
        np.asarray([times[1] - times[0]], dtype=np.float32),
    )
    postfix_pos = backend._host(postfix_pos)
    postfix_mask = backend._host(postfix_mask, np.bool_)
    v_padding = backend._host(padding - ctx["noise"], np.float32)

    cli_env = dict(os.environ)
    if backend.cudalib:
        existing = cli_env.get("LD_LIBRARY_PATH", "")
        cli_env["LD_LIBRARY_PATH"] = str(backend.cudalib) + (
            ":" + existing if existing else "")
    cli_call = 0

    def run_cli(vmfb, irpa, inputs, tmpdir, output_count):
        nonlocal cli_call
        stem = f"call_{cli_call}"
        cli_call += 1
        input_flags = []
        for index, array in enumerate(inputs):
            path = tmpdir / f"{stem}_input_{index}.npy"
            np.save(path, array)
            input_flags.append(f"--input=@{path}")
        output_paths = [tmpdir / f"{stem}_output_{index}.npy"
                        for index in range(output_count)]
        cmd = [
            str(args.run_module), f"--device={backend.device}",
            f"--module={vmfb}", f"--parameters=model={irpa}",
            "--function=main", *input_flags,
            *[f"--output=@{path}" for path in output_paths],
        ]
        completed = subprocess.run(
            cmd, cwd=REPO, env=cli_env, capture_output=True, text=True,
            timeout=600)
        if completed.returncode != 0:
            raise RuntimeError(
                f"iree-run-module failed for {stem}:\n"
                f"{completed.stdout[-1500:]}\n{completed.stderr[-1500:]}")
        return [np.load(path) for path in output_paths]

    def execute(mode, tmpdir):
        session = None
        if mode == "resident":
            from wall_x.serving.backend.iree.runtime import IreeResidentSession
            session = IreeResidentSession(
                backend.prefill_vmfb, backend.prefill_irpa,
                backend.step_vmfb, backend.step_irpa,
                device=backend.device, cudalib=backend.cudalib)
            k_past, v_past, _v0, x = session.prefill(*prefill_inputs)
        else:
            k_past, v_past, _v0, x = run_cli(
                backend.prefill_vmfb, backend.prefill_irpa,
                prefill_inputs, tmpdir, output_count=4)
        x = np.asarray(x, dtype=np.float32)
        for index in range(len(times) - 2):
            timestep = np.asarray([times[1 + index]], dtype=np.float32)
            dt = np.asarray(
                [times[2 + index] - times[1 + index]], dtype=np.float32)
            if mode == "resident":
                velocity = session.step(
                    x, timestep, postfix_pos, postfix_mask,
                    k_past, v_past, v_padding)
            else:
                velocity = run_cli(
                    backend.step_vmfb, backend.step_irpa,
                    (x, timestep, postfix_pos, postfix_mask,
                     k_past, v_past, v_padding),
                    tmpdir, output_count=1)[0]
            x = x + dt * velocity
        np.save(Path(f"/tmp/wallx_flow_{mode}.npy"), x)
        return x

    outputs = {}
    with tempfile.TemporaryDirectory(prefix="wallx_flow_") as directory:
        tmpdir = Path(directory)
        for mode in ("cli", "resident"):
            print(f"\n===== RUN {mode.upper()} =====", flush=True)
            outputs[mode] = execute(mode, tmpdir)

    a, b = outputs["cli"], outputs["resident"]
    max_abs = float(np.abs(a - b).max())
    equal = bool(np.array_equal(a, b))
    print("resident_vs_cli_final_shape:", a.shape)
    print("resident_vs_cli_final_maxabs:", max_abs)
    print("resident_vs_cli_array_equal:", equal)
    if not equal:
        print("RESIDENT-FLOW FAIL")
        return 1
    print("RESIDENT-FLOW PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
