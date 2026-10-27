#!/usr/bin/env python3
"""P1: freeze the semantic contract into a request/tensor bank + oracles.

Per sample: raw observation/instruction, preprocessed batch_inputs, full ctx
(inputs_embeds, positions, routing, mask, times, EXPLICIT noise), suffix
postfix tensors, joint-prefill per-layer KV digests, v_0, full trajectory,
final action. Oracles (asserted + recorded): time-grid closed form, mask
visibility semantics, mrope section structure, routing histogram/indices.

Usage:
  PYTHONPATH=. .venv-serve/bin/python tools/manifest/build_bank.py \
      --checkpoint <ckpt> --samples 4 --out artifacts/bank
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "iree_wallx"))

from bench_model_api import (  # noqa: E402
    _build_fake_observation, _right_gripper_dim_from_config, build_adapter)


def kv_layers(cache):
    """Return ([(k, v), ...], setter) for legacy or DynamicCache objects."""
    from transformers import DynamicCache
    if isinstance(cache, DynamicCache):
        layers = [(layer.keys, layer.values) for layer in cache.layers]

        def setter(i, k, v):
            cache.layers[i].keys = k
            cache.layers[i].values = v
        return layers, setter
    if isinstance(cache, (list, tuple)):
        layers = [tuple(tensors) for tensors in cache]

        def setter(i, k, v):
            cache[i] = (k, v)
        return layers, setter
    raise TypeError(f"unknown cache type {type(cache)}")

INSTRUCTIONS = [
    "pick up the object",
    "put the cup on the plate",
    "open the drawer and take out the towel",
    "fold the cloth and place it on the shelf",
]
NOISE_SEED = 20261023


def tdigest(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "bank")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ns = argparse.Namespace(
        checkpoint=args.checkpoint, train_config=None, norm_key="x2_normal",
        architecture="qwen2_5", cam_names=["face_view", "right_wrist_view"],
        action_horizon=None, image_size=args.image_size, seed=0)
    adapter = build_adapter(ns)
    model = adapter._model
    gd = _right_gripper_dim_from_config(getattr(adapter, "_train_config", {}))

    from wall_x._vendor.harrix.envs.libero_common import encode_proprio

    A, D = adapter._action_horizon, adapter._action_dim
    N = adapter._num_inference_timesteps
    rng = np.random.default_rng(NOISE_SEED)
    noises = [torch.from_numpy(
        rng.standard_normal((1, A, D)).astype(np.float32))
        for _ in range(args.samples)]

    oracle: dict = {"utc": datetime.now(timezone.utc).isoformat(),
                    "noise_seed": NOISE_SEED, "samples": args.samples}
    index = []

    # ---- one-shot structural oracle (sample 0) ----------------------------
    obs0 = _build_fake_observation(0, args.image_size, gd)
    ob0 = encode_proprio(obs0, adapter._train_config, A)
    pre0, post0 = adapter._get_flow_prompt(INSTRUCTIONS[0])
    bi0 = adapter._construct_model_input([ob0], [pre0], [post0])
    ctx0 = model._prepare_flow_action_inputs(
        input_ids=bi0["input_ids"], action_horizon=A, action_dim=D,
        num_inference_timesteps=N, prefix_length=None,
        attention_mask=bi0.get("attention_mask"),
        position_ids=bi0.get("position_ids"), past_key_values=None,
        inputs_embeds=None, moe_token_types=bi0["moe_token_types"],
        start_indices=None, end_indices=None, positional_masks=None,
        labels=None, use_cache=None, output_attentions=None,
        output_hidden_states=None, return_dict=None,
        pixel_values=bi0.get("pixel_values"),
        pixel_values_videos=bi0.get("pixel_values_videos"),
        image_grid_thw=bi0.get("image_grid_thw"),
        video_grid_thw=bi0.get("video_grid_thw"), action_chunk=None,
        proprioception=bi0.get("proprioception"),
        unnorm_proprioception=bi0.get("unnorm_proprioception"),
        rope_deltas=bi0.get("rope_deltas"), cache_position=None,
        second_per_grid_ts=None, dataset_names=bi0["dataset_names"],
        dof_mask=bi0["dof_mask"], agent_pos_mask=bi0.get("agent_pos_mask"),
        noise=noises[0])

    times = ctx0["times"]
    nsched = model.config.noise_scheduler
    # time_shift comes from noise_scheduler.get("time_shift", 1.0); this
    # checkpoint config has no time_shift key, so the grid is the plain
    # linspace*s form (beta_alpha/beta_beta do NOT shape the inference grid)
    shift = model.action_preprocessor.time_shift
    s = model.action_preprocessor.s
    oracle["time_grid_note"] = (
        "time_shift resolved from noise_scheduler config with default 1.0; "
        f"beta_alpha={nsched['beta_alpha']}, beta_beta={nsched['beta_beta']} "
        "are not part of the inference grid")
    t_lin = torch.linspace(0.0, 1.0, N + 1)
    times_expected = (((shift * t_lin) / (1 + (shift - 1) * t_lin)) * s
                      ).to(times.device)
    oracle["time_grid"] = {
        "N": N, "len": int(times.shape[0]), "shift": shift, "s": s,
        "values": [round(v, 6) for v in times.tolist()],
        "matches_closed_form": bool(
            torch.allclose(times, times_expected, atol=1e-6)),
    }
    assert oracle["time_grid"]["matches_closed_form"], "time grid mismatch"

    # routing oracle
    types = ctx0["moe_token_types"].flatten()
    hist = {int(k): int(v) for k, v in
            zip(*torch.unique(types, return_counts=True))}
    oracle["routing"] = {
        "hist": hist,
        "start_indices": ctx0["start_indices"].tolist(),
        "end_indices": ctx0["end_indices"].tolist(),
        "prefix_length": int(ctx0["prefix_length"]),
        "seq_len": int(bi0["input_ids"].shape[1]),
    }
    assert hist == {0: 606, 1: 32}, f"unexpected routing hist {hist}"

    # mrope oracle: position ids [3, B, L]; sections 16/24/24 per HF config
    pids = ctx0["position_ids"]
    mrope_sections = model.config.rope_scaling["mrope_section"]
    assert list(pids.shape)[0] == 3, f"pos dims {pids.shape}"
    action_pos = pids[:, :, int(ctx0["prefix_length"]):]
    oracle["mrope"] = {
        "shape": list(pids.shape),
        "sections": mrope_sections,
        "action_t_all_zero": bool((action_pos[2] == 0).all().item()),
        "prefix_t_first512_zero": bool(
            (pids[2, 0, :512] == 0).all().item()),
    }

    # ---- per-sample bank ---------------------------------------------------
    for si in range(args.samples):
        obs = _build_fake_observation(si, args.image_size, gd)
        ob = encode_proprio(obs, adapter._train_config, A)
        instr = INSTRUCTIONS[si % len(INSTRUCTIONS)]
        pre_t, post_t = adapter._get_flow_prompt(instr)
        bi = adapter._construct_model_input([ob], [pre_t], [post_t])
        noise = noises[si]
        ctx = model._prepare_flow_action_inputs(
            input_ids=bi["input_ids"], action_horizon=A, action_dim=D,
            num_inference_timesteps=N, prefix_length=None,
            attention_mask=bi.get("attention_mask"),
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
            noise=noise)

        # official vanilla run on the same ctx: trajectory + per-layer KV
        kv_digests = []
        lay = []

        def run_vanilla_with_probe(ctx_in):
            nonlocal lay
            pre_out = model.model(
                input_ids=None, attention_mask=ctx_in["attention_mask"],
                position_ids=ctx_in["position_ids"], past_key_values=None,
                inputs_embeds=ctx_in["inputs_embeds"],
                moe_token_types=ctx_in["moe_token_types"],
                start_indices=ctx_in["start_indices"],
                end_indices=ctx_in["end_indices"],
                positional_masks=ctx_in["positional_masks"],
                use_cache=True, output_attentions=False,
                output_hidden_states=False, return_dict=True,
                adarms_conds=[None, ctx_in["adarms_cond"]])
            kv = pre_out.past_key_values
            lay, _ = kv_layers(kv)
            for k, v in lay:
                kv_digests.append((tdigest(k), tdigest(v)))
            return pre_out

        pre_out = run_vanilla_with_probe(ctx)
        hid = pre_out.last_hidden_state
        ahs = hid[ctx["flow_action_mask"]].to(torch.float32)
        pred0 = model.action_preprocessor.action_proj_back(
            ahs[:, : model.action_preprocessor.action_hidden_size])
        v0 = pred0.reshape(ctx["batch_size"], A, D)

        traj, _ = model._generate_flow_action_vanilla(ctx)
        final_action = traj[-1]

        (pp_pos, pp_emb, pp_mask, pp_types, pp_ids, pp_si, pp_ei,
         _pm) = model._prepare_flow_postfix(ctx, pre_out.past_key_values)

        sample = {
            "meta": {"sample": si, "instruction": instr,
                     "obs_seed": si, "image_size": args.image_size},
            "raw": {"eef_pos": obs["eef_pos"], "eef_axisangle":
                    obs["eef_axisangle"], "gripper": obs["gripper"]},
            "input_ids": bi["input_ids"].cpu(),
            "attention_mask": (bi["attention_mask"].cpu()
                               if bi.get("attention_mask") is not None
                               else None),
            "pixel_values": bi.get("pixel_values"),
            "image_grid_thw": bi.get("image_grid_thw"),
            "moe_token_types": bi["moe_token_types"].cpu(),
            "dof_mask": bi["dof_mask"].cpu(),
            "dataset_names": bi["dataset_names"],
            "noise": noise.cpu(),
            "ctx": {
                "inputs_embeds": ctx["inputs_embeds"].cpu(),
                "position_ids": ctx["position_ids"].cpu(),
                "moe_token_types": ctx["moe_token_types"].cpu(),
                "start_indices": ctx["start_indices"].cpu(),
                "end_indices": ctx["end_indices"].cpu(),
                "times": ctx["times"].cpu(),
                "padding_action": (ctx["padding_action"].cpu()
                                    if ctx["padding_action"] is not None
                                    else None),
                "prefix_length": int(ctx["prefix_length"]),
                "flow_action_mask": ctx["flow_action_mask"].cpu(),
                "dof_mask": ctx["dof_mask"].cpu(),
            },
            "postfix": {
                "position_ids": pp_pos.cpu(), "inputs_embeds": pp_emb.cpu(),
                "attention_mask": pp_mask.cpu(),
                "moe_token_types": pp_types.cpu(), "input_ids": pp_ids.cpu(),
                "start_indices": pp_si.cpu() if pp_si is not None else None,
                "end_indices": pp_ei.cpu() if pp_ei is not None else None,
            },
            "intermediates": {
                "v0": v0.cpu(),
                "kv_digests_sha16": kv_digests,
                "kv_layer0_k": lay[0][0].cpu() if lay else None,
                "kv_layer0_v": lay[0][1].cpu() if lay else None,
                "trajectory": traj.cpu(),
                "final_action": final_action.cpu(),
            },
        }
        if sample["pixel_values"] is not None:
            sample["pixel_values"] = sample["pixel_values"].cpu()
        if sample["image_grid_thw"] is not None:
            sample["image_grid_thw"] = sample["image_grid_thw"].cpu()

        f = args.out / f"sample_{si:03d}.pt"
        torch.save(sample, f)
        index.append({
            "sample": si, "file": f.name, "instruction": instr,
            "noise_sha16": tdigest(noise),
            "final_action_sha16": tdigest(final_action),
            "v0_sha16": tdigest(v0),
            "kv_layer0_k_sha16": kv_digests[0][0],
            "bytes": f.stat().st_size,
        })
        print(f"sample {si}: saved {f.name} "
              f"({f.stat().st_size / 1e6:.1f} MB)", flush=True)

    (args.out / "index.json").write_text(json.dumps(
        {"oracle": oracle, "samples": index}, indent=2))
    print(json.dumps(oracle, indent=2)[:1200])
    print("BANK DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
