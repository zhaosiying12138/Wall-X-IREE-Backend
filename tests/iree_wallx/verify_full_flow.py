#!/usr/bin/env python3
"""Same-noise differential verification for the single full_flow artifact.

Run modes in separate processes (24GB GPU safety):
  trace: full_flow_trace + PyTorch static reference + official/bank comparisons
  production: real IreeExpandedBackend with one-output full_flow VMFB

Trace mode checks context, all 36 K/V pairs, v0/x0, every suffix velocity, every
Euler state, and normalized/unnormalized final action.  Production mode checks
shape and the one-invocation boundary; compare its saved action with trace mode
using tools/iree/compare_flow_traces.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks" / "iree_wallx"))
CKPT = os.environ.get(
    "WALLX_CKPT", "/home/zhaosiying/codebase/weights/wall-oss-0.5")


def _delta(actual, expected):
    a = np.asarray(actual, dtype=np.float32)
    e = np.asarray(expected, dtype=np.float32)
    maxabs = float(np.abs(a - e).max())
    return {"maxabs": maxabs, "meanabs": float(np.abs(a - e).mean()),
            "relmax": maxabs / max(float(np.abs(e).max()), 1e-12)}


def _mask(ctx, model):
    from wall_x.model.core.attention.mask import update_joint_attention_mask_2d
    valid = ctx["attention_mask"].bool()
    positional = dict(ctx["positional_masks"] or {})
    positional["padding_positions"] = ~valid
    mask = update_joint_attention_mask_2d(
        ctx["attention_mask"], ctx["moe_token_types"], positional,
        causal_action_attention_mask=model.config.causal_action_attention_mask
    ).bool()
    eye = torch.eye(638, dtype=torch.bool, device=mask.device).unsqueeze(0)
    return torch.where((~valid).unsqueeze(-1),
                       eye & (~valid).unsqueeze(-1), mask)


def _official_ctx(model, adapter, bi, noise):
    return model._prepare_flow_action_inputs(
        input_ids=bi["input_ids"], action_horizon=adapter._action_horizon,
        action_dim=adapter._action_dim,
        num_inference_timesteps=adapter._num_inference_timesteps,
        prefix_length=None, attention_mask=bi.get("attention_mask"),
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


def _graph_args(backend, bi, ctx, mask, postfix_pos, postfix_mask):
    padding = ctx["padding_action"]
    if padding is None:
        padding = torch.zeros(1, 26, device=mask.device)
    args = [backend._host(bi["input_ids"], np.int64),
            backend._host(bi["pixel_values"], np.float32)]
    if not backend._state_string:
        args.extend([backend._host(bi["proprioception"], np.float32),
                     backend._host(bi["agent_pos_mask"], np.float32)])
    args.extend([
        backend._host(ctx["position_ids"]), backend._host(mask, np.bool_),
        backend._host(ctx["noise"], np.float32),
        backend._host(padding.reshape(1, 1, -1), np.float32),
        backend._host(ctx["times"], np.float32),
        backend._host(postfix_pos), backend._host(postfix_mask, np.bool_),
        backend._host(ctx["dof_mask"], np.float32),
    ])
    return args


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("trace", "production"), default="trace")
    ap.add_argument("--vmfb", type=Path, default=None,
                    help="defaults to WALLX_IREE_FULL_FLOW_VMFB")
    ap.add_argument("--irpa", type=Path, default=None,
                    help="defaults to WALLX_IREE_FULL_FLOW_IRPA")
    ap.add_argument("--save", type=Path, default=None)
    ap.add_argument("--fresh-official-flow", action="store_true",
                    help="also execute the full official flow as the reference "
                         "(required for explicit eager-profile acceptance)")
    ap.add_argument("--allow-experimental", action="store_true",
                    help="report strict failure separately and return success "
                         "only for the limited final-action gate")
    ap.add_argument("--official-attn", choices=("flash", "sdpa", "eager"),
                    default="flash", help="explicit official visual reference class")
    args = ap.parse_args()
    vmfb = args.vmfb or os.environ.get("WALLX_IREE_FULL_FLOW_VMFB")
    irpa = args.irpa or os.environ.get("WALLX_IREE_FULL_FLOW_IRPA")
    save = args.save or Path(f"/tmp/wallx_full_flow_{args.mode}.npz")
    save.parent.mkdir(parents=True, exist_ok=True)

    from bench_model_api import (
        _build_fake_observation, _right_gripper_dim_from_config, build_adapter)
    from wall_x.serving.backend import create_backend
    ns = argparse.Namespace(
        checkpoint=CKPT, train_config=None, norm_key="x2_normal",
        architecture="qwen2_5", cam_names=["face_view", "right_wrist_view"],
        action_horizon=None, image_size=448, seed=0)
    adapter = build_adapter(ns)
    model = adapter._model
    if args.official_attn != "flash":
        from wall_x.model.qact.qwen2_5.modeling_qwen2_5_vl import (
            Qwen2_5_VLVisionAttention, Qwen2_5_VLVisionSdpaAttention)
        attention_cls = (Qwen2_5_VLVisionAttention
                         if args.official_attn == "eager"
                         else Qwen2_5_VLVisionSdpaAttention)
        for block in model.visual.blocks:
            previous = block.attn
            replacement = attention_cls(
                model.visual.config, model.visual.config.hidden_size,
                num_heads=model.visual.config.num_heads)
            replacement.load_state_dict(previous.state_dict())
            block.attn = replacement.to(
                device=previous.qkv.weight.device,
                dtype=previous.qkv.weight.dtype)
        model.visual.config._attn_implementation = args.official_attn
    from wall_x.model.core import ops as wall_ops
    for name in ("rmsnorm", "m_rope", "rot_pos_emb", "permute", "unpermute",
                 "get_rope_index", "get_window_index"):
        op = getattr(wall_ops, name, None)
        if op is not None:
            op._resolved_fn = op._pytorch_fallback

    bank = torch.load(REPO / "artifacts/bank/sample_000.pt",
                      map_location="cpu", weights_only=False)
    noise = bank["noise"].cuda()
    gd = _right_gripper_dim_from_config(getattr(adapter, "_train_config", {}))
    from wall_x._vendor.harrix.envs.libero_common import encode_proprio
    obs = _build_fake_observation(0, 448, gd)
    ob = encode_proprio(obs, adapter._train_config, adapter._action_horizon)
    pre_t, post_t = adapter._get_flow_prompt(bank["meta"]["instruction"])
    bi = adapter._construct_model_input([ob], [pre_t], [post_t])
    backend = create_backend(
        model, name="iree-expanded", full_flow_vmfb=vmfb, full_flow_irpa=irpa)
    bi = backend._pad_to_bucket(bi)

    if args.mode == "production":
        calls = {"vision": 0, "action_step": 0}
        def vision_hook(_module, _inputs, _output):
            calls["vision"] += 1
        hook = model.visual.register_forward_hook(vision_hook)
        original_step = model.action_preprocessor.step
        def counted_step(*step_args, **step_kwargs):
            calls["action_step"] += 1
            return original_step(*step_args, **step_kwargs)
        model.action_preprocessor.step = counted_step
        try:
            output = backend.generate_flow_action(
                action_horizon=32, action_dim=26, num_inference_timesteps=10,
                padding_action=None, noise=noise, **bi)
        finally:
            hook.remove()
            model.action_preprocessor.step = original_step
        predict = output["predict_action"].detach().cpu().numpy()
        session_calls = backend._get_session().calls
        result = {"mode": "production", "predict_shape": list(predict.shape),
                  "session_calls": session_calls, "vmfb": str(vmfb),
                  "pytorch_vision_calls": calls["vision"],
                  "pytorch_action_step_calls": calls["action_step"]}
        np.savez(save, predict_action=predict)
        print(json.dumps(result, indent=2))
        ok = (predict.shape == (1, 32, 26) and session_calls == 1
              and calls["vision"] == 0 and calls["action_step"] == 0)
        print("RESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    # Official raw context is a fresh ViT/text/state reference with the same
    # explicit noise.  Expanded metadata uses a placeholder, never this context.
    official_trajectory = None
    official_output = None
    with torch.no_grad():
        official_ctx = _official_ctx(model, adapter, bi, noise)
        if args.fresh_official_flow:
            official_trajectory, _ = model._generate_flow_action_vanilla(official_ctx)
            official_output = model._finalize_flow_action_output(
                official_trajectory, official_ctx)
        ctx = backend.prepare_ctx(32, 26, 10, None, noise, bi)
    mask = _mask(ctx, model)
    pp_pos, _e, pp_mask, _t, _i, _s, _ei, _pm = \
        model._prepare_flow_postfix(ctx, None)
    graph_args = _graph_args(backend, bi, ctx, mask, pp_pos, pp_mask)
    # Independent Python execution of the exact static context + t=0 scatter.
    from wall_x.serving.backend.static_flow import StaticVisionTextContext
    from wall_x.serving.backend.static_flow.vision import _ordered_scatter
    python_context = StaticVisionTextContext(
        model, bi["image_grid_thw"].detach().cpu(),
        attention_mode="eager-bf16").cuda().eval()
    with torch.no_grad():
        features = python_context.forward_features(
            bi["input_ids"], bi["pixel_values"])
        action_embed, _ = model.action_preprocessor.step(
            timestep=ctx["times"][0].reshape(1),
            noisy_action=ctx["noise"], dof_mask=ctx["dof_mask"])
        action_embed = action_embed.reshape(-1, features.shape[-1]).to(features.dtype)
        python_static_embeds = _ordered_scatter(
            features, bi["input_ids"] == model.action_token_id_set["action_token_id"],
            action_embed)
        python_static_np = python_static_embeds.float().detach().cpu().numpy()
    del python_context, features, action_embed, python_static_embeds
    torch.cuda.empty_cache()
    from wall_x.serving.backend.iree.runtime import IreeExpandedSession
    session = IreeExpandedSession(
        vmfb, irpa, trace=True, device=backend.device, cudalib=backend.cudalib)
    if backend._state_string:
        trace = session.run(*graph_args, with_trace=True)
    else:
        trace = session.run_tensor_state(*graph_args, with_trace=True)
    final, embeds_np, k_np, v_np, v0_np, x0_np, velocities, trajectory = trace
    print("full_flow trace invocation complete", flush=True)
    # Preserve raw unified outputs before constructing any reference/report.
    np.savez(save, final=final, embeds=embeds_np, v0=v0_np, x0=x0_np,
             velocities=velocities, trajectory=trajectory)

    # Same-context PyTorch static reference, including runtime DOF overrides.
    from wall_x.serving.backend.static_flow import (
        StaticDenoiseStep, StaticJointPrefill)
    embeds = torch.from_numpy(embeds_np).cuda()
    dof = ctx["dof_mask"]
    padding = ctx["padding_action"]
    if padding is None:
        padding = torch.zeros(1, 26, device="cuda")
    ref_pre = StaticJointPrefill(model, dof_mask=dof.detach().cpu()).cuda().eval()
    ref_step = StaticDenoiseStep(
        model, len(model.model.layers), dof_mask=dof.detach().cpu()).cuda().eval()
    times = ctx["times"]
    with torch.no_grad():
        k_ref, v_ref, v0_ref, x_ref = ref_pre(
            embeds, ctx["position_ids"], mask, ctx["noise"],
            padding, (times[1] - times[0]).reshape(1), dof_mask=dof)
        v_padding = padding - ctx["noisy_action"]
        ref_velocities, ref_trajectory = [], [x_ref]
        for i in range(9):
            velocity = ref_step(
                x_ref, times[1 + i].reshape(1), pp_pos, pp_mask,
                k_ref, v_ref, v_padding=v_padding, dof_mask=dof)
            x_ref = x_ref + (times[2 + i] - times[1 + i]) * velocity
            ref_velocities.append(velocity)
            ref_trajectory.append(x_ref)
        ref_final = x_ref.detach().cpu().numpy()
        ref_velocities = torch.stack(ref_velocities).detach().cpu().numpy()
        ref_trajectory = torch.stack(ref_trajectory).detach().cpu().numpy()
        k_ref_np = k_ref.detach().cpu().numpy()
        v_ref_np = v_ref.detach().cpu().numpy()
        v0_ref_np = v0_ref.detach().cpu().numpy()
        x0_ref_np = ref_trajectory[0]

    official_embeds_np = official_ctx["inputs_embeds"].float().detach().cpu().numpy()
    ids_np = bi["input_ids"].detach().cpu().numpy()
    image_rows = ids_np == model.config.image_token_id
    action_rows = ids_np == model.action_token_id_set["action_token_id"]
    other_rows = ~(image_rows | action_rows)
    context_categories = {
        "image": _delta(embeds_np[image_rows], official_embeds_np[image_rows]),
        "action": _delta(embeds_np[action_rows], official_embeds_np[action_rows]),
        "text_and_state": _delta(embeds_np[other_rows],
                                 official_embeds_np[other_rows]),
    }
    token_errors = np.abs(embeds_np - official_embeds_np).max(axis=(0, 2))
    worst_tokens = np.argsort(token_errors)[-8:][::-1]
    context_categories["worst_tokens"] = [
        {"token": int(i), "maxabs": float(token_errors[i]),
         "image": bool(image_rows[0, i]), "action": bool(action_rows[0, i])}
        for i in worst_tokens]
    result = {
        "mode": "trace", "session_calls": session.calls,
        "official_vision_attention": model.visual.config._attn_implementation,
        "official_vision_class": type(model.visual.blocks[0].attn).__name__,
        "official_reference": ("fresh_flow" if args.fresh_official_flow
                               else "bank_flow"),
        "shapes": {"final": list(final.shape), "embeds": list(embeds_np.shape),
                   "k": list(k_np.shape), "v": list(v_np.shape),
                   "v0": list(v0_np.shape), "x0": list(x0_np.shape),
                   "velocities": list(velocities.shape),
                   "trajectory": list(trajectory.shape)},
        "context_vs_fresh_official": _delta(embeds_np, official_embeds_np),
        "context_vs_python_static": _delta(embeds_np, python_static_np),
        "python_static_vs_official": _delta(python_static_np, official_embeds_np),
        "context_categories": context_categories,
        "context_vs_bank": _delta(
            embeds_np, bank["ctx"]["inputs_embeds"].float().detach().cpu().numpy()),
        "position_equal": bool(torch.equal(
            ctx["position_ids"], official_ctx["position_ids"])),
        "times_equal": bool(torch.equal(ctx["times"], official_ctx["times"])),
        "noise_equal": bool(torch.equal(ctx["noise"], official_ctx["noise"])),
        "v0_vs_static": _delta(v0_np, v0_ref_np),
        "x0_vs_static": _delta(x0_np, x0_ref_np),
        "final_vs_static": _delta(final, ref_final),
        "final_vs_official_bank": _delta(
            final, bank["intermediates"]["final_action"].detach().cpu().numpy()),
        "velocities_vs_static": [
            _delta(velocities[i], ref_velocities[i]) for i in range(9)],
        "trajectory_vs_static": [
            _delta(trajectory[i], ref_trajectory[i]) for i in range(10)],
        "kv_k_vs_static": [_delta(k_np[i], k_ref_np[i]) for i in range(36)],
        "kv_v_vs_static": [_delta(v_np[i], v_ref_np[i]) for i in range(36)],
    }
    fresh_normalized = None
    fresh_physical = None
    if args.fresh_official_flow:
        fresh_normalized = official_trajectory[-1].detach().cpu().numpy()
        fresh_physical = official_output["predict_action"].detach().cpu().numpy()
        result["final_vs_fresh_official"] = _delta(final, fresh_normalized)
    from wall_x.model.core.action.normalizer import (
        unnormalize_data_with_virtual_tail)
    predict = unnormalize_data_with_virtual_tail(
        model.action_preprocessor.normalizer_action,
        torch.from_numpy(final).cuda(), ctx["dataset_names"], dof)
    official_predict = unnormalize_data_with_virtual_tail(
        model.action_preprocessor.normalizer_action,
        bank["intermediates"]["final_action"].detach().cuda(),
        ctx["dataset_names"], dof)
    predict_np = predict.detach().cpu().numpy()
    result["unnormalized_vs_official_bank"] = _delta(
        predict_np, official_predict.detach().cpu().numpy())
    if args.fresh_official_flow:
        result["unnormalized_vs_fresh_official"] = _delta(
            predict_np, fresh_physical)
    result["max_velocity_rel"] = max(
        row["relmax"] for row in result["velocities_vs_static"])
    result["max_k_rel"] = max(row["relmax"] for row in result["kv_k_vs_static"])
    result["max_v_rel"] = max(row["relmax"] for row in result["kv_v_vs_static"])
    extra = {}
    if args.fresh_official_flow:
        extra = {"official_final": fresh_normalized,
                 "official_predict_action": fresh_physical}
    np.savez(save, final=final, predict_action=predict_np,
             embeds=embeds_np, official_embeds=official_embeds_np,
             python_static_embeds=python_static_np,
             v0=v0_np, x0=x0_np, velocities=velocities,
             trajectory=trajectory, reference_final=ref_final,
             reference_velocities=ref_velocities,
             reference_trajectory=ref_trajectory, **extra)
    print(json.dumps(result, indent=2))

    expected_shapes = ((1, 32, 26), (1, 638, 2048),
                       (36, 1, 2, 638, 128), (36, 1, 2, 638, 128),
                       (1, 32, 26), (1, 32, 26), (9, 1, 32, 26),
                       (10, 1, 32, 26))
    actual_shapes = tuple(tuple(row) for row in
                          (final.shape, embeds_np.shape, k_np.shape, v_np.shape,
                           v0_np.shape, x0_np.shape, velocities.shape,
                           trajectory.shape))
    final_reference_key = ("final_vs_fresh_official" if args.fresh_official_flow
                           else "final_vs_official_bank")
    physical_reference_key = ("unnormalized_vs_fresh_official"
                              if args.fresh_official_flow
                              else "unnormalized_vs_official_bank")
    ok = (actual_shapes == expected_shapes and session.calls == 1
          and result["position_equal"] and result["times_equal"]
          and result["noise_equal"]
          and result["context_vs_fresh_official"]["relmax"] < 0.02
          and result["context_vs_python_static"]["relmax"] < 0.02
          and result["python_static_vs_official"]["relmax"] < 0.02
          and result["v0_vs_static"]["relmax"] < 0.02
          and result["x0_vs_static"]["relmax"] < 0.02
          and result["max_k_rel"] < 0.06 and result["max_v_rel"] < 0.06
          and result["max_velocity_rel"] < 0.02
          and result["final_vs_static"]["relmax"] < 0.02
          and result[final_reference_key]["relmax"] < 0.03
          and result[physical_reference_key]["relmax"] < 0.03)
    print("RESULT:", "PASS" if ok else "FAIL")
    if args.allow_experimental:
        # Limited final-action acceptance; strict context/KV result remains
        # visible above and is never replaced by this experimental result.
        final_ok = (
            actual_shapes == expected_shapes and session.calls == 1
            and result["position_equal"] and result["times_equal"]
            and result["noise_equal"]
            and result["v0_vs_static"]["relmax"] < 0.02
            and result["x0_vs_static"]["relmax"] < 0.02
            and result["max_k_rel"] < 0.06
            and result["max_velocity_rel"] < 0.02
            and result["final_vs_static"]["relmax"] < 0.02
            and result[final_reference_key]["relmax"] < 0.02
            and result[physical_reference_key]["relmax"] < 0.02)
        print("STRICT_RESULT:", "PASS" if ok else "FAIL")
        print("EXPERIMENTAL_RESULT:", "PASS" if final_ok else "FAIL")
        return 0 if final_ok else 1
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
