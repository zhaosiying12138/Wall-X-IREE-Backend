#!/usr/bin/env python3
"""P2 verification: static suffix step vs official path, per layer and end to
end, on bank sample inputs with explicit noise (bitwise-reproducible).

Both sides run with wall_x ops forced to the pytorch fallback so op
implementations match; the official side additionally runs the capture-compat
patches (honest in eager mode).
"""
import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "iree_wallx"))

from bench_model_api import (  # noqa: E402
    _build_fake_observation, _right_gripper_dim_from_config, build_adapter)
from capture_compat import patch_joint_attention  # noqa: E402
from wall_x.serving.backend.static_flow import StaticDenoiseStep  # noqa: E402

CKPT = os.environ.get(
    "WALLX_CKPT", "/home/zhaosiying/codebase/weights/wall-oss-0.5")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layers", type=int, default=2,
                    help="layers to compare individually (default first 2)")
    ap.add_argument("--full", action="store_true",
                    help="also run the full 36-layer step end to end")
    args = ap.parse_args()

    ns = argparse.Namespace(
        checkpoint=CKPT, train_config=None, norm_key="x2_normal",
        architecture="qwen2_5", cam_names=["face_view", "right_wrist_view"],
        action_horizon=None, image_size=448, seed=0)
    adapter = build_adapter(ns)
    model = adapter._model
    patch_joint_attention()
    from wall_x.model.core import ops as wall_ops
    for name in ("rmsnorm", "m_rope", "rot_pos_emb", "permute", "unpermute",
                 "get_rope_index", "get_window_index"):
        op = getattr(wall_ops, name, None)
        if op is not None:
            op._resolved_fn = op._pytorch_fallback
            op._backend = "pytorch(forced)"

    bank = torch.load(REPO_ROOT / "artifacts/bank/sample_000.pt",
                      weights_only=False)
    ctx_noise = bank["noise"].cuda()
    times = bank["ctx"]["times"].cuda()
    v0 = bank["intermediates"]["v0"].cuda()
    pp_pos = bank["postfix"]["position_ids"].cuda()
    pp_mask = bank["postfix"]["attention_mask"].cuda()
    pp_emb = bank["postfix"]["inputs_embeds"].cuda()
    pp_ids = bank["postfix"]["input_ids"].cuda()
    pp_types = bank["postfix"]["moe_token_types"].cuda()
    pp_si = bank["postfix"]["start_indices"].cuda()
    pp_ei = bank["postfix"]["end_indices"].cuda()

    # rebuild ctx deterministically (explicit noise makes it bitwise)
    gd = _right_gripper_dim_from_config(getattr(adapter, "_train_config", {}))
    from wall_x._vendor.harrix.envs.libero_common import encode_proprio
    obs = _build_fake_observation(0, 448, gd)
    ob = encode_proprio(obs, adapter._train_config, adapter._action_horizon)
    pre_t, post_t = adapter._get_flow_prompt(bank["meta"]["instruction"])
    bi = adapter._construct_model_input([ob], [pre_t], [post_t])
    ctx = model._prepare_flow_action_inputs(
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
        noise=ctx_noise)

    # prefill + trim KV to prefix (official semantics)
    pre_out = model.model(
        input_ids=None, attention_mask=ctx["attention_mask"],
        position_ids=ctx["position_ids"], past_key_values=None,
        inputs_embeds=ctx["inputs_embeds"],
        moe_token_types=ctx["moe_token_types"],
        start_indices=ctx["start_indices"], end_indices=ctx["end_indices"],
        positional_masks=ctx["positional_masks"], use_cache=True,
        output_attentions=False, output_hidden_states=False,
        return_dict=True, adarms_conds=[None, ctx["adarms_cond"]])
    kv = pre_out.past_key_values
    P = int(ctx["prefix_length"])
    for l in kv.layers:
        l.keys = l.keys[:, :, :P, :]
        l.values = l.values[:, :, :P, :]

    # x at suffix step 0 = noise + dt0 * v0 (official first Euler update)
    x0 = ctx["noisy_action"] + (times[1] - times[0]) * v0

    # ---- official suffix step 0 with per-layer hooks ----------------------
    layer_outs = {}
    hooks = []
    for i, layer in enumerate(model.model.layers):
        def mk(i):
            def hook(mod, inp, out):
                layer_outs[i] = out[0].detach()
            return hook
        hooks.append(layer.register_forward_hook(mk(i)))

    am = pp_ids == model.action_token_id_set["action_token_id"]
    t1 = times[1].unsqueeze(0).repeat(1)
    emb, cond = model.action_preprocessor.step(
        timestep=t1, noisy_action=x0, dof_mask=ctx["dof_mask"])
    emb = emb.reshape(-1, pp_emb.shape[-1])
    tmp = pp_emb.clone()
    tmp[am] = emb.to(tmp.dtype)
    o = model.model(
        input_ids=None, attention_mask=pp_mask, position_ids=pp_pos,
        past_key_values=kv, inputs_embeds=tmp,
        moe_token_types=pp_types, start_indices=pp_si, end_indices=pp_ei,
        use_cache=False, output_attentions=False, output_hidden_states=False,
        return_dict=True, adarms_conds=[None, cond])
    for h in hooks:
        h.remove()
    official_hidden = o.last_hidden_state  # [1,32,2048] unpermuted view

    # ---- static module, per layer -----------------------------------------
    k_past = torch.stack([l.keys for l in kv.layers])   # [L,1,2,606,128]
    v_past = torch.stack([l.values for l in kv.layers])
    n_cmp = args.n_layers
    sd = {"n_layers_cmp": n_cmp, "layer_maxabs": [], "corridor_zero": True}
    dof = ctx["dof_mask"]
    v_padding = None
    if ctx["padding_action"] is not None:
        v_padding = ctx["padding_action"] - ctx["noisy_action"]
    static_full = StaticDenoiseStep(model, len(model.model.layers),
                                    dof_mask=dof.detach().cpu())
    static_full.cuda().eval()
    # walk layer by layer feeding official inputs
    h_s = None
    emb_s, cond_s = model.action_preprocessor.step(
        timestep=t1, noisy_action=x0, dof_mask=ctx["dof_mask"])
    hs = emb_s.reshape(pp_emb.shape[1], -1).to(pp_emb.dtype)
    from wall_x.model.core.ops import m_rope as m_rope_op
    dummy = pp_emb
    cos, sin = static_full.rotary(dummy, pp_pos)
    for i in range(n_cmp):
        h_s = static_full.layers[i](hs, cond_s, k_past[i], v_past[i],
                                    cos, sin, pp_mask)
        off_h = layer_outs[i]
        d = (h_s.float() - off_h.view(h_s.shape).float()).abs().max().item()
        sd["layer_maxabs"].append(round(d, 6))
        corridor_ok = bool((off_h.view(h_s.shape)[:, 1024:].abs() < 1e-3)
                           .all().item())
        sd["corridor_zero"] = sd["corridor_zero"] and corridor_ok
        hs = h_s  # feed static output forward (chained static stack)

    if args.full:
        v_static = static_full(x0, t1, pp_pos, pp_mask,
                               k_past.to(torch.float32),
                               v_past.to(torch.float32),
                               v_padding=v_padding)
        # localize any end-to-end gap: final-norm output vs official hidden
        fin, _ = static_full.final_norm(hs[:, : static_full.dim_e1], cond_s)
        fin_full = torch.cat(
            [fin.to(hs.dtype),
             torch.zeros(hs.shape[0], hs.shape[1] - static_full.dim_e1,
                         dtype=hs.dtype, device=hs.device)], dim=1)
        sd["final_hidden_maxabs"] = round(
            (fin_full.float()
             - official_hidden.view(fin_full.shape).float())
            .abs().max().item(), 6)
        hh = official_hidden[0].to(torch.float32)
        pr = model.action_preprocessor.action_proj_back(
            hh[:, : model.action_preprocessor.action_hidden_size])
        v_official = pr.reshape(1, -1, adapter._action_dim)
        if v_padding is not None:
            v_official = v_padding * (1.0 - dof) + v_official * dof
        d = (v_static.float() - v_official.float()).abs().max().item()
        rel = d / v_official.float().abs().max().item()
        sd["v_maxabs"] = round(d, 6)
        sd["v_relmax"] = round(rel, 6)

    print({k: v for k, v in sd.items()})
    ok = all(d < 2e-2 for d in sd["layer_maxabs"]) and sd["corridor_zero"]
    if args.full:
        ok = ok and sd.get("v_relmax", 1.0) < 5e-3
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
