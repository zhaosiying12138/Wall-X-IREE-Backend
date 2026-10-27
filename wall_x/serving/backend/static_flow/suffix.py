"""P2: static-shape suffix velocity-step module (token-routing specialized).

The official suffix forward is a 2-expert MoE over mot_opt segments; for the
fixed profile the postfix contains ONLY expert-1 (action stream) tokens, so
the static specialization is a plain transformer step:

  embed = action_preprocessor.step(t, x)          # [B, A, 2048]
  per layer: AdaRMSNorm(x1024, cond) -> QKV(e1) -> mrope -> SDPA over
  [prefix KV 606 | current 32] -> O(e1) -> gated residual -> AdaRMSNorm ->
  MLP(e1) -> gated residual
  v = proj_back(h[:, :1024])

Weights are the official model's submodules (shared references). All shapes
static for B=1, A=32, prefix=606, kv=638. Numerical fidelity is checked
against the official path by verify_static_suffix_step.py.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from wall_x.model.core.ops import m_rope  # resolves to pytorch fallback in export env


def repeat_kv(t: torch.Tensor, n: int) -> torch.Tensor:
    return t.repeat_interleave(n, dim=1)


class StaticSuffixLayer(torch.nn.Module):
    """One decoder layer specialized to the action-only postfix segment."""

    def __init__(self, model, layer_idx: int):
        super().__init__()
        L = model.model.layers[layer_idx]
        self.norm1 = L.input_layernorms[1]
        self.norm2 = L.post_attention_layernorms[1]
        self.qkv = L.self_attn.qkv_proj_experts[1]
        self.o_proj = L.self_attn.o_proj_experts[1]
        self.mlp = L.moe.experts[1]
        self.n_heads = model.config.num_attention_heads
        self.n_kv = model.config.num_key_value_heads
        self.head_dim = model.config.hidden_size // self.n_heads
        self.dim_e1 = model.config.dim_inputs[1]
        self.gqa = self.n_heads // self.n_kv
        self.mrope_section = model.config.rope_scaling["mrope_section"]
        # bind the pure-torch fallback directly: the OpsProxy lazy resolver
        # (locks/imports) is not dynamo-traceable, and the fallback is the
        # numerically-verified reference for export
        self._mrope = m_rope._pytorch_fallback
        # per-layer KV dtype follows the selective-bf16 policy (some layers
        # project in fp32); KV enters as fp32 and is cast losslessly here
        self._kv_dtype = L.self_attn.qkv_proj_experts[1].weight.dtype

    def forward(self, h: torch.Tensor, cond: torch.Tensor,
                k_past_f32: torch.Tensor, v_past_f32: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                suffix_mask: torch.Tensor) -> torch.Tensor:
        # KV arrives as fp32 (uniform, numpy-safe ABI); cast to this layer's
        # projection dtype (no-op for fp32 layers, lossless bf16->fp32->bf16
        # for bf16 layers)
        k_past = k_past_f32.to(self._kv_dtype)
        v_past = v_past_f32.to(self._kv_dtype)
        # h: [A, 2048] mot order (== original order, single expert-1 segment)
        a_len = h.shape[0]

        # 1) input norm on the expert-1 slice, zero corridor preserved
        processed, gate = self.norm1(h[:, : self.dim_e1], cond)
        hn = torch.zeros_like(h)
        hn = torch.cat([processed.to(h.dtype),
                        torch.zeros(a_len, h.shape[1] - self.dim_e1,
                                    dtype=h.dtype, device=h.device)],
                       dim=1)

        # 2) QKV from the expert-1 projection (input slice 1024)
        qkv = self.qkv(hn[:, : self.dim_e1])
        kv_dim = self.n_kv * self.head_dim
        q, k, v = torch.split(
            qkv, [self.n_heads * self.head_dim, kv_dim, kv_dim], dim=-1)
        q = q.view(1, a_len, self.n_heads, self.head_dim)
        k = k.view(1, a_len, self.n_kv, self.head_dim)
        v = v.view(1, a_len, self.n_kv, self.head_dim)

        # 3) mrope (half-width cos/sin per official _apply_rotary_pos_embed)
        q, k = self._mrope(q.contiguous(), k.contiguous(),
                      cos[..., : cos.size(3) // 2].contiguous().float(),
                      sin[..., : sin.size(3) // 2].contiguous().float(),
                      self.mrope_section)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 4) attend over [prefix | current] KV with the bool suffix mask
        k_full = torch.cat([k_past, k], dim=2)
        v_full = torch.cat([v_past, v], dim=2)
        # official casts q/k/v to the projection dtype before SDPA
        target_dtype = self.qkv.weight.dtype
        q = q.to(target_dtype)
        k_full = repeat_kv(k_full, self.gqa).to(target_dtype)
        v_full = repeat_kv(v_full, self.gqa).to(target_dtype)
        # expand the shared mask to the head dim explicitly: torch SDPA
        # broadcasts [1,1,Q,K], but the IREE attention lowering requires
        # the batch dim to match; same values, no semantic change
        mask4 = suffix_mask.unsqueeze(1).expand(
            q.shape[0], self.n_heads, suffix_mask.shape[1],
            suffix_mask.shape[2]).contiguous()
        attn = F.scaled_dot_product_attention(
            q, k_full, v_full, attn_mask=mask4, is_causal=False)
        attn = attn.transpose(1, 2).reshape(a_len, self.n_heads * self.head_dim)

        # 5) expert-1 output projection into the corridor layout
        out = self.o_proj(attn.to(self.o_proj.weight.dtype))
        out_full = torch.zeros_like(h)
        out_full = torch.cat(
            [out, torch.zeros(a_len, h.shape[1] - self.dim_e1,
                              dtype=h.dtype, device=h.device)], dim=1)

        # 6) gated residual (official _gated_residual for the e1 segment)
        h = gated_residual_static(h, out_full, gate, self.dim_e1)
        residual = h

        # 7) post-attention norm + expert-1 MLP + gated residual
        processed2, gate2 = self.norm2(h[:, : self.dim_e1], cond)
        mlp_dtype = next(self.mlp.parameters()).dtype
        mlp_out = self.mlp(processed2.to(mlp_dtype))
        mlp_full = torch.cat(
            [mlp_out[:, : self.dim_e1].to(h.dtype),
             torch.zeros(a_len, h.shape[1] - self.dim_e1,
                         dtype=h.dtype, device=h.device)], dim=1)
        h = gated_residual_static(residual, mlp_full, gate2, self.dim_e1)
        return h


def gated_residual_static(x, y, gate, dim_e1: int):
    """Official _gated_residual specialized to the single expert-1 segment:
    y[seg, :dim_e1] = (y[seg, :dim_e1].float() * gate).view(-1, dim_e1).to(y)
    """
    if gate is None:
        return x + y
    new_y = y.clone()
    sel = new_y[:, :dim_e1].to(torch.float32)
    # official views the segment as [B, A, D] before multiplying by gate
    sel = sel.view(1, -1, dim_e1) * gate
    new_y[:, :dim_e1] = sel.view(-1, dim_e1).to(y.dtype)
    return x + new_y


class StaticDenoiseStep(torch.nn.Module):
    """Full suffix velocity step: embed -> N layers -> proj_back -> v."""

    def __init__(self, model, n_layers: int, action_dim: int = 26,
                 dof_mask: torch.Tensor | None = None):
        super().__init__()
        self.ap = model.action_preprocessor
        self.layers = torch.nn.ModuleList(
            [StaticSuffixLayer(model, i) for i in range(n_layers)])
        self.rotary = model.model.rotary_emb
        self.final_norm = model.model.norms[1]
        self.dim_e1 = model.config.dim_inputs[1]
        self.action_hidden = model.action_preprocessor.action_hidden_size
        self.hidden = model.config.hidden_size
        # proj_with_mask concatenates the dof mask onto the action vector;
        # dof layout is a profile constant (robot config), stored as a buffer
        # with the TRUE values (not ones: wall-oss has inactive dofs)
        A = getattr(model.config, "action_horizon_flow", 32)
        if dof_mask is None:
            dof_mask = torch.ones(1, A, action_dim)
        self.register_buffer("dof_mask", dof_mask.to(torch.float32))

    def forward(self, noisy_action: torch.Tensor, timestep: torch.Tensor,
                postfix_pos: torch.Tensor, suffix_mask: torch.Tensor,
                k_past_f32: torch.Tensor, v_past_f32: torch.Tensor,
                v_padding: torch.Tensor | None = None,
                dof_mask: torch.Tensor | None = None) -> torch.Tensor:
        # noisy_action [1, A, D] f32; timestep [1] f32; k/v as fp32
        # [n_layers, 1, 2, P, 128] (uniform ABI; layers cast internally)
        # ABI v2 (P5): KV arrives as the FULL 638-row bucket buffer; the
        # 606-row prefix crop is a compile-time static slice at module
        # level so prefill's device buffers feed this module directly
        k_past = k_past_f32[:, :, :, :606, :]
        v_past = v_past_f32[:, :, :, :606, :]
        active_dof = self.dof_mask if dof_mask is None else dof_mask.float()
        emb, cond = self.ap.step(timestep=timestep, noisy_action=noisy_action,
                                 dof_mask=active_dof)
        h = emb.reshape(1, -1, self.hidden).to(torch.bfloat16)
        # rotary tables on postfix positions (t section of mrope)
        cos, sin = self.rotary(h, postfix_pos)
        a_len = h.shape[1]
        h = h.view(a_len, -1)
        for i, layer in enumerate(self.layers):
            h = layer(h, cond, k_past[i], v_past[i], cos, sin, suffix_mask)
        # model.forward tail: expert-wise final norm on the e1 segment
        # (gate discarded); unpermute is identity for the single segment
        fin, _ = self.final_norm(h[:, : self.dim_e1], cond)
        h = torch.cat([fin.to(h.dtype),
                       torch.zeros(a_len, h.shape[1] - self.dim_e1,
                                   dtype=h.dtype, device=h.device)], dim=1)
        pred = self.ap.action_proj_back(h[:, : self.action_hidden]
                                        .to(torch.float32))
        v = pred.reshape(1, a_len, -1)
        # official step tail: v = v_padding*(1-dof) + v*dof, where the
        # padding velocity (padding_action - initial noise, computed once at
        # the first step and closed over) is passed in by the caller
        if v_padding is not None:
            v = v_padding * (1.0 - active_dof) + v * active_dof
        return v
