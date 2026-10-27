"""P4: static-shape joint prefill (t=0) module for the fixed profile.

Dual-segment static specialization of the official prefill:
  seg0 = VLM stream    [0, 606)  hidden width 2048, plain RMSNorm (cond None)
  seg1 = action stream [606, 638) hidden width 1024, cond None at prefill
Per layer: per-segment norm -> per-segment QKV -> shared-head concat ->
mrope over all 638 -> SDPA (2D padding mask expanded to heads) ->
per-segment O -> plain residual -> per-segment norm+MLP -> residual.
Layer returns its post-rope K/V so the module emits the prefix KV cache the
suffix steps consume. Outputs (fp32 ABI): k/v [L,1,2,638,128], v0, x0.

Weights are official submodules by reference; bitwise fidelity is verified
by tests/iree_wallx/verify_static_prefill.py.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from wall_x.model.core.ops import m_rope  # fallback bound per layer


def repeat_kv(t: torch.Tensor, n: int) -> torch.Tensor:
    return t.repeat_interleave(n, dim=1)


class StaticPrefillLayer(torch.nn.Module):
    math_attention = True  # P3-R3: bf16-score math attention
    # (P3): the head-materialized bool mask forces a slow attention path in
    # the IREE lowering (~5x slower prefill); an additive [1,1,S,S] mask
    # broadcast over pure matmul/softmax keeps the same semantics

    def __init__(self, model, layer_idx: int, seg0: int = 606, seg1: int = 32):
        super().__init__()
        L = model.model.layers[layer_idx]
        self.seg0, self.seg1 = seg0, seg1
        self.n0 = L.input_layernorms[0]
        self.n1 = L.input_layernorms[1]
        self.pn0 = L.post_attention_layernorms[0]
        self.pn1 = L.post_attention_layernorms[1]
        self.qkv0 = L.self_attn.qkv_proj_experts[0]
        self.qkv1 = L.self_attn.qkv_proj_experts[1]
        self.o0 = L.self_attn.o_proj_experts[0]
        self.o1 = L.self_attn.o_proj_experts[1]
        self.mlp0 = L.moe.experts[0]
        self.mlp1 = L.moe.experts[1]
        self.n_heads = model.config.num_attention_heads
        self.n_kv = model.config.num_key_value_heads
        self.head_dim = model.config.hidden_size // self.n_heads
        self.d0 = model.config.dim_inputs[0]
        self.d1 = model.config.dim_inputs[1]
        self.gqa = self.n_heads // self.n_kv
        self.mrope_section = model.config.rope_scaling["mrope_section"]
        self._mrope = m_rope._pytorch_fallback

    def forward(self, h: torch.Tensor, cos_h: torch.Tensor, sin_h: torch.Tensor,
                mask4: torch.Tensor | None):
        """h [638, 2048]; cos_h/sin_h are the half-width float tables and
        mask4 the head-expanded bool mask, all precomputed ONCE per module
        call (P3: recomputing them per layer cost 4+ dispatches/layer of
        pure bandwidth). Returns (h', k, v) with k/v the post-rope cache
        tensors [1, 2, 638, 128]."""
        s0, s1 = self.seg0, self.seg1
        total = s0 + s1
        p0, _ = self.n0(h[:s0, : self.d0])
        p1, _ = self.n1(h[s0:, : self.d1])
        kv_dim = self.n_kv * self.head_dim
        qkv0 = self.qkv0(p0.to(self.qkv0.weight.dtype))
        qkv1 = self.qkv1(p1.to(self.qkv1.weight.dtype))
        q0, k0, v0_ = torch.split(
            qkv0, [self.n_heads * self.head_dim, kv_dim, kv_dim], dim=-1)
        q1, k1, v1_ = torch.split(
            qkv1, [self.n_heads * self.head_dim, kv_dim, kv_dim], dim=-1)
        # official path routes Q/K/V through the permute buffers and the
        # unpermute fallback, which upcasts to fp32; rope and the cache
        # therefore see fp32 (the fallback computes in fp32 anyway, so
        # casting back for SDPA is lossless and layer outputs stay bitwise)
        q = torch.cat([q0, q1], dim=0).to(torch.float32).view(
            1, total, self.n_heads, self.head_dim)
        k = torch.cat([k0, k1], dim=0).to(torch.float32).view(
            1, total, self.n_kv, self.head_dim)
        v = torch.cat([v0_, v1_], dim=0).to(torch.float32).view(
            1, total, self.n_kv, self.head_dim)
        q, k = self._mrope(q.contiguous(), k.contiguous(),
                           cos_h, sin_h, self.mrope_section)
        q = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        qf = q.to(self.qkv0.weight.dtype)
        kf = repeat_kv(k_t, self.gqa).to(self.qkv0.weight.dtype)
        vf = repeat_kv(v_t, self.gqa).to(self.qkv0.weight.dtype)
        if self.math_attention:
            # P3-R3: bf16-score math attention. The fp32 variant (round 1)
            # materialized 26MB fp32 scores = 1067ms; bf16 GEMM scores
            # (mma accumulates fp32, stores bf16) + fp32 softmax costs
            # ~0.3-0.5ms/layer vs the 8.24ms masked online_attention.
            # Extra rounding vs official: scores rounded to bf16 before
            # softmax (~0.4% rel on attention weights) - documented trade,
            # SDPA variant stays the correctness anchor.
            # P3-R4: fp16 scores. bf16 scores (8-bit mantissa) failed the
            # per-dim quality check (11.8pct); fp16 gives 10 bits at the
            # same mma speed (NV_MMA_SYNC_F32_16x8x16_F16 on sm_120).
            # bf16->fp16 is exact for activations (|q|,|k| << 65504) and
            # widens the mantissa; softmax runs fp32 on the widened scores.
            qf16 = qf.to(torch.float16)
            kf16 = kf.to(torch.float16)
            add_mask = torch.where(
                mask4[:, 0], 0.0, float("-inf")).to(
                    torch.float16).unsqueeze(1)
            sc = torch.matmul(qf16, kf16.transpose(-1, -2))                 * float(self.head_dim) ** -0.5
            sc = sc + add_mask
            p = torch.softmax(sc.to(torch.float32), dim=-1).to(vf.dtype)
            attn = torch.matmul(p, vf)
        elif mask4 is None:
            attn = F.scaled_dot_product_attention(qf, kf, vf, attn_mask=None,
                                                  is_causal=True)
        else:
            attn = F.scaled_dot_product_attention(qf, kf, vf, attn_mask=mask4,
                                                  is_causal=False)
        attn = attn.transpose(1, 2).reshape(total,
                                            self.n_heads * self.head_dim)
        o0_out = self.o0(attn[:s0].to(self.o0.weight.dtype))
        o1_out = self.o1(attn[s0:].to(self.o1.weight.dtype))
        out = torch.cat(
            [o0_out.to(h.dtype),
             torch.cat([o1_out.to(h.dtype),
                        torch.zeros(s1, self.d0 - self.d1, dtype=h.dtype,
                                    device=h.device)], dim=1)], dim=0)
        h = h + out
        residual = h
        q0n, _ = self.pn0(h[:s0, : self.d0])
        q1n, _ = self.pn1(h[s0:, : self.d1])
        m0 = self.mlp0(q0n.to(next(self.mlp0.parameters()).dtype))
        m1 = self.mlp1(q1n.to(next(self.mlp1.parameters()).dtype))
        mfull = torch.cat(
            [m0[:, : self.d0].to(h.dtype),
             torch.cat([m1[:, : self.d1].to(h.dtype),
                        torch.zeros(s1, self.d0 - self.d1, dtype=h.dtype,
                                    device=h.device)], dim=1)], dim=0)
        # cache tensors in this layer's projection dtype (post-rope K)
        return residual + mfull, k_t.contiguous(), v_t.contiguous()


class StaticJointPrefill(torch.nn.Module):
    def __init__(self, model, dof_mask: torch.Tensor | None = None,
                 n_layers: int | None = None, assume_causal: bool = True):
        super().__init__()
        self.assume_causal = assume_causal
        n = n_layers if n_layers else len(model.model.layers)
        self.layers = torch.nn.ModuleList(
            [StaticPrefillLayer(model, i) for i in range(n)])
        self.rotary = model.model.rotary_emb
        self.fn0 = model.model.norms[0]
        self.fn1 = model.model.norms[1]
        self.ap = model.action_preprocessor
        self.d0 = model.config.dim_inputs[0]
        self.d1 = model.config.dim_inputs[1]
        self.hidden = model.config.hidden_size
        self.action_hidden = model.action_preprocessor.action_hidden_size
        self.s0 = self.layers[0].seg0
        self.s1 = self.layers[0].seg1
        A = self.s1
        D = dof_mask.shape[-1] if dof_mask is not None else 26
        if dof_mask is None:
            dof_mask = torch.ones(1, A, D)
        self.register_buffer("dof_mask", dof_mask.to(torch.float32))

    def forward(self, inputs_embeds_f32: torch.Tensor,
                position_ids: torch.Tensor, mask3d: torch.Tensor,
                noise: torch.Tensor, padding_action: torch.Tensor,
                dt0: torch.Tensor, dof_mask: torch.Tensor | None = None):
        h3 = inputs_embeds_f32.to(torch.bfloat16)
        cos, sin = self.rotary(h3, position_ids)
        total = h3.shape[1]
        # P3: hoist per-layer recompute - half-width float tables and the
        # head-expanded mask are layer-invariant; values identical, so
        # numerics are unchanged (verify stays bitwise for the SDPA branch)
        cos_h = cos[..., : cos.size(3) // 2].contiguous().float()
        sin_h = sin[..., : sin.size(3) // 2].contiguous().float()
        if self.assume_causal:
            mask4 = mask3d.to(torch.bool).unsqueeze(0)
        else:
            mask4 = mask3d.to(torch.bool).unsqueeze(1).expand(
                1, self.layers[0].n_heads, total, total).contiguous()
        h = h3.view(total, -1)
        ks, vs = [], []
        for layer in self.layers:
            h, k_l, v_l = layer(h, cos_h, sin_h, mask4)
            ks.append(k_l.to(torch.float32))
            vs.append(v_l.to(torch.float32))
        k_out = torch.stack(ks)
        v_out = torch.stack(vs)
        # final expert norms per segment (gates discarded)
        f0, _ = self.fn0(h[: self.s0, : self.d0])
        f1, _ = self.fn1(h[self.s0:, : self.d1])
        hf = torch.cat(
            [f0.to(h.dtype),
             torch.cat([f1.to(h.dtype),
                        torch.zeros(self.s1, self.d0 - self.d1,
                                    dtype=h.dtype, device=h.device)],
                       dim=1)], dim=0)
        act = hf[self.s0:]
        pred = self.ap.action_proj_back(
            act[:, : self.action_hidden].to(torch.float32))
        v0 = pred.reshape(1, self.s1, -1)
        v_pad = padding_action - noise  # [1,A,D] via broadcast ([1,1,D])
        active_dof = self.dof_mask if dof_mask is None else dof_mask.float()
        v0 = v_pad * (1.0 - active_dof) + v0 * active_dof
        x0 = noise + dt0 * v0
        return k_out, v_out, v0, x0
