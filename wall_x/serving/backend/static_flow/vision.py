"""Static, exportable WALL-OSS vision + text/state context.

Graph boundary
--------------
The input is the serving processor's *tensor* output, not a JPEG or text:
``input_ids`` and flattened/normalized ``pixel_values``; tensor-state model
variants additionally take normalized proprioception and its mask.  The module
executes the complete official ViT (patch embedding, windowed/full attention,
RoPE, merger), token/state embedding lookup, and ordered image/state scatter.
It emits the ``inputs_embeds`` representation consumed by joint prefill.

The host remains responsible for JPEG decode, resize/normalization, tokenization
and state-string quantization, ``grid_thw``/window profile construction, 3-D
mRoPE positions, joint masks, noise, and action normalization.  ``StaticFullFlow``
adds the t=0 action embedding in-graph; standalone use of this context component
does not.  Neither component is an end-to-end serving graph by itself.

Static ViT profile
------------------
The real WALL-OSS-0.5 serving profile is two [1, 32, 32] grids: 2048 patches,
32 windows of 64 tokens, two 1024-token full-attention segments, and 512 merged
features.  Window/full metadata and RoPE tables are computed once from the
official operators at module construction.  Dynamic grids and uneven window
sizes are rejected instead of being silently padded or traced with host
``item()`` calls.  Export another static profile for another shape.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from wall_x.model.core.ops import get_window_index, rot_pos_emb
from wall_x.model.qact.qwen2_5.modeling_qwen2_5_vl import (
    apply_rotary_pos_emb_vision,
)


class StaticProfileError(ValueError):
    """A request cannot be represented by the compiled static profile."""


@dataclass(frozen=True)
class VisionProfile:
    grid_thw: tuple[tuple[int, int, int], ...]
    patch_tokens: int
    merged_tokens: int
    window_tokens: int
    num_windows: int
    full_segment_tokens: tuple[int, int, ...]

    @property
    def patch_features(self) -> int:
        # The WALL-OSS processor flattens C*T*H*W per patch.
        return 3 * 2 * 14 * 14


def build_vision_profile(visual, grid_thw: torch.Tensor) -> tuple[
        VisionProfile, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor]:
    """Build official static ViT metadata once on the visual device.

    Returns the profile plus window/reverse indices and already-reordered
    cos/sin tables.  No metadata is derived from tensor values inside forward.
    Computing trigonometric tables on the official visual device avoids a
    one-ULP CPU/GPU difference that can flip BF16 rotary rounding.
    """
    grid = tuple(tuple(int(v) for v in row) for row in grid_thw.cpu().tolist())
    if not grid or any(t < 1 or h < 2 or w < 2 for t, h, w in grid):
        raise StaticProfileError(f"invalid image_grid_thw: {grid}")
    merge = int(visual.spatial_merge_size)
    if any(h % merge or w % merge for _, h, w in grid):
        raise StaticProfileError(
            f"grid H/W must be divisible by spatial_merge_size={merge}: {grid}")
    patch_tokens = sum(t * h * w for t, h, w in grid)
    merged_tokens = patch_tokens // (merge * merge)
    full_lengths = tuple(t * h * w for t, h, w in grid)

    grid_cpu = torch.tensor(grid, dtype=torch.int64)
    device = visual.rotary_pos_emb.inv_freq.device
    grid_device = grid_cpu.to(device)
    # Bind the official pure-PyTorch fallbacks explicitly, but evaluate the
    # trigonometric table on the same device as the official visual forward.
    inv_freq = visual.rotary_pos_emb.inv_freq.detach().float()
    rotary = rot_pos_emb._pytorch_fallback(inv_freq, grid_device, merge)
    window_index, cu_window = get_window_index._pytorch_fallback(
        grid_thw=grid_device.to(torch.int32),
        window_size=int(visual.window_size),
        spatial_merge_size=merge,
        patch_size=int(visual.patch_size),
        spatial_merge_unit=int(visual.spatial_merge_unit),
    )
    window_index = window_index.long()
    cu_window = [int(v) for v in cu_window.cpu().tolist()]
    widths = [b - a for a, b in zip(cu_window, cu_window[1:])]
    if not widths or len(set(widths)) != 1:
        raise StaticProfileError(
            "IREE static ViT requires equal-sized windows; got "
            f"cu_window_seqlens={cu_window} for grid={grid}. "
            "Export a profile with explicit per-window buckets.")
    if int(window_index.numel()) != merged_tokens:
        raise StaticProfileError(
            f"window index has {window_index.numel()} rows, expected "
            f"{merged_tokens} for grid={grid}")
    if cu_window[-1] != patch_tokens:
        raise StaticProfileError(
            f"window coverage {cu_window[-1]} != patch tokens {patch_tokens}")

    unit = int(visual.spatial_merge_unit)
    rotary = rotary.reshape(merged_tokens, unit, -1)[window_index].reshape(
        patch_tokens, -1)
    emb = torch.cat((rotary, rotary), dim=-1)
    cos, sin = emb.cos().float().contiguous(), emb.sin().float().contiguous()
    reverse_index = torch.argsort(window_index).contiguous()
    profile = VisionProfile(
        grid_thw=grid,
        patch_tokens=patch_tokens,
        merged_tokens=merged_tokens,
        window_tokens=widths[0],
        num_windows=len(widths),
        full_segment_tokens=full_lengths,
    )
    return (profile, window_index.contiguous(), reverse_index, cos, sin,
            grid_cpu)


def _ordered_scatter(base: torch.Tensor, mask: torch.Tensor,
                     values: torch.Tensor) -> torch.Tensor:
    """Static-shape equivalent of ordered boolean masked_scatter (B=1).

    ``values`` is [N,D], ``mask`` [1,S].  cumsum computes each destination's
    source row without nonzero/dynamic output shapes.  The caller validates
    that the mask selects exactly N rows.
    """
    if base.shape[0] != 1:
        raise StaticProfileError("IREE context currently requires batch size 1")
    n = values.shape[0]
    if n < 1:
        raise StaticProfileError("ordered scatter requires at least one feature")
    rank = mask.to(torch.int64).cumsum(dim=1) - 1
    rank = rank.clamp(min=0, max=n - 1).reshape(-1)
    gathered = values.reshape(n, -1)[rank].reshape(base.shape)
    return torch.where(mask.unsqueeze(-1), gathered.to(base.dtype), base)


class StaticVisionBlock(torch.nn.Module):
    """Official ViT block with profile-static window/full attention."""

    def __init__(self, block, layer_idx: int, profile: VisionProfile,
                 attention_mode: str = "eager-bf16"):
        super().__init__()
        if attention_mode not in ("sdpa", "eager-bf16", "math-fp16", "math-fp32"):
            raise ValueError(f"unsupported static vision attention: {attention_mode}")
        self.attention_mode = attention_mode
        self.norm1 = block.norm1
        self.norm2 = block.norm2
        self.attn = block.attn
        self.mlp = block.mlp
        self.layer_idx = int(layer_idx)
        self.profile = profile
        self.num_heads = int(self.attn.num_heads)
        self.head_dim = self.norm1.hidden_size // self.num_heads

    def _attend(self, q, k, v):
        if self.attention_mode == "sdpa":
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        if self.attention_mode == "eager-bf16":
            # Exact operator order of Qwen2_5_VLVisionAttention within each
            # already-masked static segment (the masked-off logits are absent).
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            probabilities = torch.softmax(
                scores, dim=-1, dtype=torch.float32).to(q.dtype)
            return torch.matmul(probabilities, v)
        if self.attention_mode == "math-fp32":
            scores = torch.matmul(q.float(), k.float().transpose(-1, -2))
            scores = scores * float(self.head_dim) ** -0.5
            probabilities = torch.softmax(scores, dim=-1).to(v.dtype)
            return torch.matmul(probabilities, v)
        # fp16 mantissa/score variant used to approximate FA2 without the
        # masked-online-attention lowering; softmax remains fp32.
        scores = torch.matmul(q.to(torch.float16), k.to(torch.float16).transpose(
            -1, -2)) * float(self.head_dim) ** -0.5
        probabilities = torch.softmax(
            scores.to(torch.float32), dim=-1).to(v.dtype)
        return torch.matmul(probabilities, v)

    def _attention(self, q, k, v, lengths: tuple[int, ...]):
        """Attention over independent, statically shaped segments."""
        if len(set(lengths)) == 1:
            # All WALL-OSS windows (and both image-sized full segments) have
            # equal length, so batch them into one attention call rather than
            # emitting 32 separate attention dispatches per windowed layer.
            length = lengths[0]
            segments = len(lengths)
            qs = q.reshape(segments, length, self.num_heads,
                           self.head_dim).transpose(1, 2)
            ks = k.reshape(segments, length, self.num_heads,
                           self.head_dim).transpose(1, 2)
            vs = v.reshape(segments, length, self.num_heads,
                           self.head_dim).transpose(1, 2)
            out = self._attend(qs, ks, vs)
            return out.transpose(1, 2).reshape(
                segments * length, self.num_heads * self.head_dim)

        outputs = []
        offset = 0
        for length in lengths:
            # Unequal full-attention segments remain statically unrolled.
            qs = q[offset:offset + length].reshape(
                1, length, self.num_heads, self.head_dim).transpose(1, 2)
            ks = k[offset:offset + length].reshape(
                1, length, self.num_heads, self.head_dim).transpose(1, 2)
            vs = v[offset:offset + length].reshape(
                1, length, self.num_heads, self.head_dim).transpose(1, 2)
            out = self._attend(qs, ks, vs)
            outputs.append(out.transpose(1, 2).reshape(
                length, self.num_heads * self.head_dim))
            offset += length
        return torch.cat(outputs, dim=0)

    def forward_stages(self, hidden_states, cos, sin, full_attention: bool):
        """Diagnostic block-0 computation with every intermediate exposed."""
        seq = hidden_states.shape[0]
        normed, _ = self.norm1(hidden_states)
        q, k, v = self.attn.qkv(normed).reshape(
            seq, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3).unbind(0)
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        lengths = (self.profile.full_segment_tokens if full_attention else
                   ((self.profile.window_tokens,) * self.profile.num_windows))
        attn_flat = self._attention(q, k, v, lengths)
        attention = attn_flat.reshape(seq, self.num_heads, self.head_dim)
        projected = self.attn.proj(attn_flat)
        residual = hidden_states + projected
        normed2, _ = self.norm2(residual)
        mlp_out = self.mlp(normed2)
        output = residual + mlp_out
        return output, (normed, q, k, v, attention, projected, residual,
                        normed2, mlp_out, output)

    def forward(self, hidden_states, cos, sin, full_attention: bool):
        seq = hidden_states.shape[0]
        normed, _ = self.norm1(hidden_states)
        q, k, v = self.attn.qkv(normed).reshape(
            seq, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3).unbind(0)
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        if full_attention:
            lengths = self.profile.full_segment_tokens
        else:
            lengths = ((self.profile.window_tokens,)
                       * self.profile.num_windows)
        attn = self._attention(q, k, v, lengths)
        hidden_states = hidden_states + self.attn.proj(attn)
        normed, _ = self.norm2(hidden_states)
        return hidden_states + self.mlp(normed)


class StaticVisionTower(torch.nn.Module):
    """Complete static ViT using the official submodules by reference."""

    def __init__(self, visual, grid_thw: torch.Tensor,
                 fullatt_block_indexes=None, attention_mode: str = "eager-bf16"):
        super().__init__()
        (profile, window_index, reverse_index, cos, sin, _grid) = \
            build_vision_profile(visual, grid_thw)
        self.profile = profile
        self.patch_embed = visual.patch_embed
        self.merger = visual.merger
        full = set(fullatt_block_indexes if fullatt_block_indexes is not None
                   else visual.fullatt_block_indexes)
        self.blocks = torch.nn.ModuleList([
            StaticVisionBlock(block, i, profile, attention_mode=attention_mode)
            for i, block in enumerate(visual.blocks)
        ])
        self.full_attention = tuple(i in full for i in range(len(self.blocks)))
        self.register_buffer("window_index", window_index)
        self.register_buffer("reverse_index", reverse_index)
        self.register_buffer("vision_cos", cos)
        self.register_buffer("vision_sin", sin)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        p = self.profile
        hidden = self.patch_embed(pixel_values)
        if tuple(hidden.shape) != (p.patch_tokens,
                                   self.patch_embed.embed_dim):
            raise StaticProfileError(
                f"pixel_values produces {tuple(hidden.shape)} patch rows; "
                f"compiled profile requires {(p.patch_tokens, self.patch_embed.embed_dim)}")
        unit = self.merger.hidden_size // self.patch_embed.embed_dim
        hidden = hidden.reshape(p.merged_tokens, unit, -1)[
            self.window_index].reshape(p.patch_tokens, -1)
        for block, full in zip(self.blocks, self.full_attention):
            hidden = block(hidden, self.vision_cos, self.vision_sin, full)
        hidden = self.merger(hidden)
        return hidden[self.reverse_index]

    def forward_trace(self, pixel_values: torch.Tensor):
        """Diagnostic-only ViT path returning every block and final features."""
        p = self.profile
        hidden = self.patch_embed(pixel_values)
        unit = self.merger.hidden_size // self.patch_embed.embed_dim
        hidden = hidden.reshape(p.merged_tokens, unit, -1)[
            self.window_index].reshape(p.patch_tokens, -1)
        block_outputs = []
        for block, full in zip(self.blocks, self.full_attention):
            hidden = block(hidden, self.vision_cos, self.vision_sin, full)
            block_outputs.append(hidden)
        hidden = self.merger(hidden)
        return hidden[self.reverse_index], torch.stack(block_outputs)

    def forward_first_block_trace(self, pixel_values: torch.Tensor):
        """Return patch, every block-0 stage, and external metadata."""
        p = self.profile
        patch = self.patch_embed(pixel_values)
        unit = self.merger.hidden_size // self.patch_embed.embed_dim
        hidden = patch.reshape(p.merged_tokens, unit, -1)[
            self.window_index].reshape(p.patch_tokens, -1)
        _output, stages = self.blocks[0].forward_stages(
            hidden, self.vision_cos, self.vision_sin, self.full_attention[0])
        return ((patch.float(),) + tuple(value.float() for value in stages)
                + (self.vision_cos, self.vision_sin,
                   self.window_index, self.reverse_index))


class StaticVisionTextContext(torch.nn.Module):
    """ViT + token embeddings for WALL-OSS state-string configurations.

    In the real WALL-OSS-0.5 checkpoint ``use_state_string_representation`` is
    true: normalized robot state has already been quantized/tokenized by the
    host processor.  Those state tokens are compiled here through exactly the
    same embedding lookup as instruction tokens; no nonexistent propri_proj is
    invented.
    """

    def __init__(self, model, grid_thw: torch.Tensor,
                 attention_mode: str = "eager-bf16"):
        super().__init__()
        self.vision = StaticVisionTower(
            model.visual, grid_thw, attention_mode=attention_mode)
        self.embed_tokens = model.model.embed_tokens
        self.image_token_id = int(model.config.image_token_id)

    def _image_context(self, input_ids: torch.Tensor,
                       pixel_values: torch.Tensor) -> torch.Tensor:
        base = self.embed_tokens(input_ids)
        image_features = self.vision(pixel_values)
        image_mask = input_ids == self.image_token_id
        return _ordered_scatter(base, image_mask, image_features)

    def forward_features(self, input_ids: torch.Tensor,
                         pixel_values: torch.Tensor) -> torch.Tensor:
        return self._image_context(input_ids, pixel_values)

    def forward_vision_trace(self, input_ids: torch.Tensor,
                             pixel_values: torch.Tensor):
        """Diagnostic context + final ViT + per-block outputs (f32)."""
        base = self.embed_tokens(input_ids)
        image_features, block_outputs = self.vision.forward_trace(pixel_values)
        image_mask = input_ids == self.image_token_id
        base = _ordered_scatter(base, image_mask, image_features)
        return (base.float(), image_features.float(), block_outputs.float(),
                self.vision.vision_cos, self.vision.vision_sin,
                self.vision.window_index, self.vision.reverse_index)

    def forward(self, input_ids: torch.Tensor,
                pixel_values: torch.Tensor) -> torch.Tensor:
        return self.forward_features(input_ids, pixel_values).float()


class StaticVisionContextTrace(torch.nn.Module):
    """Standalone diagnostic wrapper with unified-IRPA-compatible names."""

    def __init__(self, model, grid_thw: torch.Tensor,
                 attention_mode: str = "eager-bf16"):
        super().__init__()
        self.context = StaticVisionTextContext(
            model, grid_thw, attention_mode=attention_mode)

    def forward(self, input_ids, pixel_values):
        return self.context.forward_vision_trace(input_ids, pixel_values)


class StaticVisionFirstBlockTrace(torch.nn.Module):
    """Standalone block-0 stage probe with unified-IRPA-compatible names."""

    def __init__(self, model, grid_thw: torch.Tensor,
                 attention_mode: str = "eager-bf16"):
        super().__init__()
        self.context = StaticVisionTextContext(
            model, grid_thw, attention_mode=attention_mode)

    def forward(self, pixel_values):
        return self.context.vision.forward_first_block_trace(pixel_values)


class StaticVisionTextStateContext(StaticVisionTextContext):
    """Variant for models with tensor (not tokenized) proprioception state."""

    def __init__(self, model, grid_thw: torch.Tensor,
                 attention_mode: str = "eager-bf16"):
        if getattr(model.config, "use_state_string_representation", False):
            raise StaticProfileError(
                "use_state_string_representation=True needs "
                "StaticVisionTextContext, not the tensor-state variant")
        if not getattr(model.config, "proj_with_mask", False):
            raise StaticProfileError(
                "This static ABI expects proj_with_mask=True and separate "
                "proprioception/agent_pos_mask tensors")
        super().__init__(model, grid_thw, attention_mode=attention_mode)
        self.propri_proj = model.action_preprocessor.propri_proj
        self.propri_token_id = int(model.action_token_id_set["propri_token_id"])
        self.state_hidden_size = int(
            model.action_preprocessor.state_hidden_size)
        self.hidden_size = int(model.config.hidden_size)

    def forward_features(self, input_ids: torch.Tensor,
                         pixel_values: torch.Tensor,
                         proprioception: torch.Tensor,
                         agent_pos_mask: torch.Tensor) -> torch.Tensor:
        base = self._image_context(input_ids, pixel_values)

        # Exact ActionProcessor.proprioception_proj tensor math for this
        # profile (proj_with_mask=True, state_hidden_size==hidden_size).
        propri = torch.cat([proprioception, agent_pos_mask], dim=-1)
        propri = propri.to(self.propri_proj.weight.dtype)
        state = self.propri_proj(propri)
        if self.state_hidden_size < self.hidden_size:
            pad = torch.zeros(
                (*state.shape[:-1], self.hidden_size - self.state_hidden_size),
                dtype=state.dtype, device=state.device)
            state = torch.cat([state, pad], dim=-1)
        state = state.reshape(-1, self.hidden_size)
        state_mask = input_ids == self.propri_token_id
        base = _ordered_scatter(base, state_mask, state)
        return base

    def forward(self, input_ids: torch.Tensor, pixel_values: torch.Tensor,
                proprioception: torch.Tensor,
                agent_pos_mask: torch.Tensor) -> torch.Tensor:
        return self.forward_features(
            input_ids, pixel_values, proprioception, agent_pos_mask).float()
