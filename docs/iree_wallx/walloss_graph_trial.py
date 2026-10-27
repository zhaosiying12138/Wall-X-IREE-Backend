#!/usr/bin/env python3
"""CPU/meta shape-only WALL-OSS-0.5 graph visualization trial.

This is a structural surrogate, not a numerical model and not an ONNX export
of WALL-OSS. Shapes and operator boundaries mirror the audited static flow.
No checkpoint is loaded and no CUDA/GPU operation is performed.
"""
from __future__ import annotations

import argparse
import os
from functools import lru_cache
from pathlib import Path
import re

# Must be set before torch import. The script never selects a CUDA device.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch import nn
from torchviz import make_dot
from torchview import draw_graph

OUT = Path(os.environ.get(
    'WALLX_VIS_OUT', Path(__file__).resolve().parent / 'implementation_evidence'))
OUT.mkdir(parents=True, exist_ok=True)

Shape = tuple[int, ...]


def shape_text(shape: Shape) -> str:
    return "[" + ",".join(str(x) for x in shape) + "]"


def shapes_text(shapes: tuple[Shape, ...]) -> str:
    return " ; ".join(shape_text(s) for s in shapes)


@lru_cache(maxsize=None)
def view_module_type(name: str) -> type[nn.Module]:
    """Give Torchview an informative module name without manual DOT labels."""
    return type(name, (nn.Module,), {})


@lru_cache(maxsize=None)
def viz_function_type(label: str) -> type[torch.autograd.Function]:
    """One named autograd boundary per structural operator for Torchviz."""
    return type(label, (torch.autograd.Function,), {
        "forward": staticmethod(_viz_forward),
        "backward": staticmethod(_viz_backward),
    })


def _viz_forward(ctx, output_shapes, *inputs):
    ctx.output_shapes = output_shapes
    outputs = tuple(torch.empty(s, device="meta") for s in output_shapes)
    return outputs[0] if len(outputs) == 1 else outputs


def _viz_backward(ctx, *grad_outputs):
    return (None, *(None for _ in ctx.output_shapes))


class ShapeEmitter:
    """Execute only shape construction through Torchview or Torchviz."""

    def __init__(self, mode: str):
        self.mode = mode

    def __call__(self, name: str, inputs, output_shapes):
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        if not output_shapes:
            raise ValueError(name)
        output_shapes = tuple(tuple(s) for s in output_shapes)
        if self.mode == "view":
            output_shapes_holder = output_shapes

            def forward(_self, *args):
                del args
                # CPU outputs preserve RecorderTensor tracing without evaluating operators.
                source = inputs[0]
                outputs = tuple(
                    source.new_empty(s) for s in output_shapes_holder
                )
                return outputs[0] if len(outputs) == 1 else outputs

            module_type = view_module_type(name)
            module = module_type()
            # Bind a fresh forward closure; class identity still controls labels.
            module.forward = forward.__get__(module, module_type)
            return module(*inputs)
        if self.mode == "viz":
            label = name + "\n" + shapes_text(output_shapes)
            return viz_function_type(label).apply(output_shapes, *inputs)
        raise ValueError(self.mode)


class FullFlowShape(nn.Module):
    """StaticFullFlow structure, compact or with nine suffix/Euler calls."""

    def __init__(self, mode: str, unroll: bool):
        super().__init__()
        self.emit = ShapeEmitter(mode)
        self.unroll = unroll

    def forward(self, pixel_values, context_ids, noise, dof_mask,
                padding_action, times):
        e = self.emit
        context = e("VisionTextContext_ViT32", (pixel_values, context_ids),
                    ((1, 638, 2048),))
        action_embed = e("ActionEmbed_52_1024_2048", (noise, dof_mask, times),
                         ((1, 32, 2048),))
        embeds = e("OrderedScatter_606VL_32Action", (context, action_embed),
                   ((1, 638, 2048),))
        k, v, v0_theta = e(
            "JointPrefill_MoT36", embeds,
            ((36, 1, 2, 638, 128), (36, 1, 2, 638, 128), (1, 32, 26)))
        v_pad = e("PaddingVelocity", (padding_action, noise), ((1, 32, 26),))
        v0 = e("DOFMix_v0", (v0_theta, v_pad, dof_mask), ((1, 32, 26),))
        x = e("Euler0", (noise, v0, times), ((1, 32, 26),))
        if self.unroll:
            for i in range(1, 10):
                velocity = e(f"Suffix{i}_ActionMoT36_DOFMix", (x, k, v, times),
                             ((1, 32, 26),))
                x = e(f"Euler{i}", (x, velocity, times), ((1, 32, 26),))
        else:
            x = e("NineSuffixEuler_ActionMoT36_9x", (x, k, v, times, v_pad, dof_mask),
                  ((1, 32, 26),))
        return x


class MoTLayerShape(nn.Module):
    """One prefill MoT layer with independent VL/Action expert parameters."""

    def __init__(self, mode: str):
        super().__init__()
        self.emit = ShapeEmitter(mode)

    def forward(self, vl, action):
        e = self.emit
        vn = e("VL_RMSNorm", vl, ((606, 2048),))
        an = e("Action_RMSNorm", action[:, :1024], ((32, 1024),))
        vqkv = e("VL_QKV_weight_2560x2048", vn, ((606, 2560),))
        aqkv = e("Action_QKV_weight_2560x1024", an, ((32, 2560),))
        va, aa, k, v = e(
            "JointSelfAttention_Q16_KV2_mRoPE", (vqkv, aqkv),
            ((606, 2048), (32, 2048), (1, 2, 638, 128), (1, 2, 638, 128)))
        vo = e("VL_O_weight_2048x2048", va, ((606, 2048),))
        ao = e("Action_O_weight_1024x2048", aa, ((32, 1024),))
        ao_full = e("Action_ZeroCorridor_O_1024_2048", ao, ((32, 2048),))
        vr1 = e("VL_Residual1", (vl, vo), ((606, 2048),))
        ar1 = e("Action_Residual1", (action, ao_full), ((32, 2048),))
        vpn = e("VL_PostAttentionNorm", vr1, ((606, 2048),))
        apn = e("Action_PostAttentionNorm", ar1[:, :1024], ((32, 1024),))
        vm = e("VL_MLP_2048_11008_2048", vpn, ((606, 2048),))
        am = e("Action_MLP_1024_2048_1024", apn, ((32, 1024),))
        am_full = e("Action_ZeroCorridor_MLP_1024_2048", am, ((32, 2048),))
        vl_out = e("VL_Residual2", (vr1, vm), ((606, 2048),))
        action_out = e("Action_Residual2", (ar1, am_full), ((32, 2048),))
        return vl_out, action_out, k, v


class ActionDOFShape(nn.Module):
    """Action embedding, zero corridor, velocity DOF mixing, and Euler."""

    def __init__(self, mode: str):
        super().__init__()
        self.emit = ShapeEmitter(mode)

    def forward(self, state, initial_noise, dof_mask, timestep, dt,
                padding_action):
        e = self.emit
        concat = e("Concat_noisy_action_DOF", (state, dof_mask), ((1, 32, 52),))
        w1 = e("Action_w1_weight_1024x52", concat, ((1, 32, 1024),))
        temb = e("TimeEmbed_1024_repeat32", timestep, ((1, 32, 1024),))
        cat_time = e("Concat_action_time", (w1, temb), ((1, 32, 2048),))
        w2 = e("Action_w2_weight_1024x2048", cat_time, ((1, 32, 1024),))
        w3 = e("SiLU_Action_w3_weight_1024x1024", w2, ((1, 32, 1024),))
        corridor = e("ZeroCorridor_1024valid_1024zero", w3, ((1, 32, 2048),))
        action_h = e("ActionExpert_MoT36", corridor, ((1, 32, 1024),))
        v_theta = e("ActionProjBack_weight_26x1024", action_h, ((1, 32, 26),))
        v_pad = e("PaddingVelocity_padding_minus_initial_noise",
                  (padding_action, initial_noise), ((1, 32, 26),))
        velocity = e("DOFMix_vpad1minusmask_vthetamask",
                     (v_theta, v_pad, dof_mask), ((1, 32, 26),))
        return e("Euler_state_plus_dt_velocity", (state, velocity, dt),
                 ((1, 32, 26),))


class BackendArchitectureShape(nn.Module):
    """Serving backend selection; unit tensors are opaque request handles."""

    def __init__(self, mode: str):
        super().__init__()
        self.emit = ShapeEmitter(mode)

    def forward(self, request, shared_contract):
        e = self.emit
        processed = e("HostProcessor<BR/>JPEG_tokenizer_metadata", request, ((1,),))
        contract = e("SharedContract<BR/>H_D_noise_masks_times_KV",
                     shared_contract, ((1,),))
        routed = e("BackendRegistry<BR/>explicit_backend_selection",
                   (processed, contract), ((1,),))
        candidates = []
        for name in (
            "Eager<BR/>official_model",
            "TorchCompile<BR/>static_prefill_and_suffix",
            "staged_IREE<BR/>2_VMFB_Device_KV",
            "expanded_IREE<BR/>1_full_flow_VMFB",
        ):
            candidates.append(e(name, routed, ((1, 32, 26),)))
        finalized = e("OfficialFinalizer<BR/>unnormalize_and_virtual_tail_DOF",
                      tuple(candidates), ((1, 32, 26),))
        return e("WebSocket<BR/>physical_action_[1,32,26]", finalized,
                 ((1, 32, 26),))


class FlowTimelineShape(nn.Module):
    """The complete 1+9 flow schedule with ten explicit Euler updates."""

    def __init__(self, mode: str):
        super().__init__()
        self.emit = ShapeEmitter(mode)

    def forward(self, z0, official_times):
        e = self.emit
        action_shape = ((1, 32, 26),)
        velocity = e("Prefill36<BR/>v0_at_t0", (z0, official_times), action_shape)
        state = e("z1_Euler0_1<BR/>z0_plus_dt_t0_t1_v0_eq_z1",
                  (z0, velocity), action_shape)
        for i in range(1, 10):
            velocity = e(f"Suffix{i}_MoT36<BR/>v{i}_at_t{i}", state, action_shape)
            state = e(
                f"z{i + 1}_Euler{i}_{i + 1}<BR/>z{i}_plus_dt_t{i}_t{i + 1}_v{i}_eq_z{i + 1}",
                (state, velocity), action_shape)
        return state


class CausalMaskShape(nn.Module):
    """Static prefill/suffix mask construction and visibility semantics."""

    def __init__(self, mode: str):
        super().__init__()
        self.emit = ShapeEmitter(mode)

    def forward(self, token_positions, padding_query_rows):
        e = self.emit
        prefill_q = e("PrefillQueryPositions<BR/>range_[0,638)",
                      token_positions, ((638, 1),))
        prefill_k = e("PrefillKeyPositions<BR/>range_[0,638)",
                      token_positions, ((1, 638),))
        prefill_compare = e("PrefillCompare<BR/>key_LE_query_638x638",
                            (prefill_q, prefill_k), ((638, 638),))
        prefill_rules = e(
            "PrefillMaskRules<BR/>VL_to_VL_causal<BR/>Action_to_prefix_all_visible"
            "<BR/>Action_to_Action_causal<BR/>prefix_to_future_Action_-inf",
            prefill_compare, ((638, 638),))
        prefill_pad = e("PadQueryOverride<BR/>diagonal_self_loop_only",
                        (prefill_rules, padding_query_rows), ((638, 638),))
        prefill_mask = e("PrefillMask3D<BR/>unsqueeze_batch_[1,638,638]",
                         prefill_pad, ((1, 638, 638),))

        suffix_q = e("SuffixAbsoluteQuery<BR/>range_[606,638)",
                     token_positions, ((32, 1),))
        suffix_k = e("SuffixKeyPositions<BR/>range_[0,638)",
                     token_positions, ((1, 638),))
        suffix_compare = e("SuffixCompare<BR/>key_LE_absolute_query_32x638",
                           (suffix_q, suffix_k), ((32, 638),))
        suffix_rules = e(
            "SuffixMaskRules<BR/>Action_to_prefix_all_visible"
            "<BR/>Action_to_Action_causal<BR/>future_Action_-inf",
            suffix_compare, ((32, 638),))
        suffix_mask = e("SuffixMask<BR/>unsqueeze_batch_[1,32,638]",
                        suffix_rules, ((1, 32, 638),))
        return e("AdditiveAttentionMask<BR/>True_to_0_False_to_-inf",
                 (prefill_mask, suffix_mask),
                 ((1, 638, 638), (1, 32, 638)))


def graph_spec(kind: str, mode: str):
    if kind == "full_flow":
        shapes = [(2048, 1176), (1, 638), (1, 32, 26),
                  (1, 32, 26), (1, 1, 26), (11,)]
        names = ["pixel_values", "context_ids", "initial_noise", "dof_mask",
                 "padding_action", "times"]
        model = FullFlowShape(mode, unroll=False)
        direction = "TB"
    elif kind == "full_flow_unrolled":
        shapes = [(2048, 1176), (1, 638), (1, 32, 26),
                  (1, 32, 26), (1, 1, 26), (11,)]
        names = ["pixel_values", "context_ids", "initial_noise", "dof_mask",
                 "padding_action", "times"]
        model = FullFlowShape(mode, unroll=True)
        direction = "TB"
    elif kind == "mot_layer":
        shapes = [(606, 2048), (32, 2048)]
        names = ["VL_606x2048", "Action_corridor_32x2048"]
        model = MoTLayerShape(mode)
        direction = "TB"
    elif kind == "action_dof":
        shapes = [(1, 32, 26), (1, 32, 26), (1, 32, 26),
                  (1,), (1,), (1, 1, 26)]
        names = ["state", "initial_noise", "dof_mask", "timestep", "dt",
                 "padding_action"]
        model = ActionDOFShape(mode)
        direction = "TB"
    elif kind == "backend_architecture":
        # Opaque request handles are used only for control-flow rendering.
        # Shapes are hidden for this figure; physical action shape is in labels.
        shapes = [(1,), (1,)]
        names = ["request", "shared_contract"]
        model = BackendArchitectureShape(mode)
        direction = "TB"
    elif kind == "flow_timeline":
        shapes = [(1, 32, 26), (11,)]
        names = ["z0_initial_noise", "official_times_t0_through_t10"]
        model = FlowTimelineShape(mode)
        direction = "TB"
    elif kind == "causal_masks":
        shapes = [(638,), (1, 638)]
        names = ["token_positions", "padding_query_rows"]
        model = CausalMaskShape(mode)
        direction = "TB"
    else:
        raise ValueError(kind)
    return model, shapes, names, direction


def tidy_dot(dot, direction: str, title: str):
    dot.graph_attr.update({
        "rankdir": direction,
        "bgcolor": "white",
        "pad": "0.25",
        "nodesep": "0.32",
        "ranksep": "0.65",
        "splines": "ortho",
        "fontname": "Helvetica",
        "labelloc": "t",
        "label": title,
        "fontsize": "20",
        "dpi": "144",
    })
    dot.graph_attr.pop("size", None)
    dot.node_attr.update({"fontname": "Helvetica", "fontsize": "10"})
    # Torchview exposes implementation depth, which is irrelevant in this view.
    dot.body = [re.sub(r"<BR/>depth:-?\d+", "", line) for line in dot.body]
    return dot


def name_view_inputs(dot, names):
    """Replace generic Torchview root labels; graph topology is unchanged."""
    remaining = list(names)
    renamed = []
    for line in dot.body:
        if "<TR><TD>input-tensor</TD>" in line and remaining:
            line = line.replace("<TR><TD>input-tensor</TD>",
                                f"<TR><TD>{remaining.pop(0)}</TD>")
        renamed.append(line)
    dot.body = renamed
    return dot


def prune_viz_accumulators(dot):
    """Remove autograd-only AccumulateGrad nodes, reconnecting named leaves."""
    accumulators = set()
    for line in dot.body:
        match = re.match(r"\s*(\d+) \[label=AccumulateGrad\]", line)
        if match:
            accumulators.add(match.group(1))
    if not accumulators:
        return dot
    predecessors = {node: [] for node in accumulators}
    successors = {node: [] for node in accumulators}
    kept = []
    edge_pattern = re.compile(r"\s*(\d+) -> (\d+)(?:\s.*)?$")
    for line in dot.body:
        match = edge_pattern.match(line)
        if match and (match.group(1) in accumulators or
                      match.group(2) in accumulators):
            source, target = match.groups()
            if target in accumulators:
                predecessors[target].append(source)
            if source in accumulators:
                successors[source].append(target)
            continue
        if any(re.match(rf"\s*{node} \[label=AccumulateGrad\]", line)
               for node in accumulators):
            continue
        kept.append(line)
    reconnections = []
    for node in accumulators:
        for source in predecessors[node]:
            for target in successors[node]:
                reconnections.append(f"\t{source} -> {target}")
    dot.body = kept[:-1] + reconnections + kept[-1:]
    dot.body = [line.replace("SliceBackward0", "ActionSlice_valid1024")
                    .replace("Backward", "") for line in dot.body]
    return dot


def render_view(kind: str):
    model, shapes, names, direction = graph_spec(kind, "view")
    graph = draw_graph(
        model, input_size=shapes, device="cpu", graph_name=f"torchview_{kind}",
        graph_dir=direction, depth=1, expand_nested=False,
        hide_inner_tensors=True, hide_module_functions=False, roll=False,
        show_shapes=(kind != "backend_architecture"), save_graph=False, mode="eval",
    )
    dot = tidy_dot(graph.visual_graph, direction,
                   f"WALL-OSS-0.5 | {kind} | Torchview shape-only surrogate")
    dot = name_view_inputs(dot, names)
    if kind == "flow_timeline":
        dot.graph_attr["ranksep"] = "0.38"
    stem = OUT / f"torchview_{kind}"
    dot.save(filename=stem.name + ".dot", directory=OUT)
    dot.render(filename=stem.name + ".dot", directory=OUT, format="svg", cleanup=False)
    dot.render(filename=stem.name + ".dot", directory=OUT, format="png", cleanup=False)


def run_viz_model(model, shapes):
    inputs = [torch.empty(s, device="meta", requires_grad=True) for s in shapes]
    with torch.enable_grad():
        outputs = model(*inputs)
    return inputs, outputs


def render_viz(kind: str):
    model, shapes, names, direction = graph_spec(kind, "viz")
    inputs, outputs = run_viz_model(model, shapes)
    named_inputs = dict(zip(names, inputs))
    dot = make_dot(outputs, params=named_inputs, show_attrs=False, show_saved=False)
    dot = prune_viz_accumulators(dot)
    dot = tidy_dot(dot, direction,
                   f"WALL-OSS-0.5 | {kind} | Torchviz shape-only surrogate")
    stem = OUT / f"torchviz_{kind}"
    dot.save(filename=stem.name + ".dot", directory=OUT)
    dot.render(filename=stem.name + ".dot", directory=OUT, format="svg", cleanup=False)
    dot.render(filename=stem.name + ".dot", directory=OUT, format="png", cleanup=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", choices=["torchview", "torchviz", "both"], default="both")
    parser.add_argument("--kind", choices=["full_flow", "full_flow_unrolled",
                                            "mot_layer", "action_dof",
                                            "backend_architecture", "flow_timeline",
                                            "causal_masks", "all"],
                        default="all")
    args = parser.parse_args()
    tools = ["torchview", "torchviz"] if args.tool == "both" else [args.tool]
    for tool in tools:
        kinds = ["full_flow", "full_flow_unrolled", "mot_layer", "action_dof"]
        if tool == "torchview":
            kinds += ["backend_architecture", "flow_timeline", "causal_masks"]
        if args.kind != "all":
            kinds = [args.kind]
        for kind in kinds:
            print(f"rendering {tool} {kind}", flush=True)
            (render_view if tool == "torchview" else render_viz)(kind)
    print("done; CUDA_VISIBLE_DEVICES=", repr(os.environ["CUDA_VISIBLE_DEVICES"]))


if __name__ == "__main__":
    main()
