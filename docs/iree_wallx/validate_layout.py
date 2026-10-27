#!/usr/bin/env python3
"""Validate Torchview DOT semantics and Graphviz SVG layout geometry."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET

from PIL import ImageFont

OUT = Path(os.environ.get(
    'WALLX_VIS_OUT', Path(__file__).resolve().parent / 'implementation_evidence'))
KINDS = ("full_flow", "mot_layer", "action_dof", "backend_architecture",
         "flow_timeline", "causal_masks")
SVG = "{http://www.w3.org/2000/svg}"
FONT_PATH = "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf"


def bbox_union(boxes):
    boxes = list(boxes)
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def intersection(a, b, tolerance=0.5):
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return width > tolerance and height > tolerance


def polygon_bbox(element):
    points = []
    for polygon in element.findall(SVG + "polygon"):
        for pair in polygon.attrib.get("points", "").split():
            x, y = pair.split(",")
            points.append((float(x), float(y)))
    if not points:
        return None
    return (min(x for x, _ in points), min(y for _, y in points),
            max(x for x, _ in points), max(y for _, y in points))


def text_bbox(text_element):
    value = "".join(text_element.itertext()).strip()
    if not value:
        return None
    font_size = float(text_element.attrib.get("font-size", "14"))
    font = ImageFont.truetype(FONT_PATH, max(1, round(font_size)))
    x = float(text_element.attrib["x"])
    baseline = float(text_element.attrib["y"])
    left, top, right, bottom = font.getbbox(value, anchor="ls")
    anchor = text_element.attrib.get("text-anchor", "start")
    if anchor == "middle":
        x0 = x - (right - left) / 2
    elif anchor == "end":
        x0 = x - (right - left)
    else:
        x0 = x + left
    return (x0, baseline + top, x0 + (right - left), baseline + bottom)


def pair_overlaps(items, tolerance=0.5):
    collisions = []
    for index, (name_a, box_a) in enumerate(items):
        for name_b, box_b in items[index + 1:]:
            if intersection(box_a, box_b, tolerance):
                collisions.append([name_a, name_b])
    return collisions


def validate_semantics(kind, dot):
    if kind == "full_flow":
        required = ("VisionTextContext_ViT32", "OrderedScatter_606VL_32Action",
                    "JointPrefill_MoT36", "NineSuffixEuler_ActionMoT36_9x",
                    "output-tensor")
        assert all(item in dot for item in required)
        return {"full_flow": True, "layers": 36, "suffix_euler_steps": 9}
    if kind == "mot_layer":
        required = ("VL_QKV_weight_2560x2048", "Action_QKV_weight_2560x1024",
                    "JointSelfAttention_Q16_KV2_mRoPE", "Action_ZeroCorridor",
                    "VL_MLP_2048_11008_2048", "Action_MLP_1024_2048_1024")
        assert all(item in dot for item in required)
        return {"independent_expert_weights": True, "zero_corridor": True}
    if kind == "action_dof":
        required = ("Concat_noisy_action_DOF", "Action_w1_weight_1024x52",
                    "SiLU_Action_w3_weight_1024x1024", "ZeroCorridor",
                    "ActionExpert_MoT36", "DOFMix_vpad1minusmask_vthetamask",
                    "Euler_state_plus_dt_velocity")
        assert all(item in dot for item in required)
        return {"action_layout": "52->1024->2048", "silu": True,
                "dof_and_euler": True}
    if kind == "backend_architecture":
        required = ("HostProcessor", "BackendRegistry", "Eager", "TorchCompile",
                    "staged_IREE", "expanded_IREE", "OfficialFinalizer",
                    "WebSocket", "SharedContract")
        assert all(item in dot for item in required)
        return {"backend_candidates": 4, "shared_contract": True}
    if kind == "flow_timeline":
        for i in range(10):
            assert f"v{i}_at_t{i}" in dot
            assert f"z{i + 1}_Euler{i}_{i + 1}" in dot
        for i in range(11):
            assert re.search(rf"t{i}(?!\d)", dot), f"missing t{i}"
        velocities = len(re.findall(r"v\d+_at_t\d+", dot))
        eulers = len(re.findall(r"z\d+_Euler\d+_\d+", dot))
        assert velocities == 10 and eulers == 10
        assert "(11,)" in dot and dot.count("(1, 32, 26)") >= 40
        return {"velocity_evaluations": velocities, "euler_updates": eulers,
                "time_points": 11, "time_grid_shape": [11]}
    if kind == "causal_masks":
        required = ("638x638", "32x638", "VL_to_VL_causal",
                    "Action_to_prefix_all_visible", "Action_to_Action_causal",
                    "prefix_to_future_Action_-inf", "diagonal_self_loop_only",
                    "True_to_0_False_to_-inf", "(1, 638, 638)", "(1, 32, 638)")
        assert all(item in dot for item in required)
        assert dot.count("Action_to_prefix_all_visible") >= 2
        return {"prefill_shape": [638, 638], "suffix_shape": [32, 638],
                "pad_query_diagonal_self_loop": True,
                "additive_rule": "True->0; False->-inf"}
    raise ValueError(kind)


def validate_layout(kind):
    stem = OUT / f"torchview_{kind}.dot"
    dot = stem.read_text(encoding="utf-8")
    svg_path = OUT / f"torchview_{kind}.dot.svg"
    png_path = OUT / f"torchview_{kind}.dot.png"
    root = ET.parse(svg_path).getroot()
    nodes = []
    node_text = []
    outside_node = []
    for group in root.iter(SVG + "g"):
        if group.attrib.get("class") != "node":
            continue
        title = group.find(SVG + "title")
        name = title.text if title is not None else f"node-{len(nodes)}"
        bounds = polygon_bbox(group)
        assert bounds is not None, f"{kind}: no polygon bounds for {name}"
        nodes.append((name, bounds))
        for text in group.findall(SVG + "text"):
            box = text_bbox(text)
            if box is None:
                continue
            label = "".join(text.itertext()).strip()
            node_text.append((f"{name}:{label}", box))
            tolerance = 3.0
            if (box[0] < bounds[0] - tolerance or box[1] < bounds[1] - tolerance or
                    box[2] > bounds[2] + tolerance or box[3] > bounds[3] + tolerance):
                outside_node.append(f"{name}:{label}")
    edge_text = []
    for group in root.iter(SVG + "g"):
        if group.attrib.get("class") != "edge":
            continue
        title = group.find(SVG + "title")
        edge_name = title.text if title is not None else "edge"
        for text in group.findall(SVG + "text"):
            box = text_bbox(text)
            if box is not None:
                edge_text.append((f"{edge_name}:{''.join(text.itertext()).strip()}", box))

    # Graphviz uses a graph-level scale/translate. Collision tests stay in
    # local coordinates; the background polygon supplies the local canvas.
    graph_group = next(group for group in root.iter(SVG + "g")
                       if group.attrib.get("class") == "graph")
    canvas = polygon_bbox(graph_group)
    assert canvas is not None
    all_boxes = nodes + node_text + edge_text
    clipped = [name for name, box in all_boxes if not (
        box[0] >= canvas[0] - 0.5 and box[1] >= canvas[1] - 0.5 and
        box[2] <= canvas[2] + 0.5 and box[3] <= canvas[3] + 0.5)]
    node_overlaps = pair_overlaps(nodes)
    text_overlaps = pair_overlaps(node_text + edge_text)
    edge_label_overlaps = pair_overlaps(edge_text)
    edge_labels_in_nodes = []
    for edge_name, edge_box in edge_text:
        for node_name, node_box in nodes:
            if intersection(edge_box, node_box):
                edge_labels_in_nodes.append([edge_name, node_name])
    png_data = png_path.read_bytes()
    width, height = struct.unpack(">II", png_data[16:24])
    semantics = validate_semantics(kind, dot)
    result = {
        "dot": str(stem), "svg": str(svg_path), "png": str(png_path),
        "png_size": [width, height], "svg_viewBox": root.attrib["viewBox"],
        "node_count": len(nodes), "edge_label_count": len(edge_text),
        "node_overlaps": node_overlaps,
        "text_overlaps": text_overlaps,
        "edge_label_overlaps": edge_label_overlaps,
        "edge_labels_in_nodes": edge_labels_in_nodes,
        "text_outside_node": outside_node, "clipped": clipped,
        "semantics": semantics,
    }
    result["passed"] = not any((node_overlaps, text_overlaps, edge_label_overlaps,
                                edge_labels_in_nodes, outside_node, clipped))
    return result


def main():
    results = [validate_layout(kind) for kind in KINDS]
    payload = {
        "renderer": "Torchview 0.2.7 -> Graphviz dot 14.1.2",
        "method": ("SVG node-polygon intersection; Nimbus Sans baseline-aware text-box "
                   "collision, node-containment and local-canvas checks; required DOT semantics"),
        "figures": results,
        "all_passed": all(item["passed"] for item in results),
    }
    destination = OUT / "torchview_layout_validation.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
