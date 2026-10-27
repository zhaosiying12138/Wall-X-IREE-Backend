#!/usr/bin/env python3
"""CPU tests for expanded static modules; no checkpoint or GPU required."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from wall_x.serving.backend.static_flow import (
    StaticDenoiseEulerLoop,
    StaticFullFlow,
    StaticFullFlowTrace,
    StaticProfileError,
    StaticVisionTextContext,
    StaticVisionTextStateContext,
    StaticVisionTower,
    build_vision_profile,
)
from wall_x.serving.backend.static_flow.vision import _ordered_scatter
from wall_x.model.core import ops as wall_ops  # noqa: E402
from wall_x.model.qact.qwen2_5.configuration_qwen2_5_vl import (  # noqa: E402
    Qwen2_5_VLVisionConfig)
from wall_x.model.qact.qwen2_5.modeling_qwen2_5_vl import (  # noqa: E402
    Qwen2_5_VisionTransformerPretrainedModel)


def _force_official_fallbacks():
    for name in ("rot_pos_emb", "get_window_index"):
        op = getattr(wall_ops, name)
        op._resolved_fn = op._pytorch_fallback


def _tiny_visual(depth=3):
    config = Qwen2_5_VLVisionConfig(
        depth=depth, hidden_size=32, intermediate_size=64, num_heads=4,
        in_channels=3, out_hidden_size=64, patch_size=14,
        spatial_merge_size=2, temporal_patch_size=2, window_size=112,
        fullatt_block_indexes=[1], _attn_implementation="eager")
    return Qwen2_5_VisionTransformerPretrainedModel(config).eval()


class ExpandedStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _force_official_fallbacks()

    def test_tiny_vision_matches_official(self):
        torch.manual_seed(11)
        visual = _tiny_visual()
        grid = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.int64)
        pixels = torch.randn(32, 3 * 2 * 14 * 14)
        static = StaticVisionTower(visual, grid).eval()
        self.assertEqual(static.profile.patch_tokens, 32)
        self.assertEqual(static.profile.merged_tokens, 8)
        self.assertEqual(static.profile.num_windows, 2)
        self.assertEqual(static.profile.window_tokens, 16)
        self.assertEqual(static.profile.full_segment_tokens, (16, 16))
        with torch.no_grad():
            expected = visual(pixels, grid_thw=grid)
            actual = static(pixels)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_tiny_context_matches_official_scatter(self):
        torch.manual_seed(12)
        visual = _tiny_visual()
        grid = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.int64)
        image_id, propri_id = 40, 41
        ids = torch.tensor([[5, 40, 6, 40, 40, 41, 40, 7, 40, 40, 40, 40]])
        embedding = torch.nn.Embedding(50, 64)
        propri_proj = torch.nn.Linear(6, 64, bias=False)
        model = SimpleNamespace(
            visual=visual,
            model=SimpleNamespace(embed_tokens=embedding),
            config=SimpleNamespace(
                use_state_string_representation=False, proj_with_mask=True,
                image_token_id=image_id, hidden_size=64),
            action_token_id_set={"propri_token_id": propri_id},
            action_preprocessor=SimpleNamespace(
                propri_proj=propri_proj, state_hidden_size=64),
        )
        static = StaticVisionTextStateContext(model, grid).eval()
        pixels = torch.randn(32, 3 * 2 * 14 * 14)
        propri = torch.randn(1, 1, 3)
        agent_mask = torch.randint(0, 2, (1, 1, 3)).float()
        with torch.no_grad():
            actual = static(ids, pixels, propri, agent_mask)
            expected = embedding(ids)
            image = visual(pixels, grid_thw=grid).to(expected.dtype)
            image_mask = (ids == image_id).unsqueeze(-1).expand_as(expected)
            expected = expected.masked_scatter(image_mask, image)
            state = propri_proj(torch.cat([propri, agent_mask], dim=-1))
            expected[ids == propri_id] = state.reshape(-1, 64).to(expected.dtype)
            expected = expected.float()
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_state_string_context_uses_token_embedding(self):
        torch.manual_seed(13)
        visual = _tiny_visual(depth=1)
        grid = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.int64)
        embedding = torch.nn.Embedding(50, 64)
        model = SimpleNamespace(
            visual=visual, model=SimpleNamespace(embed_tokens=embedding),
            config=SimpleNamespace(
                use_state_string_representation=True, image_token_id=40))
        static = StaticVisionTextContext(model, grid).eval()
        ids = torch.tensor([[40, 40, 40, 40, 40, 40, 40, 40, 9, 8, 7]])
        pixels = torch.randn(32, 3 * 2 * 14 * 14)
        with torch.no_grad():
            actual = static(ids, pixels)
            expected = embedding(ids)
            image = visual(pixels, grid_thw=grid).to(expected.dtype)
            mask = (ids == 40).unsqueeze(-1).expand_as(expected)
            expected = expected.masked_scatter(mask, image).float()
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_ordered_scatter_is_masked_scatter(self):
        base = torch.randn(1, 7, 4)
        values = torch.randn(3, 4)
        mask = torch.tensor([[False, True, False, False, True, False, True]])
        expanded = mask.unsqueeze(-1).expand_as(base)
        expected = base.masked_scatter(expanded, values)
        actual = _ordered_scatter(base, mask, values)
        self.assertTrue(torch.equal(actual, expected))

    def test_uneven_windows_have_clear_error(self):
        visual = _tiny_visual(depth=1)
        # 4x6 merged grid produces one full and one partial window.
        with self.assertRaisesRegex(StaticProfileError, "equal-sized windows"):
            build_vision_profile(
                visual, torch.tensor([[1, 8, 12]], dtype=torch.int64))

    def test_unrolled_euler_matches_independent_loop(self):
        class Velocity(torch.nn.Module):
            def forward(self, x, timestep, pos, mask, k, v, v_padding=None,
                        dof_mask=None):
                del pos, mask, k, v, dof_mask
                return x * 0 + timestep.reshape(1, 1, 1) + v_padding

        loop = StaticDenoiseEulerLoop.__new__(StaticDenoiseEulerLoop)
        torch.nn.Module.__init__(loop)
        loop.num_inference_timesteps = 10
        loop.step = Velocity()
        times = torch.tensor(
            [0.0, 0.1, 0.2, 0.35, 0.45, 0.6, 0.7, 0.8, 0.9, 0.95, 0.999])
        x0 = torch.randn(1, 3, 2)
        v_padding = torch.randn(1, 3, 2)
        final, velocities, trajectory = loop(
            x0, times, None, None, None, None, v_padding)
        final_only = loop.forward_final(
            x0, times, None, None, None, None, v_padding)
        expected_x = x0.clone()
        expected_v, expected_xs = [], [x0]
        for i in range(9):
            t = times[1 + i].reshape(1)
            v = expected_x * 0 + t.reshape(1, 1, 1) + v_padding
            expected_x = expected_x + (times[2 + i] - times[1 + i]) * v
            expected_v.append(v)
            expected_xs.append(expected_x)
        torch.testing.assert_close(velocities, torch.stack(expected_v),
                                   rtol=0, atol=0)
        torch.testing.assert_close(trajectory, torch.stack(expected_xs),
                                   rtol=0, atol=0)
        torch.testing.assert_close(final, expected_x, rtol=0, atol=0)
        torch.testing.assert_close(final_only, expected_x, rtol=0, atol=0)
        self.assertEqual(velocities.shape, (9, 1, 3, 2))
        self.assertEqual(trajectory.shape, (10, 1, 3, 2))
    def test_production_and_trace_output_boundary(self):
        tensors = tuple(torch.full((1,), float(i)) for i in range(8))
        flow = StaticFullFlow.__new__(StaticFullFlow)
        torch.nn.Module.__init__(flow)
        flow._forward_all = lambda *args: tensors
        trace = StaticFullFlowTrace.__new__(StaticFullFlowTrace)
        torch.nn.Module.__init__(trace)
        trace._forward_all = lambda *args: tensors
        args = tuple(None for _ in range(10))
        self.assertIs(flow(*args), tensors[0])
        self.assertEqual(trace(*args), tensors)

    def test_session_only_reads_production_final(self):
        from wall_x.serving.backend.iree.runtime import IreeExpandedSession

        class Device:
            def __init__(self, value):
                self.value = value
                self.host_reads = 0

            def to_host(self):
                self.host_reads += 1
                return self.value

        session = IreeExpandedSession.__new__(IreeExpandedSession)
        session.trace = False
        outputs = [Device(np.zeros((1, 32, 26), dtype="f4"))]
        final = session._finish(outputs, with_trace=False)
        self.assertEqual(final.shape, (1, 32, 26))
        self.assertEqual(outputs[0].host_reads, 1)
        with self.assertRaisesRegex(RuntimeError, "separate full_flow_trace"):
            session._finish(outputs, with_trace=True)
        with self.assertRaisesRegex(RuntimeError, "expected 1"):
            session._finish([Device(np.zeros(1))] * 8, with_trace=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
