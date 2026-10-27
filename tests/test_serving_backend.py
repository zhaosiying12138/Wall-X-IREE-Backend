"""CPU-only contracts for the independent Wall-X serving backends."""
from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

import torch

from wall_x.serving.backend.registry import (
    available_backends,
    backend_class,
    create_backend,
)


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.env_names = (
            "WALLX_BACKEND",
            "WALLX_IREE_PREFILL_VMFB",
            "WALLX_IREE_PREFILL_IRPA",
            "WALLX_IREE_STEP_VMFB",
            "WALLX_IREE_STEP_IRPA",
            "WALLX_IREE_FULL_FLOW_VMFB",
            "WALLX_IREE_FULL_FLOW_IRPA",
        )
        self.saved = {name: os.environ.get(name) for name in self.env_names}
        for name in self.env_names:
            os.environ.pop(name, None)

    def tearDown(self):
        for name, value in self.saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_eager_is_default_and_preserves_explicit_noise(self):
        class Model:
            def generate_flow_action(self, **kwargs):
                return kwargs

        noise = torch.randn(1, 32, 26)
        backend = create_backend(Model())
        result = backend.generate_flow_action(noise=noise, padding_action=None)
        self.assertEqual(backend.name, "eager")
        self.assertIs(result["noise"], noise)

    def test_unknown_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown Wall-X serving backend"):
            create_backend(object(), name="not-registered")

    def test_wrapper_does_not_fall_back_for_unknown_backend(self):
        from wall_x._vendor.harrix.serving._wallx_infer.model_wrapper import (
            WallxModelWrapper,
        )

        wrapper = WallxModelWrapper.__new__(WallxModelWrapper)
        wrapper.model = object()
        with patch.dict(os.environ, {"WALLX_BACKEND": "not-registered"}):
            with self.assertRaisesRegex(ValueError, "unknown Wall-X serving backend"):
                wrapper._get_serving_backend()
        self.assertFalse(hasattr(wrapper, "_serving_backend"))

    def test_iree_requires_all_artifacts_before_runtime_import(self):
        with self.assertRaisesRegex(ValueError, "explicit artifact configuration"):
            create_backend(object(), name="iree")
        self.assertNotIn("iree.runtime", sys.modules)

    def test_registry_loads_iree_model_lazily_without_runtime(self):
        self.assertIn("torch-compile", available_backends())
        backend_class("iree-expanded")
        self.assertNotIn("iree.runtime", sys.modules)


class _Prefill(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.k = torch.full((1, 1, 1, 1, 1), 3.0)
        self.v = torch.full((1, 1, 1, 1, 1), 5.0)
        self.x0 = torch.full((1, 1, 1), 7.0)
        self.suffix_calls = 0

    def forward(self, *args, dof_mask=None):
        del args, dof_mask
        return self.k, self.v, torch.zeros_like(self.x0), self.x0


class _Suffix(torch.nn.Module):
    def __init__(self, prefill):
        super().__init__()
        self.prefill = prefill
        self.calls = 0

    def forward(
        self,
        x,
        timestep,
        postfix_positions,
        postfix_mask,
        k,
        v,
        v_padding=None,
        dof_mask=None,
    ):
        del postfix_positions, postfix_mask, dof_mask
        self.assert_clone_boundary(x, k, v)
        velocity = x * 0 + timestep.reshape(1, 1, 1) + v_padding
        # Simulate reuse/invalidation of reduce-overhead graph-owned outputs.
        self.prefill.k.fill_(-100)
        self.prefill.v.fill_(-100)
        self.prefill.x0.fill_(-100)
        self.prefill.suffix_calls += 1
        self.calls += 1
        return velocity

    def assert_clone_boundary(self, x, k, v):
        if self.calls == 0:
            assert x.data_ptr() != self.prefill.x0.data_ptr()
        assert k.data_ptr() != self.prefill.k.data_ptr()
        assert v.data_ptr() != self.prefill.v.data_ptr()
        torch.testing.assert_close(k, torch.full_like(k, 3.0))
        torch.testing.assert_close(v, torch.full_like(v, 5.0))


class TorchCompileFlowTests(unittest.TestCase):
    def test_static_flow_clones_cross_compiled_region_outputs(self):
        from wall_x.serving.backend.torch_compile import execute_static_flow

        prefill = _Prefill()
        suffix = _Suffix(prefill)
        noise = torch.ones(1, 1, 1)
        padding = torch.zeros(1, 1, 1)
        times = torch.tensor([0.0, 0.1, 0.2, 0.4])
        result = execute_static_flow(
            prefill,
            suffix,
            inputs_embeds=torch.zeros(1),
            position_ids=torch.zeros(1),
            mask3d=torch.zeros(1),
            noise=noise,
            padding_action=padding,
            times=times,
            postfix_positions=torch.zeros(1),
            postfix_mask=torch.zeros(1),
            dof_mask=torch.ones(1, 1, 1),
        )
        expected = torch.full((1, 1, 1), 7.0)
        expected = expected + (times[2] - times[1]) * (times[1] - 1)
        expected = expected + (times[3] - times[2]) * (times[2] - 1)
        torch.testing.assert_close(result, expected)
        self.assertEqual(suffix.calls, 2)

    def test_backend_compiles_prefill_and_suffix_as_real_peers(self):
        from wall_x.serving.backend.torch_compile import TorchCompileFlowBackend

        prefill = _Prefill()
        suffix = _Suffix(prefill)
        calls = []

        def fake_compile(module, **kwargs):
            calls.append((module, kwargs))
            return module

        backend = TorchCompileFlowBackend(
            torch.nn.Linear(1, 1),
            prefill=prefill,
            suffix=suffix,
            compile_fn=fake_compile,
            mode="reduce-overhead",
        )
        self.assertEqual(backend.name, "torch-compile")
        self.assertEqual([call[0] for call in calls], [prefill, suffix])
        for _module, options in calls:
            self.assertEqual(options["mode"], "reduce-overhead")
            self.assertFalse(options["dynamic"])


class IreeHostConversionTests(unittest.TestCase):
    def test_bf16_tensor_is_cast_before_numpy(self):
        import numpy as np
        from wall_x.serving.backend.iree.model import IreeFlowBackend

        value = IreeFlowBackend._host(
            torch.tensor([1.5, -2.0], dtype=torch.bfloat16), np.float32)
        self.assertEqual(value.dtype, np.float32)
        np.testing.assert_allclose(value, [1.5, -2.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
