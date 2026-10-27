"""Lazy, in-process sessions backed by stock upstream ``iree.runtime``."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def _runtime(driver_name: str, cudalib=None):
    if cudalib:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = str(cudalib) + (
            ":" + existing if existing else ""
        )
    # Deliberately imported only when an IREE backend is initialized.  No local
    # compiler/runtime patches or compatibility shims are required.
    import iree.runtime as rt

    return rt, rt.Config(driver_name=driver_name)


class IreeResidentSession:
    """Mmap staged artifacts once and retain prefill K/V on the device."""

    def __init__(
        self,
        prefill_vmfb,
        prefill_irpa,
        step_vmfb,
        step_irpa,
        *,
        device="cuda",
        cudalib=None,
    ):
        rt, config = _runtime(device, cudalib)
        instance = config.vm_instance

        def load(vmfb, irpa):
            index = rt.ParameterIndex()
            index.load(file_path=str(irpa))
            provider = index.create_provider(scope="model")
            io_module = rt.create_io_parameters_module(instance, provider)
            vm = rt.VmModule.mmap(instance, str(vmfb))
            bound = rt.load_vm_modules(io_module, vm, config=config)[-1]
            return bound, provider

        self.prefill_mod, self._prefill_provider = load(prefill_vmfb, prefill_irpa)
        self.step_mod, self._step_provider = load(step_vmfb, step_irpa)
        self._rt = rt
        self._device = config.device
        self.calls = 0

    def _dev(self, value):
        if type(value).__name__ == "DeviceArray":
            return value
        return self._rt.asdevicearray(self._device, value)

    def prefill(self, embeds, pos, mask3d, noise, padding, dt0):
        outputs = self.prefill_mod.main(
            *[self._dev(value) for value in (embeds, pos, mask3d, noise, padding, dt0)]
        )
        outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        if len(outputs) != 4:
            raise RuntimeError(f"joint_prefill returned {len(outputs)} outputs; expected 4")
        k_dev, v_dev, v0_dev, x0_dev = outputs
        self.calls += 1
        return (
            k_dev,
            v_dev,
            np.asarray(v0_dev.to_host()),
            np.asarray(x0_dev.to_host()),
        )

    def step(self, x, timestep, pos, suffix_mask, k_dev, v_dev, v_padding):
        outputs = self.step_mod.main(
            *[
                self._dev(value)
                for value in (x, timestep, pos, suffix_mask, k_dev, v_dev, v_padding)
            ]
        )
        outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        if len(outputs) != 1:
            raise RuntimeError(f"denoise_step returned {len(outputs)} outputs; expected 1")
        self.calls += 1
        return np.asarray(outputs[0].to_host())


class IreeExpandedSession:
    """One-artifact full-flow session with a production or trace ABI."""

    def __init__(
        self,
        full_flow_vmfb,
        full_flow_irpa,
        *,
        trace=False,
        device="cuda",
        cudalib=None,
    ):
        rt, config = _runtime(device, cudalib)
        instance = config.vm_instance
        index = rt.ParameterIndex()
        index.load(file_path=str(full_flow_irpa))
        self._provider = index.create_provider(scope="model")
        io_module = rt.create_io_parameters_module(instance, self._provider)
        vm = rt.VmModule.mmap(instance, str(full_flow_vmfb))
        self.full_flow_mod = rt.load_vm_modules(io_module, vm, config=config)[-1]
        self._rt = rt
        self._device = config.device
        self.trace = bool(trace)
        self.calls = 0

    def _dev(self, value):
        if type(value).__name__ == "DeviceArray":
            return value
        return self._rt.asdevicearray(self._device, value)

    @staticmethod
    def _outputs(value):
        return value if isinstance(value, (list, tuple)) else [value]

    def run(
        self,
        input_ids,
        pixel_values,
        position_ids,
        mask3d,
        noise,
        padding_action,
        times,
        postfix_pos,
        suffix_mask,
        dof_mask,
        with_trace=False,
    ):
        output = self._outputs(
            self.full_flow_mod.main(
                *[
                    self._dev(value)
                    for value in (
                        input_ids,
                        pixel_values,
                        position_ids,
                        mask3d,
                        noise,
                        padding_action,
                        times,
                        postfix_pos,
                        suffix_mask,
                        dof_mask,
                    )
                ]
            )
        )
        self.calls += 1
        return self._finish(output, with_trace)

    def run_tensor_state(
        self,
        input_ids,
        pixel_values,
        proprioception,
        agent_pos_mask,
        position_ids,
        mask3d,
        noise,
        padding_action,
        times,
        postfix_pos,
        suffix_mask,
        dof_mask,
        with_trace=False,
    ):
        output = self._outputs(
            self.full_flow_mod.main(
                *[
                    self._dev(value)
                    for value in (
                        input_ids,
                        pixel_values,
                        proprioception,
                        agent_pos_mask,
                        position_ids,
                        mask3d,
                        noise,
                        padding_action,
                        times,
                        postfix_pos,
                        suffix_mask,
                        dof_mask,
                    )
                ]
            )
        )
        self.calls += 1
        return self._finish(output, with_trace)

    def _finish(self, output, with_trace):
        expected = 8 if self.trace else 1
        if len(output) != expected:
            mode = "trace" if self.trace else "production"
            raise RuntimeError(
                f"full_flow {mode} module returned {len(output)} outputs; "
                f"expected {expected}. Select the matching VMFB explicitly."
            )
        if with_trace and not self.trace:
            raise RuntimeError(
                "Trace reads require the separate full_flow_trace VMFB; "
                "production full_flow has only the final-action output"
            )
        final = np.asarray(output[0].to_host())
        if not with_trace:
            return final
        return (final,) + tuple(
            np.asarray(value.to_host()) for value in output[1:]
        )
