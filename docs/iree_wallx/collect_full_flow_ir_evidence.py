#!/usr/bin/env python3
"""Collect focused, line-numbered IR evidence for the current full_flow build."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(os.environ.get('WALLX_FULL_FLOW_PHASES', '/tmp/wallx_native_full_flow_phases'))
OUT = ROOT / 'docs/iree_wallx/implementation_evidence/ir/full_flow'
NATIVE = ROOT / 'docs/iree_wallx/implementation_evidence/native_logs'


def lines(path):
    return path.read_text(encoding='utf-8', errors='replace').splitlines()


def write_matches(source: Path, dest: Path, predicates, limit=8):
    result = [f'# Source: {source}', f'# Bytes: {source.stat().st_size}']
    for predicate in predicates:
        hits = [(i, line) for i, line in enumerate(lines(source), 1)
                if predicate(line)]
        result.append(f'\n# Pattern: {predicates.index(predicate)}; hits={len(hits)}')
        for i, line in hits[:limit]:
            result.append(f'{i}: {line[:360]}')
    dest.write_text('\n'.join(result) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write_matches(SRC / 'module.1.input.mlir', OUT / 'input_main.txt',
                  [lambda l: 'util.func public @main' in l,
                   lambda l: 'util.call @main$async' in l,
                   lambda l: 'iree_linalg_ext.attention' in l], limit=4)
    write_matches(ROOT / 'artifacts/ir/full_flow.mlir', OUT / 'torch_main.txt',
                  [lambda l: 'func.func @main' in l,
                   lambda l: 'torch.aten.slice.Tensor' in l,
                   lambda l: 'torch.aten.conv' in l], limit=4)
    write_matches(SRC / 'module.6.flow.mlir', OUT / 'flow.txt',
                  [lambda l: 'flow.executable private' in l,
                   lambda l: 'flow.dispatch @main' in l,
                   lambda l: 'iree_linalg_ext.attention' in l], limit=5)
    write_matches(SRC / 'module.7.stream.mlir', OUT / 'stream.txt',
                  [lambda l: 'util.func public @main$async' in l,
                   lambda l: 'stream.cmd.execute' in l,
                   lambda l: 'stream.cmd.dispatch @main' in l], limit=6)
    write_matches(SRC / 'module.9.executable-configurations.mlir',
                  OUT / 'configuration.txt',
                  [lambda l: '#translation' in l and 'pipeline =' in l,
                   lambda l: 'online_attention' in l,
                   lambda l: 'TileAndFuse' in l or 'Distribute' in l or 'VectorDistribute' in l], limit=8)
    write_matches(SRC / 'module.12.vm.mlir', OUT / 'vm.txt',
                  [lambda l: 'vm.export @main' in l,
                   lambda l: 'vm.func private @main(' in l,
                   lambda l: 'hal.fence.await' in l,
                   lambda l: 'hal.device.queue.execute' in l], limit=5)

    production_mlir = ROOT / 'artifacts/ir/full_flow.mlir'
    production_vmfb = ROOT / 'artifacts/bin/full_flow.vmfb'
    trace_mlir = ROOT / 'artifacts/ir/full_flow_trace.mlir'
    trace_vmfb = ROOT / 'artifacts/bin/full_flow_trace.vmfb'
    irpa = ROOT / 'artifacts/bin/full_flow.weights.irpa'
    compile_log = NATIVE / 'full_flow_iree_compile.console.log'
    compile_command = next(
        line[2:] for line in compile_log.read_text(encoding='utf-8').splitlines()
        if line.startswith('+ .venv-serve/bin/iree-compile '))
    manifest = {
        'input': 'artifacts/ir/full_flow.mlir',
        'compile_command': compile_command,
        'phase_dump': str(SRC),
        'production': {
            'mlir_bytes': production_mlir.stat().st_size,
            'vmfb_bytes': production_vmfb.stat().st_size,
            'irpa_bytes': irpa.stat().st_size,
            'inputs': 10, 'outputs': 1, 'output_shape': [1, 32, 26],
            'session_calls': 1,
            'mlir_sha256': sha(production_mlir),
            'vmfb_sha256': sha(production_vmfb),
            'irpa_sha256': sha(irpa),
        },
        'trace': {
            'mlir_bytes': trace_mlir.stat().st_size,
            'vmfb_bytes': trace_vmfb.stat().st_size,
            'mlir_sha256': sha(trace_mlir),
            'vmfb_sha256': sha(trace_vmfb),
        },
        'verdict': 'experimental-only: final normalized/physical action gates pass; context and V-cache strict gates fail',
        'note': 'Complete native compiler console logs are retained under native_logs. This directory contains line-numbered IR excerpts from the same build.',
        'files': {},
    }
    for path in sorted(OUT.glob('*.txt')):
        text = path.read_text(encoding='utf-8', errors='replace')
        path.write_text('\n'.join(line.rstrip() for line in text.splitlines()) + '\n',
                        encoding='utf-8')
    for path in sorted(OUT.glob('*.txt')):
        manifest['files'][path.name] = {
            'bytes': path.stat().st_size, 'sha256': sha(path)}
    (OUT / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('full-flow IR evidence:', OUT)


if __name__ == '__main__':
    main()
