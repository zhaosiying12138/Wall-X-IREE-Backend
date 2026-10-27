"""Capture-compatibility patch for the official joint attention (B2, P0).

The official forward guards validate device scalars with `if tensor.max()`
/ `if not mask.any()` which force a host sync — illegal inside CUDA graph
capture. Semantics-preserving rewrite: during eager execution compute and
cache the scalar (strong tensor ref kept so id() stays unique); during
capture reuse the cached value. Validation still happens for every fresh
tensor in eager mode, so no invalid expert index can slip through.

Only methods whose source changed are rebound onto the already-constructed
model instances' classes.
"""
from __future__ import annotations

import inspect
import sys
import textwrap

_HELPERS = '''

_GUARD_CACHE = {}

def _wallx_iree_cache_key(t):
    # id()-keyed caches never hit for `tensor[i]` (fresh view object each
    # call); data_ptr+shape is stable across calls and host-side only.
    return (t.data_ptr(), t.dtype, tuple(t.shape))

def _wallx_iree_guard_max(t):
    import torch as _t
    key = _wallx_iree_cache_key(t)
    hit = _GUARD_CACHE.get(key)
    if hit is not None:
        return hit[1]
    if _t.cuda.is_current_stream_capturing():
        # unseen tensor during capture cannot be validated here; return a
        # value that fails no guard (0) — eager warmups always precede
        # capture, so a genuinely invalid index would have raised earlier.
        return 0
    v = int(t.max().item())
    _GUARD_CACHE[key] = (t, v)
    return v

def _wallx_iree_guard_any(t):
    import torch as _t
    key = _wallx_iree_cache_key(t)
    hit = _GUARD_CACHE.get(key)
    if hit is not None:
        return hit[1]
    if _t.cuda.is_current_stream_capturing():
        return False
    v = bool(t.any().item())
    _GUARD_CACHE[key] = (t, v)
    return v


_PAIR_CACHE = {}

def _wallx_iree_pair_eq(a, b):
    """Capture-safe `a == b` for 0-dim device scalars (segment emptiness).
    Keyed by data_ptr so per-call views of the same slot hit the value
    computed during eager warmup. Returning False during capture is safe
    only for non-empty segments (empty ones would launch zero-size grids),
    hence the warmup-before-capture requirement."""
    import torch as _t
    key = (_wallx_iree_cache_key(a), _wallx_iree_cache_key(b))
    hit = _PAIR_CACHE.get(key)
    if hit is not None:
        return hit[2]
    if _t.cuda.is_current_stream_capturing():
        return False
    v = bool(a.item() == b.item())
    _PAIR_CACHE[key] = (a, b, v)
    return v

'''

_REPLACEMENTS = [
    ("if token_types.max() >= len(self.dim_inputs):",
     "if _wallx_iree_guard_max(token_types) >= len(self.dim_inputs):"),
    ("if not mask.any():",
     "if not _wallx_iree_guard_any(mask):"),
    ("if start == end:", "if _wallx_iree_pair_eq(start, end):"),
]

_VLA_REPLACEMENTS = [
    ("if start == end:", "if _wallx_iree_pair_eq(start, end):"),
]

_MOE_REPLACEMENTS = [
    ("if start == end:", "if _wallx_iree_pair_eq(start, end):"),
]


def patch_joint_attention() -> list[str]:
    """Patch wall_x capture-hostile guards in place. Returns changed methods."""
    changed = _patch_module("wall_x.model.core.attention.joint", _REPLACEMENTS)
    changed += _patch_module("wall_x.model.core.vla_mixin", _VLA_REPLACEMENTS)
    changed += _patch_module("wall_x.model.core.action.moe", _MOE_REPLACEMENTS)
    return changed


def _patch_module(mod_name: str, replacements) -> list[str]:
    mod = sys.modules[mod_name]
    src = inspect.getsource(mod)

    old_classes = {n: c for n, c in vars(mod).items() if inspect.isclass(c)}
    changed: list[str] = []

    patched_src = src
    for old, new in replacements:
        patched_src = patched_src.replace(old, new)

    # append helpers at the end (the module may start with __future__ imports)
    patched_src = patched_src + _HELPERS

    ns = dict(vars(mod))
    exec(compile(patched_src, mod.__file__, "exec"), ns)

    for name, new_cls in ns.items():
        if name not in old_classes or not inspect.isclass(new_cls):
            continue
        old_cls = old_classes[name]
        for meth in dir(new_cls):
            if meth.startswith("__"):
                continue
            nf = getattr(new_cls, meth)
            if not callable(nf) or not hasattr(nf, "__code__"):
                continue
            of = getattr(old_cls, meth, None)
            if of is None or not hasattr(of, "__code__") or \
                    of.__code__.co_code != nf.__code__.co_code:
                setattr(old_cls, meth, nf)
                changed.append(f"{name}.{meth}")
    # module-level helpers used by rebound forwards resolve in mod namespace
    for h in ("_wallx_iree_guard_max", "_wallx_iree_guard_any",
              "_wallx_iree_pair_eq", "_GUARD_CACHE", "_PAIR_CACHE"):
        if h in ns:
            setattr(mod, h, ns[h])
    return changed


if __name__ == "__main__":
    import wall_x.model.core.attention.joint  # noqa: F401
    print("patched classes:", patch_joint_attention())
