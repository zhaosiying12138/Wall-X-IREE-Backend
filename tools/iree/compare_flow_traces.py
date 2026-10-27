#!/usr/bin/env python3
"""CPU-only comparison of two saved full-flow trace NPZ files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def delta(a, b):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    d = float(np.abs(a - b).max())
    return {"maxabs": d, "relmax": d / max(float(np.abs(b).max()), 1e-12)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("actual", type=Path)
    ap.add_argument("reference", type=Path)
    ap.add_argument("--final-relmax", type=float, default=0.03)
    args = ap.parse_args()
    a = np.load(args.actual)
    b = np.load(args.reference)
    result = {"actual": str(args.actual), "reference": str(args.reference)}
    common = [key for key in ("final", "predict_action", "v0", "x0")
              if key in a and key in b]
    if not common:
        raise ValueError("trace files share none of final/predict_action/v0/x0")
    for key in common:
        result[key] = delta(a[key], b[key])
    if "velocities" in a and "velocities" in b:
        av, bv = a["velocities"], b["velocities"]
        if av.shape != bv.shape or av.shape[0] != 9:
            raise ValueError(f"velocity shape mismatch {av.shape} vs {bv.shape}")
        result["velocities"] = [delta(av[i], bv[i]) for i in range(9)]
    if "trajectory" in a and "trajectory" in b:
        at, bt = a["trajectory"], b["trajectory"]
        if at.shape != bt.shape or at.shape[0] != 10:
            raise ValueError(f"trajectory shape mismatch {at.shape} vs {bt.shape}")
        result["trajectory"] = [delta(at[i], bt[i]) for i in range(10)]
    print(json.dumps(result, indent=2))
    key = "final" if "final" in result else "predict_action"
    finite = all(np.isfinite(a[k]).all() and np.isfinite(b[k]).all()
                 for k in common)
    ok = finite and result[key]["relmax"] < args.final_relmax
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
