#!/usr/bin/env python3
"""Build artifacts/manifests/model.json — frozen model/workload manifest (P0-2).

Static section: repo SHA, checkpoint file hashes, resolved HF config, tokenizer
metadata, safetensors dtype/shape census, environment freeze.
Runtime section (grid_thw, load_state_dict keys, latency) is merged later by
probe scripts via ``--merge-runtime``.

Usage (from repo root, PYTHONPATH=.):
    python tools/manifest/build_model_manifest.py \
        --checkpoint /path/to/wall-oss-0.5 --output artifacts/manifests/model.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HF_REPO = "x-square-robot/wall-oss-0.5"
HF_CONFIG_REVISION = "b7dc23a82dbf70a65859c9fdd596703234597028"


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_meta(path: Path, with_hash: bool = True) -> dict:
    st = path.stat()
    meta = {"size_bytes": st.st_size, "mtime": datetime.fromtimestamp(
        st.st_mtime, tz=timezone.utc).isoformat()}
    if with_hash:
        meta["sha256"] = sha256_file(path)
    return meta


def git_info(repo: Path) -> dict:
    def run(*args: str) -> str:
        return subprocess.run(("git", "-C", str(repo), *args),
                              capture_output=True, text=True, check=True).stdout.strip()
    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_files": len(run("status", "--porcelain").splitlines()),
    }


def env_freeze() -> dict:
    import torch
    versions = {"python": platform.python_version(), "torch": torch.__version__,
                "torch_cuda": torch.version.cuda}
    for mod_name in ("transformers", "safetensors", "flash_attn", "numpy",
                     "accelerate", "msgpack", "websockets"):
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "?")
        except Exception as exc:  # noqa: BLE001
            versions[mod_name] = f"unavailable: {type(exc).__name__}"
    try:
        import wall_x  # noqa: F401
        versions["wall_x"] = "import-ok (PYTHONPATH source tree)"
    except Exception as exc:  # noqa: BLE001
        versions["wall_x"] = f"import-failed: {exc}"
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        versions["gpu"] = {
            "name": props.name, "capability": f"{props.major}.{props.minor}",
            "sm_count": props.multi_processor_count,
            "total_mem_gb": round(props.total_memory / 2**30, 2),
            "driver_version": torch.cuda.driver_dump.get("driver_version") if isinstance(getattr(torch.cuda, "driver_dump", None), dict) else None,
        }
    return versions


def safetensors_census(model_file: Path) -> dict:
    from safetensors import safe_open
    tensors = {}
    with safe_open(str(model_file), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for k in keys:
            t = f.get_slice(k)
            tensors[k] = {"dtype": str(t.get_dtype()), "shape": list(t.get_shape())}
    dtypes = Counter(v["dtype"] for v in tensors.values())
    n_params = sum(math.prod(v["shape"]) for v in tensors.values())
    return {"num_tensors": len(tensors), "dtypes": dict(dtypes),
            "total_params": n_params, "tensors": tensors}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--skip-model-hash", action="store_true",
                    help="skip sha256 of the big safetensors (offline re-runs)")
    ap.add_argument("--model-hash-from", type=Path, default=None,
                    help="file containing '<sha256>  model.safetensors' to reuse")
    args = ap.parse_args()

    ckpt = args.checkpoint.resolve()
    repo_root = Path(__file__).resolve().parents[2]

    manifest: dict = {
        "manifest_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "hf_repo": HF_REPO,
            "hf_config_revision_audited": HF_CONFIG_REVISION,
            "local_path": str(ckpt),
            "downloaded_via": "official checkpoint snapshot",
        },
        "repo": git_info(repo_root),
    }

    print("[1/5] environment freeze ...", file=sys.stderr)
    manifest["environment"] = env_freeze()

    print("[2/5] checkpoint files ...", file=sys.stderr)
    files = {}
    for name in sorted(os.listdir(ckpt)):
        p = ckpt / name
        if not p.is_file():
            continue
        big = p.suffix in {".safetensors", ".pth", ".json"} and p.stat().st_size > 50_000_000
        if p.name == "model.safetensors":
            if args.model_hash_from and args.model_hash_from.exists():
                h = args.model_hash_from.read_text().split()[0]
                files[p.name] = {**file_meta(p, with_hash=False), "sha256": h,
                                 "sha256_source": str(args.model_hash_from)}
            elif args.skip_model_hash:
                files[p.name] = file_meta(p, with_hash=False)
            else:
                files[p.name] = file_meta(p)
        elif big:
            files[p.name] = file_meta(p, with_hash=False)
        else:
            files[p.name] = file_meta(p)
    manifest["checkpoint_files"] = files

    print("[3/5] resolved config ...", file=sys.stderr)
    cfg = json.loads((ckpt / "config.json").read_text())
    head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
    manifest["resolved_config"] = {
        "raw": cfg,
        "derived": {
            "joint_backbone_layers": cfg["num_hidden_layers"],
            "hidden_size": cfg["hidden_size"],
            "num_q_heads": cfg["num_attention_heads"],
            "num_kv_heads": cfg["num_key_value_heads"],
            "head_dim": head_dim,
            "gqa_group_size": cfg["num_attention_heads"] // cfg["num_key_value_heads"],
            "expert_hidden_sizes": [e["hidden_size"] for e in cfg["experts"]],
            "expert_inter_sizes": [e["intermediate_size"] for e in cfg["experts"]],
            "vision_layers": cfg["vision_config"]["depth"],
            "vision_fullatt_block_indexes": cfg["vision_config"]["fullatt_block_indexes"],
            "vocab_size_config": cfg["vocab_size"],
            "causal_action_attention_mask": cfg["causal_action_attention_mask"],
            "mot_opt": cfg["mot_opt"],
            "use_x_pred": cfg["use_x_pred"],
            "num_inference_timesteps": cfg["noise_scheduler"]["num_inference_timesteps"],
            "attn_implementation_text": cfg.get("_attn_implementation"),
            "attn_implementation_vision": cfg["vision_config"].get("_attn_implementation"),
        },
    }

    print("[4/5] tokenizer / processor ...", file=sys.stderr)
    tok_cfg = json.loads((ckpt / "tokenizer_config.json").read_text())
    manifest["tokenizer"] = {
        "vocab_size_file": len(json.loads((ckpt / "vocab.json").read_text())),
        "special_tokens": {k: tok_cfg.get(k) for k in
                           ("bos_token", "eos_token", "pad_token")},
        "processor": json.loads((ckpt / "preprocessor_config.json").read_text()),
    }

    print("[5/5] safetensors census ...", file=sys.stderr)
    manifest["weights"] = safetensors_census(ckpt / "model.safetensors")

    manifest["runtime_probes"] = {}  # filled by later probe scripts

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=False))
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
