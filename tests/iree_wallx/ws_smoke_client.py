#!/usr/bin/env python3
"""P7 smoke: validate a real IREE WebSocket action response."""
import argparse
import asyncio
import base64
import math

import cv2  # noqa: F401
import msgpack
import msgpack_numpy
import numpy as np
import websockets

msgpack_numpy.patch()


def b64_jpeg(seed: int) -> str:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (448, 448, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


async def main(port: int, print_response: bool = False) -> int:
    uri = f"ws://127.0.0.1:{port}"
    async with websockets.connect(uri, max_size=64 * 2**20) as ws:
        meta_raw = await asyncio.wait_for(ws.recv(), timeout=120)
        meta = msgpack.unpackb(meta_raw, raw=False)
        if not isinstance(meta, dict):
            print("WS-SMOKE FAIL: metadata is not a dictionary")
            return 1
        print("metadata keys:", list(meta)[:8])

        rng = np.random.default_rng(0)
        obs = {
            "instruction": "pick up the object",
            "state": {
                "eef_pos": rng.normal(size=(3,)).tolist(),
                "eef_axisangle": rng.normal(size=(3,)).tolist(),
                "gripper": rng.normal(size=(1,)).tolist(),
            },
            "views": {
                "face_view": b64_jpeg(1),
                "right_wrist_view": b64_jpeg(2),
                "left_wrist_view": b64_jpeg(3),
            },
            "infer_mode": "flow",
        }
        await ws.send(msgpack.packb(obs, use_bin_type=True))
        resp = await asyncio.wait_for(ws.recv(), timeout=600)
        if not isinstance(resp, bytes):
            print("WS-SMOKE FAIL: server returned an error frame")
            print(str(resp)[:1000])
            return 1
        out = msgpack.unpackb(resp, raw=False)
        if not isinstance(out, dict):
            print("WS-SMOKE FAIL: response is not a dictionary")
            return 1
        if print_response:
            # Do not summarize or truncate the action array in diagnostic mode.
            np.set_printoptions(threshold=np.inf)
            print("COMPLETE decoded response:")
            print(out)
        print("response keys:", list(out)[:10])
        if "predict_action" not in out or "server_timing" not in out:
            print("WS-SMOKE FAIL: required response fields are missing")
            return 1
        try:
            action = np.asarray(out["predict_action"], dtype=np.float32)
            timing = out["server_timing"]
            infer_ms = float(timing["infer_ms"])
        except (KeyError, TypeError, ValueError) as exc:
            print("WS-SMOKE FAIL: cannot decode action/timing:", exc)
            return 1
        print(f"predict_action: shape={action.shape} "
              f"min={action.min():.6f} max={action.max():.6f}")
        print(f"server_timing.infer_ms: {infer_ms:.3f}")
        if action.shape != (1, 32, 26):
            print("WS-SMOKE FAIL: unexpected action shape")
            return 1
        if not np.isfinite(action).all():
            print("WS-SMOKE FAIL: action contains NaN or infinity")
            return 1
        if not math.isfinite(infer_ms) or infer_ms < 0:
            print("WS-SMOKE FAIL: invalid inference timing")
            return 1
        print("WS-SMOKE PASS")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("port", nargs="?", type=int, default=44661)
    parser.add_argument(
        "--print-response",
        action="store_true",
        help="print the complete decoded response without NumPy truncation",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.port, args.print_response)))
