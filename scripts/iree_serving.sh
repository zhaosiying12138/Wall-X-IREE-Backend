#!/usr/bin/env bash
# Launch staged/expanded IREE serving without hiding native server output.
# Usage: WALLX_CKPT=/path/to/checkpoint bash scripts/iree_serving.sh expanded_serve
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
STAGE="${1:-expanded_serve}"
CKPT="${WALLX_CKPT:?set WALLX_CKPT to the checkpoint directory}"
PY="${PYTHON_BIN:-python3}"
PORT="${PORT:-44661}"
LOG="${SERVER_LOG:-/tmp/wallx_${STAGE}.log}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ENABLE_FAST_PREPROCESS=false

case "$STAGE" in
  serve)
    export WALLX_BACKEND=iree
    export WALLX_VISION_ATTN_IMPLEMENTATION="${WALLX_VISION_ATTN_IMPLEMENTATION:-flash_attention_2}"
    ;;
  expanded_serve)
    export WALLX_BACKEND=iree-expanded
    export WALLX_VISION_ATTN_IMPLEMENTATION="${WALLX_VISION_ATTN_IMPLEMENTATION:-eager}"
    ;;
  torch_compile_serve)
    export WALLX_BACKEND=torch-compile
    export WALLX_VISION_ATTN_IMPLEMENTATION="${WALLX_VISION_ATTN_IMPLEMENTATION:-flash_attention_2}"
    ;;
  *) printf 'Unknown stage: %s\n' "$STAGE" >&2; exit 2 ;;
esac

# Process substitution tees native stdout/stderr to the console and the log while
# $! remains the actual server PID (unlike a `server | tee &` pipeline).
PYTHON_BIN="$PY" bash scripts/run_serving.sh \
  --checkpoint-path "$CKPT" \
  --train-config-path "$CKPT/config.yml" \
  --port "$PORT" --env LIBERO --robot-type desktop \
  --no-serialize-actions -- \
  --model-config.cam-names face_view right_wrist_view \
  > >(tee "$LOG") 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

ready=0
for _ in $(seq 1 50); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    wait "$SERVER_PID" || true
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "$ready" != 1 ]]; then
  printf 'Server did not become healthy; see %s\n' "$LOG" >&2
  exit 1
fi
curl -fsS "http://127.0.0.1:$PORT/healthz"
printf '\n'

ws_args=("$PORT")
if [[ "${WS_PRINT_RESPONSE:-0}" == 1 ]]; then
  ws_args+=(--print-response)
fi
"$PY" tests/iree_wallx/ws_smoke_client.py "${ws_args[@]}"
# Match the actual selected backend name, not only the staged activation text.
grep -F "serving backend ACTIVE (name=$WALLX_BACKEND" "$LOG"
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
trap - EXIT
printf 'server_log: %s\n' "$LOG"
