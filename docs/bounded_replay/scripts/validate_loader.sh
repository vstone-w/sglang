#!/usr/bin/env bash
# One-shot M1 validation: baseline (replay off) vs decoder replay on.
#
#   ./validate.sh --dry-run     # print the plan, start nothing
#   ./validate.sh               # both phases: probe + prefix(smoke,full) + GSM8K
#   ./validate.sh --quick       # both phases but prefix smoke only
#   SKIP_GSM8K=1 ./validate.sh  # functional checks only
#
# The two phases run sequentially on HOST:PORT. A server that this script did
# not start is never killed (KILL_STALE=0, preflight refuses if one is up).
#
# Results: outputs/validation_<stamp>/{baseline,replay,report.md,compare.txt}

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

SGLANG_ROOT=${SGLANG_ROOT:-/home/w00889861/dsv4/sglang_bouned_replay/sglang}
MODEL_PATH=${MODEL_PATH:-/mnt/paas/weights/DeepSeek-V4.1-Flash-1017-w4a8}
KERNEL_PATH=${KERNEL_PATH:-/home/w00889861/dsv4/sgl_kernel_npu_2026.11.0}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8001}
BASE_URL="http://$HOST:$PORT"
# DSV4.1 o_groups=8 with a 528GB checkpoint on 16x64GB dies leaves one viable
# shape on one node: all 16 dies with attention TP 1, i.e. DP16/TP16. Without
# DP attention TP would have to be <= 8, which does not fit the weights.
TP_SIZE=${TP_SIZE:-16}
DP_SIZE=${DP_SIZE:-16}
# Eager decode: the TP16 expert intermediate K=144 is not divisible by 64, so
# the WeightNz grouped matmul used during decode graph capture rejects it.
DISABLE_CUDA_GRAPH=${DISABLE_CUDA_GRAPH:-1}
# A3 canonical MoE backend: "none" keeps all experts per rank and fails at
# TP16 (expert intermediate K=144); DeepEP shards experts by EP=TP.
MOE_A2A_BACKEND=${MOE_A2A_BACKEND:-deepep}
# Weights (target + draft) need >=0.813 at DP16; DeepEP's HCCL buffers need
# (1 - fraction) * 64GB >= 5.24GB, i.e. fraction <= 0.918. 0.90 is the middle
# of the viable window.
MEM_FRACTION=${MEM_FRACTION:-0.90}

# Small by default so a full baseline+replay sweep finishes quickly; raise
# them when a tighter accuracy number is needed.
NUM_QUESTIONS=${NUM_QUESTIONS:-50}
NUM_THREADS=${NUM_THREADS:-32}
MAX_TOKENS=${MAX_TOKENS:-4096}
THINKING=${THINKING:-0}
# Quick-pass thresholds for the 50-question default; raise with NUM_QUESTIONS
# when a tighter accuracy number is needed.
GSM8K_MIN=${GSM8K_MIN:-0.90}
GSM8K_DELTA=${GSM8K_DELTA:-0.04}
MIN_CACHE_HIT_PCT=${MIN_CACHE_HIT_PCT:-30}
SERVER_TIMEOUT=${SERVER_TIMEOUT:-2400}

QUICK=0
DRY_RUN=0
PREFLIGHT_ONLY=0
SKIP_GSM8K=${SKIP_GSM8K:-0}
SKIP_PREFIX=${SKIP_PREFIX:-0}
SKIP_PROBE=${SKIP_PROBE:-0}
SKIP_REPLAY=${SKIP_REPLAY:-0}

for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --preflight-only) PREFLIGHT_ONLY=1 ;;
    --skip-gsm8k) SKIP_GSM8K=1 ;;
    --skip-prefix) SKIP_PREFIX=1 ;;
    --skip-probe) SKIP_PROBE=1 ;;
    --skip-replay) SKIP_REPLAY=1 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR=${OUT_DIR:-$SCRIPT_DIR/outputs/validation_$STAMP}
SERVER_PID=

log() { echo "[validate] $*"; }
die() { echo "[validate] ERROR: $*" >&2; exit 1; }

stop_server() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    log "stopping server pid=$SERVER_PID"
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 90); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null || true
    sleep 5
  fi
  SERVER_PID=
}
trap 'stop_server' EXIT INT TERM

preflight() {
  [ -d "$MODEL_PATH" ] || die "MODEL_PATH missing: $MODEL_PATH"
  [ -d "$KERNEL_PATH" ] || die "KERNEL_PATH missing: $KERNEL_PATH (see README)"
  PYTHONPATH="$KERNEL_PATH" python3 -c \
    "from sgl_kernel_npu.operators.indexer import lightning_indexer_v2" \
    || die "KERNEL_PATH lacks lightning_indexer_v2; install the paired sgl-kernel-npu wheel"
  if curl -sf --max-time 2 "$BASE_URL/health" >/dev/null 2>&1; then
    die "a server already answers on $BASE_URL; stop it or set another PORT"
  fi
  if pgrep -f "[s]glang.launch_server" >/dev/null 2>&1; then
    die "sglang.launch_server is already running; refusing to interfere"
  fi
  if [ "$DP_SIZE" = 1 ]; then
    [ "$TP_SIZE" -le 8 ] || die "TP_SIZE=$TP_SIZE invalid without DP attention (o_groups=8)"
  else
    [ "$DP_SIZE" = "$TP_SIZE" ] || die "DP attention with replay needs DP_SIZE == TP_SIZE; got DP_SIZE=$DP_SIZE TP_SIZE=$TP_SIZE"
  fi
  local out
  out=$(TP_SIZE="$TP_SIZE" DP_SIZE="$DP_SIZE" MOE_A2A_BACKEND="$MOE_A2A_BACKEND" \
    MEM_FRACTION="$MEM_FRACTION" DISABLE_CUDA_GRAPH="$DISABLE_CUDA_GRAPH" \
    DRY_RUN=1 ./run_dsv4.1.sh 2>&1) \
    || { echo "$out" >&2; die "baseline command preflight failed"; }
  out=$(TP_SIZE="$TP_SIZE" DP_SIZE="$DP_SIZE" MOE_A2A_BACKEND="$MOE_A2A_BACKEND" \
    MEM_FRACTION="$MEM_FRACTION" DISABLE_CUDA_GRAPH="$DISABLE_CUDA_GRAPH" \
    DRY_RUN=1 ENABLE_DECODER_REPLAY=1 ./run_dsv4.1.sh 2>&1) \
    || { echo "$out" >&2; die "replay command preflight failed (gate?)"; }
}

start_server() {
  local log_file=$1
  shift
  log "launching: env $* ./run_dsv4.1.sh"
  setsid env KILL_STALE=0 HOST="$HOST" PORT="$PORT" TP_SIZE="$TP_SIZE" DP_SIZE="$DP_SIZE" \
    MOE_A2A_BACKEND="$MOE_A2A_BACKEND" MEM_FRACTION="$MEM_FRACTION" \
    DISABLE_CUDA_GRAPH="$DISABLE_CUDA_GRAPH" EXTRA_ARGS="${SERVER_EXTRA_ARGS:-}" "$@" \
    ./run_dsv4.1.sh >"$log_file" 2>&1 &
  SERVER_PID=$!
  local waited=0
  while [ "$waited" -lt "$SERVER_TIMEOUT" ]; do
    if curl -sf --max-time 2 "$BASE_URL/health" >/dev/null 2>&1; then
      log "server ready after ${waited}s"
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "server died during startup; last log lines:"
      tail -40 "$log_file" >&2
      SERVER_PID=
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
  log "server startup timed out after ${SERVER_TIMEOUT}s"
  tail -40 "$log_file" >&2
  stop_server
  return 1
}

run_sanity() {
  local dir=$1
  log "sanity generate"
  curl -sf --max-time 600 -X POST "$BASE_URL/generate" \
    -H 'Content-Type: application/json' \
    -d '{"text": "Say pong.", "sampling_params": {"max_new_tokens": 8, "temperature": 0.0}}' \
    >"$dir/sanity.json" || return 1
  python3 - "$dir/sanity.json" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1]))
text = (data.get("text") or "").strip()
print(f"[validate] sanity text={text[:80]!r}")
PY
}

run_probe() {
  local dir=$1
  if [ "$SKIP_PROBE" = 1 ]; then
    log "probe skipped"
    return 0
  fi
  log "probe"
  python3 "$SCRIPT_DIR/probe_replay.py" "$BASE_URL" "$dir/probe.json" \
    2>&1 | tee "$dir/probe.log"
  local status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    log "WARNING: probe failed (status $status); continuing to the other checks"
  fi
  return 0
}

run_prefix() {
  local dir=$1
  if [ "$SKIP_PREFIX" = 1 ]; then
    log "prefix skipped"
    return 0
  fi
  log "prefix smoke"
  HOST="$HOST" PORT="$PORT" \
    GSP_NUM_GROUPS=1 GSP_PROMPTS_PER_GROUP=4 GSP_SYSTEM_PROMPT_LEN=8192 \
    GSP_QUESTION_LEN=512 GSP_OUTPUT_LEN=256 MAX_CONCURRENCY=4 \
    MIN_CACHE_HIT_PCT=0 \
    LOG_FILE="$dir/prefix_smoke.log" JSON_FILE="$dir/prefix_smoke.jsonl" \
    ./prefix.sh || log "WARNING: prefix smoke failed; continuing"
  if [ "$QUICK" = 1 ]; then
    log "quick mode: full prefix skipped"
    return 0
  fi
  log "prefix full"
  HOST="$HOST" PORT="$PORT" \
    GSP_NUM_GROUPS=${GSP_NUM_GROUPS:-1} \
    GSP_PROMPTS_PER_GROUP=${GSP_PROMPTS_PER_GROUP:-32} \
    GSP_SYSTEM_PROMPT_LEN=${GSP_SYSTEM_PROMPT_LEN:-32768} \
    GSP_QUESTION_LEN=${GSP_QUESTION_LEN:-4096} \
    GSP_OUTPUT_LEN=${GSP_OUTPUT_LEN:-512} \
    MAX_CONCURRENCY=${MAX_CONCURRENCY:-16} \
    MIN_CACHE_HIT_PCT="$MIN_CACHE_HIT_PCT" \
    LOG_FILE="$dir/prefix.log" JSON_FILE="$dir/prefix.jsonl" \
    ./prefix.sh || log "WARNING: prefix full failed; continuing to GSM8K"
}

run_gsm8k() {
  local dir=$1
  if [ "$SKIP_GSM8K" = 1 ]; then
    log "gsm8k skipped"
    return 0
  fi
  log "gsm8k num_questions=$NUM_QUESTIONS"
  HOST="$HOST" PORT="$PORT" \
    OUT_DIR="$dir" LOG_FILE="$dir/gsm8k.log" \
    NUM_QUESTIONS="$NUM_QUESTIONS" NUM_THREADS="$NUM_THREADS" \
    MAX_TOKENS="$MAX_TOKENS" THINKING="$THINKING" MIN_SCORE=0 \
    ./gsm8k.sh || return 1
}

run_phase() {
  local phase=$1
  local replay=$2
  local dir="$OUT_DIR/$phase"
  mkdir -p "$dir"
  log "===== phase $phase (replay=$replay) ====="
  local extra=()
  if [ "$replay" = 1 ]; then
    extra=(ENABLE_DECODER_REPLAY=1)
  fi
  start_server "$dir/server.log" "${extra[@]}" || return 1
  run_sanity "$dir" || return 1
  run_probe "$dir" || return 1
  run_prefix "$dir" || return 1
  run_gsm8k "$dir" || return 1
  stop_server
}

write_env() {
  {
    echo "stamp=$STAMP"
    echo "git=$(git -C "$SGLANG_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "branch=$(git -C "$SGLANG_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "model=$MODEL_PATH"
    echo "kernel_path=$KERNEL_PATH"
    echo "topology=TP$TP_SIZE DP$DP_SIZE (attention TP $((TP_SIZE / DP_SIZE)))"
    echo "decode_graph=$([ "$DISABLE_CUDA_GRAPH" = 1 ] && echo disabled-eager || echo enabled)"
    echo "moe_a2a_backend=$MOE_A2A_BACKEND"
    echo "mem_fraction=$MEM_FRACTION"
    echo "listen=$HOST:$PORT"
    echo "gsm8k=questions=$NUM_QUESTIONS threads=$NUM_THREADS max_tokens=$MAX_TOKENS thinking=$THINKING"
    echo "encoder_replay=$ENABLE_ENCODER_REPLAY"
    echo "decoder_replay=$ENABLE_DECODER_REPLAY"
  } >"$OUT_DIR/env.txt"
}

main() {
  log "out_dir=$OUT_DIR"
  if [ "$DRY_RUN" = 1 ]; then
    local prefix_plan="prefix smoke + full"
    if [ "$QUICK" = 1 ]; then
      prefix_plan="prefix smoke only"
    fi
    cat <<EOF
plan:
  1. preflight: model / kernel wheel / port free / both launch commands
  2. baseline phase (replay off): sanity -> probe -> $prefix_plan -> gsm8k N=$NUM_QUESTIONS
  3. replay phase (ENABLE_DECODER_REPLAY=1): same steps
  4. compare + report -> $OUT_DIR/report.md
nothing is started in --dry-run.
EOF
    return 0
  fi
  preflight
  mkdir -p "$OUT_DIR"
  write_env
  if [ "$PREFLIGHT_ONLY" = 1 ]; then
    log "preflight ok (model, kernel wheel, port, both launch commands)"
    return 0
  fi
  run_phase baseline 0 || die "baseline phase failed; see $OUT_DIR/baseline"
  if [ "$SKIP_REPLAY" = 1 ]; then
    log "replay phase skipped"
  else
    run_phase replay 1 || die "replay phase failed; see $OUT_DIR/replay"
  fi
  if [ -f "$OUT_DIR/baseline/probe.json" ] && [ -f "$OUT_DIR/replay/probe.json" ]; then
    python3 "$SCRIPT_DIR/compare_replay.py" \
      "$OUT_DIR/baseline/probe.json" "$OUT_DIR/replay/probe.json" \
      >"$OUT_DIR/compare.txt" 2>&1 || true
  fi
  NUM_QUESTIONS="$NUM_QUESTIONS" MIN_CACHE_HIT_PCT="$MIN_CACHE_HIT_PCT" \
    GSM8K_MIN="$GSM8K_MIN" GSM8K_DELTA="$GSM8K_DELTA" \
    python3 "$SCRIPT_DIR/summarize_replay.py" "$OUT_DIR" "$STAMP" \
    >"$OUT_DIR/report.md"
  log "report: $OUT_DIR/report.md"
  cat "$OUT_DIR/report.md"
}

main
