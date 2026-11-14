#!/usr/bin/env bash
# Functional prefix-cache test for DeepSeek-V4.1 bounded replay.
#
# Sends a generated-shared-prefix workload (one shared system prompt per group)
# and asserts that the radix cache actually serves prefix hits. With decoder
# replay on, the last 128 tokens of a matched prefix are reprefilled, so the
# hit rate stays near full but never reaches 100%.
#
#   ./prefix.sh                              # default 1 group x 128 prompts
#   GSP_NUM_GROUPS=2 MAX_CONCURRENCY=64 ./prefix.sh
#   MIN_CACHE_HIT_PCT=0 ./prefix.sh          # report only, never fail

set -uo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

SGLANG_ROOT=${SGLANG_ROOT:-/home/w00889861/dsv4/sglang_bouned_replay/sglang}
export PYTHONPATH="$SGLANG_ROOT/python:${PYTHONPATH:-}"

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8001}
MODEL_PATH=${MODEL_PATH:-/mnt/paas/weights/DeepSeek-V4.1-Flash-1017-w4a8}
MODEL_NAME=${MODEL_NAME:-$MODEL_PATH}

GSP_NUM_GROUPS=${GSP_NUM_GROUPS:-1}
GSP_PROMPTS_PER_GROUP=${GSP_PROMPTS_PER_GROUP:-128}
GSP_SYSTEM_PROMPT_LEN=${GSP_SYSTEM_PROMPT_LEN:-117965}
GSP_QUESTION_LEN=${GSP_QUESTION_LEN:-13107}
GSP_OUTPUT_LEN=${GSP_OUTPUT_LEN:-1024}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-128}
SEED=${SEED:-100}
WARMUP_REQUESTS=${WARMUP_REQUESTS:-1}
# Expected hit rate is ~90% at these lengths; anything near 0 means the radix
# cache is off or broken.
MIN_CACHE_HIT_PCT=${MIN_CACHE_HIT_PCT:-30}
SHOW_SERVER_STATS=${SHOW_SERVER_STATS:-1}
EXTRA_ARGS=${EXTRA_ARGS:-}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT_DIR=${OUT_DIR:-$SCRIPT_DIR/outputs}
mkdir -p "$OUT_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE=${LOG_FILE:-$OUT_DIR/prefix_$STAMP.log}
JSON_FILE=${JSON_FILE:-$OUT_DIR/prefix_$STAMP.jsonl}

show_server_stats() {
  curl -sf --max-time 30 "http://$HOST:$PORT/server_info" | python3 -c '
import json
import sys

info = json.load(sys.stdin)
states = info.get("internal_states") or []
accept = [
    state.get("avg_spec_accept_length")
    for state in states
    if state.get("avg_spec_accept_length") is not None
]
if accept:
    print(f"[prefix] dspark avg_spec_accept_length per dp: {accept}")
else:
    print("[prefix] dspark accept length not reported (DSpark off or no drafts yet)")
' 2>/dev/null || true
}

check_cache_report() {
  python3 - "$JSON_FILE" "$MIN_CACHE_HIT_PCT" <<'PY'
import json
import sys

path, min_pct = sys.argv[1], float(sys.argv[2])
last = None
with open(path) as f:
    for line in f:
        if line.strip():
            last = line
if last is None:
    raise SystemExit("[prefix] FAIL: no benchmark record written")
record = json.loads(last)
report = record.get("cache_report")
if not report:
    raise SystemExit("[prefix] FAIL: --cache-report produced no cache_report")
hit = float(report["cache_hit_rate_pct"])
print(
    "[prefix] cache hit rate: "
    f"{hit:.2f}% (cached {report['total_cached_tokens']} / "
    f"prompt {report['total_prompt_tokens']})"
)
if hit < min_pct:
    raise SystemExit(f"[prefix] FAIL: hit rate {hit:.2f}% < MIN_CACHE_HIT_PCT {min_pct}")
print("[prefix] PASS")
PY
}

main() {
  if ! curl -sf --max-time 10 "http://$HOST:$PORT/health" >/dev/null; then
    echo "[prefix] ERROR: no server at http://$HOST:$PORT (start run_dsv4.1.sh first)" >&2
    return 2
  fi
  # Measure this run's hits, not the previous run's leftovers.
  curl -sf --max-time 60 "http://$HOST:$PORT/flush_cache" >/dev/null || \
    echo "[prefix] warning: flush_cache failed" >&2

  echo "[prefix] base_url=http://$HOST:$PORT model=$MODEL_NAME"
  echo "[prefix] groups=$GSP_NUM_GROUPS prompts/group=$GSP_PROMPTS_PER_GROUP system=$GSP_SYSTEM_PROMPT_LEN question=$GSP_QUESTION_LEN output=$GSP_OUTPUT_LEN concurrency=$MAX_CONCURRENCY"
  echo "[prefix] log=$LOG_FILE jsonl=$JSON_FILE"

  read -r -a EXTRA_ARGS_ARR <<< "$EXTRA_ARGS"
  python3 -m sglang.benchmark.serving \
    --backend sglang \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups "$GSP_NUM_GROUPS" \
    --gsp-prompts-per-group "$GSP_PROMPTS_PER_GROUP" \
    --gsp-system-prompt-len "$GSP_SYSTEM_PROMPT_LEN" \
    --gsp-question-len "$GSP_QUESTION_LEN" \
    --gsp-output-len "$GSP_OUTPUT_LEN" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --seed "$SEED" \
    --warmup-requests "$WARMUP_REQUESTS" \
    --host "$HOST" \
    --port "$PORT" \
    --model "$MODEL_NAME" \
    --tokenizer "$MODEL_PATH" \
    --output-file "$JSON_FILE" \
    --cache-report \
    "${EXTRA_ARGS_ARR[@]}"
  local status=$?
  if [ "$status" -ne 0 ]; then
    echo "[prefix] FAIL: benchmark exited with status $status" >&2
    return "$status"
  fi

  check_cache_report || return $?
  if [ "$SHOW_SERVER_STATS" = 1 ]; then
    show_server_stats
  fi
}

main 2>&1 | tee "$LOG_FILE"
exit "${PIPESTATUS[0]}"
