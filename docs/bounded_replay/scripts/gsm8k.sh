#!/usr/bin/env bash
# GSM8K accuracy test for the DeepSeek-V4.1 server.
#
# Uses the same sgl-eval path as the repo's DSV4.1 CI (python/sglang/test/
# dsv41_flash_common.py). Falls back to the legacy five-shot scorer only when
# sgl-eval is not installed; set STRICT_SGL_EVAL=1 to forbid the fallback.
#
# Run against a server started by run_dsv4.1.sh:
#   ./gsm8k.sh
#   PORT=8002 NUM_QUESTIONS=200 ./gsm8k.sh     # second server, quick pass

set -uo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

SGLANG_ROOT=${SGLANG_ROOT:-/home/w00889861/dsv4/sglang_bouned_replay/sglang}
export PYTHONPATH="$SGLANG_ROOT/python:${PYTHONPATH:-}"

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8001}
BASE_URL=${BASE_URL:-http://$HOST:$PORT/v1}

MODEL_PATH=${MODEL_PATH:-/mnt/paas/weights/DeepSeek-V4.1-Flash-1017-w4a8}
MODEL_NAME=${MODEL_NAME:-$MODEL_PATH}

NUM_QUESTIONS=${NUM_QUESTIONS:-1319}
NUM_THREADS=${NUM_THREADS:-64}
MAX_TOKENS=${MAX_TOKENS:-4096}
# auto: keep the server's thinking default; 1/0: force the chat-template key.
THINKING=${THINKING:-auto}
TEMPERATURE=${TEMPERATURE:-}
TOP_P=${TOP_P:-}
REASONING_EFFORT=${REASONING_EFFORT:-}
MIN_SCORE=${MIN_SCORE:-0}

NUM_SHOTS=${NUM_SHOTS:-5}
DATA_PATH=${DATA_PATH:-/home/w00889861/dsv4/dataset/test.jsonl}
# NS-shape jsonl ({problem, expected_answer}) for sgl-eval; generated from
# DATA_PATH when unset so no GitHub download is needed.
NS_DATASET=${NS_DATASET:-}
# auto: sgl-eval (CLI, else in-process) when available; otherwise legacy.
# cli / inproc / legacy force one harness.
HARNESS=${HARNESS:-auto}
SGL_EVAL_BIN=${SGL_EVAL_BIN:-}
STRICT_SGL_EVAL=${STRICT_SGL_EVAL:-0}
FLUSH_CACHE=${FLUSH_CACHE:-1}
SHOW_SERVER_STATS=${SHOW_SERVER_STATS:-1}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT_DIR=${OUT_DIR:-$SCRIPT_DIR/outputs}
mkdir -p "$OUT_DIR"
LOG_FILE=${LOG_FILE:-$OUT_DIR/gsm8k_$(date +%Y%m%d_%H%M%S).log}

if [ -z "$SGL_EVAL_BIN" ]; then
  if command -v sgl-eval >/dev/null 2>&1; then
    SGL_EVAL_BIN=$(command -v sgl-eval)
  elif [ -x /home/w00889861/dsv4/sgl_eval_venv/bin/sgl-eval ]; then
    SGL_EVAL_BIN=/home/w00889861/dsv4/sgl_eval_venv/bin/sgl-eval
  fi
fi

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
    print(f"[gsm8k] dspark avg_spec_accept_length per dp: {accept}")
else:
    print("[gsm8k] dspark accept length not reported (DSpark off or no drafts yet)")
' 2>/dev/null || true
}

run_sgl_eval() {
  OUT_DIR="$OUT_DIR" BASE_URL="$BASE_URL" MODEL_NAME="$MODEL_NAME" \
  NUM_QUESTIONS="$NUM_QUESTIONS" NUM_THREADS="$NUM_THREADS" \
  MAX_TOKENS="$MAX_TOKENS" THINKING="$THINKING" \
  TEMPERATURE="$TEMPERATURE" TOP_P="$TOP_P" \
  REASONING_EFFORT="$REASONING_EFFORT" MIN_SCORE="$MIN_SCORE" \
  python3 - <<'PY'
import os
from types import SimpleNamespace

from sglang.test.sgl_eval_utils import run_sgl_eval


def env_float(name):
    value = os.environ.get(name, "").strip()
    return float(value) if value else None


thinking = os.environ.get("THINKING", "auto").strip().lower()
chat_template_kwargs = None
if thinking in ("1", "0"):
    chat_template_kwargs = {"thinking": thinking == "1"}

args = SimpleNamespace(
    eval_name="gsm8k",
    base_url=os.environ["BASE_URL"],
    model=os.environ["MODEL_NAME"],
    num_examples=int(os.environ["NUM_QUESTIONS"]),
    num_threads=int(os.environ["NUM_THREADS"]),
    max_tokens=int(os.environ["MAX_TOKENS"]),
    temperature=env_float("TEMPERATURE"),
    top_p=env_float("TOP_P"),
    reasoning_effort=os.environ.get("REASONING_EFFORT") or None,
    chat_template_kwargs=chat_template_kwargs,
    sgl_eval_out_dir=os.path.join(os.environ["OUT_DIR"], "sgl_eval"),
)
metrics = run_sgl_eval(args)
score = metrics["score"]
min_score = float(os.environ.get("MIN_SCORE") or 0)
print(f"[gsm8k] sgl-eval score={score:.4f} (MIN_SCORE={min_score})")
if score < min_score:
    raise SystemExit(f"[gsm8k] FAIL: score {score:.4f} < MIN_SCORE {min_score}")
PY
}

ensure_ns_dataset() {
  local src="$DATA_PATH"
  local dst="$OUT_DIR/gsm8k_ns.jsonl"
  if [ -n "$NS_DATASET" ]; then
    printf '%s\n' "$NS_DATASET"
    return 0
  fi
  if [ ! -f "$src" ]; then
    return 0
  fi
  if [ ! -f "$dst" ] || [ "$src" -nt "$dst" ]; then
    python3 - "$src" "$dst" <<'PY'
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
count = 0
with open(src) as fin, open(dst, "w") as fout:
    for line in fin:
        if not line.strip():
            continue
        row = json.loads(line)
        problem = row.get("problem") or row["question"]
        answer = row.get("expected_answer")
        if answer is None:
            text = row["answer"].split("####")[-1].strip().replace(",", "")
            try:
                value = float(text)
                answer = int(value) if value.is_integer() else value
            except ValueError:
                answer = text
        fout.write(json.dumps({"problem": problem, "expected_answer": answer}) + "\n")
        count += 1
print(f"[gsm8k] wrote {count} examples to {dst}", file=sys.stderr)
PY
    [ -f "$dst" ] || return 0
  fi
  printf '%s\n' "$dst"
}

run_sgl_eval_cli() {
  local dataset
  dataset=$(ensure_ns_dataset)
  local eval_args=(
    run gsm8k
    --base-url "$BASE_URL"
    --num-examples "$NUM_QUESTIONS"
    --num-threads "$NUM_THREADS"
    --max-tokens "$MAX_TOKENS"
    --out-dir "$OUT_DIR/sgl_eval_cli"
  )
  if [ -n "$dataset" ]; then
    eval_args+=(--from-dataset "$dataset")
  fi
  if [ -n "$MODEL_NAME" ]; then
    eval_args+=(--model "$MODEL_NAME")
  fi
  if [ -n "$TEMPERATURE" ]; then
    eval_args+=(--temperature "$TEMPERATURE")
  fi
  if [ -n "$TOP_P" ]; then
    eval_args+=(--top-p "$TOP_P")
  fi
  if [ -n "$REASONING_EFFORT" ]; then
    eval_args+=(--reasoning-effort "$REASONING_EFFORT")
  fi
  if [ "$THINKING" = 1 ]; then
    eval_args+=(--thinking)
  elif [ "$THINKING" = 0 ]; then
    eval_args+=(--no-thinking)
  fi
  "$SGL_EVAL_BIN" "${eval_args[@]}"
  local status=$?
  if [ "$status" -ne 0 ]; then
    return "$status"
  fi
  python3 - "$OUT_DIR/sgl_eval_cli" "$MIN_SCORE" <<'PY'
import glob
import json
import os
import sys

root, min_score = sys.argv[1], float(sys.argv[2])
paths = sorted(
    glob.glob(os.path.join(root, "sgl_eval_gsm8k_*", "metrics.json")),
    key=os.path.getmtime,
)
if not paths:
    raise SystemExit("[gsm8k] FAIL: sgl-eval wrote no metrics.json")
with open(paths[-1]) as f:
    metrics = json.load(f)
score = float(metrics["aggregate"]["score"])
print(f"[gsm8k] sgl-eval score={score:.4f} (MIN_SCORE={min_score}) metrics={paths[-1]}")
if score < min_score:
    raise SystemExit(f"[gsm8k] FAIL: score {score:.4f} < MIN_SCORE {min_score}")
PY
}

run_legacy_gsm8k() {
  echo "[gsm8k] WARNING: sgl_eval is not installed; using the deprecated five-shot scorer."
  echo "[gsm8k]          install it with: pip install 'sgl-eval==0.1.3'"
  if [ "$STRICT_SGL_EVAL" = 1 ]; then
    echo "[gsm8k] STRICT_SGL_EVAL=1, refusing the legacy fallback" >&2
    return 3
  fi
  local extra=()
  if [ -n "$TEMPERATURE" ]; then
    extra+=(--temperature "$TEMPERATURE")
  fi
  python3 -m sglang.test.few_shot_gsm8k \
    --num-questions "$NUM_QUESTIONS" \
    --num-shots "$NUM_SHOTS" \
    --max-new-tokens "$MAX_TOKENS" \
    --parallel "$NUM_THREADS" \
    --host "$HOST" \
    --port "$PORT" \
    --data-path "$DATA_PATH" \
    "${extra[@]}"
}

main() {
  if ! curl -sf --max-time 10 "http://$HOST:$PORT/health" >/dev/null; then
    echo "[gsm8k] ERROR: no server at http://$HOST:$PORT (start run_dsv4.1.sh first)" >&2
    return 2
  fi
  if [ "$FLUSH_CACHE" = 1 ]; then
    curl -sf --max-time 60 "http://$HOST:$PORT/flush_cache" >/dev/null || \
      echo "[gsm8k] warning: flush_cache failed" >&2
  fi

  echo "[gsm8k] base_url=$BASE_URL model=$MODEL_NAME"
  echo "[gsm8k] num_questions=$NUM_QUESTIONS num_threads=$NUM_THREADS max_tokens=$MAX_TOKENS"
  echo "[gsm8k] thinking=$THINKING temperature=${TEMPERATURE:-<harness default>} top_p=${TOP_P:-<harness default>} reasoning_effort=${REASONING_EFFORT:-<harness default>}"

  local have_cli=0
  local have_inproc=0
  if [ -n "$SGL_EVAL_BIN" ] && [ -x "$SGL_EVAL_BIN" ]; then
    have_cli=1
  fi
  if python3 -c "import sgl_eval" >/dev/null 2>&1; then
    have_inproc=1
  fi

  case "$HARNESS" in
    auto)
      if [ "$have_cli" = 1 ]; then
        echo "[gsm8k] harness: sgl-eval CLI ($SGL_EVAL_BIN)"
        run_sgl_eval_cli
      elif [ "$have_inproc" = 1 ]; then
        echo "[gsm8k] harness: sgl-eval in-process (repo CI path)"
        run_sgl_eval
      else
        run_legacy_gsm8k
      fi
      ;;
    cli)
      if [ "$have_cli" != 1 ]; then
        echo "[gsm8k] ERROR: HARNESS=cli but no sgl-eval binary found (set SGL_EVAL_BIN)" >&2
        return 4
      fi
      echo "[gsm8k] harness: sgl-eval CLI ($SGL_EVAL_BIN)"
      run_sgl_eval_cli
      ;;
    inproc)
      if [ "$have_inproc" != 1 ]; then
        echo "[gsm8k] ERROR: HARNESS=inproc but sgl_eval is not importable" >&2
        return 4
      fi
      echo "[gsm8k] harness: sgl-eval in-process (repo CI path)"
      run_sgl_eval
      ;;
    sgl_eval)
      if [ "$have_cli" = 1 ]; then
        echo "[gsm8k] harness: sgl-eval CLI ($SGL_EVAL_BIN)"
        run_sgl_eval_cli
      elif [ "$have_inproc" = 1 ]; then
        echo "[gsm8k] harness: sgl-eval in-process (repo CI path)"
        run_sgl_eval
      else
        echo "[gsm8k] ERROR: HARNESS=sgl_eval but no sgl-eval found" >&2
        return 4
      fi
      ;;
    legacy)
      run_legacy_gsm8k
      ;;
    *)
      echo "[gsm8k] ERROR: HARNESS must be auto|cli|inproc|sgl_eval|legacy (got $HARNESS)" >&2
      return 5
      ;;
  esac
  local status=$?

  if [ "$SHOW_SERVER_STATS" = 1 ]; then
    show_server_stats
  fi
  return "$status"
}

main 2>&1 | tee "$LOG_FILE"
exit "${PIPESTATUS[0]}"
