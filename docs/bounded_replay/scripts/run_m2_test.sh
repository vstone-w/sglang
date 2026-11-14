#!/bin/bash
# One-shot A3 M2 smoke: eager decode, decoder replay, DSpark off, two waves of
# one >128-token fresh prompt per DP rank (capture then replay).
set -u

SCRIPTS=/home/w00889861/dsv4/scripts/dsv41
LOG=/tmp/m2_server.log
OUT=/tmp/m2_probe.log
SERVER_PID=""

npu_hbm_mb() {
  npu-smi info 2>/dev/null | grep -oE '[0-9]+ */ *65536' \
    | awk -F'/' '{gsub(/ /, "", $1); print $1}'
}

free_ok() {
  local vals n max
  vals=$(npu_hbm_mb)
  n=$(printf '%s\n' "$vals" | sed '/^$/d' | wc -l)
  [ "$n" -eq 16 ] || return 1
  max=$(printf '%s\n' "$vals" | sort -n | tail -1)
  [ -n "$max" ] && [ "$max" -le 6000 ]
}

wait_free() {
  local streak=0
  while [ "$streak" -lt 3 ]; do
    if free_ok; then streak=$((streak + 1)); else streak=0; fi
    echo "$(date +%H:%M:%S) wait streak=$streak max_hbm=$(npu_hbm_mb | sort -n | tail -1)" >> "$LOG.wait"
    [ "$streak" -ge 3 ] && return 0
    sleep 30
  done
}

stop_server() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
  fi
  # The setsid wrapper may be gone while the launcher lives; sweep its group.
  local actual
  actual=$(pgrep -of "[p]ython3 -m sglang.launch_server" 2>/dev/null || true)
  if [ -n "$actual" ]; then
    kill -TERM -- "-$actual" 2>/dev/null || true
    sleep 3
    kill -KILL -- "-$actual" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap 'stop_server' EXIT INT TERM

wait_free
echo "$(date +%H:%M:%S) starting M2 smoke" > "$LOG.wait"
( cd "$SCRIPTS" && setsid env KILL_STALE=0 HOST=127.0.0.1 PORT=8001 \
    TP_SIZE=${M2_TP_SIZE:-16} DP_SIZE=${M2_DP_SIZE:-16} MOE_A2A_BACKEND=deepep MEM_FRACTION=${M2_MEM_FRACTION:-0.86} \
    DISABLE_CUDA_GRAPH=${M2_DISABLE_GRAPH:-1} ENABLE_DECODER_REPLAY=1 ENABLE_ENCODER_REPLAY=0 \
    ENABLE_DSPARK=${M2_DSPARK:-0} MAX_RUNNING_REQUESTS=${M2_MAX_RUNNING_REQUESTS:-16} \
    SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS=${M2_MAX_ROWS:-2048} \
    SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \
    EXTRA_ARGS='--model-loader-extra-config {"num_threads":2}' \
    ${M2_RUN_SCRIPT:-./run_dsv4.1.sh} > "$LOG" 2>&1 & echo $! > /tmp/m2_server.pid )
sleep 2
SERVER_PID=$(cat /tmp/m2_server.pid 2>/dev/null || true)

waited=0
while [ "$waited" -lt 1500 ]; do
  if curl -sf --max-time 2 http://127.0.0.1:8001/health >/dev/null 2>&1; then
    SERVER_PID=$(pgrep -of "python3 -m sglang.launch_server" 2>/dev/null || echo "$SERVER_PID")
    echo "$(date +%H:%M:%S) ready after ${waited}s pid=$SERVER_PID" >> "$LOG.wait"
    break
  fi
  # The setsid wrapper may exit before the launcher; a missing launcher (after
  # the env-setup grace) is the real startup failure.
  if [ "$waited" -ge 10 ] && ! pgrep -f "[p]ython3 -m sglang.launch_server" >/dev/null 2>&1; then
    echo "$(date +%H:%M:%S) server died during startup" >> "$LOG.wait"
    tail -40 "$LOG" >> "$LOG.wait"
    exit 1
  fi
  sleep 5
  waited=$((waited + 5))
done
if ! curl -sf --max-time 2 http://127.0.0.1:8001/health >/dev/null 2>&1; then
  echo "$(date +%H:%M:%S) startup timeout" >> "$LOG.wait"
  exit 1
fi

python3 /tmp/m2_probe.py ${M2_REQUESTS:-16} > "$OUT" 2>&1
rc=$?
echo "$(date +%H:%M:%S) probe rc=$rc" >> "$LOG.wait"
cat "$OUT" >> "$LOG.wait"
stop_server
echo "$(date +%H:%M:%S) done" >> "$LOG.wait"
exit "$rc"
