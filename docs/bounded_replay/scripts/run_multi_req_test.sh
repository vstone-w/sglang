#!/bin/bash
# Priority 2 smoke: encoder replay prefill with several running requests per
# attention-DP rank.  32 concurrent requests over DP16 => 2 requests/rank.
set -u

SCRIPTS=/home/w00889861/dsv4/scripts/dsv41
LOG=/tmp/multi_req_server.log
OUT=/tmp/multi_req_probe.log
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
  SERVER_PID=""
}
trap 'stop_server' EXIT INT TERM

wait_free
echo "$(date +%H:%M:%S) starting multi-request encoder replay smoke" > "$LOG.wait"
( cd "$SCRIPTS" && setsid env KILL_STALE=0 HOST=127.0.0.1 PORT=8001 \
    TP_SIZE=16 DP_SIZE=16 MOE_A2A_BACKEND=deepep MEM_FRACTION=0.90 \
    DISABLE_CUDA_GRAPH=1 ENABLE_DECODER_REPLAY=0 ENABLE_ENCODER_REPLAY=1 \
    ENABLE_DSPARK=0 MAX_RUNNING_REQUESTS=32 \
    EXTRA_ARGS='--model-loader-extra-config {"num_threads":2}' \
    ./run_dsv4.1.sh > "$LOG" 2>&1 & echo $! > /tmp/multi_req_server.pid )
sleep 2
SERVER_PID=$(cat /tmp/multi_req_server.pid 2>/dev/null || true)

waited=0
while [ "$waited" -lt 1500 ]; do
  if curl -sf --max-time 2 http://127.0.0.1:8001/health >/dev/null 2>&1; then
    SERVER_PID=$(pgrep -of "python3 -m sglang.launch_server" 2>/dev/null || echo "$SERVER_PID")
    echo "$(date +%H:%M:%S) ready after ${waited}s pid=$SERVER_PID" >> "$LOG.wait"
    break
  fi
  if [ -n "$SERVER_PID" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
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

: > "$OUT"
echo "$(date +%H:%M:%S) hit-accuracy 32 concurrent (2 per rank)" >> "$LOG.wait"
python3 "$SCRIPTS/e3_hit_accuracy.py" 32 512 >> "$OUT" 2>&1
hit_rc=$?
echo "$(date +%H:%M:%S) e3_probe 32" >> "$LOG.wait"
python3 "$SCRIPTS/e3_probe.py" 32 >> "$OUT" 2>&1
probe_rc=$?
echo "$(date +%H:%M:%S) results: hit_rc=$hit_rc probe_rc=$probe_rc" >> "$LOG.wait"
cat "$OUT" >> "$LOG.wait"
stop_server
echo "$(date +%H:%M:%S) done" >> "$LOG.wait"
[ "$hit_rc" -eq 0 ] && [ "$probe_rc" -eq 0 ]
