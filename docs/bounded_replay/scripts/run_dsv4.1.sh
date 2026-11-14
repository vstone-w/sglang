#!/usr/bin/env bash
# Launch DeepSeek-V4.1-Flash on Ascend A3 (16 dies, single node).
#
# Bounded replay verification (see docs/bounded_replay/04_verification.md):
#   ENABLE_DECODER_REPLAY=1 DP_SIZE=1 ./run_dsv4.1.sh
#
# Every setting is an environment override with the default shown.
# DRY_RUN=1 runs the preflight checks and prints the command without launching.

unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY no_proxy NO_PROXY

SGLANG_ROOT=${SGLANG_ROOT:-/home/w00889861/dsv4/sglang_bouned_replay/sglang}
MODEL_PATH=${MODEL_PATH:-/mnt/paas/weights/DeepSeek-V4.1-Flash-1017-w4a8}

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8001}

TP_SIZE=${TP_SIZE:-16}
# Attention DP. Bounded replay M1 rejects DP attention on NPU, so replay runs
# need DP_SIZE=1.
DP_SIZE=${DP_SIZE:-8}
MEM_FRACTION=${MEM_FRACTION:-}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-16}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-4096}
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-131072}
DECODE_GRAPH_BS=${DECODE_GRAPH_BS:-"1 2"}
# TP16 shards the expert intermediate to K=144, which the WeightNz grouped
# matmul used by graph capture rejects (K must be divisible by 64); eager
# decode avoids that path.
DISABLE_CUDA_GRAPH=${DISABLE_CUDA_GRAPH:-0}

ENABLE_DSPARK=${ENABLE_DSPARK:-1}
DSPARK_BLOCK_SIZE=${DSPARK_BLOCK_SIZE:-5}
# The bundled DSpark head ships BF16 (mtp.* is FLOAT in the ModelSlim
# description), so the draft must load unquantized.
DRAFT_QUANT=${DRAFT_QUANT:-unquant}
ENABLE_DECODER_REPLAY=${ENABLE_DECODER_REPLAY:-0}
ENABLE_ENCODER_REPLAY=${ENABLE_ENCODER_REPLAY:-0}
DISABLE_RADIX_CACHE=${DISABLE_RADIX_CACHE:-0}

# DSpark keeps the draft verify graphs resident; 0.96 OOMs during capture.
if [ -z "$MEM_FRACTION" ]; then
  if [ "$ENABLE_DSPARK" = 1 ] && [ "$DISABLE_CUDA_GRAPH" = 0 ]; then
    MEM_FRACTION=0.84
  else
    MEM_FRACTION=0.96
  fi
fi
# The A3 canonical MoE backend: with "none" every rank must hold all experts,
# so TP16 shards the expert intermediate to K=144 and the grouped matmul
# rejects it (K must divide 64). DeepEP shards experts by EP=TP (K=2304).
MOE_A2A_BACKEND=${MOE_A2A_BACKEND:-deepep}

KILL_STALE=${KILL_STALE:-1}
FRESH_JIT_CACHE=${FRESH_JIT_CACHE:-0}
DSV4_DEBUG=${DSV4_DEBUG:-0}
DRY_RUN=${DRY_RUN:-0}
EXTRA_ARGS=${EXTRA_ARGS:-}

log() { echo "[run_dsv4.1] $*"; }
die() { echo "[run_dsv4.1] ERROR: $*" >&2; exit 1; }

[ -d "$SGLANG_ROOT/python/sglang" ] || die "SGLANG_ROOT=$SGLANG_ROOT has no python/sglang"
[ -d "$MODEL_PATH" ] || die "MODEL_PATH=$MODEL_PATH does not exist"

# DSV4.1 has o_groups=8; without DP attention each rank needs 8 // TP groups.
if [ "$DP_SIZE" = 1 ] && [ "$TP_SIZE" -gt 8 ]; then
  die "DSV4.1 needs attn_tp<=8 when DP attention is off (o_groups=8); set TP_SIZE=8 (got $TP_SIZE)"
fi

# Encoder replay prefill groups window rows per request, so a DP rank may host
# several running requests; the pool is sized from MAX_RUNNING_REQUESTS.

if [ "$ENABLE_DECODER_REPLAY" = 1 ] || [ "$ENABLE_ENCODER_REPLAY" = 1 ]; then
  # The replay DP path (enter_late_layer_dp) is only wired for attention TP 1.
  if [ "$DP_SIZE" -gt 1 ]; then
    [ "$DP_SIZE" = "$TP_SIZE" ] || die "DP attention with replay needs DP_SIZE == TP_SIZE (attention TP 1); got DP_SIZE=$DP_SIZE TP_SIZE=$TP_SIZE"
    case "$MOE_A2A_BACKEND" in
      none|deepep) ;;
      *) die "DP attention with replay needs MOE_A2A_BACKEND=none or deepep (got $MOE_A2A_BACKEND)" ;;
    esac
  fi
  if [ "$ENABLE_ENCODER_REPLAY" = 1 ] && [ "$DISABLE_RADIX_CACHE" = 1 ]; then
    die "encoder replay rebuilds windows on prefix hits and needs the radix cache; unset DISABLE_RADIX_CACHE"
  fi
fi

SERVER_ARGS=(
  --model-path "$MODEL_PATH"
  --page-size 128
  --tp-size "$TP_SIZE"
  --trust-remote-code
  --attention-backend dsv4
  --device npu
  --watchdog-timeout 9000
  --host "$HOST"
  --port "$PORT"
  --mem-fraction-static "$MEM_FRACTION"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --max-prefill-tokens "$MAX_PREFILL_TOKENS"
  --kv-cache-dtype auto
  --moe-a2a-backend "$MOE_A2A_BACKEND"
  --reasoning-parser deepseek-v4
  --language-model-only
  --json-model-override-args '{"architectures": ["DeepseekV4ForCausalLM"], "quantization_config": null}'
)

if [ "$DISABLE_CUDA_GRAPH" = 1 ]; then
  SERVER_ARGS+=(--disable-cuda-graph)
else
  read -r -a DECODE_BS_ARR <<< "$DECODE_GRAPH_BS"
  SERVER_ARGS+=(--cuda-graph-bs-decode "${DECODE_BS_ARR[@]}")
fi

if [ "$DP_SIZE" -gt 1 ]; then
  SERVER_ARGS+=(--enable-dp-attention --dp-size "$DP_SIZE" --enable-dp-lm-head)
fi
if [ "$DISABLE_RADIX_CACHE" = 1 ]; then
  SERVER_ARGS+=(--disable-radix-cache)
fi
if [ "$ENABLE_DECODER_REPLAY" = 1 ]; then
  SERVER_ARGS+=(--enable-decoder-swa-bounded-replay)
fi
if [ "$ENABLE_ENCODER_REPLAY" = 1 ]; then
  SERVER_ARGS+=(--enable-encoder-swa-bounded-replay)
fi
if [ "$ENABLE_DSPARK" = 1 ]; then
  SERVER_ARGS+=(
    --speculative-algorithm DSPARK
    --speculative-draft-model-path "$MODEL_PATH"
    --speculative-draft-model-quantization "$DRAFT_QUANT"
    --speculative-draft-kv-cache-dtype bf16
    --speculative-dspark-block-size "$DSPARK_BLOCK_SIZE"
  )
fi
read -r -a EXTRA_ARGS_ARR <<< "$EXTRA_ARGS"
SERVER_ARGS+=("${EXTRA_ARGS_ARR[@]}")

radix_state=$([ "$DISABLE_RADIX_CACHE" = 1 ] && echo off || echo on)
log "repo: $SGLANG_ROOT"
log "model: $MODEL_PATH"
log "listen: $HOST:$PORT  tp=$TP_SIZE dp=$DP_SIZE  dspark=$ENABLE_DSPARK  radix=$radix_state"
log "decoder_replay=$ENABLE_DECODER_REPLAY encoder_replay=$ENABLE_ENCODER_REPLAY debug=$DSV4_DEBUG"
printf '[run_dsv4.1] command:'
printf ' %q' python3 -m sglang.launch_server "${SERVER_ARGS[@]}"
printf '\n'

if [ "$DRY_RUN" = 1 ]; then
  log "DRY_RUN=1, not launching"
  exit 0
fi

if [ "$KILL_STALE" = 1 ]; then
  pkill -9 -f "sglang.launch_server" 2>/dev/null || true
fi
if [ "${KILL_ALL_PYTHON:-0}" = 1 ]; then
  pkill -9 python 2>/dev/null || true
fi

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null 2>&1 || true
sysctl -w vm.swappiness=0 >/dev/null 2>&1 || true
sysctl -w kernel.numa_balancing=0 >/dev/null 2>&1 || true

for env_script in \
  /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash \
  /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash \
  /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
  /usr/local/Ascend/nnal/atb/set_env.sh; do
  if [ -f "$env_script" ]; then
    # shellcheck disable=SC1090
    source "$env_script"
  fi
done

if [ "$FRESH_JIT_CACHE" = 1 ]; then
  TORCH_EXTENSIONS_DIR=$(mktemp -d /tmp/dsv41-jit.XXXXXX)
else
  TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/tmp/dsv41-jit}
  mkdir -p "$TORCH_EXTENSIONS_DIR"
fi
export TORCH_EXTENSIONS_DIR
export CANN_OP_BUILDER_NO_FPRINT_CACHE=1

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_SET_CPU_AFFINITY=1
export TASK_QUEUE_ENABLE=1
export INF_NAN_MODE_FORCE_DISABLE=1

export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-128}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-lo}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
export DEEPEP_NORMAL_LONG_SEQ_ROUND=16
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=2048
# DeepEP on A3 (matches the canonical DSV4 NPU recipe).
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export DEEPEP_HCCL_BUFFSIZE=${DEEPEP_HCCL_BUFFSIZE:-2500}
# Decode graph capture runs a low-latency dispatch over the tail bucket; the
# workspace must cover at least one 128-row bucket or aclnn tiling fails.
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-256}
export DEEPEP_HYBRID_DEPLOYMENT=1

# Default serving behavior for DSV4.1 reasoning.
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=max

# Skip GPU-only fused paths that have no A3 kernel.
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export SGLANG_DSV4_FP4_EXPERTS=True
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_OPT_FP8_WO_A_GEMM=False

export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_GRAPH_REPLAY_SYNC=1
export SGLANG_DSV4_GRAPH_MAX_SEQ=131072
export SGLANG_SHALLOW_FUSED_ROPE_FQ4=0
export SGLANG_COMPRESSOR_PREFILL_USE_TRITON=1
export SGLANG_COMPRESSOR_DECODE_USE_TRITON=1
export SGLANG_RAGGED_VERIFY_MODE=static

if [ "$DSV4_DEBUG" = 1 ]; then
  export ASCEND_LAUNCH_BLOCKING=1
  export SGLANG_DSV4_GRAPH_META_DEBUG=1
  export SGLANG_DSV4_DBG_ATTN=1
fi

# The DSV4.1 A3 path needs the paired sgl-kernel-npu operators
# (lightning_indexer_v2 / compressor_v2); keep them in a private target dir so
# the shared site-packages stays untouched.
KERNEL_PATH=${KERNEL_PATH:-/home/w00889861/dsv4/sgl_kernel_npu_2026.11.0}
if [ -n "$KERNEL_PATH" ] && [ -d "$KERNEL_PATH" ]; then
  export PYTHONPATH="$KERNEL_PATH:$SGLANG_ROOT/python:${PYTHONPATH:-}"
else
  export PYTHONPATH="$SGLANG_ROOT/python:${PYTHONPATH:-}"
fi

exec python3 -m sglang.launch_server "${SERVER_ARGS[@]}"
