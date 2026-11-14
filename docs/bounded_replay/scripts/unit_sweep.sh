#!/usr/bin/env bash
# CPU-only sweep of the DSV4.1 / bounded-replay unit files.
#
# Each file runs in its own pytest process: several tests extract ASTs and
# patch imports, which contaminates later modules when all files share one
# process (collection errors that do not reproduce in isolation).
#
# Known baseline failure on clean upstream/main in this environment:
#   - test_pool_configurator.py: 18 cases fail identically on a clean
#     upstream/main checkout (pricing/floor tests, not a replay regression).
# The A5 operator paths need an sgl-kernel-npu image that ships
# kv_compress_epilog_v2; set KERNEL_PATH to that wheel if it is not the one in
# site-packages.
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SGLANG_ROOT=${SGLANG_ROOT:-/home/w00889861/dsv4/sglang_bouned_replay/sglang}
KERNEL_PATH=${KERNEL_PATH:-/home/w00889861/dsv4/sgl_kernel_npu_2026.11.0}
PER_FILE_TIMEOUT=${PER_FILE_TIMEOUT:-900}
KNOWN_BASELINE_FAILURES=(
  "test_pool_configurator.py"
)

# Vendor env scripts reference unset variables; keep nounset off while sourcing.
set +u
if [ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh >/dev/null 2>&1
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/nnal/atb/set_env.sh >/dev/null 2>&1
fi
set -u
VENDOR_PATH=""
if [ -d "$KERNEL_PATH" ]; then
  VENDOR_PATH="$KERNEL_PATH:"
fi
export PYTHONPATH="$SGLANG_ROOT/python:${VENDOR_PATH}${PYTHONPATH:-}"

cd "$SGLANG_ROOT" || exit 1

files=(
  test/registered/unit/npu/attention/test_npu_dsv41_decoder_tail.py
  test/registered/unit/npu/attention/test_dsv41_a3_backend.py
  test/registered/unit/npu/attention/test_dsv41_a3_cache.py
  test/registered/unit/npu/attention/test_dsv41_a3_dspark_attention.py
  test/registered/unit/npu/attention/test_dsv41_a3_dspark_cache.py
  test/registered/unit/npu/attention/test_dsv41_a3_dspark_draft_attention.py
  test/registered/unit/npu/attention/test_dsv41_dspark_metadata_rebase.py
  test/registered/unit/npu/attention/test_npu_dsv41_request_window.py
  test/registered/unit/npu/attention/test_npu_dsv41_low_ratio_backend.py
  test/registered/unit/npu/attention/test_npu_ascend_dsv4_backend.py
  test/registered/unit/layers/test_dsv41_candidate_blocks.py
  test/registered/unit/layers/test_dsv41_indexer_tp_shard.py
  test/registered/unit/layers/test_dsv41_late_layer_tail_readers.py
  test/registered/unit/model_executor/test_encoder_swa_replay.py
  test/registered/unit/model_executor/test_num_token_non_padded_localization.py
  test/registered/unit/model_executor/test_pool_configurator.py
  test/registered/unit/models/test_deepseek_v4_platform_dispatch.py
  test/registered/unit/npu/attention/test_npu_dsv41_tail_graph_guards.py
  test/registered/unit/npu/test_dsv41_a3_dspark_model.py
  test/registered/unit/npu/test_npu_dsv41_mhc_moe.py
  test/registered/unit/server_args/test_bounded_replay_platform_gates.py
)

fails=0
known=0
for f in "${files[@]}"; do
  log=$(mktemp)
  timeout "$PER_FILE_TIMEOUT" python3 -m pytest -q "$f" >"$log" 2>&1
  status=$?
  echo "$(basename "$f"): $(tail -1 "$log")"
  if [ "$status" -ne 0 ]; then
    if printf '%s\n' "${KNOWN_BASELINE_FAILURES[@]}" \
      | grep -qx "$(basename "$f")"; then
      known=$((known + 1))
      echo "  (known pre-existing failures on clean upstream/main; not counted)"
    else
      fails=$((fails + 1))
      echo "  FAILED: see $log"
      continue
    fi
  fi
  rm -f "$log"
done

echo "NEW_FAILED_FILES=$fails KNOWN_BASELINE_FAILED_FILES=$known"
exit $([ "$fails" -eq 0 ] && echo 0 || echo 1)
