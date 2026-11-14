#!/usr/bin/env bash
# CPU-only sweep of the DSV4.1 / bounded-replay unit files.
#
# Each file runs in its own pytest process: several tests extract ASTs and
# patch imports, which contaminates later modules when all files share one
# process (collection errors that do not reproduce in isolation).
#
# Known baseline failure: test_npu_dsv41_low_ratio_backend.py fails on clean
# base 6309017ada in this environment (torch_npu/version dependent), so it is
# reported but not counted as a regression.

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SGLANG_ROOT=${SGLANG_ROOT:-/home/w00889861/dsv4/sglang_bouned_replay/sglang}
PER_FILE_TIMEOUT=${PER_FILE_TIMEOUT:-900}
KNOWN_BASELINE_FAILURE="test_npu_dsv41_low_ratio_backend.py"

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
export PYTHONPATH="$SGLANG_ROOT/python:${PYTHONPATH:-}"

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
  test/registered/unit/layers/test_dsv41_candidate_blocks.py
  test/registered/unit/layers/test_dsv41_indexer_tp_shard.py
  test/registered/unit/layers/test_dsv41_late_layer_tail_readers.py
  test/registered/unit/model_executor/test_encoder_swa_replay.py
  test/registered/unit/model_executor/test_num_token_non_padded_localization.py
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
    if [ "$(basename "$f")" = "$KNOWN_BASELINE_FAILURE" ]; then
      known=$((known + 1))
      echo "  (known pre-existing failures on base 6309017ada; not counted)"
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
