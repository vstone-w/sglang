# Decoder tail row-bucket graph (archived)

This directory archives the NPU **decoder late-layer tail row-bucket graph**
(`NpuDecoderReplayTail`): it captures one NPU graph per tail row bucket and
replays the late layers of a prefill extend from it.

Status: **not part of the active path; re-add after the prefill breakable CUDA
graph (BCG) work is complete**. `restore_tail_graph.patch` contains the exact
reverse diff (re-add both files, the platform hook and the gate-test NPU case);
apply it from the repo root with `git apply
docs/bounded_replay/tail_graph/restore_tail_graph.patch`. The runner is only allowed to engage
when decoder replay runs without the encoder replay request window; with the
validated all-features configuration (encoder + decoder replay, DSpark, decode
graphs) `plan()` returned `(False, None)` because the request window is active
(`encoder_swa_row_floor` set / `pool.request_window` not None), so the tail
always ran eagerly. On the rebased A3 tree the graph path has not been
validated, so it was archived here instead of shipped active.

The active decoder bounded replay still trims the late layers to the tail rows;
it just runs them eagerly (see `02_npu_decoder_design.md` and the backend's
`_build_late_layer_tail_metadata` / `enter_late_layer_tail`).

Files:

- `npu_decoder_replay_tail.py` — the removed runner (restore to
  `python/sglang/srt/hardware_backend/npu/dsv4/`).
- `test_npu_dsv41_tail_graph_guards.py` — its CPU guard tests (restore to
  `test/registered/unit/npu/attention/`).

To restore, apply the patch from the repo root:

```bash
git apply docs/bounded_replay/tail_graph/restore_tail_graph.patch
```

It re-adds both files, `NPUSRTPlatform.create_decoder_replay_tail` in
`python/sglang/srt/platforms/npu.py`, and the NPU expectation in
`test/registered/unit/server_args/test_bounded_replay_platform_gates.py`.
Add the guard test to `docs/bounded_replay/scripts/unit_sweep.sh` if desired.
