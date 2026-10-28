"""Manual spike: capture the DSpark verify-epilogue accept (static and
compact paths) in an NPU graph and verify replay correctness with
re-staged inputs.

Single-card portion of the docs/graph M0 milestone: validates
capture-safety of the accept op chain (argmax / eq / cumsum / index /
copy_ into static buffers; compact adds the triton-ascend
scatter_compact_to_strided kernel) under
torch.npu.graph(auto_dispatch_capture=True) and post-replay readability of
the epilogue buffers. The in-graph HCCL broadcast (TP > 1) and the
folds_commit KV-write chain need the full multi-card model environment and
stay with the 8p M0 run.

Run: PYTHONPATH=python python3 test/manual/ascend/spike_dspark_npu_epilogue_capture.py
"""

from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401

from sglang.kernels.ops.speculative.dspark.dspark_accept import (
    AcceptGreedy,
    AcceptSampling,
)
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    scatter_compact_to_strided_into,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    DsparkVerifyEpilogue,
    build_accept_outs,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite

DEVICE = "npu"
BS, STRIDE, VOCAB, HIDDEN = 5, 4, 64, 16
FIELDS = (
    "correct_len",
    "bonus",
    "cap_trim_lens",
    "commit_lens",
    "new_seq_lens",
    "out_tokens",
)


class _SingleRankTpGroup:
    world_size = 1
    rank_in_group = 0


def _stage(gen):
    return (
        torch.randint(0, VOCAB, (BS * STRIDE,), dtype=torch.int64, generator=gen).to(
            DEVICE
        ),
        torch.randn(BS * STRIDE, VOCAB, generator=gen, dtype=torch.float32).to(DEVICE),
        torch.randn(BS * STRIDE, HIDDEN, generator=gen, dtype=torch.float32).to(DEVICE),
        torch.randint(1, 32, (BS,), dtype=torch.int64, generator=gen).to(DEVICE),
    )


def _eager_reference(logits, input_ids, seq_lens):
    candidates = input_ids.view(BS, STRIDE)
    correct_len, bonus, cap_trim_lens = AcceptGreedy.execute(
        candidates=candidates,
        target_logits=logits,
        verify_num_draft_tokens=STRIDE,
        cutoff_verify_lens=None,
        fused_argmax=False,
    )
    return build_accept_outs(
        correct_len=correct_len,
        bonus=bonus,
        cap_trim_lens=cap_trim_lens,
        prefix_lens=seq_lens[:BS],
        draft_tokens=candidates[:, 1:].contiguous(),
        gamma=STRIDE - 1,
        verify_num_draft_tokens=STRIDE,
        tp_sync=SpecTpSync(tp_group=_SingleRankTpGroup()),
        sync_site=SpecTpSyncSite.DSPARK_ACCEPT_GRAPH,
    )


def _compact_eager_reference(
    compact_logits, compact_hidden, input_ids, seq_lens, verify_lens, draft_tokens
):
    bs = verify_lens.shape[0]
    gamma = STRIDE - 1
    strided_logits = scatter_compact_to_strided_into(
        compact=compact_logits,
        verify_lens=verify_lens,
        out=torch.empty(
            bs * STRIDE,
            compact_logits.shape[1],
            dtype=compact_logits.dtype,
            device=DEVICE,
        ),
        stride=STRIDE,
        fill_value=0.0,
    )
    candidates = scatter_compact_to_strided_into(
        compact=input_ids.view(-1, 1),
        verify_lens=verify_lens,
        out=torch.empty(bs * STRIDE, 1, dtype=input_ids.dtype, device=DEVICE),
        stride=STRIDE,
        fill_value=0,
    ).view(bs, STRIDE)
    correct_len, bonus, cap_trim_lens = AcceptGreedy.execute(
        candidates=candidates,
        target_logits=strided_logits,
        verify_num_draft_tokens=STRIDE,
        cutoff_verify_lens=verify_lens,
        fused_argmax=False,
    )
    return build_accept_outs(
        correct_len=correct_len,
        bonus=bonus,
        cap_trim_lens=cap_trim_lens,
        prefix_lens=seq_lens[:bs],
        draft_tokens=draft_tokens.view(bs, gamma),
        gamma=gamma,
        verify_num_draft_tokens=STRIDE,
        tp_sync=SpecTpSync(tp_group=_SingleRankTpGroup()),
        sync_site=SpecTpSyncSite.DSPARK_ACCEPT_GRAPH,
    )


def compact_spike():
    """Capture the compact-mode epilogue __call__ (scatter + accept) and
    replay with a re-staged ragged layout (same bucket, different
    per-request verify_lens)."""
    epilogue = DsparkVerifyEpilogue(
        max_bs=BS,
        verify_num_draft_tokens=STRIDE,
        device=DEVICE,
        tp_sync=SpecTpSync(tp_group=_SingleRankTpGroup()),
        fused_argmax=False,
    )

    # Bucket: 12 packed tokens across 5 requests (max_bs-sized slot table).
    def _stage(seed, verify_lens_list):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        num_tokens = sum(verify_lens_list)
        verify_lens = torch.tensor(verify_lens_list, dtype=torch.int64, device=DEVICE)
        return (
            verify_lens,
            torch.randint(0, VOCAB, (num_tokens,), dtype=torch.int64, generator=gen).to(
                DEVICE
            ),
            torch.randn(num_tokens, VOCAB, generator=gen, dtype=torch.float32).to(
                DEVICE
            ),
            torch.randn(num_tokens, HIDDEN, generator=gen, dtype=torch.float32).to(
                DEVICE
            ),
            torch.randint(1, 32, (BS,), dtype=torch.int64, generator=gen).to(DEVICE),
            torch.randint(
                0, VOCAB, (BS * (STRIDE - 1),), dtype=torch.int64, generator=gen
            ).to(DEVICE),
        )

    verify_lens, input_ids, logits, hidden, seq_lens, draft_tokens = _stage(
        7, [4, 2, 3, 2, 1]
    )
    static_ids, static_logits = input_ids.clone(), logits.clone()
    static_hidden, static_seq_lens = hidden.clone(), seq_lens.clone()
    static_draft_tokens = draft_tokens.clone()
    req_pool_indices = torch.zeros(BS, dtype=torch.int64, device=DEVICE)

    def _run_tail():
        epilogue(
            compact_logits=static_logits,
            compact_hidden=static_hidden,
            input_ids=static_ids,
            seq_lens=static_seq_lens,
            req_pool_indices=req_pool_indices,
            bs=BS,
        )

    # Warmup outside capture: allocates strided buffers, compiles the
    # triton-ascend scatter kernel.
    epilogue.begin_step(verify_lens, armed=False)
    epilogue.draft_tokens_buf[: BS * (STRIDE - 1)].copy_(static_draft_tokens)
    torch.npu.synchronize()
    _run_tail()
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        with torch.npu.graph(
            graph, pool=torch.npu.graph_pool_handle(), auto_dispatch_capture=True
        ):
            _run_tail()
    torch.npu.current_stream().wait_stream(stream)

    # Replay with a different ragged layout in the same bucket.
    verify_lens2, ids2, logits2, hidden2, seq_lens2, draft2 = _stage(
        99, [1, 4, 2, 3, 2]
    )
    static_ids[: ids2.shape[0]].copy_(ids2)
    static_ids[ids2.shape[0] :].zero_()
    static_logits[: logits2.shape[0]].copy_(logits2)
    static_logits[logits2.shape[0] :].zero_()
    static_hidden[: hidden2.shape[0]].copy_(hidden2)
    static_hidden[hidden2.shape[0] :].zero_()
    static_seq_lens.copy_(seq_lens2)
    epilogue.begin_step(verify_lens2, armed=False)
    epilogue.draft_tokens_buf[: BS * (STRIDE - 1)].copy_(draft2)
    graph.replay()
    torch.npu.synchronize()

    ref = _compact_eager_reference(
        static_logits,
        static_hidden,
        static_ids,
        static_seq_lens,
        verify_lens2,
        draft2,
    )
    got = epilogue.read_accept(BS)
    for field in FIELDS:
        assert torch.equal(getattr(got, field), getattr(ref, field)), (
            f"compact replay: field {field} diverges"
        )
    print("compact replay (re-staged ragged layout): all fields bit-equal: OK")


def sampling_spike():
    """Feasibility probe for folding the SAMPLING accept into the NPU
    verify graph (today sampling batches take the eager path).

    Captures the temperature-only sampling chain (SoftmaxTemp -> rand draws
    -> triton-ascend chain sampler -> bonus gather) and checks:
      1. capture and replay run;
      2. outputs are structurally valid;
      3. whether in-graph RNG advances across replays -- frozen draws would
         make every replay sample the same tokens, a correctness trap.
    """
    temperatures = torch.full((BS,), 0.8, dtype=torch.float32, device=DEVICE)
    sampling_info = SimpleNamespace(
        temperatures=temperatures,
        need_top_k_sampling=False,
        need_top_p_sampling=False,
    )
    draft_input = SimpleNamespace(max_top_k=1, uniform_top_k_value=None)

    gen = torch.Generator(device="cpu").manual_seed(3)
    candidates = (
        torch.randint(0, VOCAB, (BS, STRIDE), dtype=torch.int64, generator=gen)
        .to(DEVICE)
        .contiguous()
    )
    logits = (
        torch.randn(BS * STRIDE, VOCAB, generator=gen, dtype=torch.float32)
        .to(DEVICE)
        .contiguous()
    )
    draft_probs = torch.softmax(
        torch.randn(BS, STRIDE, VOCAB, generator=gen, dtype=torch.float32).to(DEVICE),
        dim=-1,
    ).contiguous()

    static_candidates = candidates.clone()
    static_logits = logits.clone()
    static_probs = draft_probs.clone()

    def _run():
        return AcceptSampling.torch(
            candidates=static_candidates,
            target_logits=static_logits,
            draft_probs=static_probs,
            sampling_info=sampling_info,
            draft_input=draft_input,
            gamma=STRIDE - 1,
            verify_num_draft_tokens=STRIDE,
            cutoff_verify_lens=None,
        )

    torch.npu.synchronize()
    warm = _run()
    torch.npu.synchronize()
    print(
        f"eager sampling accept: correct_len={warm[0].tolist()} "
        f"bonus_ok={bool((warm[1] < VOCAB).all())}"
    )

    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        with torch.npu.graph(
            graph, pool=torch.npu.graph_pool_handle(), auto_dispatch_capture=True
        ):
            captured_out = _run()
    torch.npu.current_stream().wait_stream(stream)

    draws = []
    for step in range(2):
        graph.replay()
        torch.npu.synchronize()
        correct_len, bonus = captured_out[0], captured_out[1]
        assert bool((correct_len >= 0).all() and (correct_len <= STRIDE - 1).all()), (
            f"replay {step}: correct_len out of range: {correct_len.tolist()}"
        )
        assert bool((bonus >= 0).all() and (bonus < VOCAB).all()), (
            f"replay {step}: bonus out of range"
        )
        draws.append(correct_len.tolist())
        print(f"replay {step}: correct_len={correct_len.tolist()} (valid)")
    if draws[0] == draws[1]:
        print(
            "WARNING: in-graph RNG is FROZEN across replays (same draws) -- "
            "folding sampling accept would repeat the same samples."
        )
    else:
        print("in-graph RNG advances across replays: fresh draws per replay.")


def main():
    torch.npu.set_device(0)
    epilogue = DsparkVerifyEpilogue(
        max_bs=8,
        verify_num_draft_tokens=STRIDE,
        device=DEVICE,
        tp_sync=SpecTpSync(tp_group=_SingleRankTpGroup()),
        fused_argmax=False,
    )

    gen = torch.Generator(device="cpu").manual_seed(7)
    input_ids, logits, hidden, seq_lens = _stage(gen)

    static_ids = input_ids.clone()
    static_logits = logits.clone()
    static_hidden = hidden.clone()
    static_seq_lens = seq_lens.clone()

    forward_batch = SimpleNamespace(
        batch_size=BS, input_ids=static_ids, seq_lens=static_seq_lens
    )
    out = SimpleNamespace(next_token_logits=static_logits, hidden_states=static_hidden)

    # Warmup outside capture, mirroring NPUCudaGraphBackend's double warmup.
    torch.npu.synchronize()
    epilogue.begin_static_step(BS, armed=False)
    epilogue._static_epilogue(out, forward_batch)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        with torch.npu.graph(
            graph, pool=torch.npu.graph_pool_handle(), auto_dispatch_capture=True
        ):
            epilogue._static_epilogue(out, forward_batch)
    torch.npu.current_stream().wait_stream(stream)

    ref = _eager_reference(logits, input_ids, seq_lens)
    got = epilogue.read_accept(BS)
    for field in FIELDS:
        assert torch.equal(getattr(got, field), getattr(ref, field)), field
    print("capture-time outputs match eager: OK")

    for step, seed in enumerate((101, 202), start=1):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        input_ids, logits, hidden, seq_lens = _stage(gen)
        static_ids.copy_(input_ids)
        static_logits.copy_(logits)
        static_hidden.copy_(hidden)
        static_seq_lens.copy_(seq_lens)
        epilogue.begin_static_step(BS, armed=False)
        graph.replay()
        torch.npu.synchronize()
        ref = _eager_reference(logits, input_ids, seq_lens)
        got = epilogue.read_accept(BS)
        for field in FIELDS:
            assert torch.equal(getattr(got, field), getattr(ref, field)), (
                f"replay step {step}: field {field} diverges"
            )
        print(f"replay step {step} (fresh inputs): all fields bit-equal: OK")

    print(
        "SPIKE PASSED: epilogue accept chain is NPUGraph-capturable and "
        "replay-correct on re-staged inputs."
    )


if __name__ == "__main__":
    main()
    compact_spike()
    sampling_spike()
