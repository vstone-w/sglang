"""NPU parity: the DSpark static verify-epilogue accept vs the eager accept.

The epilogue folds the accept computation into the NPU verify graph's tail
hook and copies results into preallocated static buffers. This guards that,
on identical inputs, the buffered route (``_static_epilogue`` +
``read_accept``) matches the eager route (``AcceptGreedy`` +
``build_accept_outs``) field-for-field, and that a second step does not read
stale buffer contents from the first. A divergence here surfaces on NPU as
silently wrong accept lengths or tokens during graph replay.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.speculative.dspark.dspark_accept import AcceptGreedy
from sglang.srt.speculative.dspark_components.dspark_verify import (
    DsparkVerifyEpilogue,
    build_accept_outs,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=120, suite="nightly-1-npu-a3", nightly=True)

DEVICE = "npu"
BS = 5
STRIDE = 4  # verify_num_draft_tokens; gamma = STRIDE - 1
VOCAB = 64
HIDDEN = 16


class _SingleRankTpGroup:
    world_size = 1
    rank_in_group = 0


class _NoopBroadcastTpGroup:
    """world_size > 1 with a no-op broadcast: the rank-0 view, so the packed
    accept broadcast path runs and its roundtrip is observable."""

    world_size = 2
    rank_in_group = 0

    def broadcast(self, tensor, src=0):
        return tensor


def _run_step(epilogue, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = torch.randint(
        0, VOCAB, (BS * STRIDE,), dtype=torch.int64, generator=gen
    ).to(DEVICE)
    logits = torch.randn(BS * STRIDE, VOCAB, generator=gen, dtype=torch.float32).to(
        DEVICE
    )
    hidden = torch.randn(BS * STRIDE, HIDDEN, generator=gen, dtype=torch.float32).to(
        DEVICE
    )
    seq_lens = torch.randint(1, 32, (BS,), generator=gen, dtype=torch.int64).to(DEVICE)

    forward_batch = SimpleNamespace(
        batch_size=BS, input_ids=input_ids, seq_lens=seq_lens
    )
    out = SimpleNamespace(next_token_logits=logits, hidden_states=hidden)
    epilogue.begin_static_step(BS, armed=False)
    epilogue._static_epilogue(out, forward_batch)
    buffered = epilogue.read_accept(BS)

    candidates = input_ids.view(BS, STRIDE)
    correct_len, bonus, cap_trim_lens = AcceptGreedy.execute(
        candidates=candidates,
        target_logits=logits,
        verify_num_draft_tokens=STRIDE,
        cutoff_verify_lens=None,
    )
    eager = build_accept_outs(
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
    return buffered, eager


class TestNpuDsparkEpilogueParity(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)
        cls.epilogue = DsparkVerifyEpilogue(
            max_bs=8,
            verify_num_draft_tokens=STRIDE,
            device=DEVICE,
            tp_sync=SpecTpSync(tp_group=_SingleRankTpGroup()),
        )

    def test_static_epilogue_matches_eager_accept(self):
        for seed in (11, 29):
            with self.subTest(seed=seed):
                buffered, eager = _run_step(self.epilogue, seed)
                for field in (
                    "correct_len",
                    "bonus",
                    "cap_trim_lens",
                    "commit_lens",
                    "new_seq_lens",
                    "out_tokens",
                ):
                    self.assertTrue(
                        torch.equal(getattr(buffered, field), getattr(eager, field)),
                        f"field {field} differs between epilogue and eager "
                        f"accept at seed={seed}",
                    )

    def test_packed_accept_broadcast_roundtrip(self):
        """build_accept_outs packs the three decision tensors into one int64
        broadcast when the sync site is enabled; the packed roundtrip must
        return exactly the no-packing result (values and dtypes), or every
        TP>1 deployment silently corrupts the accept decision."""
        gen = torch.Generator(device="cpu").manual_seed(41)
        # int32 correct_len exercises the narrow-side of the int64 roundtrip.
        correct_len = torch.randint(
            0, STRIDE - 1, (BS,), dtype=torch.int32, generator=gen
        ).to(DEVICE)
        bonus = torch.randint(0, VOCAB, (BS,), dtype=torch.int64, generator=gen).to(
            DEVICE
        )
        cap_trim_lens = torch.randint(
            0, STRIDE - 1, (BS,), dtype=torch.int64, generator=gen
        ).to(DEVICE)
        prefix_lens = torch.randint(1, 32, (BS,), dtype=torch.int64, generator=gen).to(
            DEVICE
        )
        draft_tokens = torch.randint(
            0, VOCAB, (BS, STRIDE - 1), dtype=torch.int64, generator=gen
        ).to(DEVICE)
        kwargs = dict(
            correct_len=correct_len,
            bonus=bonus,
            cap_trim_lens=cap_trim_lens,
            prefix_lens=prefix_lens,
            draft_tokens=draft_tokens,
            gamma=STRIDE - 1,
            verify_num_draft_tokens=STRIDE,
            sync_site=SpecTpSyncSite.DSPARK_ACCEPT_GRAPH,
        )
        packed_sync = SpecTpSync(tp_group=_NoopBroadcastTpGroup())
        self.assertTrue(packed_sync.enabled(SpecTpSyncSite.DSPARK_ACCEPT_GRAPH))
        packed = build_accept_outs(tp_sync=packed_sync, **kwargs)
        plain = build_accept_outs(
            tp_sync=SpecTpSync(tp_group=_SingleRankTpGroup()), **kwargs
        )
        for field in (
            "correct_len",
            "bonus",
            "cap_trim_lens",
            "commit_lens",
            "new_seq_lens",
            "out_tokens",
        ):
            self.assertTrue(
                torch.equal(getattr(packed, field), getattr(plain, field)),
                f"field {field} differs between packed and plain broadcast",
            )
        for field in ("correct_len", "bonus", "cap_trim_lens"):
            self.assertIs(getattr(packed, field).dtype, getattr(plain, field).dtype)


if __name__ == "__main__":
    unittest.main()
