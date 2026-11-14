"""CPU references for A3 DSV4.1 mHC and clipped W8A8 activations."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.srt.hardware_backend.npu.dsv4 import dsv4_mhc as npu_mhc
from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _hc_fixture(tokens=3):
    torch.manual_seed(12)
    cfg = mhc.HcConfig(4, 20, 1e-6, 1e-20, 8, True, False)
    hc = SimpleNamespace(
        cfg=cfg,
        fn=torch.randn(24, 32),
        scale=torch.tensor([0.2, 0.4, 0.7]),
        base=torch.randn(24),
        norm=torch.nn.Identity(),
    )
    return hc, torch.randn(tokens, 4, 8, dtype=torch.bfloat16)


class TestA3Mhc(unittest.TestCase):
    def test_triplet_matches_fused_sinkhorn_formula(self):
        hc, x = _hc_fixture()
        flat = x.flatten(1).float()
        mixes = (
            F.linear(flat, hc.fn)
            / (flat.square().mean(-1, keepdim=True) + hc.cfg.rms_eps).sqrt()
        )
        pre_ref = torch.sigmoid(mixes[:, :4] * hc.scale[0] + hc.base[:4]) + hc.cfg.eps
        post_ref = 2 * torch.sigmoid(mixes[:, 4:8] * hc.scale[1] + hc.base[4:8])
        mat = (mixes[:, 8:] * hc.scale[2] + hc.base[8:]).view(-1, 4, 4)
        mat = torch.exp(mat - mat.amax(-1, keepdim=True))
        mat = mat / mat.sum(-1, keepdim=True) + hc.cfg.eps
        mat = mat / (mat.sum(-2, keepdim=True) + hc.cfg.eps)
        for _ in range(19):
            mat = mat / (mat.sum(-1, keepdim=True) + hc.cfg.eps)
            mat = mat / (mat.sum(-2, keepdim=True) + hc.cfg.eps)
        actual = npu_mhc.mix_stats_torch(hc, x)
        for result, expected in zip(actual, (pre_ref, post_ref, mat)):
            torch.testing.assert_close(result, expected)
            self.assertEqual(result.dtype, torch.float32)

    def test_combine_keeps_previous_pre_and_caches_current_stats(self):
        hc, x = _hc_fixture()
        for previous in (None, torch.rand(3, 4)):
            with self.subTest(identity=previous is None):
                state = SimpleNamespace(residual=x, pre=previous, stats=None)
                with (
                    patch.object(mhc, "_is_npu", True),
                    patch.object(mhc, "_is_npu_a3", True),
                ):
                    actual = mhc._combine(hc, state, None)
                expected = (
                    x[:, 0]
                    if previous is None
                    else (x.float() * previous.unsqueeze(-1)).sum(1).bfloat16()
                )
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                for result, reference in zip(
                    state.stats, npu_mhc.mix_stats_torch(hc, x)
                ):
                    torch.testing.assert_close(result, reference)

    def test_post_matrix_orientation_and_bf16_output(self):
        hc, residual = _hc_fixture()
        _, post_mix, comb = npu_mhc.mix_stats_torch(hc, residual)
        x = torch.randn(3, 8, dtype=torch.bfloat16)
        expected = (
            post_mix.unsqueeze(-1) * x.float().unsqueeze(1)
            + (comb.unsqueeze(-1) * residual.float().unsqueeze(2)).sum(1)
        ).bfloat16()
        with (
            patch.object(mhc, "_is_npu", True),
            patch.object(mhc, "_is_npu_a3", True),
        ):
            actual = mhc.post(hc.cfg, x, residual, post_mix, comb)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_idle_premix_has_no_kernel_call(self):
        hc, x = _hc_fixture(0)
        state = SimpleNamespace(residual=x, pre=None, stats=None)
        actual = mhc._npu_combine_premix(hc, state, x, None)
        self.assertEqual(tuple(actual.shape), (0, 8))
        self.assertEqual(
            [tuple(v.shape) for v in state.stats], [(0, 4), (0, 4), (0, 4, 4)]
        )

    def test_a5_uses_the_fused_premix_and_post(self):
        # A5 keeps the fused vendor ops; only A3 falls back to torch.
        hc, residual = _hc_fixture()
        previous = torch.rand(3, 4)
        pre = torch.rand(3, 4)
        post_mix = torch.rand(3, 4)
        comb = torch.rand(3, 4, 4)
        mixed = residual[:, 0].clone()
        state = SimpleNamespace(stats=None)
        with (
            patch.object(mhc, "_is_npu_a3", False),
            patch(
                "sgl_kernel_npu.operators.hyper_conn.mhc_pre_dpskv4",
                create=True,
                return_value=(mixed, post_mix, comb, pre),
            ) as fused,
        ):
            actual = mhc._npu_combine_premix(hc, state, residual, previous)
        self.assertIs(actual, mixed)
        self.assertIs(fused.call_args.kwargs["pre_mix"], previous)
        self.assertIs(state.stats[0], pre)

        x = torch.randn(3, 8, dtype=torch.bfloat16)
        posted = residual + 1.0
        fused_post = SimpleNamespace(mhc_post=Mock(return_value=posted))
        with (
            patch.object(mhc, "_is_npu", True),
            patch.object(mhc, "_is_npu_a3", False),
            patch.object(torch.ops, "cann_ops_transformer", fused_post, create=True),
        ):
            out = mhc.post(hc.cfg, x, residual, post_mix, comb)
        self.assertIs(out, posted)
        self.assertEqual(
            fused_post.mhc_post.call_args.args, (residual, comb, x, post_mix)
        )


if __name__ == "__main__":
    unittest.main()
