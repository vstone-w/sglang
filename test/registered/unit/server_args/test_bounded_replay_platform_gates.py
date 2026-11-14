"""Platform gates and model-config validation for DSV4.1 bounded replay.

The platform predicates replace per-device checks in the resolver; these cases
pin the existing CUDA/ROCm truth table and the NPU arch gate. The config
validation keeps a missing or inconsistent V4.1 source-layer layout from
reaching model construction.
"""

import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[4]
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

from sglang.srt.arg_groups import deepseek_v4_hook
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.platforms.cuda import CudaSRTPlatform
from sglang.srt.platforms.interface import SRTPlatform
from sglang.srt.platforms.npu import NPUSRTPlatform
from sglang.srt.platforms.rocm import RocmSRTPlatform


class TestPlatformGates(unittest.TestCase):
    def test_cuda_supports_both_for_v41(self):
        platform = CudaSRTPlatform()
        self.assertTrue(
            platform.supports_decoder_swa_bounded_replay(model_type="deepseek_v41")
        )
        self.assertTrue(
            platform.supports_encoder_swa_bounded_replay(model_type="deepseek_v41")
        )
        self.assertFalse(
            platform.supports_decoder_swa_bounded_replay(model_type="deepseek_v3")
        )

    def test_base_platform_stays_conservative(self):
        platform = SRTPlatform()
        self.assertFalse(
            platform.supports_decoder_swa_bounded_replay(model_type="deepseek_v41")
        )
        self.assertFalse(
            platform.supports_encoder_swa_bounded_replay(model_type="deepseek_v41")
        )

    def test_rocm_encoder_needs_gfx95(self):
        platform = RocmSRTPlatform()
        with patch("sglang.srt.utils.device.is_gfx95_supported", return_value=False):
            self.assertFalse(
                platform.supports_encoder_swa_bounded_replay(model_type="deepseek_v41")
            )
        with patch("sglang.srt.utils.device.is_gfx95_supported", return_value=True):
            self.assertTrue(
                platform.supports_encoder_swa_bounded_replay(model_type="deepseek_v41")
            )

    def test_npu_arch_gate(self):
        platform = NPUSRTPlatform()
        # A3 and A5 share the row-based window; only the row format differs.
        for arch35 in (False, True):
            with patch(
                "sglang.srt.hardware_backend.npu.utils.is_npu_arch35",
                return_value=arch35,
            ):
                self.assertTrue(
                    platform.supports_decoder_swa_bounded_replay(
                        model_type="deepseek_v41"
                    )
                )
                self.assertTrue(
                    platform.supports_encoder_swa_bounded_replay(
                        model_type="deepseek_v41"
                    )
                )
        self.assertFalse(
            platform.supports_decoder_swa_bounded_replay(model_type="deepseek_v3")
        )

    def test_npu_request_window_nbytes_uses_row_layout(self):
        from sglang.srt.hardware_backend.npu.dsv4.npu_request_window import (
            NPURequestWindowSpec,
        )

        with patch(
            "sglang.srt.hardware_backend.npu.utils.is_npu_arch35",
            return_value=False,
        ):
            got = NPUSRTPlatform().request_window_nbytes(
                num_slots=2,
                layers=40,
                page_size=128,
                capacity=134,
                workspace_rows=4096,
                qk_nope_head_dim=512,
                qk_rope_head_dim=64,
                draft_ring_size=256,
                draft_ring_layers=3,
            )
        spec = NPURequestWindowSpec.create(
            num_slots=2,
            layers=40,
            page_size=128,
            capacity=134,
            workspace_rows=4096,
            row_dim=576,
            row_dtype=torch.bfloat16,
        )
        # DSV41NPUSingleKVPool.create_buffer pads to whole pages with floor
        # division and keeps one spare page.
        ring_pages = (2 * 256 + 128 + 1) // 128
        expected = spec.nbytes + ring_pages * 128 * 3 * 576 * 2
        self.assertEqual(got, expected)

    def test_hccp_warmup_is_safe_off_npu(self):
        import sglang.srt.hardware_backend.npu.utils as npu_utils

        with (
            patch.object(npu_utils, "_hccp_initialized", False),
            patch("torch.distributed.all_reduce") as all_reduce,
        ):
            npu_utils.ensure_hccp_initialized(device="cpu")
            all_reduce.assert_not_called()

    def test_decoder_replay_dp_moe_backends(self):
        self.assertEqual(
            CudaSRTPlatform().decoder_replay_dp_moe_backends(), ("none", "megamoe")
        )
        self.assertEqual(RocmSRTPlatform().decoder_replay_dp_moe_backends(), ("none",))
        self.assertEqual(
            NPUSRTPlatform().decoder_replay_dp_moe_backends(), ("none", "deepep")
        )
        self.assertEqual(SRTPlatform().decoder_replay_dp_moe_backends(), ())

    def test_decoder_replay_tail_factory_is_platform_seam(self):
        # The shared model calls one factory; CUDA and NPU must opt in with
        # their own runner and the existing env knob must still disable both.
        from sglang.srt.hardware_backend.npu.dsv4.npu_decoder_replay_tail import (
            NpuDecoderReplayTail,
        )
        from sglang.srt.models.deepseek_v4_replay_tail import DecoderReplayTail

        model = SimpleNamespace(
            tail_writes_source_kv=True,
            late_layer_start=20,
            end_layer=40,
            dspark_layers_to_capture=None,
        )
        self.assertIsNone(SRTPlatform().create_decoder_replay_tail(model))

        max_rows = "sglang.srt.environ.envs.SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS.get"
        with (
            patch("sglang.srt.models.deepseek_v4_replay_tail.is_cuda", return_value=True),
            patch(max_rows, return_value=2048),
            patch(
                "sglang.srt.environ.envs.SGLANG_ENABLE_GRAPH_POOL_BORROW.get",
                return_value=False,
            ),
        ):
            self.assertIsInstance(
                CudaSRTPlatform().create_decoder_replay_tail(model),
                DecoderReplayTail,
            )
        with (
            patch(
                "sglang.srt.hardware_backend.npu.utils.is_npu_arch35",
                return_value=False,
            ),
            patch(max_rows, return_value=2048),
        ):
            self.assertIsInstance(
                NPUSRTPlatform().create_decoder_replay_tail(model),
                NpuDecoderReplayTail,
            )
        with (
            patch("sglang.srt.models.deepseek_v4_replay_tail.is_cuda", return_value=True),
            patch(max_rows, return_value=0),
        ):
            self.assertIsNone(CudaSRTPlatform().create_decoder_replay_tail(model))
            self.assertIsNone(NPUSRTPlatform().create_decoder_replay_tail(model))

    def test_npu_encoder_replay_arg_validation(self):
        def cfg(**overrides):
            values = dict(
                speculative_algorithm=None,
                disaggregation_mode="null",
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(backend=Backend.DISABLED)
                ),
                max_running_requests=16,
                attn_dp_size=16,
            )
            values.update(overrides)
            return SimpleNamespace(**values)

        from sglang.srt.runtime_context import override_platform

        platform = NPUSRTPlatform()
        with override_platform(is_npu=True):
            platform.validate_encoder_swa_bounded_replay_args(cfg())
            platform.validate_encoder_swa_bounded_replay_args(
                cfg(speculative_algorithm="DSPARK")
            )
            # PD is allowed: the connector consumes the window's ring pages
            # through resume_units and the handshake checks page sizes.
            platform.validate_encoder_swa_bounded_replay_args(
                cfg(disaggregation_mode="prefill")
            )
            platform.validate_encoder_swa_bounded_replay_args(
                cfg(
                    cuda_graph_config=SimpleNamespace(
                        decode=SimpleNamespace(backend=Backend.FULL)
                    )
                )
            )
            # Prefill groups rows per request, so several running requests per
            # DP rank are allowed; the pool is sized from this value.
            platform.validate_encoder_swa_bounded_replay_args(
                cfg(max_running_requests=32)
            )


def _hf(**overrides):
    ratios = [0, 0] + [2] * 18 + [1] * 20
    values = dict(
        model_type="deepseek_v41",
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20],
        candidate_source_layer_id=20,
        compress_ratios=ratios,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _validate(hf):
    with patch.object(
        deepseek_v4_hook,
        "model_config_of",
        return_value=SimpleNamespace(hf_config=hf),
    ):
        deepseek_v4_hook._validate_decoder_swa_bounded_replay_config(object())


class TestDecoderReplayConfig(unittest.TestCase):
    def test_valid_source_layout_passes(self):
        _validate(_hf())

    def test_missing_kv_sources_raise(self):
        with self.assertRaisesRegex(ValueError, "kv_source_layer_ids"):
            _validate(_hf(kv_source_layer_ids=[]))

    def test_last_source_must_be_an_index_source(self):
        with self.assertRaisesRegex(ValueError, "index source"):
            _validate(_hf(index_source_layer_ids=[2, 8, 14]))

    def test_candidate_must_match_the_last_source(self):
        with self.assertRaisesRegex(ValueError, "candidate_source_layer_id"):
            _validate(_hf(candidate_source_layer_id=14))

    def test_last_source_must_compress(self):
        ratios = [0, 0] + [2] * 18 + [0] * 20
        with self.assertRaisesRegex(ValueError, "compress"):
            _validate(_hf(compress_ratios=ratios))


if __name__ == "__main__":
    unittest.main()
