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
        # Each request stages into a page-aligned segment, so the budget adds
        # one alignment page per request slot.
        spec = NPURequestWindowSpec.create(
            num_slots=2,
            layers=40,
            page_size=128,
            capacity=134,
            workspace_rows=4096 + 2 * 128,
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

    def test_decoder_replay_tail_factory_is_cuda_only(self):
        # The shared model calls one factory. CUDA opts in with its runner; the
        # NPU tail row-bucket graph is archived (docs/bounded_replay/tail_graph),
        # so every other platform keeps the eager tail.
        from sglang.srt.models.deepseek_v4_replay_tail import DecoderReplayTail

        model = SimpleNamespace(
            tail_writes_source_kv=True,
            late_layer_start=20,
            end_layer=40,
            dspark_layers_to_capture=None,
        )
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
                DecoderReplayTail.maybe_create(model), DecoderReplayTail
            )
        with (
            patch("sglang.srt.models.deepseek_v4_replay_tail.is_cuda", return_value=False),
            patch(max_rows, return_value=2048),
        ):
            self.assertIsNone(DecoderReplayTail.maybe_create(model))
        with (
            patch("sglang.srt.models.deepseek_v4_replay_tail.is_cuda", return_value=True),
            patch(max_rows, return_value=0),
        ):
            self.assertIsNone(DecoderReplayTail.maybe_create(model))

    def test_dsv4_kernel_num_heads_uses_the_platform_singleton(self):
        # get_platform() exposes runtime *facts* only; the kernel head width is
        # a platform method. Reading it through the facts object raised
        # "unknown platform fact" on every A3/A5 load.
        from sglang.srt.models import deepseek_v4

        attention = deepseek_v4.MqaAttentionBase.__new__(deepseek_v4.MqaAttentionBase)
        attention.attn_tp_size = 8
        attention.n_local_heads = 32
        attention.is_dsv41 = True
        with patch.object(
            type(deepseek_v4.current_platform),
            "dsv4_kernel_num_heads",
            return_value=32,
        ) as heads:
            self.assertEqual(attention._kernel_num_heads(1), 32)
        heads.assert_called_once_with(attention, 1)

    def test_hc_pre_from_prev_binds_the_attention_backend(self):
        # The replay tail path reads attn_backend in several branches; the
        # guarded tail block no longer defines it, so the function itself must
        # bind it or a replay prefill raises UnboundLocalError.
        import ast
        from pathlib import Path

        from sglang.srt.models import deepseek_v4

        tree = ast.parse(Path(deepseek_v4.__file__).read_text())
        fn = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_forward_layers_hc_pre_from_prev"
        )
        stored = {
            node.id
            for node in ast.walk(fn)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        self.assertIn("attn_backend", stored)

    def test_graph_metadata_binds_the_request_window_last(self):
        # _init_dsv41_graph_metadata reallocates dsv41_win_sparse_indices and
        # dsv41_win_topk_length; binding the request-window layout before it
        # leaves the captured graph reading zero-length SWA windows on A5.
        import ast
        from pathlib import Path

        root = Path(__file__).resolve()
        while not (root / "python/sglang").is_dir():
            root = root.parent
        source = (
            root
            / "python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py"
        ).read_text()
        fn = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef)
            and node.name == "_init_dsv4_graph_metadata"
        )
        calls = {
            node.func.attr: node.lineno
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr
            in ("_init_request_window_graph_metadata", "_init_dsv41_graph_metadata")
        }
        self.assertIn("_init_request_window_graph_metadata", calls)
        self.assertIn("_init_dsv41_graph_metadata", calls)
        self.assertGreater(
            calls["_init_request_window_graph_metadata"],
            calls["_init_dsv41_graph_metadata"],
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
