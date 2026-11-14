"""CPU contract checks for the NPU decoder SWA bounded replay tail.

The production methods are AST-extracted and run with CPU tensors and mocks so
no torch_npu or ACL is needed. These tests pin the row layout, the batch swap,
the index-K guard, and the tail-only SWA write path; they do not replace the
on-device parity and accuracy tests.
"""

import ast
import runpy
import sys
import types
import unittest
from copy import copy
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import torch

from sglang.srt.hardware_backend.npu.dsv4.ring_layout import (
    RingPageTableCache,
    ring_row_loc,
)

ROOT = Path(__file__).resolve().parents[5]
BACKEND = (
    ROOT
    / "python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend_a3.py"
)
TAIL_MODULE = (
    ROOT / "python/sglang/srt/layers/attention/dsv4/late_layer_tail.py"
)

register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=6, stage="base-a", runner_config="cpu")


class LateLayerTail:
    """Field-for-field stand-in for the production msgspec Struct."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _async_h2d(values, dtype, device):
    return [torch.tensor(value, dtype=dtype, device=device) for value in values]


def _load_tail_module():
    namespace = dict(
        torch=torch,
        List=List,
        Optional=Optional,
        CaptureHiddenMode=SimpleNamespace(FULL="full"),
        SWA_WINDOW=128,
        async_h2d=_async_h2d,
    )
    wanted = {
        "_tail_rows",
        "_late_layer_whole_extend",
        "late_layer_whole_extend",
        "late_layer_num_tokens",
        "late_layer_tail_lens",
        "late_layer_tail_layout",
        "_whole_extend_tail_layout",
        "apply_encoder_row_floor",
    }
    body = [
        ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0)
    ]
    for node in ast.parse(TAIL_MODULE.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            if node.name == "late_layer_num_tokens":
                # The backend method of the same name calls this helper; keep
                # the helper under a distinct test name to avoid self-recursion.
                node.name = "_tail_num_tokens"
            body.append(node)
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(TAIL_MODULE), "exec"), namespace)
    return namespace


def _extract_methods(source_path, class_names, method_names):
    tree = ast.parse(source_path.read_text())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in class_names:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in method_names:
                    found[item.name] = item
    missing = method_names - set(found)
    assert not missing, missing
    return found


def _load_backend(tail_ns):
    methods = _extract_methods(
        BACKEND,
        {"DeepseekV4AscendAttnBackend", "LowRatioCompressAscendBackendMixin"},
        {
            "_tail_forward_batch_view",
            "_build_late_layer_tail_metadata",
            "_decoder_tail_needs_full_extend",
            "_tail_compress_rows",
            "late_layer_num_tokens",
            "enter_late_layer_tail",
            "exit_late_layer_tail",
            "get_swa_out_cache_loc",
            "write_tail_source_kv",
            "_dsv41_index_k_store",
            "_dsv41_window_indices",
            "_dsv41_a3_attention_kwargs",
            "_dsv41_a3_kv_lengths",
            "_build_dspark_sparse_indices",
            "_draft_logical_pages",
            "_draft_ring_page_table",
            "_draft_swa_ring_size",
        },
    )
    namespace = dict(
        torch=torch,
        SimpleNamespace=SimpleNamespace,
        LateLayerTail=LateLayerTail,
        copy=copy,
        list=list,
        zip=zip,
        is_npu_arch35=lambda: False,
        ring_row_loc=ring_row_loc,
        BuildBlockSeqLensCausal=None,
        BuildDsparkSwaPageIndices=None,
        BuildDsparkSwaRingIndices=None,
        ComputeDsparkWindowGather=None,
    )
    namespace.update(tail_ns)
    body = [
        ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0)
    ]
    for name, node in methods.items():
        if name == "late_layer_num_tokens":
            node.name = "_late_layer_num_tokens_method"
        body.append(node)
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(BACKEND), "exec"), namespace)
    namespace["late_layer_num_tokens"] = namespace["_tail_num_tokens"]
    renamed = {"late_layer_num_tokens": "_late_layer_num_tokens_method"}
    return {
        name: namespace[renamed.get(name, name)] for name in methods
    }


class _Pool:
    full_to_swa_index_mapping = object()

    def translate_loc_from_full_to_swa(self, loc):
        return loc + 1000


class _BackendBase:
    def __init__(self, dspark=False, window=True):
        self._dsv4_sliding_window_size = 128
        self._is_dspark_algorithm = dspark
        self.enable_decoder_swa_bounded_replay = True
        self.forward_metadata = "full"
        self.token_to_kv_pool = _Pool()
        # The request window is the only A3 path that stages the decoder floor;
        # the default fixture represents that validated combination.
        self.request_window = object() if window else None
        self.tail_forward_metadata = None
        self.tail_source_kv_layer_id = None
        self._build_npu_forward_metadata = lambda view: SimpleNamespace()
        self._draft_page_tables = RingPageTableCache()
        self.window_activations = 0

    def _activate_request_window(self):
        # The enter/exit contract rebinds the request window to the metadata
        # the next layer will read; a stale layout corrupts prewrite staging.
        self.window_activations += 1

    def _dsv41_live_rows_for(self, fm, forward_batch, t):
        return None


def _make_backend(methods, *, dspark=False, window=True):
    cls = type("BackendUnderTest", (_BackendBase,), dict(methods))
    return cls(dspark=dspark, window=window)


def _batch(*, extend_lens, seq_lens=None, return_logprob=False, logprob_starts=None,
           capture_hidden_mode=None, encoder_swa_row_floor=None,
           encoder_swa_compress_rows=None):
    total = sum(extend_lens)
    seq_lens = extend_lens if seq_lens is None else seq_lens
    return SimpleNamespace(
        encoder_swa_row_floor=encoder_swa_row_floor,
        encoder_swa_compress_rows=encoder_swa_compress_rows,
        positions=torch.arange(total, dtype=torch.int64),
        out_cache_loc=torch.arange(total, dtype=torch.int64) + 10,
        input_ids=torch.zeros(total, dtype=torch.int64),
        extend_seq_lens=torch.tensor(extend_lens, dtype=torch.int32),
        extend_seq_lens_cpu=list(extend_lens),
        extend_prefix_lens=torch.zeros(len(extend_lens), dtype=torch.int64),
        extend_prefix_lens_cpu=[0] * len(extend_lens),
        extend_start_loc=torch.tensor([0] * len(extend_lens)),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int64),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int64),
        forward_mode=SimpleNamespace(
            is_extend_without_speculative=lambda: True,
            is_decode=lambda: False,
            is_idle=lambda: False,
        ),
        return_logprob=return_logprob,
        extend_logprob_start_lens_cpu=logprob_starts,
        capture_hidden_mode=capture_hidden_mode,
    )


class TestDecoderTailLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tail = _load_tail_module()

    def test_mixed_requests_take_their_last_window(self):
        indices, lens, floor = self.tail["late_layer_tail_layout"](
            extend_lens_cpu=[256, 50],
            seq_lens_cpu=[256, 50],
            tail_len=128,
            device=torch.device("cpu"),
        )
        self.assertEqual(lens, [128, 50])
        self.assertEqual(indices[:2].tolist(), [128, 129])
        self.assertEqual(indices[-1].item(), 305)
        self.assertEqual(floor[:2].tolist(), [128, 128])
        self.assertEqual(floor[-1].item(), 0)

    def test_prompt_logprob_runs_the_whole_extend(self):
        whole = self.tail["late_layer_whole_extend"](
            _batch(
                extend_lens=[256],
                return_logprob=True,
                logprob_starts=[10],
            ),
            dspark=False,
        )
        self.assertEqual(whole, [True])
        indices, lens, _ = self.tail["late_layer_tail_layout"](
            extend_lens_cpu=[256],
            seq_lens_cpu=[256],
            tail_len=128,
            device=torch.device("cpu"),
            whole_extend=whole,
        )
        self.assertEqual(lens, [256])
        self.assertEqual(indices[0].item(), 0)

    def test_encoder_row_floor_clamps_the_tail(self):
        start = torch.tensor([96, 96, 0], dtype=torch.int32)
        out = self.tail["apply_encoder_row_floor"](
            start,
            token_indices=torch.tensor([0, 1, 2]),
            row_floor=torch.tensor([50, 120, 0, 0]),
        )
        self.assertEqual(out.tolist(), [96, 120, 0])

    def test_dspark_full_capture_keeps_the_tail(self):
        batch = _batch(extend_lens=[256], capture_hidden_mode="full")
        self.assertIsNone(
            self.tail["late_layer_whole_extend"](batch, dspark=True)
        )
        self.assertEqual(
            self.tail["late_layer_whole_extend"](batch, dspark=False), [True]
        )


class TestDecoderTailBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tail = _load_tail_module()
        cls.methods = _load_backend(cls.tail)

    def test_tail_view_swaps_rows_and_lengths(self):
        batch = _batch(extend_lens=[10])
        view = self.methods["_tail_forward_batch_view"](
            batch,
            positions=torch.arange(3),
            out_cache_loc=torch.tensor([7, 8, 9]),
            tail_lens_cpu=[3],
            tail_start_lens=[7],
        )
        self.assertEqual(view.positions.tolist(), [0, 1, 2])
        self.assertEqual(view.extend_seq_lens.tolist(), [3])
        self.assertEqual(view.extend_seq_lens_cpu, [3])
        self.assertEqual(view.extend_prefix_lens.tolist(), [7])
        self.assertEqual(view.extend_prefix_lens_cpu, [7])
        self.assertEqual(view.extend_start_loc.tolist(), [0])

    def test_build_tail_metadata_restores_the_full_metadata(self):
        backend = _make_backend(self.methods)
        batch = _batch(extend_lens=[256, 50])
        tail_metadata = backend._build_late_layer_tail_metadata(batch)
        self.assertEqual(backend.forward_metadata, "full")
        tail = tail_metadata.late_layer_tail
        self.assertEqual(tail.extend_seq_lens_cpu, [128, 50])
        self.assertEqual(tail.token_indices[:2].tolist(), [128, 129])
        self.assertEqual(tail.positions[:2].tolist(), [128, 129])
        self.assertEqual(tail.full_out_cache_loc[:2].tolist(), [138, 139])
        self.assertEqual(tail.swa_out_cache_loc[:2].tolist(), [1138, 1139])
        self.assertEqual(tail_metadata.dsv41_win_floor.tolist(), [128] * 128 + [0] * 50)

    def test_single_short_extend_uses_a_contiguous_view(self):
        backend = _make_backend(self.methods)
        tail = backend._build_late_layer_tail_metadata(
            _batch(extend_lens=[40])
        ).late_layer_tail
        self.assertEqual(tail.extend_seq_lens_cpu, [40])
        self.assertEqual(tail.contiguous_start, 0)

    def test_enter_exit_restore_positions_and_locations(self):
        backend = _make_backend(self.methods)
        batch = _batch(extend_lens=[256])
        backend.tail_forward_metadata = backend._build_late_layer_tail_metadata(
            batch
        )
        saved = backend.enter_late_layer_tail(batch)
        self.assertEqual(batch.positions.tolist(), list(range(128, 256)))
        self.assertEqual(batch.out_cache_loc.tolist(), list(range(138, 266)))
        backend.exit_late_layer_tail(saved, batch)
        self.assertEqual(batch.positions.tolist(), list(range(256)))
        self.assertEqual(batch.out_cache_loc.tolist(), list(range(10, 266)))
        self.assertEqual(backend.forward_metadata, "full")
        self.assertEqual(backend.window_activations, 2)

    def test_late_layer_num_tokens_selects_tail_rows(self):
        backend = _make_backend(self.methods)
        batch = _batch(extend_lens=[256, 50])
        self.assertEqual(backend.late_layer_num_tokens(batch), 178)
        backend.enable_decoder_swa_bounded_replay = False
        self.assertIsNone(backend.late_layer_num_tokens(batch))

    def test_paged_tail_falls_back_when_the_floor_is_non_zero(self):
        # Without the request window the A3 operator cannot express the floor,
        # so a long extend must run every late-layer row instead of the tail.
        backend = _make_backend(self.methods, window=False)
        batch = _batch(extend_lens=[256, 50])
        self.assertIsNone(backend.late_layer_num_tokens(batch))
        self.assertIsNone(backend._build_late_layer_tail_metadata(batch))
        # A batch whose extends fit the window needs no floor at all.
        short = _make_backend(self.methods, window=False)
        short_batch = _batch(extend_lens=[100, 50])
        self.assertEqual(short.late_layer_num_tokens(short_batch), 150)
        self.assertIsNotNone(short._build_late_layer_tail_metadata(short_batch))

    def test_whole_extend_floor_depends_on_the_prefix(self):
        backend = _make_backend(self.methods, window=False)
        # Prompt-logprob rows run the whole extend; a fresh long prompt has a
        # replay start of 0, so its late layers are already complete.
        fresh = _batch(
            extend_lens=[256],
            seq_lens=[256],
            return_logprob=True,
            logprob_starts=[0],
        )
        self.assertEqual(backend.late_layer_num_tokens(fresh), 256)
        self.assertIsNotNone(backend._build_late_layer_tail_metadata(fresh))
        # A prefix longer than one window is not covered by this extend.
        prefixed = _batch(
            extend_lens=[256],
            seq_lens=[512],
            return_logprob=True,
            logprob_starts=[0],
        )
        self.assertIsNone(backend.late_layer_num_tokens(prefixed))
        self.assertIsNone(backend._build_late_layer_tail_metadata(prefixed))

    def test_tail_metadata_applies_the_encoder_row_floor(self):
        backend = _make_backend(self.methods)
        seen = {}

        def build(view):
            seen["floor"] = view.encoder_swa_row_floor
            return SimpleNamespace()

        backend._build_npu_forward_metadata = build
        batch = _batch(
            extend_lens=[10],
            seq_lens=[10],
            encoder_swa_row_floor=torch.full((10,), 7, dtype=torch.int64),
        )
        metadata = backend._build_late_layer_tail_metadata(batch)
        self.assertEqual(metadata.dsv41_win_floor.tolist(), [7] * 10)
        # The window layout must floor at the same combined value; a larger
        # window would stage history the late layers never computed.
        self.assertEqual(seen["floor"].tolist(), [7] * 10)

    def test_tail_window_floor_reaches_the_window_layout(self):
        # The late layers run only the tail, so rows before the decoder replay
        # start have no late-layer SWA KV. The request-window layout must be
        # floored there; staging them reads uninitialized ring rows (the
        # combined decoder+encoder path crashed on this).
        backend = _make_backend(self.methods)
        seen = {}

        def build(view):
            seen["floor"] = view.encoder_swa_row_floor
            return SimpleNamespace()

        backend._build_npu_forward_metadata = build
        backend._build_late_layer_tail_metadata(_batch(extend_lens=[256, 50]))
        self.assertEqual(seen["floor"].tolist(), [128] * 128 + [0] * 50)

    def test_tail_view_remaps_the_fold_compress_rows(self):
        # encoder_swa_compress_rows indexes the full folded extend; the tail
        # metadata must get tail-local indices, or the ratio metadata gathers
        # out of bounds (the combined decoder+encoder path crashed on this).
        backend = _make_backend(self.methods)
        seen = {}

        def build(view):
            seen["keep"] = view.encoder_swa_compress_rows
            return SimpleNamespace()

        backend._build_npu_forward_metadata = build
        # 128 replay rows then 256 original rows: the last-128 tail is original.
        backend._build_late_layer_tail_metadata(
            _batch(
                extend_lens=[384],
                seq_lens=[384],
                encoder_swa_compress_rows=torch.arange(128, 384, dtype=torch.int64),
            )
        )
        self.assertEqual(seen["keep"].tolist(), list(range(128)))
        # 128 replay rows then 50 original rows: the 128-row tail starts at
        # folded row 50, so its first 78 rows are replay rows.
        backend._build_late_layer_tail_metadata(
            _batch(
                extend_lens=[178],
                seq_lens=[178],
                encoder_swa_compress_rows=torch.arange(128, 178, dtype=torch.int64),
            )
        )
        self.assertEqual(seen["keep"].tolist(), list(range(78, 128)))

    def test_window_floor_clamps_the_left_edge(self):
        backend = _make_backend(self.methods)
        fm = SimpleNamespace(
            dsv41_win_floor=torch.tensor([128, 128], dtype=torch.int32)
        )
        backend.forward_metadata = fm
        batch = SimpleNamespace(positions=torch.tensor([128, 255]))
        indices, lengths = backend._dsv41_window_indices(batch)
        self.assertEqual(lengths.reshape(-1).tolist(), [1, 128])
        self.assertEqual(indices[0, 0, 0].item(), 128)
        self.assertEqual(indices[0, 0, 1].item(), -1)
        self.assertEqual(indices[1, 0, 0].item(), 128)

    def test_index_k_store_guard(self):
        backend = _make_backend(self.methods)
        backend.tail_source_kv_layer_id = 20
        backend._dsv41_index_k_store(SimpleNamespace(layer_id=20), None)
        with self.assertRaises(RuntimeError):
            backend._dsv41_index_k_store(SimpleNamespace(layer_id=21), None)

    def test_tail_source_writes_compressed_kv_and_index_k(self):
        backend = _make_backend(self.methods)
        calls = []
        backend._low_ratio_compress = lambda layer, x, pos, fb: calls.append(
            "compress"
        ) or "latent"
        backend._dsv41_index_k_store = lambda layer, latent, fb=None: calls.append(
            ("k_store", latent)
        )
        layer = SimpleNamespace(
            layer_id=20,
            compress_ratio=1,
            compressor=object(),
            indexer=SimpleNamespace(owns_k=True),
        )
        backend.write_tail_source_kv(
            layer=layer,
            x=torch.zeros(4, 8),
            positions=torch.arange(4),
            forward_batch=_batch(extend_lens=[4]),
        )
        self.assertEqual(calls, ["compress", ("k_store", "latent")])
        self.assertEqual(backend.tail_source_kv_layer_id, 20)

    def test_swa_write_locations_use_the_tail(self):
        backend = _make_backend(self.methods)
        batch = _batch(extend_lens=[256])
        backend.tail_forward_metadata = backend._build_late_layer_tail_metadata(
            batch
        )
        backend.forward_metadata = backend.tail_forward_metadata
        loc = backend.get_swa_out_cache_loc(batch)
        self.assertEqual(loc.tolist()[:2], [1138, 1139])
        self.assertEqual(loc.numel(), 128)


class TestEncoderWindowAddressing(unittest.TestCase):
    """The staged window is addressed relatively, the compressed side absolutely.

    A3 derives the sliding window from ``seqused_ori_kv``; a folded extend
    stages compact rows from a nonzero replay start, so passing the full
    sequence length there reads the wrong rows (verified on device). The
    window side therefore takes the layout's staged span, while the compressed
    side keeps the real sequence length.
    """

    @classmethod
    def setUpClass(cls):
        cls.tail = _load_tail_module()
        cls.methods = _load_backend(cls.tail)

    @classmethod
    def _backend(cls, seq_lens, window_lens=None):
        backend = _make_backend(cls.methods)
        fm = SimpleNamespace(actual_seq_lengths_q_pa=torch.tensor([0, 2]))
        if window_lens is not None:
            fm.dsv41_window_kv_lens = torch.tensor(window_lens, dtype=torch.int32)
        backend.forward_metadata = fm
        batch = SimpleNamespace(seq_lens=torch.tensor(seq_lens, dtype=torch.int32))
        return backend, batch

    def test_window_lens_drive_the_original_side_only(self):
        backend, batch = self._backend([100, 200], window_lens=[7, 9])
        kwargs = backend._dsv41_a3_attention_kwargs(batch, ratio=2)
        self.assertEqual(kwargs["seqused_ori_kv"].tolist(), [7, 9])
        self.assertEqual(kwargs["seqused_cmp_kv"].tolist(), [50, 100])
        self.assertEqual(kwargs["cmp_residual_kv"].tolist(), [0, 0])

    def test_no_window_keeps_the_full_sequence_length(self):
        backend, batch = self._backend([100, 200])
        kwargs = backend._dsv41_a3_attention_kwargs(batch, ratio=2)
        self.assertEqual(kwargs["seqused_ori_kv"].tolist(), [100, 200])
        self.assertEqual(kwargs["seqused_cmp_kv"].tolist(), [50, 100])


class TestDsparkDraftRing(unittest.TestCase):
    """The draft ring's window is bounded by the request's ``valid_from``.

    Encoder replay rebuilds only the target window; the draft ring still holds
    the previous slot owner's rows below the request's written-from mark, so
    the draft window must never reach past it.
    """

    @classmethod
    def setUpClass(cls):
        cls.tail = _load_tail_module()
        cls.methods = _load_backend(cls.tail)

    def test_valid_from_clamps_the_draft_window_start(self):
        backend = _make_backend(self.methods)
        backend.token_to_kv_pool = SimpleNamespace(
            swa_valid_from=torch.tensor([0, 6], dtype=torch.int64)
        )
        indices = backend._build_dspark_sparse_indices(
            seq_lens=torch.tensor([10]),
            req_pool_indices=torch.tensor([1]),
            out_cache_loc=torch.arange(2),
            block_size=2,
        )
        # start = max(10 - 128, valid_from[1] = 6) = 6; context = 4.
        self.assertEqual(indices.shape[0], 2)
        self.assertEqual(indices[0, :6].tolist(), [6, 7, 8, 9, 10, 11])
        self.assertTrue((indices[:, 6:] == -1).all())
        self.assertTrue(torch.equal(indices[0], indices[1]))

    def test_missing_valid_from_keeps_the_sliding_window(self):
        backend = _make_backend(self.methods)
        backend.token_to_kv_pool = SimpleNamespace()
        indices = backend._build_dspark_sparse_indices(
            seq_lens=torch.tensor([10]),
            req_pool_indices=torch.tensor([1]),
            out_cache_loc=torch.arange(2),
            block_size=2,
        )
        self.assertEqual(indices[0, :12].tolist(), list(range(12)))
        self.assertTrue((indices[:, 12:] == -1).all())

    def test_graph_swa_loc_prefers_the_captured_buffer(self):
        backend = _make_backend(self.methods)
        backend.token_to_kv_pool = SimpleNamespace(swa_req_ring_size=8)
        backend.forward_metadata = SimpleNamespace(swa_loc=torch.tensor([11, 12]))
        batch = SimpleNamespace(
            out_cache_loc=torch.arange(2),
            positions=torch.tensor([3, 4]),
            req_pool_indices=torch.tensor([0, 1]),
            forward_mode=SimpleNamespace(is_idle=lambda: False),
        )
        self.assertEqual(
            backend.get_swa_out_cache_loc(batch).tolist(), [11, 12]
        )

    def test_eager_ring_swa_loc_uses_the_slot_ring(self):
        backend = _make_backend(self.methods)
        backend.token_to_kv_pool = SimpleNamespace(swa_req_ring_size=8)
        backend.forward_metadata = SimpleNamespace()
        batch = SimpleNamespace(
            out_cache_loc=torch.arange(2),
            positions=torch.tensor([3, 4]),
            req_pool_indices=torch.tensor([0, 1]),
            forward_mode=SimpleNamespace(is_idle=lambda: False),
        )
        self.assertEqual(backend.get_swa_out_cache_loc(batch).tolist(), [3, 12])

    def test_ring_page_table_maps_positions_into_the_slot_ring(self):
        backend = _make_backend(self.methods)
        backend.page_size = 4
        backend.token_to_kv_pool = SimpleNamespace(swa_req_ring_size=8)
        logical_pages = backend._draft_logical_pages(torch.tensor([4]), 2)
        # ring_pages = 2; the width covers committed length + the draft block.
        self.assertEqual(logical_pages, 2)
        table = backend._draft_ring_page_table(
            req_pool_indices=torch.tensor([2]),
            logical_pages=logical_pages,
        )
        # slot 2 -> pages 4, 5.
        self.assertEqual(table.tolist(), [[4, 5]])


if __name__ == "__main__":
    unittest.main()
