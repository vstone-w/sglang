"""CPU contracts for the shared A3/A5 DSpark metadata lifecycle.

Execute production metadata/capture/forward methods; only external device
operators and the large cache allocation are replaced. Not an NPU kernel test.
"""

import ast
import runpy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Optional
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[5]
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")
BACKEND = (
    ROOT / "python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend_a3.py"
)


def _mode(*, idle=False):
    return SimpleNamespace(
        is_target_verify=lambda: not idle,
        is_draft_extend_v2=lambda: False,
        is_decode=lambda: False,
        is_decode_or_idle=lambda: idle,
        is_idle=lambda: idle,
        is_extend_without_speculative=lambda: False,
    )


@pytest.fixture
def build_backend():
    metadata_module = ModuleType("sglang.srt.layers.attention.dsv4.metadata")
    metadata_source = ROOT / "python/sglang/srt/layers/attention/dsv4/metadata.py"
    expand = next(
        node
        for node in ast.parse(metadata_source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "expand_index_page_table"
    )
    namespace = {"torch": torch}
    exec(
        compile(
            ast.Module(body=[expand], type_ignores=[]), str(metadata_source), "exec"
        ),
        namespace,
    )
    metadata_module.expand_index_page_table = namespace[expand.name]
    pool_module = ModuleType("sglang.srt.mem_cache.deepseek_v4_memory_pool")
    pool_module.DSV41_INDEX_PAGE_SIZE = 64
    ring_path = ROOT / "python/sglang/srt/hardware_backend/npu/dsv4/ring_layout.py"
    ring_module = ModuleType("sglang.srt.hardware_backend.npu.dsv4.ring_layout")
    exec(compile(ring_path.read_text(), str(ring_path), "exec"), ring_module.__dict__)
    dspark_path = ROOT / "python/sglang/srt/hardware_backend/npu/dsv4/dspark_ring.py"
    dspark_module = ModuleType("sglang.srt.hardware_backend.npu.dsv4.dspark_ring")
    dspark_namespace = {
        "__name__": dspark_module.__name__,
        "Optional": Optional,
        "torch": torch,
        "make_ring_page_table": ring_module.make_ring_page_table,
    }
    dspark_source = dspark_path.read_text().replace(
        "from sglang.srt.hardware_backend.npu.dsv4.ring_layout import "
        "make_ring_page_table",
        "",
    )
    exec(compile(dspark_source, str(dspark_path), "exec"), dspark_namespace)
    dspark_module.__dict__.update(dspark_namespace)

    def make(arch35, width, *, draft=False, bs=3, ring=None, pages=2):
        source = ast.parse(BACKEND.read_text())
        low_ratio = next(
            node
            for node in source.body
            if isinstance(node, ast.ClassDef)
            and node.name == "LowRatioCompressAscendBackendMixin"
        )
        main = next(
            node
            for node in source.body
            if isinstance(node, ast.ClassDef)
            and node.name == "DeepseekV4AscendAttnBackend"
        )
        names = {
            "_init_dsv4_graph_buffers",
            "_init_dsv4_graph_metadata",
            "_build_a3_capture_metadata",
            "_build_a3_draft_capture_metadata",
            "_dsv41_low_ratios",
            "_seq_lens_cpu_int_view",
            "_build_dsv4_graph_replay_ctx",
            "_refresh_graph_seq_metadata",
            "_refresh_graph_verify_rows",
            "_refresh_graph_kernel_metadata",
            "_kernel_metadata_from_parts",
            "_dsv41_a3_draft_attention_kwargs",
            "_dsv41_a3_draft_kernel_metadata",
            "_forward_swa",
            "_draft_swa_ring_size",
            "_seed_draft_graph_page_table",
            "_refresh_graph_swa_metadata_direct",
            "_refresh_graph_dspark_sparse_metadata",
            "_build_dspark_sparse_indices",
            "_draft_logical_pages",
            "_draft_ring_page_table",
            "_copy_page_table_into_graph",
            "_copy_1d_with_zero_tail",
        }
        main.bases = [ast.Name(id=low_ratio.name, ctx=ast.Load())]
        main.body = [
            node
            for node in main.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        op = SimpleNamespace(
            attention=Mock(side_effect=lambda q, **kw: (q + 1, None)),
            metadata=Mock(
                side_effect=lambda *a, **kw: torch.full((2048,), 7, dtype=torch.int32)
            ),
        )
        a5 = SimpleNamespace(
            mixed_quant_sparse_flash_mla_metadata=Mock(
                side_effect=lambda **kw: torch.full((48,), 5, dtype=torch.int32)
            ),
            quant_lightning_indexer_metadata=Mock(
                side_effect=lambda **kw: torch.full((1024,), 9, dtype=torch.int32)
            ),
        )
        host_metadata = Mock(
            side_effect=lambda **kw: torch.full((1024,), 11, dtype=torch.int32)
        )
        legacy_attention = Mock(side_effect=lambda q, **kw: (q + 2, None))
        ns = dict(
            torch=torch,
            F=F,
            SimpleNamespace=SimpleNamespace,
            ForwardMetadata=lambda: SimpleNamespace(block_tables_swa=None),
            is_npu_arch35=lambda: arch35,
            ensure_hccp_initialized=lambda *args, **kwargs: None,
            _dsv41_a3_ops=lambda: op,
            _ds41_ops=lambda: a5,
            _sparse_attn_ops=lambda *a, **kw: (host_metadata, legacy_attention),
            _sparse_attn_kv_quant_kwargs=lambda: {"kv_quant_mode": 1} if arch35 else {},
            ring_row_loc=ring_module.ring_row_loc,
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                low_ratio,
                main,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(BACKEND), "exec"), ns)
        backend = ns[main.name]()
        backend.device = torch.device("cpu")
        backend.page_size = 128
        backend.is_hybrid_swa = False
        backend._dsv4_has_low_ratio = not draft
        backend._dsv4_has_c4 = backend._dsv4_has_c128 = False
        backend._dsv4_compress_ratios = () if draft else (0, 1, 2)
        backend._dsv4_unique_compress_ratios = backend._dsv4_compress_ratios
        backend._dsv4_q_head_num = 64 if arch35 else 8
        backend._dsv4_kv_head_num = 1
        backend._dsv4_head_dim = 512
        backend._dsv4_index_n_heads = 32
        backend._dsv4_index_head_dim = 128
        backend._dsv4_index_topk = 512
        backend._dsv41_candidate_source_layer = -1
        backend._dsv4_sliding_window_size = 128
        backend._dsv4_graph_tokens_per_req = width
        backend._is_dspark_algorithm = True
        backend._is_dspark_draft_worker = draft
        backend._dsv4_state_pools_by_ratio = {}
        backend._capture_seed_inputs = []
        backend._SEQ_LENS_CPU_BUF_ROWS = 512
        backend._dspark_ragged_verify_layout = lambda _: None
        # Encoder-replay window hooks have their own CPU tests.
        backend.enable_encoder_swa_bounded_replay = False
        backend._init_request_window_graph_metadata = lambda *a, **k: None
        backend.forward_metadata = SimpleNamespace(stale_sentinel=True)
        backend.token_to_kv_pool = SimpleNamespace(
            swa_req_ring_size=None,
            get_swa_buffer=lambda _: torch.zeros(8, 128, 1, 512, dtype=torch.bfloat16),
        )
        if ring is None:
            backend._seed_draft_graph_page_table = lambda *a, **k: None
        else:
            backend.token_to_kv_pool.swa_req_ring_size = ring
            backend.token_to_kv_pool.num_req_slots = 16
            backend.token_to_kv_pool.swa_valid_from = torch.zeros(16, dtype=torch.int64)
            backend._draft_page_tables = ring_module.RingPageTableCache()
            backend._graph_table_high_water = {}
        backend.graph_metadata = {
            "block_tables": torch.tensor(
                [[2 + row for row in range(pages)] for _ in range(bs)],
                dtype=torch.int32,
            )
        }
        backend._init_dsv4_graph_buffers(max_bs=bs, max_num_tokens=bs * width)
        backend.test_ops = SimpleNamespace(
            a3=op, a5=a5, host=host_metadata, legacy=legacy_attention
        )
        return backend

    with patch.dict(
        "sys.modules",
        {
            metadata_module.__name__: metadata_module,
            pool_module.__name__: pool_module,
            ring_module.__name__: ring_module,
            dspark_module.__name__: dspark_module,
        },
    ):
        yield make


def _batch(prefixes, width, *, raw_bs=None, idle=False, tail=0, req_slots=None):
    bs = len(prefixes)
    live = torch.tensor(prefixes, dtype=torch.int32)
    positions = torch.cat([torch.arange(p, p + width) for p in prefixes])
    slots = torch.cat(
        [torch.arange(256 + r * 256, 256 + r * 256 + width) for r in range(bs)]
    )
    raw_bs = bs if raw_bs is None else raw_bs
    req_slots = torch.arange(bs) if req_slots is None else torch.tensor(req_slots)
    return SimpleNamespace(
        batch_size=bs,
        forward_mode=_mode(),
        actual_forward_mode=_mode(idle=idle),
        num_padding=bs if idle else bs - raw_bs,
        seq_lens=live,
        seq_lens_cpu=live + width,
        spec_info=SimpleNamespace(draft_token_num=width, live_seq_lens_cpu=live),
        positions=F.pad(positions, (0, tail)),
        out_cache_loc=slots[: 0 if idle else raw_bs * width],
        req_pool_indices=req_slots,
    )


def _capture(backend, batch):
    backend._init_dsv4_graph_metadata(
        metadata_key=batch.batch_size, forward_batch=batch, ragged_layout=None
    )
    metadata = backend.graph_metadata[batch.batch_size]
    backend.forward_metadata = metadata
    return metadata


def _refresh(backend, batch):
    ctx = backend._build_dsv4_graph_replay_ctx(batch)
    backend._refresh_graph_seq_metadata(ctx)
    backend._refresh_graph_verify_rows(ctx)
    if not ctx.is_idle_replay:
        backend._refresh_graph_dsv41_index_page_tables(ctx.fm)
    backend._refresh_graph_kernel_metadata(ctx)
    return ctx


@pytest.mark.parametrize("arch35", [False, True])
@pytest.mark.parametrize("width", [1, 5, 6])
@pytest.mark.parametrize("bs", [1, 3])
def test_shared_verify_capture_idle_live_idle_lifecycle(
    build_backend, arch35, width, bs
):
    b = build_backend(arch35, width, bs=bs)
    prefixes = [128 + row for row in range(bs)]
    fm = _capture(b, _batch(prefixes, width))
    fields = [
        "start_pos",
        "seqused",
        "actual_seq_lengths_q_pa",
        "actual_seq_lengths_kv",
    ]
    fields += [
        f"c{ratio}_{suffix}"
        for ratio in (1, 2)
        for suffix in ("loc", "index_page_table")
    ]
    pointers = {name: getattr(fm, name).data_ptr() for name in fields}
    kernel_pointers = {
        name: tensor.data_ptr() for name, tensor in fm.kernel_metadata.items()
    }
    for idle, raw_bs in ((True, 0), (False, bs), (False, 1), (True, 0)):
        batch = _batch(prefixes, width, raw_bs=raw_bs, idle=idle)
        _refresh(b, batch)
        live_count = 0 if idle else raw_bs
        # A5 keeps upstream prefix-derived scalars; A3 also masks raw_bs
        # padding. Both filter absent output slots in the shared row planner.
        scalar_rows = 0 if idle else (bs if arch35 else raw_bs)
        assert fm.seqused.tolist() == [width] * scalar_rows + [0] * (bs - scalar_rows)
        assert fm.start_pos.tolist() == prefixes[:scalar_rows] + [0] * (
            bs - scalar_rows
        )
        for ratio in (1, 2):
            keep = []
            positions = []
            for request in range(bs):
                for token in range(width):
                    position = prefixes[request] + token
                    valid = request < live_count and position % ratio == ratio - 1
                    slot = (256 + request * 256 + token) // ratio
                    if ratio == 1 or valid:
                        keep.append(slot if valid else -1)
                        positions.append(position // ratio * ratio if valid else 0)
            size = min(bs * width, bs * width // ratio + bs)
            keep = keep[:size] + [-1] * max(0, size - len(keep))
            positions = positions[:size] + [0] * max(0, size - len(positions))
            assert getattr(fm, f"c{ratio}_loc").tolist() == keep
            assert getattr(fm, f"positions_cmp_padding_c{ratio}").tolist() == positions
        assert pointers == {name: getattr(fm, name).data_ptr() for name in fields}
        assert kernel_pointers == {
            name: tensor.data_ptr() for name, tensor in fm.kernel_metadata.items()
        }


@pytest.mark.parametrize("arch35", [False, True])
def test_target_verify_tp_tail_is_a3_only(build_backend, arch35):
    b = build_backend(arch35, 6, bs=1)
    batch = _batch([128], 6, tail=2)
    fm = _capture(b, batch)
    if arch35:
        with pytest.raises(RuntimeError, match="verify replay rows"):
            _refresh(b, batch)
    else:
        _refresh(b, batch)
        assert fm.c1_loc.tolist() == list(range(256, 262))
        assert fm.c2_loc.tolist() == [128, 129, 130, -1]
        assert fm.actual_seq_lengths_q_pa.tolist() == [0, 6]


def test_a3_draft_capture_initializes_own_metadata_and_replay_keeps_addresses(
    build_backend,
):
    b = build_backend(False, 5, draft=True, bs=2)
    stale = b.forward_metadata
    batch = _batch([128, 129], 5)
    fm = _capture(b, batch)
    assert not hasattr(stale, "kernel_metadata")
    assert set(fm.kernel_metadata) == {"c1a_metadata"}
    tensor = fm.kernel_metadata["c1a_metadata"]
    assert tensor.numel() == 2048
    assert torch.all(tensor != 0)
    pointer = tensor.data_ptr()
    cpu_pointer = None
    for idle in (False, True, False):
        replay = _batch([130, 131], 5, idle=idle)
        ctx = b._build_dsv4_graph_replay_ctx(replay)
        b._refresh_graph_seq_metadata(ctx)
        # No c4 pool exists: refresh must not unconditionally clear c4 indices.
        b._refresh_graph_kernel_metadata(ctx)
        assert tensor.data_ptr() == pointer
        if cpu_pointer is None:
            cpu_pointer = fm.seq_lens_cpu_int.data_ptr()
        assert fm.seq_lens_cpu_int.data_ptr() == cpu_pointer
        q = torch.zeros(10, b._dsv4_q_head_num, 512, dtype=torch.bfloat16)
        result = b._forward_swa(
            q,
            SimpleNamespace(layer_id=0, scaling=512**-0.5),
            replay,
            torch.zeros(b._dsv4_q_head_num),
        )
        assert result.shape == q.shape
        operator = b.test_ops.a3.attention
        assert operator.call_args.kwargs["metadata"].data_ptr() == pointer
    assert b.test_ops.a3.attention.call_args.kwargs["ori_mask_mode"] == 0
    b.test_ops.host.assert_not_called()


def test_a5_draft_capture_preserves_upstream_initialization(build_backend):
    # A3's standalone metadata initialization must not be imposed on A5. This
    # checks the unchanged upstream contract, not an end-to-end A5 graph run.
    b = build_backend(True, 5, draft=True, bs=2)
    fm = _capture(b, _batch([128, 129], 5))
    assert "kernel_metadata_c1a" not in b.graph_metadata
    assert fm.kernel_metadata == {}
    b.test_ops.host.assert_not_called()
    b.test_ops.a3.metadata.assert_not_called()


def _refresh_draft_ring(backend, batch):
    ctx = backend._build_dsv4_graph_replay_ctx(batch)
    backend._refresh_graph_seq_metadata(ctx)
    backend._refresh_graph_swa_metadata_direct(ctx)
    backend._refresh_graph_dspark_sparse_metadata(ctx)
    backend._refresh_graph_kernel_metadata(ctx)
    return ctx


def _expected_draft_ring_indices(live, valid_from, *, window, width):
    columns = (window + width + 127) // 128 * 128
    start = max(0, live - window, valid_from)
    valid = min(max(0, live - start + width), columns)
    indices = [start + column for column in range(valid)]
    return indices + [-1] * (columns - valid)


def test_a3_draft_ring_graph_replay_refreshes_in_place(build_backend):
    window, width, ring = 128, 5, 256
    b = build_backend(False, width, draft=True, bs=2, ring=ring, pages=8)
    b.token_to_kv_pool.swa_valid_from[7] = 200
    prefixes = [300, 133]
    fm = _capture(b, _batch(prefixes, width, req_slots=[7, 3]))

    def replay(live, *, idle=False, raw_bs=None):
        batch = _batch(live, width, idle=idle, raw_bs=raw_bs, req_slots=[7, 3])
        _refresh_draft_ring(b, batch)
        return batch

    replay(prefixes)
    indices = fm.ori_sparse_indices
    topk = fm.dsv41_draft_topk_length
    swa_loc = fm.swa_loc
    page_table = b.graph_metadata["swa_page_table"]
    pointers = (indices.data_ptr(), topk.data_ptr(), swa_loc.data_ptr())
    kernel_pointer = fm.kernel_metadata["c1a_metadata"].data_ptr()

    assert indices[:, 0, 0].tolist() == [200] * width + [5] * width
    assert indices[0, 0].tolist() == _expected_draft_ring_indices(
        300, 200, window=window, width=width
    )
    assert indices[width, 0].tolist() == _expected_draft_ring_indices(
        133, 0, window=window, width=width
    )
    assert topk.flatten().tolist() == [105] * width + [133] * width
    # Draft rows address the request ring directly.
    assert swa_loc.tolist() == [
        7 * ring + position % ring for position in range(300, 305)
    ] + [3 * ring + position % ring for position in range(133, 138)]
    # Logical page p of slot s maps to ring page s * ring_pages + p % ring_pages;
    # the table is as wide as the longest request needs.
    assert page_table[:2, :3].tolist() == [[14, 15, 14], [6, 7, 6]]

    # A later replay with different prefixes rewrites the same buffers and grows
    # the page table; wider columns past the longest request wrap onto ring pages.
    replay([250, 400])
    assert (indices.data_ptr(), topk.data_ptr(), swa_loc.data_ptr()) == pointers
    assert fm.kernel_metadata["c1a_metadata"].data_ptr() == kernel_pointer
    assert indices[0, 0].tolist() == _expected_draft_ring_indices(
        250, 200, window=window, width=width
    )
    assert indices[width, 0].tolist() == _expected_draft_ring_indices(
        400, 0, window=window, width=width
    )
    assert page_table[:2, :6].tolist() == [
        [14, 15, 14, 15, 14, 15],
        [6, 7, 6, 7, 6, 7],
    ]

    # An idle replay neutralizes the rows it captured and commits nothing.
    replay([250, 400], idle=True)
    assert torch.all(indices[:, 0, 0] == 0)
    assert torch.all(indices[:, :, 1:] == -1)
    assert (indices.data_ptr(), topk.data_ptr(), swa_loc.data_ptr()) == pointers
