"""CPU contract checks for the NPU row-style request window (encoder E1).

Pins the ring addressing, history copy, commit, wrap-around and reset without
torch_npu: the window is plain torch over the shared WindowLayout math.
"""

import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import InvariantCheckLevel
from sglang.srt.hardware_backend.npu.dsv4 import npu_request_window as npu_window
from sglang.srt.hardware_backend.npu.dsv4.npu_request_window import (
    NPURequestWindowController,
    NPURequestWindowMemory,
    NPURequestWindowSpec,
)
from sglang.srt.hardware_backend.npu.dsv4.ring_layout import (
    RingPageTableCache,
    sink_dead_rows,
)

ROOT = Path(__file__).resolve().parents[5]
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=4, stage="base-a", runner_config="cpu")


class NPURequestWindowTest(unittest.TestCase):
    def make_window(self, *, slots=1, layers=1, capacity=8, workspace_rows=64):
        spec = NPURequestWindowSpec.create(
            num_slots=slots,
            layers=layers,
            page_size=4,
            capacity=capacity,
            workspace_rows=workspace_rows,
            row_dim=2,
            row_dtype=torch.float32,
        )
        return NPURequestWindowMemory(spec, device=torch.device("cpu")).view

    @staticmethod
    def layout(win, req, pos, *, num_groups=1, **kwargs):
        return win.build_prefill_layout(
            torch.tensor(req), torch.tensor(pos), num_groups=num_groups, **kwargs
        )

    @staticmethod
    def rows(values):
        return torch.tensor(values, dtype=torch.float32).reshape(-1, 2)

    def test_prefill_window_is_narrower_than_the_ring(self):
        # DSpark widens the ring with draft rows (capacity 16); the attention
        # window stays 4, so history/index width must not follow capacity.
        win = self.make_window(capacity=16)
        layout = self.layout(win, [0, 0, 0, 0], [10, 11, 12, 13], window=4)
        self.assertEqual(layout.history_pos.tolist(), [6, 7, 8, 9])
        self.assertEqual(layout.indices.shape[1], 4)
        # Query at 10 attends positions 7..10: locals 1..4.
        self.assertEqual(layout.indices[0].tolist(), [1, 2, 3, 4])
        self.assertEqual(layout.lengths.tolist(), [4, 4, 4, 4])
        # The window side's seqused is the staged span (history + live).
        self.assertEqual(layout.kv_lens.tolist(), [8])
        # Storage still commits the last ``capacity`` rows.
        self.assertTrue(layout.commit_mask.all())

    def test_prefill_indices_are_a_left_aligned_valid_prefix(self):
        # Explicit-index consumers (A5) read indices[:length]; a short first
        # window must not lead with sentinels or they attend the wrong rows.
        win = self.make_window(capacity=8)
        layout = self.layout(win, [0, 0, 0], [0, 1, 2], window=4)
        self.assertEqual(layout.indices.tolist()[0], [0, -1, -1, -1])
        self.assertEqual(layout.indices.tolist()[1], [0, 1, -1, -1])
        self.assertEqual(layout.indices.tolist()[2], [0, 1, 2, -1])
        self.assertEqual(layout.lengths.tolist(), [1, 2, 3])
        for row, length in zip(layout.indices, layout.lengths.tolist()):
            self.assertTrue(bool((row[:length] >= 0).all()))
            self.assertTrue(bool((row[length:] < 0).all()))

    def test_decode_window_is_narrower_than_the_ring(self):
        # Eager decode reads the ring directly; the graph decode layout is the
        # staged counterpart and keeps the same compact window geometry.
        win = self.make_window(capacity=16)
        layout = win.build_graph_decode_layout(1, window=4)
        win.refresh_graph_decode_layout(
            layout,
            torch.tensor([0]),
            torch.tensor([10]),
            window=4,
            live=torch.tensor([True]),
        )
        self.assertEqual(layout.size, 4)
        self.assertEqual(layout.indices.tolist(), [[0, 1, 2, 3]])
        self.assertEqual(layout.lengths.tolist(), [4])
        self.assertEqual(layout.history_pos.tolist(), [7, 8, 9, 10])
        # The new K lands at the window's last row, not at capacity-1.
        self.assertEqual(layout.write_loc.tolist(), [3])

    def test_graph_decode_layout_refreshes_in_place(self):
        win = self.make_window(capacity=16)
        layout = win.build_graph_decode_layout(2, window=4)
        win.refresh_graph_decode_layout(
            layout,
            torch.tensor([0, 1]),
            torch.tensor([5, 10]),
            window=4,
            live=torch.tensor([True, False]),
        )
        # Indices stay frame-local; the per-request page table maps them onto
        # that request's own page-aligned workspace segment.
        self.assertEqual(layout.indices.tolist(), [[0, 1, 2, 3], [0, 1, 2, 3]])
        self.assertEqual(layout.lengths.tolist(), [4, 4])
        self.assertEqual(layout.write_loc.tolist(), [3, 7])
        self.assertEqual(layout.commit_mask.tolist(), [True, False])
        # row_loc = slot * capacity + position % capacity.
        self.assertEqual(layout.history_src.tolist(), [2, 3, 4, 5, 23, 24, 25, 26])
        self.assertEqual(layout.history_loc.tolist(), [0, 1, 2, 3, 4, 5, 6, 7])
        # Refreshing again reuses the same storage (graph capture holds it).
        ptr = layout.indices.data_ptr()
        win.refresh_graph_decode_layout(
            layout,
            torch.tensor([0, 1]),
            torch.tensor([1, 2]),
            window=4,
            live=torch.tensor([True, True]),
        )
        self.assertEqual(layout.indices.data_ptr(), ptr)
        self.assertEqual(layout.lengths.tolist(), [2, 3])
        self.assertEqual(layout.write_loc.tolist(), [1, 6])
        self.assertEqual(layout.indices.tolist(), [[0, 1, -1, -1], [0, 1, 2, -1]])

    def test_graph_verify_layout_refreshes_in_place(self):
        win = self.make_window(capacity=16)
        layout = win.build_graph_verify_layout(1, 2, window=4)
        win.refresh_graph_verify_layout(
            layout,
            torch.tensor([0, 0]),
            torch.tensor([10, 11]),
            torch.tensor([True, True]),
            num_draft=2,
            window=4,
        )
        # The 5-row span is padded to two 4-row pages for page addressing.
        self.assertEqual(layout.size, 8)
        self.assertEqual(layout.kv_lens.tolist(), [5])
        self.assertEqual(layout.indices.tolist(), [[0, 1, 2, 3], [1, 2, 3, 4]])
        self.assertEqual(layout.lengths.tolist(), [4, 4])
        self.assertEqual(layout.write_loc.tolist(), [3, 4])
        self.assertEqual(layout.history_src.tolist(), [7, 8, 9])
        self.assertEqual(layout.history_loc.tolist(), [0, 1, 2])
        self.assertEqual(layout.history_pos.tolist(), [7, 8, 9])

    def test_graph_page_table_covers_the_sequence_and_wraps_the_ring(self):
        # The operator scans the table up to the request's sequence length, not
        # the staged layout's; columns past a request's own segment wrap onto
        # it, and a sentinel column would make a long sequence read an invalid
        # row.
        win = self.make_window(slots=2, capacity=16, workspace_rows=64)
        layout = win.build_graph_verify_layout(2, 2, window=4)
        table = torch.full((2, 8), -1, dtype=torch.int32)
        written = []

        def copy_page_table(key, src):
            written.append((key, src))
            table[: src.shape[0], : src.shape[1]].copy_(src)

        controller = NPURequestWindowController(
            window=win,
            page_size=4,
            sliding_window=4,
            device=torch.device("cpu"),
            graph_metadata=lambda: {"swa_page_table": table},
            graph_tokens_per_req=2,
            copy_page_table=copy_page_table,
        )
        controller.refresh_graph_page_table(2, layout, max_seq=21)
        self.assertEqual(written[0][0], "swa_page_table")
        # Two 5-row spans pad to two 2-page segments; 21 tokens need 6 logical
        # pages, so the extra columns wrap inside each request's segment.
        self.assertEqual(table.tolist(), [[0, 1, 0, 1, 0, 1, -1, -1],
                                          [2, 3, 2, 3, 2, 3, -1, -1]])

    def test_graph_page_table_points_each_request_at_its_own_pages(self):
        # Staged graph layouts own one page-aligned segment per request; the
        # published table must follow them (one shared row would alias both).
        win = self.make_window(slots=2, capacity=16, workspace_rows=64)
        layout = win.build_graph_verify_layout(2, 2, window=4)
        table = torch.full((2, 8), -1, dtype=torch.int32)

        def copy_page_table(key, src):
            table[: src.shape[0], : src.shape[1]].copy_(src)

        controller = NPURequestWindowController(
            window=win,
            page_size=4,
            sliding_window=4,
            device=torch.device("cpu"),
            graph_metadata=lambda: {"swa_page_table": table},
            graph_tokens_per_req=2,
            copy_page_table=copy_page_table,
        )
        controller.refresh_graph_page_table(2, layout, max_seq=10)
        # Two 5-row spans pad to two 8-row (2-page) segments; 10 tokens need
        # 3 pages, so only the first three columns are written.
        self.assertEqual(layout.size, 16)
        self.assertEqual(table[:, :3].tolist(), [[0, 1, 0], [2, 3, 2]])

    def test_graph_verify_idle_refresh_clears_commit_without_rows(self):
        # DP idle replay has no verify rows; the fixed-shape layout must keep
        # its captured dummy indices but commit nothing.
        win = self.make_window(slots=1, capacity=8, workspace_rows=64)
        layout = win.build_graph_verify_layout(1, 2, window=4)
        controller = NPURequestWindowController(
            window=win,
            page_size=4,
            sliding_window=4,
            device=torch.device("cpu"),
            graph_metadata=lambda: {},
            graph_tokens_per_req=2,
            copy_page_table=lambda *args, **kwargs: None,
        )
        ctx = SimpleNamespace(
            graph_mode=SimpleNamespace(
                is_decode=lambda: False, is_target_verify=lambda: True
            ),
            forward_batch=SimpleNamespace(
                positions=torch.empty(0, dtype=torch.int64),
                out_cache_loc=torch.empty(0, dtype=torch.int64),
                req_pool_indices=torch.zeros(1, dtype=torch.int64),
            ),
            fm=SimpleNamespace(request_window_layout=layout),
            bs=1,
            tokens_per_bs=2,
            raw_bs=0,
            device=torch.device("cpu"),
        )
        controller.refresh_graph(ctx)
        self.assertFalse(bool(layout.commit_mask.any()))
        # The idle step must not keep reading the previous window: a zero
        # length with a legal first index, no history rows, and a zero span.
        self.assertEqual(layout.lengths.tolist(), [0, 0])
        self.assertEqual(layout.indices.tolist(), [[0, -1, -1, -1], [0, -1, -1, -1]])
        self.assertFalse(bool(layout.history_valid.any()))
        self.assertEqual(layout.kv_lens.tolist(), [0])
        self.assertEqual(int(layout.write_loc.max()), 63)

    def test_graph_idle_refresh_neutralizes_the_previous_window(self):
        # live -> idle -> live on one bucket: the idle step publishes a
        # zero-length read instead of the live step's window, and the next
        # live step restores a real one.
        win = self.make_window(slots=2, capacity=16, workspace_rows=64)
        layout = win.build_graph_decode_layout(2, window=4)
        controller = NPURequestWindowController(
            window=win,
            page_size=4,
            sliding_window=4,
            device=torch.device("cpu"),
            graph_metadata=lambda: {},
            graph_tokens_per_req=1,
            copy_page_table=lambda *args, **kwargs: None,
        )

        def ctx_for(raw_bs, positions, out_loc):
            return SimpleNamespace(
                graph_mode=SimpleNamespace(
                    is_decode=lambda: True, is_target_verify=lambda: False
                ),
                forward_batch=SimpleNamespace(
                    positions=positions,
                    out_cache_loc=out_loc,
                    req_pool_indices=torch.zeros(2, dtype=torch.int64),
                ),
                fm=SimpleNamespace(request_window_layout=layout),
                bs=2,
                tokens_per_bs=1,
                raw_bs=raw_bs,
                device=torch.device("cpu"),
            )

        live = ctx_for(2, torch.tensor([5, 9]), torch.tensor([16, 17]))
        controller.refresh_graph(live)
        self.assertEqual(layout.lengths.tolist(), [4, 4])
        idle = ctx_for(
            0,
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
        )
        controller.refresh_graph(idle)
        self.assertEqual(layout.lengths.tolist(), [0, 0])
        self.assertEqual(layout.indices.tolist(), [[0, -1, -1, -1]] * 2)
        self.assertFalse(bool(layout.commit_mask.any()))
        controller.refresh_graph(live)
        self.assertEqual(layout.lengths.tolist(), [4, 4])
        self.assertTrue(bool(layout.commit_mask.all()))

    def test_graph_decode_idle_refresh_clears_commit_without_rows(self):
        win = self.make_window(slots=1, capacity=8, workspace_rows=64)
        layout = win.build_graph_decode_layout(1, window=4)
        controller = NPURequestWindowController(
            window=win,
            page_size=4,
            sliding_window=4,
            device=torch.device("cpu"),
            graph_metadata=lambda: {},
            graph_tokens_per_req=1,
            copy_page_table=lambda *args, **kwargs: None,
        )
        ctx = SimpleNamespace(
            graph_mode=SimpleNamespace(
                is_decode=lambda: True, is_target_verify=lambda: False
            ),
            forward_batch=SimpleNamespace(
                positions=torch.empty(0, dtype=torch.int64),
                out_cache_loc=torch.empty(0, dtype=torch.int64),
                req_pool_indices=torch.zeros(1, dtype=torch.int64),
            ),
            fm=SimpleNamespace(request_window_layout=layout),
            bs=1,
            tokens_per_bs=1,
            raw_bs=0,
            device=torch.device("cpu"),
        )
        controller.refresh_graph(ctx)
        self.assertFalse(bool(layout.commit_mask.any()))
        self.assertEqual(layout.lengths.tolist(), [0])
        self.assertEqual(layout.indices.tolist(), [[0, -1, -1, -1]])
        self.assertFalse(bool(layout.history_valid.any()))

    def test_graph_decode_partial_bucket_refreshes_the_live_prefix(self):
        # A rank holding fewer requests than the DP-wide graph bucket replays
        # the larger graph: positions are padded, out_cache_loc is not. The
        # live request still needs a fresh window; its padded neighbour must
        # never read a stale window or commit.
        win = self.make_window(slots=2, capacity=16, workspace_rows=64)
        layout = win.build_graph_decode_layout(2, window=4)
        win.refresh_graph_decode_layout(
            layout,
            torch.tensor([0, 1]),
            torch.tensor([5, 10]),
            window=4,
            live=torch.tensor([True, True]),
        )
        controller = NPURequestWindowController(
            window=win,
            page_size=4,
            sliding_window=4,
            device=torch.device("cpu"),
            graph_metadata=lambda: {},
            graph_tokens_per_req=1,
            copy_page_table=lambda *args, **kwargs: None,
        )
        ctx = SimpleNamespace(
            graph_mode=SimpleNamespace(
                is_decode=lambda: True, is_target_verify=lambda: False
            ),
            forward_batch=SimpleNamespace(
                positions=torch.tensor([5, 0], dtype=torch.int64),
                out_cache_loc=torch.tensor([16], dtype=torch.int64),
                req_pool_indices=torch.tensor([0, 0], dtype=torch.int64),
            ),
            fm=SimpleNamespace(request_window_layout=layout),
            bs=2,
            tokens_per_bs=1,
            raw_bs=1,
            device=torch.device("cpu"),
        )
        controller.refresh_graph(ctx)
        self.assertEqual(layout.commit_mask.tolist(), [True, False])
        self.assertEqual(layout.lengths.tolist(), [4, 1])
        self.assertEqual(layout.write_loc.tolist(), [3, 4])
        self.assertEqual(layout.indices.tolist(), [[0, 1, 2, 3], [0, -1, -1, -1]])
        # A real row whose KV slot was not allocated this step stays dead.
        ctx.forward_batch.out_cache_loc = torch.tensor([3], dtype=torch.int64)
        controller.refresh_graph(ctx)
        self.assertEqual(layout.commit_mask.tolist(), [False, False])

    def test_graph_verify_partial_bucket_keeps_padded_requests_dead(self):
        # Same DP-bucket padding for a DSpark verify replay: the padded
        # request's block stages nothing and commits nothing, while the live
        # request's block is refreshed.
        win = self.make_window(slots=2, capacity=16, workspace_rows=64)
        layout = win.build_graph_verify_layout(2, 2, window=4)
        controller = NPURequestWindowController(
            window=win,
            page_size=4,
            sliding_window=4,
            device=torch.device("cpu"),
            graph_metadata=lambda: {},
            graph_tokens_per_req=2,
            copy_page_table=lambda *args, **kwargs: None,
        )
        ctx = SimpleNamespace(
            graph_mode=SimpleNamespace(
                is_decode=lambda: False, is_target_verify=lambda: True
            ),
            forward_batch=SimpleNamespace(
                positions=torch.tensor([4, 5, 0, 0], dtype=torch.int64),
                out_cache_loc=torch.tensor([16, 17], dtype=torch.int64),
                req_pool_indices=torch.tensor([0, 0], dtype=torch.int64),
            ),
            fm=SimpleNamespace(request_window_layout=layout),
            bs=2,
            tokens_per_bs=2,
            raw_bs=1,
            device=torch.device("cpu"),
        )
        controller.refresh_graph(ctx)
        self.assertEqual(layout.commit_mask.tolist(), [True, True, False, False])
        # The window is capped at 4 rows, so the second verify row reads 4.
        self.assertEqual(layout.lengths.tolist(), [4, 4, 1, 2])
        self.assertEqual(layout.write_loc.tolist(), [3, 4, 8, 9])

    def test_graph_commit_masked_rows_go_to_the_sink(self):
        win = self.make_window(capacity=16)
        layout = self.layout(win, [0, 0], [2, 3])
        layout.commit_mask.zero_()
        win.graph_capture = True
        win.activate(layout)
        win.workspace[layout.write_loc] = self.rows([[1, 1], [2, 2]])
        win.commit(0)
        # Neither real ring row nor the sink's own zero rows are needed; only
        # the sink may hold the masked writes.
        self.assertEqual(win.tags[0, 0].item(), -1)
        self.assertEqual(win.tags[0, 1].item(), -1)
        self.assertNotEqual(win.tags[0, win.sink_row].item(), -1)

    def test_eager_commit_skips_masked_rows(self):
        win = self.make_window(capacity=16)
        layout = self.layout(win, [0, 0], [2, 3])
        layout.commit_mask.zero_()
        win.graph_capture = False
        win.activate(layout)
        win.workspace[layout.write_loc] = self.rows([[1, 1], [2, 2]])
        win.commit(0)
        # The eager path never touches the sink row.
        self.assertEqual(win.tags[0, 0].item(), -1)
        self.assertEqual(win.tags[0, 1].item(), -1)
        self.assertEqual(win.tags[0, win.sink_row].item(), -1)

    def test_verify_layout_stages_context_then_the_block(self):
        win = self.make_window(capacity=16)
        layout = win.build_graph_verify_layout(1, 2, window=4)
        win.refresh_graph_verify_layout(
            layout,
            torch.tensor([0, 0]),
            torch.tensor([10, 11]),
            torch.tensor([True, True]),
            num_draft=2,
            window=4,
        )
        # Context is positions 7..9 (window start 7), then the verify rows.
        self.assertEqual(layout.history_pos.tolist(), [7, 8, 9])
        self.assertEqual(layout.history_loc.tolist(), [0, 1, 2])
        # The 5-row span is padded to two 4-row pages for page addressing.
        self.assertEqual(layout.size, 8)
        self.assertEqual(layout.indices[0].tolist(), [0, 1, 2, 3])
        self.assertEqual(layout.indices[1].tolist(), [1, 2, 3, 4])
        self.assertEqual(layout.lengths.tolist(), [4, 4])
        self.assertEqual(layout.kv_lens.tolist(), [5])
        self.assertEqual(layout.write_loc.tolist(), [3, 4])
        self.assertTrue(layout.commit_mask.all())

    def test_verify_layout_skips_dead_rows_on_commit(self):
        win = self.make_window(capacity=16)
        layout = win.build_graph_verify_layout(1, 2, window=4)
        win.refresh_graph_verify_layout(
            layout,
            torch.tensor([0, 0]),
            torch.tensor([0, 1]),
            torch.tensor([True, False]),
            num_draft=2,
            window=4,
        )
        self.assertEqual(layout.indices[0].tolist(), [0, -1, -1, -1])
        self.assertEqual(layout.indices[1].tolist(), [0, 1, -1, -1])
        self.assertEqual(layout.commit_mask.tolist(), [True, False])
        # Graph layouts commit with the fixed-shape sink scatter.
        win.graph_capture = True
        win.activate(layout)
        win.workspace[layout.write_loc] = self.rows([[7, 7], [8, 8]])
        win.commit(0)
        # Only the live verify row reaches the ring.
        self.assertEqual(win.tags[0, 0].item(), 0)
        self.assertEqual(win.tags[0, 1].item(), -1)

    def test_first_extend_commits_compact_live_rows(self):
        win = self.make_window()
        layout = self.layout(win, [0, 0, 0, 0], [0, 1, 2, 3])
        self.assertEqual(layout.size, 4)
        self.assertEqual(layout.history_src.numel(), 0)

        win.activate(layout)
        buffer = win.buffer(0)

        written = self.rows([[1, 1], [2, 2], [3, 3], [4, 4]])
        buffer[layout.write_loc] = written
        win.commit(0)

        self.assertTrue(torch.equal(win.state[0][:4], written))
        self.assertEqual(win.tags[0][:4].tolist(), [0, 1, 2, 3])

    def test_continuation_copies_the_cached_history(self):
        win = self.make_window()
        first = self.layout(win, [0, 0, 0, 0], [0, 1, 2, 3])
        win.activate(first)
        win.workspace[first.write_loc] = self.rows([[1, 1], [2, 2], [3, 3], [4, 4]])
        win.commit(0)

        second = self.layout(win, [0, 0], [4, 5])
        self.assertEqual(second.history_src.tolist(), [0, 1, 2, 3])
        win.activate(second)
        buffer = win.buffer(0)
        self.assertTrue(
            torch.equal(buffer[:4], self.rows([[1, 1], [2, 2], [3, 3], [4, 4]]))
        )

        buffer[second.write_loc] = self.rows([[5, 5], [6, 6]])
        win.commit(0)
        self.assertTrue(torch.equal(win.state[0][4:6], self.rows([[5, 5], [6, 6]])))
        self.assertEqual(win.tags[0][:6].tolist(), [0, 1, 2, 3, 4, 5])

    def test_ring_wraps_after_capacity(self):
        win = self.make_window(capacity=8)
        first = self.layout(win, [0, 0, 0, 0], [0, 1, 2, 3])
        win.activate(first)
        win.workspace[first.write_loc] = self.rows([[1, 1]] * 4)
        win.commit(0)
        second = self.layout(win, [0, 0, 0, 0], [4, 5, 6, 7])
        win.activate(second)
        win.workspace[second.write_loc] = self.rows([[2, 2]] * 4)
        win.commit(0)

        third = self.layout(win, [0, 0], [8, 9])
        self.assertEqual(third.history_src.tolist(), [0, 1, 2, 3, 4, 5, 6, 7])
        win.activate(third)
        win.buffer(0)
        win.workspace[third.write_loc] = self.rows([[3, 3], [4, 4]])
        win.commit(0)
        # positions 8 and 9 land on ring rows 0 and 1.
        self.assertTrue(torch.equal(win.state[0][:2], self.rows([[3, 3], [4, 4]])))
        self.assertEqual(win.tags[0][:2].tolist(), [8, 9])
        self.assertEqual(win.tags[0][2:8].tolist(), [2, 3, 4, 5, 6, 7])

    def test_two_requests_use_separate_rings(self):
        win = self.make_window(slots=2)
        layout = self.layout(win, [0, 0, 1, 1], [0, 1, 0, 1], num_groups=2)
        win.activate(layout)
        written = self.rows([[1, 1], [2, 2], [3, 3], [4, 4]])
        win.workspace[layout.write_loc] = written
        win.commit(0)

        self.assertTrue(torch.equal(win.state[0][0:2], written[:2]))
        self.assertTrue(torch.equal(win.state[0][8:10], written[2:]))
        self.assertEqual(win.tags[0][0:2].tolist(), [0, 1])
        self.assertEqual(win.tags[0][8:10].tolist(), [0, 1])

    def test_reset_clears_only_the_reset_slot(self):
        win = self.make_window(slots=2)
        layout = self.layout(win, [0, 0, 1, 1], [0, 1, 0, 1], num_groups=2)
        win.activate(layout)
        win.workspace[layout.write_loc] = self.rows([[1, 1]] * 4)
        win.commit(0)
        win.reset(torch.tensor([0]))
        self.assertEqual(win.tags[0][:8].tolist(), [-1] * 8)
        self.assertEqual(win.tags[0][8:10].tolist(), [0, 1])

    def test_compact_layout_keeps_indices_below_seqused(self):
        # The A3 operator drops explicit indices >= seqused_kv; a fixed history
        # region pushed live rows past it for short prompts and silently zeroed
        # the window. Every index must stay below both the layout and seqused.
        win = self.make_window(slots=2, capacity=8)
        layout = win.build_graph_decode_layout(1, window=4)
        for p in range(20):
            win.refresh_graph_decode_layout(
                layout,
                torch.tensor([0]),
                torch.tensor([p]),
                window=4,
                live=torch.tensor([True]),
            )
            seqused = p + 1
            valid = layout.indices >= 0
            self.assertTrue(bool(valid.any()))
            self.assertLess(int(layout.indices[valid].max()), layout.size)
            self.assertLess(int(layout.indices[valid].max()), seqused)
            self.assertLess(int(layout.write_loc.max()), layout.size)
            self.assertLess(int(layout.write_loc.max()), seqused)

    def test_decode_layout_keeps_indices_below_seqused(self):
        win = self.make_window(slots=1, capacity=8)
        layout = win.build_graph_decode_layout(1, window=8)
        for p in (4, 19):
            win.refresh_graph_decode_layout(
                layout,
                torch.tensor([0]),
                torch.tensor([p]),
                window=8,
                live=torch.tensor([True]),
            )
            valid = layout.indices >= 0
            self.assertEqual(int(valid.sum()), min(8, p + 1))
            self.assertLess(int(layout.indices[valid].max()), p + 1)
            self.assertLess(int(layout.write_loc.max()), p + 1)
            self.assertEqual(int(layout.write_loc[0]), int(valid.sum()) - 1)

    def test_sink_dead_rows_never_wraps_to_a_real_slot(self):
        # A negative store location must not index-wrap onto another request's
        # ring row; every dead row goes to the unused sink.
        loc = torch.tensor([5, -1, 0, -1], dtype=torch.int64)
        self.assertEqual(sink_dead_rows(loc, 7).tolist(), [5, 7, 0, 7])

    def test_ring_page_table_cache_rewrites_in_place_and_wraps(self):
        cache = RingPageTableCache()
        table = cache.table(
            req_pool_indices=torch.tensor([0, 2]),
            ring_pages=2,
            logical_pages=3,
        )
        # slot * ring_pages + logical page % ring_pages.
        self.assertEqual(table.tolist(), [[0, 1, 0], [4, 5, 4]])
        pointer = table.data_ptr()
        same = cache.table(
            req_pool_indices=torch.tensor([1, 0]),
            ring_pages=2,
            logical_pages=2,
        )
        # A narrower step reuses the same storage and overwrites every row.
        self.assertEqual(
            same.untyped_storage().data_ptr(), table.untyped_storage().data_ptr()
        )
        self.assertEqual(same.data_ptr(), pointer)
        self.assertEqual(same.shape[1], 2)
        self.assertEqual(same.tolist(), [[2, 3], [0, 1]])
        grown = cache.table(
            req_pool_indices=torch.tensor([0, 2]),
            ring_pages=2,
            logical_pages=5,
        )
        # The cache may double internally, but callers see the valid width.
        self.assertEqual(grown.shape[1], 5)
        self.assertEqual(grown.tolist(), [[0, 1, 0, 1, 0], [4, 5, 4, 5, 4]])

    def test_multi_request_history_is_packed_per_request(self):
        # Two requests with history: the history gather must stay request-major
        # and each request's rows must map into its own packed region.
        win = self.make_window(slots=2, capacity=8, workspace_rows=64)
        layout = win.build_prefill_layout(
            torch.tensor([0, 0, 0, 1, 1]),
            torch.tensor([10, 11, 12, 6, 7]),
            num_groups=2,
            window=4,
        )
        self.assertEqual(layout.history_pos.tolist(), [6, 7, 8, 9, 2, 3, 4, 5])
        self.assertEqual(layout.history_src.tolist(), [6, 7, 0, 1, 10, 11, 12, 13])
        # Request 0 spans 4 history + 3 rows (7, padded to 8); request 1 starts
        # at row 8, so page 0 holds only request 0.
        self.assertEqual(layout.history_loc.tolist(), [0, 1, 2, 3, 8, 9, 10, 11])
        self.assertEqual(layout.base.tolist(), [0, 8])
        self.assertEqual(layout.group_pages.tolist(), [2, 2])
        self.assertEqual(layout.write_loc.tolist(), [4, 5, 6, 12, 13])
        self.assertEqual(layout.kv_lens.tolist(), [7, 6])

    def test_long_extend_commits_only_the_last_capacity_rows(self):
        # The group end comes from the request run's boundary, not from a
        # duplicate-index scatter whose winner is implementation-defined.
        win = self.make_window(capacity=8)
        layout = win.build_prefill_layout(
            torch.zeros(10, dtype=torch.int64),
            torch.arange(10),
            num_groups=1,
            window=8,
        )
        self.assertEqual(
            layout.commit_mask.tolist(), [False, False] + [True] * 8
        )

    def test_prefill_page_table_points_each_request_at_its_own_pages(self):
        # The staged rows are page-aligned per request; the published table must
        # follow them, or request 1 reads request 0's workspace rows.
        win = self.make_window(slots=2, capacity=8, workspace_rows=64)
        layout = win.build_prefill_layout(
            torch.tensor([0, 0, 0, 1, 1]),
            torch.tensor([10, 11, 12, 6, 7]),
            num_groups=2,
            window=4,
        )
        controller = NPURequestWindowController(
            window=win,
            page_size=4,
            sliding_window=4,
            device=torch.device("cpu"),
            graph_metadata=lambda: {},
            graph_tokens_per_req=1,
            copy_page_table=lambda *args, **kwargs: None,
        )
        table = controller._page_table(
            layout, 2, torch.tensor([10, 11]), max_seq=12
        )
        # Request 0 owns pages 0..1, request 1 owns pages 2..3; columns past a
        # request's segment wrap inside that segment. 12 tokens need 3 pages.
        self.assertEqual(table.tolist(), [[0, 1, 0], [2, 3, 2]])

    def test_reset_slot_fails_the_history_ownership_invariant(self):
        # A reset slot must not be read back as history; the tags record which
        # position each ring row currently holds.
        win = self.make_window(slots=2, capacity=8, workspace_rows=64)
        layout = win.build_prefill_layout(
            torch.tensor([0, 0, 0, 1, 1]),
            torch.tensor([10, 11, 12, 6, 7]),
            num_groups=2,
            window=4,
        )
        win.activate(layout)
        win.tags[0, layout.history_src] = layout.history_pos
        win.reset(torch.tensor([1]))
        win.prepared = None
        with (
            patch.object(
                npu_window, "resolve_level", return_value=InvariantCheckLevel.WARN
            ),
            patch.object(npu_window, "expect", wraps=npu_window.expect) as spy,
        ):
            win.buffer(0)
        ok = spy.call_args.args[1]
        self.assertFalse(bool(ok.all()))

    def test_dummy_history_populates_tags_for_capture(self):
        win = self.make_window()
        layout = self.layout(win, [0, 0], [4, 5])
        win.activate(layout)
        win.initialize_dummy_history()
        loc = layout.history_src.to(torch.int64)
        self.assertEqual(win.tags[0][loc].tolist(), layout.history_pos.tolist())
        self.assertEqual(win.state[0].abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
