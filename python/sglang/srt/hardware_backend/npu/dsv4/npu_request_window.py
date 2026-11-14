"""Row-style request window for encoder SWA bounded replay on NPU.

The CUDA request window stores packed pages and copies/commits them with Triton
kernels. NPU rows are plain ``[dim]`` BF16 (A3) or quantized (A5) rows, so this
keeps the same WindowLayout math but moves rows with torch gather/scatter:

- ``state[layer]``: ``num_slots * capacity`` request rows plus a zero row and a
  sink row;
- ``workspace``: shared staging rows; ``window_layout`` addresses the history
  rows first, then the live query rows;
- ``tags[layer]``: the position each request row holds, used to validate a
  history row before it is copied.

The attention backend consumes ``workspace`` as the layer's SWA buffer and
writes the extend's rows there; ``commit`` stores them back into the request
ring after attention.
"""

from typing import Optional

import msgspec
import torch

from sglang.srt.hardware_backend.npu.dsv4.ring_layout import (
    RingPageTableCache,
    ring_row_loc,
)
from sglang.srt.mem_cache.dsv41_request_window import WindowLayout, window_layout


class NPURequestWindowSpec(msgspec.Struct, frozen=True, kw_only=True):
    num_slots: int
    layers: int
    page_size: int
    capacity: int
    workspace_rows: int
    row_dim: int
    row_dtype: torch.dtype

    def __post_init__(self):
        if (
            min(
                self.num_slots,
                self.layers,
                self.page_size,
                self.capacity,
                self.workspace_rows,
                self.row_dim,
            )
            < 1
        ):
            raise ValueError("request-window extents must be positive")
        if self.capacity % self.page_size or self.workspace_rows % self.page_size:
            raise ValueError("request-window extents must contain whole pages")

    @property
    def tag_rows(self) -> int:
        return self.num_slots * self.capacity + self.page_size

    @property
    def state_rows(self) -> int:
        # Page-aligned so decode can view the ring as a paged PA_ND buffer; the
        # zero and sink rows sit right after the request rows.
        return self.num_slots * self.capacity + self.page_size

    @property
    def nbytes(self) -> int:
        row_bytes = torch.empty(0, dtype=self.row_dtype).element_size() * self.row_dim
        return (
            self.layers * self.state_rows + self.workspace_rows
        ) * row_bytes + self.layers * self.tag_rows * 8

    @classmethod
    def create(
        cls,
        *,
        num_slots: int,
        layers: int,
        page_size: int,
        capacity: int,
        workspace_rows: int,
        row_dim: int,
        row_dtype: torch.dtype,
    ):
        return cls(
            num_slots=num_slots,
            layers=layers,
            page_size=page_size,
            capacity=-(-capacity // page_size) * page_size,
            workspace_rows=-(-workspace_rows // page_size) * page_size,
            row_dim=row_dim,
            row_dtype=row_dtype,
        )


class NPUDecodeWindowLayout(msgspec.Struct, frozen=True):
    """One decode step staged into the workspace.

    The ring's last ``capacity`` rows are non-monotonic when they wrap, and the
    A3 sparse-attention operator needs ascending row indices. ``buffer_decode``
    gathers the ring window into request-major workspace rows ``0..capacity-1``
    and the attention reads those with ascending indices; ``commit_decode``
    then writes the new K back into the ring.
    """

    indices: torch.Tensor  # [B, capacity] workspace rows ascending, -1 invalid
    lengths: torch.Tensor  # [B]
    write_loc: torch.Tensor  # [B] workspace row receiving the new K
    commit_mask: torch.Tensor  # [B] false rows commit to the unused sink row
    history_src: torch.Tensor  # [B * capacity] ring rows to gather
    history_loc: torch.Tensor  # [B * capacity] workspace rows
    history_pos: torch.Tensor
    history_valid: torch.Tensor
    req: torch.Tensor
    positions: torch.Tensor
    size: int


class NPUWindowLayout(msgspec.Struct, frozen=True):
    """Compact prefill window: only valid history rows occupy the workspace.

    A3's sparse-attention operator drops explicit indices >= ``seqused_kv``, so
    the fixed 128-row history region of the CUDA layout cannot be used: the
    history region is sized to the positions actually replayable and the live
    rows follow immediately, keeping every index below ``seqused``.
    ``kv_lens`` is each request's staged row span; the window side feeds it as
    ``seqused_ori_kv`` so the operator addresses the workspace relatively.
    """

    indices: torch.Tensor
    lengths: torch.Tensor
    kv_lens: torch.Tensor
    write_loc: torch.Tensor
    commit_mask: torch.Tensor
    history_src: torch.Tensor
    history_loc: torch.Tensor
    history_pos: torch.Tensor
    history_valid: torch.Tensor
    size: int
    req: torch.Tensor
    positions: torch.Tensor


def _cpu_max(values, rows: int) -> int:
    """Largest of the first ``rows`` host values (``None`` and empty are 0)."""
    if values is None:
        return 0
    values = values[:rows]
    if torch.is_tensor(values):
        return int(values.max()) if values.numel() else 0
    values = list(values)
    return max(values) if values else 0


def _history_rows(
    *,
    total: int,
    hist_start: torch.Tensor,
    hist_len: torch.Tensor,
    group_req: torch.Tensor,
    base: torch.Tensor,
    row_loc,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack every request's replayable history rows into flat gather lists.

    ``hist_len`` is a device tensor, so the group of each flat row comes from a
    device-side search over its prefix sum: one host read of the caller's total
    replaces one device-to-host scalar read per request.
    """
    if total == 0:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return empty, empty, empty, torch.empty(0, dtype=torch.bool, device=device)
    group_end = hist_len.cumsum(0)
    flat = torch.arange(total, dtype=torch.int64, device=device)
    group = torch.searchsorted(group_end, flat, right=True)
    offset = flat - (group_end - hist_len)[group]
    positions = hist_start[group] + offset
    return (
        row_loc(group_req[group], positions),
        base[group] + offset,
        positions,
        torch.ones(total, dtype=torch.bool, device=device),
    )


def _row_view(t: torch.Tensor) -> torch.Tensor:
    """Row gather/scatter view of a request-window tensor.

    A5 keeps fp8 rows, and the NPU ``index_select`` / ``index_copy_`` kernels
    reject ``DT_FLOAT8_E4M3FN``.  A byte view moves the same bytes and supports
    both ops; wider rows are returned unchanged.
    """
    return t.view(torch.uint8) if t.element_size() == 1 else t


class NPURequestWindowMemory:
    def __init__(self, spec: NPURequestWindowSpec, *, device: torch.device):
        self.spec = spec
        self.state = torch.empty(
            (spec.layers, spec.state_rows, spec.row_dim),
            dtype=spec.row_dtype,
            device=device,
        )
        # Invalid history rows read the zero row, so it must start zeroed.
        # A5 fp8 rows do not implement fill_ (aclnnInplaceFillScalar), so zero
        # the trailing rows through a byte view, which works for every dtype.
        self.state.view(torch.uint8)[:, spec.num_slots * spec.capacity :].zero_()
        self.tags = torch.full(
            (spec.layers, spec.tag_rows), -1, dtype=torch.int64, device=device
        )
        self.workspace = torch.empty(
            (spec.workspace_rows, spec.row_dim),
            dtype=spec.row_dtype,
            device=device,
        )
        # The sparse-attention operator may scan whole pages past the rows a
        # layout stages; uninitialized workspace rows would then read as NaN.
        # A5 fp8 rows do not implement fill_, so zero through a byte view.
        self.workspace.view(torch.uint8).zero_()
        self.view = NPURequestWindow(
            spec, state=self.state, tags=self.tags, workspace=self.workspace
        )

    def reset_rows(self, rows) -> None:
        indices = torch.as_tensor(rows, dtype=torch.int64, device=self.view.tags.device)
        self.view.reset(indices)


class NPURequestWindow:
    def __init__(self, spec, *, state, tags, workspace):
        self.spec = spec
        self.capacity = spec.capacity
        self.page_size = spec.page_size
        self.num_slots = spec.num_slots
        self.rows = spec.num_slots * spec.capacity
        self.zero_row = self.rows
        self.sink_row = self.rows + 1
        self.state = state
        self.tags = tags
        self.workspace = workspace
        self.layout: Optional[WindowLayout] = None
        self.prepared = None
        self._idle_layout = None
        # Captured graphs cannot branch on the commit mask; eager commits do.
        self.graph_capture = False
        # Prefill stages history in the workspace; decode reads and writes the
        # persistent request ring directly.
        self.prefill = True

    def set_phase(self, *, prefill: bool) -> None:
        if self.prefill != prefill:
            self.prefill = prefill
            self.prepared = None

    def state_view(self, layer: int) -> torch.Tensor:
        """The layer's request ring as a paged PA_ND buffer."""
        page_size = self.page_size
        return self.state[layer].view(-1, page_size, 1, self.state.shape[-1])

    def row_loc(self, slots, positions):
        return ring_row_loc(slots, positions, self.capacity)

    def page_indices(self, slot: int, start: int, end: int):
        """Physical page ids of ``slot``'s ring rows in [start, end).

        Mirrors the paged request window: capacity is whole pages, so a page's
        rows keep a paged pool's offsets and pair with decode's paged SWA pages
        one for one during PD transfer.
        """
        assert start % self.page_size == 0, (start, self.page_size)
        positions = torch.arange(start, end, self.page_size)
        slot = torch.as_tensor(slot, dtype=torch.int64)
        return (
            (self.row_loc(slot, positions) // self.page_size)
            .to(torch.int32)
            .numpy()
        )

    def activate(self, layout: WindowLayout) -> None:
        if self.layout is layout:
            return
        if self.workspace.shape[0] < layout.size:
            raise RuntimeError(
                f"request-window workspace too small: {self.workspace.shape[0]} rows "
                f"for a layout of {layout.size}"
            )
        self.layout = layout
        self.prepared = None

    def build_prefill_layout(
        self, req, positions, floor=None, num_groups=None, window=None
    ):
        capacity = self.capacity
        # The ring stores ``capacity`` positions (window + draft rows), but the
        # attention window stays ``window`` wide; conflating them over-attends.
        window = capacity if window is None else int(window)
        device = positions.device
        n = positions.numel()
        req = req.to(torch.int64)
        pos = positions.to(torch.int64)
        offset = torch.arange(n, dtype=torch.int64, device=device)
        starts = torch.ones(n, dtype=torch.bool, device=device)
        starts[1:] = req[1:] != req[:-1]
        group = starts.cumsum(0) - 1
        group_first = torch.cummax(
            torch.where(starts, offset, torch.zeros_like(offset)), dim=0
        ).values
        first_pos = pos - (offset - group_first)
        floor = (
            torch.zeros(n, dtype=torch.int64, device=device)
            if floor is None
            else floor.reshape(-1).to(torch.int64)
        )
        groups = int(num_groups) if num_groups is not None else int(group[-1]) + 1
        g_first = torch.zeros(groups, dtype=torch.int64, device=device).scatter_(
            0, group, first_pos
        )
        g_floor = torch.zeros(groups, dtype=torch.int64, device=device).scatter_(
            0, group, floor
        )
        g_req = torch.zeros(groups, dtype=torch.int64, device=device).scatter_(
            0, group, req
        )
        g_count = torch.zeros(groups, dtype=torch.int64, device=device).scatter_add_(
            0, group, torch.ones(n, dtype=torch.int64, device=device)
        )
        # History rows are the replayable positions just before the extend:
        # [max(0, floor, first - window), first). A fresh request (or a folded
        # one, whose floor equals its first position) has none.
        hist_start = torch.clamp(g_first - window, min=0)
        hist_start = torch.maximum(hist_start, torch.clamp(g_floor, min=0))
        hist_len = torch.clamp(g_first - hist_start, min=0, max=window)
        spans = hist_len + g_count
        base = torch.cumsum(spans, 0) - spans
        size = int(spans.sum())

        history_src, history_loc, history_pos, history_valid = _history_rows(
            total=size - n,
            hist_start=hist_start,
            hist_len=hist_len,
            group_req=g_req,
            base=base,
            row_loc=self.row_loc,
            device=device,
        )

        base_row = base[group]
        hist_row = hist_len[group]
        hist_lo = hist_start[group]
        write_loc = (base_row + hist_row + (pos - first_pos)).to(torch.int32)
        offsets = torch.arange(window, dtype=torch.int64, device=device)
        seen = pos.unsqueeze(1) - (window - 1) + offsets.unsqueeze(0)
        valid = (seen >= 0) & (seen >= hist_lo.unsqueeze(1))
        old_rows = base_row.unsqueeze(1) + (seen - hist_lo.unsqueeze(1))
        new_rows = (
            base_row.unsqueeze(1)
            + hist_row.unsqueeze(1)
            + (seen - first_pos.unsqueeze(1))
        )
        rows = torch.where(seen < first_pos.unsqueeze(1), old_rows, new_rows)
        indices = torch.where(valid, rows, rows.new_full((), -1)).to(torch.int32)
        lengths = valid.sum(-1, dtype=torch.int32)
        group_last = torch.zeros(groups, dtype=torch.int64, device=device).scatter_(
            0, group, offset
        )
        commit_mask = (group_last[group] - offset) < capacity
        return NPUWindowLayout(
            indices=indices,
            lengths=lengths,
            kv_lens=spans.to(torch.int32),
            write_loc=write_loc,
            commit_mask=commit_mask,
            history_src=history_src,
            history_loc=history_loc,
            history_pos=history_pos,
            history_valid=history_valid,
            size=size,
            req=req,
            positions=pos,
        )

    def build_verify_layout(
        self,
        req,
        positions,
        *,
        num_draft: int,
        window: int,
        live=None,
    ) -> NPUWindowLayout:
        """Stage a DSpark verify block on top of its trailing context window.

        ``positions`` are request-major verify positions ``[bs * num_draft]``;
        rows ``0..ctx-1`` stage the context from the ring, rows ``ctx..ctx+n-1``
        are the verify rows the target KV store writes before attention. Each
        query attends from its own window start through its own row, and only
        live verify rows commit back to the ring.
        """
        device = positions.device
        req = req.to(torch.int64)
        pos = positions.to(torch.int64)
        num_rows = pos.numel()
        if num_rows % num_draft:
            raise ValueError(
                f"verify rows {num_rows} are not a multiple of {num_draft}"
            )
        ctx_start, ctx_len, step, group, indices, lengths = self._verify_window(
            pos, num_draft=num_draft, window=window
        )
        spans = ctx_len + num_draft
        base = torch.cumsum(spans, 0) - spans
        size = int(spans.sum())

        history_src, history_loc, history_pos, history_valid = _history_rows(
            total=size - num_rows,
            hist_start=ctx_start,
            hist_len=ctx_len,
            group_req=req[::num_draft],
            base=base,
            row_loc=self.row_loc,
            device=device,
        )

        write_loc = base[group] + ctx_len[group] + step
        commit_mask = (
            torch.ones(num_rows, dtype=torch.bool, device=device)
            if live is None
            else live.to(torch.bool).reshape(-1)
        )
        return NPUWindowLayout(
            indices=indices,
            lengths=lengths,
            kv_lens=spans.to(torch.int32),
            write_loc=write_loc,
            commit_mask=commit_mask,
            history_src=history_src,
            history_loc=history_loc,
            history_pos=history_pos,
            history_valid=history_valid,
            size=size,
            req=req,
            positions=pos,
        )

    def _verify_window(self, positions, *, num_draft: int, window: int):
        """Compact window rows of a verify block on its staged context.

        Each query's window starts at ``max(0, pos - window + 1)``; it lies at
        or after the staged context start by construction.
        """
        device = positions.device
        rows = positions.numel()
        first = positions[::num_draft]
        group = torch.arange(rows, device=device) // num_draft
        step = torch.arange(rows, device=device) % num_draft
        ctx_start = torch.clamp(first - (window - 1), min=0)
        ctx_len = first - ctx_start
        offsets = torch.arange(window, dtype=torch.int64, device=device)
        local_lo = torch.clamp(positions - (window - 1), min=0) - ctx_start[group]
        row_idx = local_lo.unsqueeze(1) + offsets.unsqueeze(0)
        valid = offsets.unsqueeze(0) < (ctx_len[group] + step + 1).unsqueeze(1)
        indices = torch.where(valid, row_idx, row_idx.new_full((), -1)).to(torch.int32)
        lengths = valid.sum(-1, dtype=torch.int32)
        return ctx_start, ctx_len, step, group, indices, lengths

    def _decode_window(self, req, positions, window):
        """Compact decode window: valid rows fill 0..L-1 for every query.

        Compact indices stay below seqused, which the A3 operator requires.
        """
        device = positions.device
        offsets = torch.arange(window, dtype=torch.int64, device=device)
        start = torch.clamp(positions - (window - 1), min=0)
        win_pos = start.unsqueeze(1) + offsets.unsqueeze(0)
        valid = win_pos <= positions.unsqueeze(1)
        src = self.row_loc(req.unsqueeze(1), win_pos)
        src = torch.where(valid, src, torch.full_like(src, self.zero_row))
        indices = torch.where(
            valid,
            offsets.unsqueeze(0).expand_as(valid),
            torch.full((), -1, dtype=torch.int64, device=device),
        )
        lengths = valid.sum(-1, dtype=torch.int32)
        return indices.to(torch.int32), lengths, src, win_pos, valid

    def build_decode_layout(self, req, positions, window=None) -> NPUDecodeWindowLayout:
        capacity = self.capacity
        # Ring rows are addressed by ``capacity``; the query attends ``window``.
        window = capacity if window is None else int(window)
        device = positions.device
        batch = positions.numel()
        indices, lengths, src, win_pos, valid = self._decode_window(
            req, positions, window
        )
        rows = lengths.to(torch.int64) - 1
        return NPUDecodeWindowLayout(
            indices=indices,
            lengths=lengths,
            write_loc=rows,
            commit_mask=torch.ones(batch, dtype=torch.bool, device=device),
            history_src=src.reshape(-1),
            history_loc=torch.arange(batch * window, dtype=torch.int64, device=device),
            history_pos=win_pos.reshape(-1),
            history_valid=valid.reshape(-1),
            req=req,
            positions=positions,
            size=batch * window,
        )

    def _graph_scratch_row(self) -> int:
        """The workspace row graph layouts scatter their padding into."""
        return self.workspace.shape[0] - 1

    def build_graph_decode_layout(
        self, batch: int, *, window: int
    ) -> NPUDecodeWindowLayout:
        """Fixed-shape decode layout for NPU graph capture.

        Every tensor is graph-owned; ``refresh_graph_decode_layout`` rewrites
        its contents in place before each replay.
        """
        device = self.tags.device
        rows = batch * window
        if rows >= self.workspace.shape[0]:
            raise RuntimeError(
                f"request-window workspace too small for a graph decode layout: "
                f"{rows} rows for {self.workspace.shape[0]}"
            )
        return NPUDecodeWindowLayout(
            indices=torch.full((batch, window), -1, dtype=torch.int32, device=device),
            lengths=torch.ones(batch, dtype=torch.int32, device=device),
            write_loc=torch.zeros(batch, dtype=torch.int64, device=device),
            commit_mask=torch.ones(batch, dtype=torch.bool, device=device),
            history_src=torch.full(
                (rows,), self.zero_row, dtype=torch.int64, device=device
            ),
            history_loc=torch.arange(rows, dtype=torch.int64, device=device),
            history_pos=torch.zeros(rows, dtype=torch.int64, device=device),
            history_valid=torch.zeros(rows, dtype=torch.bool, device=device),
            req=torch.zeros(batch, dtype=torch.int64, device=device),
            positions=torch.zeros(batch, dtype=torch.int64, device=device),
            size=rows,
        )

    @torch.no_grad()
    def refresh_graph_decode_layout(
        self, layout: NPUDecodeWindowLayout, req, positions, *, window: int, live=None
    ) -> None:
        device = layout.indices.device
        req = req.to(device=device, dtype=torch.int64).reshape(-1)
        positions = positions.to(device=device, dtype=torch.int64).reshape(-1)
        layout.req.copy_(req)
        layout.positions.copy_(positions)
        indices, lengths, src, win_pos, valid = self._decode_window(
            req, positions, window
        )
        layout.indices.copy_(indices)
        layout.lengths.copy_(lengths)
        layout.write_loc.copy_(lengths.to(torch.int64) - 1)
        layout.commit_mask.copy_(
            torch.ones_like(valid[:, 0])
            if live is None
            else live.to(device=device, dtype=torch.bool).reshape(-1)
        )
        layout.history_src.copy_(src.reshape(-1))
        layout.history_pos.copy_(win_pos.reshape(-1))
        layout.history_valid.copy_(valid.reshape(-1))

    def build_graph_verify_layout(
        self, batch: int, num_draft: int, *, window: int
    ) -> NPUWindowLayout:
        """Fixed-shape verify layout (context + block) for graph capture."""
        device = self.tags.device
        span = window - 1 + num_draft
        rows = batch * num_draft
        hist = batch * (window - 1)
        if batch * span > self._graph_scratch_row():
            raise RuntimeError(
                f"request-window workspace too small for a graph verify layout: "
                f"{batch * span} rows for {self.workspace.shape[0]}"
            )
        scratch = self._graph_scratch_row()
        return NPUWindowLayout(
            indices=torch.full((rows, window), -1, dtype=torch.int32, device=device),
            lengths=torch.zeros(rows, dtype=torch.int32, device=device),
            kv_lens=torch.zeros(batch, dtype=torch.int32, device=device),
            write_loc=torch.zeros(rows, dtype=torch.int32, device=device),
            commit_mask=torch.zeros(rows, dtype=torch.bool, device=device),
            history_src=torch.full(
                (hist,), self.zero_row, dtype=torch.int64, device=device
            ),
            history_loc=torch.full((hist,), scratch, dtype=torch.int64, device=device),
            history_pos=torch.zeros(hist, dtype=torch.int64, device=device),
            history_valid=torch.zeros(hist, dtype=torch.bool, device=device),
            size=batch * span,
            req=torch.zeros(rows, dtype=torch.int64, device=device),
            positions=torch.zeros(rows, dtype=torch.int64, device=device),
        )

    @torch.no_grad()
    def refresh_graph_verify_layout(
        self,
        layout: NPUWindowLayout,
        req,
        positions,
        live,
        *,
        num_draft: int,
        window: int,
    ) -> None:
        device = layout.indices.device
        req = req.to(device=device, dtype=torch.int64).reshape(-1)
        positions = positions.to(device=device, dtype=torch.int64).reshape(-1)
        rows = positions.numel()
        groups = rows // num_draft
        span = window - 1 + num_draft
        ctx_start, ctx_len, step, group, indices, lengths = self._verify_window(
            positions, num_draft=num_draft, window=window
        )
        g_base = torch.arange(groups, dtype=torch.int64, device=device) * span

        layout.req.copy_(req)
        layout.positions.copy_(positions)
        layout.kv_lens.copy_((ctx_len + num_draft).to(torch.int32))
        layout.write_loc.copy_((group * span + ctx_len[group] + step).to(torch.int32))
        layout.commit_mask.copy_(live.to(device=device, dtype=torch.bool).reshape(-1))
        layout.indices.copy_(indices)
        layout.lengths.copy_(lengths)

        # History slots: each request owns ``window - 1`` rows; unused ones
        # gather the zero row into the scratch workspace row.
        hist_rows = window - 1
        h_group = (
            torch.arange(groups * hist_rows, dtype=torch.int64, device=device)
            // hist_rows
        )
        h_off = (
            torch.arange(groups * hist_rows, dtype=torch.int64, device=device)
            % hist_rows
        )
        h_valid = h_off < ctx_len[h_group]
        h_pos = ctx_start[h_group] + h_off
        h_src = self.row_loc(req[h_group * num_draft], h_pos)
        scratch = h_pos.new_full((), self._graph_scratch_row())
        layout.history_src.copy_(
            torch.where(h_valid, h_src, h_src.new_full((), self.zero_row))
        )
        layout.history_loc.copy_(torch.where(h_valid, g_base[h_group] + h_off, scratch))
        layout.history_pos.copy_(torch.where(h_valid, h_pos, torch.zeros_like(h_pos)))
        layout.history_valid.copy_(h_valid)

    def activate_decode(self, layout: NPUDecodeWindowLayout) -> None:
        if self.layout is layout:
            return
        # Decode stages through the workspace, so it uses the prefill gather path.
        self.prefill = True
        if self.workspace.shape[0] < layout.size:
            raise RuntimeError(
                f"request-window workspace too small: {self.workspace.shape[0]} rows "
                f"for a decode layout of {layout.size}"
            )
        self.layout = layout
        self.prepared = None

    def _scatter_commit(self, layer, req, positions, write_loc, mask) -> None:
        if self.graph_capture:
            # Fixed-shape scatter: masked rows land in the unused sink row, so
            # the same op is valid inside a captured graph.
            dst = self.row_loc(req, positions)
            dst = torch.where(mask, dst, dst.new_full((), self.sink_row))
            _row_view(self.state[layer]).index_copy_(
                0,
                dst,
                _row_view(self.workspace).index_select(
                    0, write_loc.to(torch.int64)
                ),
            )
            self.tags[layer].index_copy_(0, dst, positions.to(torch.int64))
        else:
            dst = self.row_loc(req[mask], positions[mask])
            _row_view(self.state[layer]).index_copy_(
                0,
                dst,
                _row_view(self.workspace).index_select(
                    0, write_loc[mask].to(torch.int64)
                ),
            )
            self.tags[layer].index_copy_(0, dst, positions[mask].to(torch.int64))
        self.prepared = None

    def commit_decode(self, layer: int, layout: NPUDecodeWindowLayout) -> None:
        self._scatter_commit(
            layer,
            layout.req,
            layout.positions,
            layout.write_loc,
            layout.commit_mask,
        )

    def activate_idle(self, num_rows: int) -> torch.Tensor:
        layout = self._idle_layout
        if layout is None or layout.pos.numel() != num_rows:
            zeros = torch.zeros(num_rows, dtype=torch.int64, device=self.tags.device)
            layout = window_layout(
                zeros,
                zeros,
                zero_row=self.zero_row,
                capacity=self.capacity,
                num_groups=1,
            )
            self._idle_layout = layout
        self.activate(layout)
        return layout.write_loc

    def reset(self, slots) -> None:
        loc = slots.to(torch.int64)[:, None] * self.capacity + torch.arange(
            self.capacity, device=slots.device
        )
        self.tags.index_fill_(1, loc.flatten(), -1)
        self.prepared = None

    def initialize_dummy_history(self) -> None:
        layout = self.layout
        self.tags.fill_(-1)
        # See __init__: float8 rows cannot use fill_ on A5.
        self.state.view(torch.uint8).zero_()
        if layout.history_src.numel():
            self.tags[:, layout.history_src.to(torch.int64)] = layout.history_pos
        self.prepared = None

    def buffer(self, layer: int) -> torch.Tensor:
        """Copy the layer's history rows into the workspace and return it."""
        if self.prepared != layer:
            layout = self.layout
            if layout is None:
                raise RuntimeError("request-window metadata was not activated")
            src = layout.history_src.to(torch.int64)
            _row_view(self.workspace).index_copy_(
                0,
                layout.history_loc.to(torch.int64),
                _row_view(self.state[layer]).index_select(0, src),
            )
            self.prepared = layer
        return self.workspace

    def commit(self, layer: int) -> None:
        layout = self.layout
        self._scatter_commit(
            layer,
            layout.req,
            layout.positions,
            layout.write_loc,
            layout.commit_mask,
        )


class NPURequestWindowController:
    """Bind forward metadata to the row-style window for one attention backend."""

    def __init__(
        self,
        *,
        window: NPURequestWindow,
        page_size: int,
        sliding_window: int,
        device: torch.device,
        graph_metadata,
        graph_tokens_per_req: int,
        copy_page_table,
    ) -> None:
        self.window = window
        self.page_size = page_size
        self.sliding_window = sliding_window
        self.device = device
        # The backend rebinds graph_metadata when graph buffers are built.
        self.graph_metadata = graph_metadata
        self.graph_tokens_per_req = graph_tokens_per_req
        self.copy_page_table = copy_page_table
        self.ring_pages = window.capacity // page_size
        self._ring_page_tables = RingPageTableCache()
        self._graph_page_patterns: dict[tuple, torch.Tensor] = {}

    def _page_table(self, layout, batch_size, positions, max_seq: int):
        window = self.window
        workspace_pages = window.workspace.shape[0] // window.page_size
        layout_pages = -(-layout.size // window.page_size)
        seq_pages = -(-max_seq // window.page_size)
        num_pages = max(layout_pages, seq_pages, 1)
        ident = torch.arange(num_pages, dtype=torch.int32, device=positions.device)
        ident = torch.where(
            ident < layout_pages, ident % workspace_pages, torch.zeros_like(ident)
        )
        # The ds41 DSL attention kernel rejects a broadcast block table
        # (stride[0] == 0), so materialize the rows.
        return ident.unsqueeze(0).expand(batch_size, -1).contiguous()

    def _max_seq_len(self, fm, forward_batch, rows: int) -> int:
        """Largest sequence length from host metadata.

        Reads the device only when no host mirror exists; the CPU lengths are
        the scheduler's own copy, so this does not add a D2H copy.
        """
        max_seq = _cpu_max(forward_batch.seq_lens_cpu, rows)
        mirror = getattr(fm, "seq_lens_cpu_int", None)
        max_seq = max(max_seq, _cpu_max(mirror, rows))
        if max_seq > 0:
            return max_seq
        seq_lens = forward_batch.seq_lens
        return int(seq_lens.max()) if seq_lens.numel() else 1

    def init_prefill(self, fm, forward_batch) -> None:
        window = self.window
        window.graph_capture = False
        window.set_phase(prefill=True)
        # Metadata is host-side: expand requests from the CPU lengths instead of
        # a device repeat_interleave, which is not reliable on NPU.
        lens_cpu = forward_batch.extend_seq_lens_cpu
        if not isinstance(lens_cpu, list):
            lens_cpu = lens_cpu.tolist()
        num_rows = sum(lens_cpu)
        req_cpu = forward_batch.req_pool_indices_cpu
        if req_cpu is None:
            req_cpu = forward_batch.req_pool_indices.cpu()
        req = torch.repeat_interleave(
            req_cpu.to(torch.int64),
            torch.tensor(lens_cpu, dtype=torch.int64),
        ).to(forward_batch.positions.device)
        positions = forward_batch.positions[:num_rows].to(torch.int64)
        floor = forward_batch.encoder_swa_row_floor
        if floor is not None:
            floor = floor.reshape(-1).to(torch.int64)
            if floor.numel() < num_rows:
                floor = torch.cat([floor, floor.new_zeros(num_rows - floor.numel())])
            floor = floor[:num_rows]
        layout = window.build_prefill_layout(
            req,
            positions,
            floor=floor,
            num_groups=forward_batch.batch_size,
            window=self.sliding_window,
        )
        window.activate(layout)
        fm.request_window_layout = layout
        fm.dsv41_window_kv_lens = layout.kv_lens
        fm.swa_loc = layout.write_loc
        indices = layout.indices.unsqueeze(1).contiguous()
        lengths = layout.lengths.unsqueeze(1).contiguous()
        fm.dsv41_win_sparse_indices = indices
        fm.dsv41_win_topk_length = lengths
        fm.swa_page_table = self._page_table(
            layout,
            forward_batch.batch_size,
            positions,
            self._max_seq_len(fm, forward_batch, forward_batch.batch_size),
        )

    def init_decode(self, fm, forward_batch) -> None:
        window = self.window
        window.graph_capture = False
        positions = forward_batch.positions.reshape(-1).to(torch.int64)
        req = forward_batch.req_pool_indices.reshape(-1).to(torch.int64)
        if req.numel() != positions.numel():
            req = torch.repeat_interleave(req, positions.numel() // req.numel())
        layout = window.build_decode_layout(req, positions, window=self.sliding_window)
        window.activate_decode(layout)
        fm.request_window_layout = layout
        fm.request_window_decode = True
        fm.dsv41_window_kv_lens = layout.lengths
        fm.dsv41_win_sparse_indices = layout.indices.unsqueeze(1).contiguous()
        fm.dsv41_win_topk_length = layout.lengths.unsqueeze(1).contiguous()
        fm.swa_page_table = self._page_table(
            layout,
            positions.numel(),
            positions,
            self._max_seq_len(fm, forward_batch, positions.numel()),
        )

    def init_verify(self, fm, forward_batch) -> None:
        window = self.window
        window.graph_capture = False
        window.set_phase(prefill=True)
        num_draft = int(forward_batch.spec_info.draft_token_num)
        if self.sliding_window - 1 + num_draft > window.capacity:
            raise RuntimeError(
                "request-window capacity "
                f"{window.capacity} cannot stage a {num_draft}-row verify block "
                f"with a {self.sliding_window}-token window"
            )
        bs = int(forward_batch.batch_size)
        num_rows = bs * num_draft
        req = forward_batch.req_pool_indices.reshape(-1).to(torch.int64)
        req = torch.repeat_interleave(req, num_draft)[:num_rows]
        positions = forward_batch.positions.reshape(-1)[:num_rows].to(torch.int64)
        live = forward_batch.out_cache_loc.reshape(-1)[:num_rows] >= self.page_size
        layout = window.build_verify_layout(
            req,
            positions,
            num_draft=num_draft,
            window=self.sliding_window,
            live=live,
        )
        window.activate(layout)
        fm.request_window_layout = layout
        fm.dsv41_window_kv_lens = layout.kv_lens
        fm.swa_loc = layout.write_loc
        indices = layout.indices.unsqueeze(1).contiguous()
        lengths = layout.lengths.unsqueeze(1).contiguous()
        fm.dsv41_win_sparse_indices = indices
        fm.dsv41_win_topk_length = lengths
        fm.swa_page_table = self._page_table(
            layout,
            bs,
            positions,
            self._max_seq_len(fm, forward_batch, bs),
        )

    def _logical_pages(self, fm, forward_batch, rows: int) -> int:
        # The operator scans the table up to seqused_kv, which can exceed the
        # committed length by a speculative step; one spare page keeps that
        # scan in bounds (extra entries wrap onto valid ring pages).
        return max(
            1, -(-self._max_seq_len(fm, forward_batch, rows) // self.page_size) + 1
        )

    def init_decode_direct(self, fm, forward_batch) -> None:
        """Decode reads and writes the request ring directly (no staging)."""
        window = self.window
        window.graph_capture = False
        window.set_phase(prefill=False)
        positions = forward_batch.positions.reshape(-1).to(torch.int64)
        req = forward_batch.req_pool_indices.reshape(-1).to(torch.int64)
        if req.numel() != positions.numel():
            req = torch.repeat_interleave(req, positions.numel() // req.numel())
        fm.swa_loc = window.row_loc(req, positions).to(torch.int64)
        fm.swa_page_table = self._ring_page_tables.table(
            req_pool_indices=req,
            ring_pages=self.ring_pages,
            logical_pages=self._logical_pages(fm, forward_batch, int(req.numel())),
        )

    def init_verify_direct(self, fm, forward_batch) -> None:
        """Verify writes its block rows and reads the ring directly."""
        window = self.window
        window.graph_capture = False
        window.set_phase(prefill=False)
        num_draft = int(forward_batch.spec_info.draft_token_num)
        bs = int(forward_batch.batch_size)
        rows = bs * num_draft
        req_pool = forward_batch.req_pool_indices.reshape(-1).to(torch.int64)
        req = torch.repeat_interleave(req_pool, num_draft)[:rows]
        positions = forward_batch.positions.reshape(-1)[:rows].to(torch.int64)
        live = forward_batch.out_cache_loc.reshape(-1)[:rows] >= self.page_size
        row_loc = window.row_loc(req, positions).to(torch.int64)
        fm.swa_loc = torch.where(live, row_loc, row_loc.new_full((), -1))
        fm.swa_page_table = self._ring_page_tables.table(
            req_pool_indices=req_pool,
            ring_pages=self.ring_pages,
            logical_pages=self._logical_pages(fm, forward_batch, bs),
        )

    def init_graph_metadata(self, metadata, forward_batch) -> None:
        window = self.window
        window.graph_capture = True
        window.set_phase(prefill=True)
        bs = int(forward_batch.batch_size)
        mode = forward_batch.forward_mode
        table_rows = bs
        if mode.is_decode():
            layout = window.build_graph_decode_layout(bs, window=self.sliding_window)
            live = (
                torch.arange(bs, device=self.device)
                < bs - int(getattr(forward_batch, "num_padding", 0) or 0)
            ) & (forward_batch.out_cache_loc[:bs] >= self.page_size)
            window.refresh_graph_decode_layout(
                layout,
                forward_batch.req_pool_indices[:bs],
                forward_batch.positions[:bs],
                window=self.sliding_window,
                live=live,
            )
            window.activate_decode(layout)
            metadata.request_window_decode = True
            metadata.dsv41_window_kv_lens = layout.lengths
        elif mode.is_target_verify():
            num_draft = self.graph_tokens_per_req
            rows = bs * num_draft
            table_rows = rows
            layout = window.build_graph_verify_layout(
                bs, num_draft, window=self.sliding_window
            )
            window.refresh_graph_verify_layout(
                layout,
                forward_batch.req_pool_indices[:bs].repeat_interleave(num_draft),
                forward_batch.positions[:rows],
                forward_batch.out_cache_loc[:rows] >= self.page_size,
                num_draft=num_draft,
                window=self.sliding_window,
            )
            window.activate(layout)
            metadata.request_window_decode = False
            metadata.dsv41_window_kv_lens = layout.kv_lens
        else:
            return
        metadata.request_window_layout = layout
        metadata.swa_loc = layout.write_loc
        metadata.dsv41_win_sparse_indices = layout.indices.unsqueeze(1).contiguous()
        metadata.dsv41_win_topk_length = layout.lengths.unsqueeze(1).contiguous()
        self.refresh_graph_page_table(
            bs, layout, self._max_seq_len(None, forward_batch, table_rows)
        )

    def refresh_graph_page_table(self, bs: int, layout, max_seq: int = 0) -> None:
        window = self.window
        workspace_pages = window.workspace.shape[0] // window.page_size
        layout_pages = max(1, -(-layout.size // window.page_size))
        # The operator scans the table up to the request's sequence length, not
        # just the staged verify block, so cover both. Every column below the
        # width wraps onto the workspace ring; leaving the rest at the -1
        # sentinel makes long sequences read invalid rows.
        seq_pages = max(1, -(-int(max_seq) // window.page_size))
        num_pages = max(layout_pages, seq_pages)
        width = min(self.graph_metadata()["swa_page_table"].shape[1], num_pages)
        pattern = self._graph_page_pattern(workspace_pages)[:width]
        self.copy_page_table(
            "swa_page_table", pattern.unsqueeze(0).expand(int(bs), -1)
        )

    def _graph_page_pattern(self, workspace_pages: int) -> torch.Tensor:
        """Graph SWA table columns: ring page ``p % workspace_pages`` at ``p``.

        Cached because it only changes with the graph table's width, and the
        replay path must not allocate per step.
        """
        width = int(self.graph_metadata()["swa_page_table"].shape[1])
        key = (workspace_pages, width, self.device)
        pattern = self._graph_page_patterns.get(key)
        if pattern is None:
            pages = torch.arange(width, dtype=torch.int32, device=self.device)
            pattern = pages % workspace_pages
            self._graph_page_patterns[key] = pattern
        return pattern

    @staticmethod
    def _has_graph_rows(fb, rows: int) -> bool:
        """Whether a replay batch carries enough rows to refresh a layout."""
        return fb.positions.numel() >= rows and fb.out_cache_loc.numel() >= rows

    def refresh_graph(self, ctx) -> None:
        layout = getattr(ctx.fm, "request_window_layout", None)
        if layout is None:
            return
        fb = ctx.forward_batch
        if ctx.graph_mode.is_decode():
            if not self._has_graph_rows(fb, ctx.bs):
                # Idle replay: keep the captured dummy rows and commit nothing.
                layout.commit_mask.fill_(False)
                return
            live = (torch.arange(ctx.bs, device=ctx.device) < ctx.raw_bs) & (
                fb.out_cache_loc[: ctx.bs] >= self.page_size
            )
            self.window.refresh_graph_decode_layout(
                layout,
                fb.req_pool_indices[: ctx.bs],
                fb.positions[: ctx.bs],
                window=self.sliding_window,
                live=live,
            )
        elif ctx.graph_mode.is_target_verify():
            num_draft = ctx.tokens_per_bs
            rows = ctx.bs * num_draft
            if not self._has_graph_rows(fb, rows):
                # Idle replay: keep the captured dummy rows and commit nothing.
                layout.commit_mask.fill_(False)
                return
            self.window.refresh_graph_verify_layout(
                layout,
                fb.req_pool_indices[: ctx.bs].repeat_interleave(num_draft),
                fb.positions[:rows],
                fb.out_cache_loc[:rows] >= self.page_size,
                num_draft=num_draft,
                window=self.sliding_window,
            )

    def activate(self, metadata) -> None:
        layout = getattr(metadata, "request_window_layout", None)
        if layout is None:
            return
        if getattr(metadata, "request_window_decode", False):
            self.window.activate_decode(layout)
        else:
            self.window.activate(layout)
