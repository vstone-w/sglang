"""Per-request ring addressing for the NPU SWA request window and draft ring.

A request's logical position ``p`` lives at row ``slot * ring + p % ring`` of
its ring; the A3 sparse-attention operator reaches it through a per-request
page table, one row of which maps logical page ``q`` to ring page
``q % ring_pages``.
"""

import torch


def ring_row_loc(slots, positions, ring: int) -> torch.Tensor:
    """Physical row of a request slot's ring for logical ``positions``."""
    return slots.to(torch.int64) * ring + positions.to(torch.int64) % ring


def sink_dead_rows(loc: torch.Tensor, sink: int) -> torch.Tensor:
    """Map dead (-1) store rows to an unused sink row without a host sync."""
    return torch.where(loc >= 0, loc, loc.new_full((), sink))


def _ring_page_base(req_pool_indices: torch.Tensor, ring_pages: int) -> torch.Tensor:
    return req_pool_indices.reshape(-1).to(torch.int32) * ring_pages


def _ring_page_pattern(ring_pages: int, width: int, device) -> torch.Tensor:
    return torch.arange(width, dtype=torch.int32, device=device) % ring_pages


def make_ring_page_table(
    *,
    req_pool_indices: torch.Tensor,
    ring_pages: int,
    logical_pages: int,
) -> torch.Tensor:
    base = _ring_page_base(req_pool_indices, ring_pages)
    pattern = _ring_page_pattern(ring_pages, logical_pages, req_pool_indices.device)
    return base[:, None] + pattern[None, :]


class RingPageTableCache:
    """Reusable per-request ring page tables.

    Logical page ``p`` of request slot ``s`` maps to ring page ``p % ring_pages``
    of that slot's ring, i.e. row ``s * ring_pages + p % ring_pages``. The table
    is rewritten in place each step on the compute stream, so one buffer per
    (rows, ring_pages, device) removes the per-step allocation; the buffer grows
    by doubling when a batch needs more logical pages. Extra columns past the
    caller's width wrap onto valid ring pages and are never read below
    ``seqused_kv``.
    """

    def __init__(self) -> None:
        self._tables: dict[tuple, torch.Tensor] = {}
        self._patterns: dict[tuple, torch.Tensor] = {}

    def table(
        self,
        *,
        req_pool_indices: torch.Tensor,
        ring_pages: int,
        logical_pages: int,
    ) -> torch.Tensor:
        req = req_pool_indices.reshape(-1)
        device = req.device
        ring_pages = int(ring_pages)
        key = (int(req.numel()), ring_pages, device)
        table = self._tables.get(key)
        if table is None or table.shape[1] < logical_pages:
            width = (
                int(logical_pages)
                if table is None
                else max(int(logical_pages), 2 * table.shape[1])
            )
            table = torch.empty((key[0], width), dtype=torch.int32, device=device)
            self._tables[key] = table
        pattern = self._patterns.get((ring_pages, table.shape[1], device))
        if pattern is None:
            pattern = _ring_page_pattern(ring_pages, table.shape[1], device)
            self._patterns[(ring_pages, table.shape[1], device)] = pattern
        table.copy_(_ring_page_base(req, ring_pages)[:, None])
        table.add_(pattern)
        # The cache doubles its width; callers copy this into a graph buffer
        # sized for the live table, so expose only the valid columns.
        return table[:, : int(logical_pages)]
