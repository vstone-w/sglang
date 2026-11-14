"""DSpark draft SWA ring addressing for the NPU draft worker.

The draft stores each row at ``slot * ring_size + position % ring_size`` and
reads it through a per-request page table for the A3 sparse-attention operator.
"""

from typing import Optional

import torch

from sglang.srt.hardware_backend.npu.dsv4.ring_layout import make_ring_page_table


def draft_swa_ring_size(pool) -> Optional[int]:
    # The base pool sets this attribute only for a draft ring.
    return getattr(pool, "swa_req_ring_size", None)


def draft_ring_page_table(
    *,
    ring_size: int,
    page_size: int,
    req_pool_indices: torch.Tensor,
    logical_pages: int,
) -> torch.Tensor:
    return make_ring_page_table(
        req_pool_indices=req_pool_indices,
        ring_pages=ring_size // page_size,
        logical_pages=logical_pages,
    )


def seed_draft_graph_page_table(
    *,
    copy_page_table,
    ring_size: int,
    page_size: int,
    bs: int,
    logical_pages: int,
    device: torch.device,
) -> None:
    table = draft_ring_page_table(
        ring_size=ring_size,
        page_size=page_size,
        req_pool_indices=torch.arange(bs, device=device),
        logical_pages=logical_pages,
    )
    copy_page_table("swa_page_table", table)
