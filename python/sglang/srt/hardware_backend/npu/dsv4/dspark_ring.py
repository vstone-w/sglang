"""DSpark draft SWA ring addressing for the NPU draft worker.

The draft stores each row at ``slot * ring_size + position % ring_size`` and
reads it through a per-request page table for the A3 sparse-attention operator.
"""

from typing import Optional


def draft_swa_ring_size(pool) -> Optional[int]:
    return getattr(pool, "swa_req_ring_size", None)
