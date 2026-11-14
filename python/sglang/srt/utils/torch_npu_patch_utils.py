import sys
from collections.abc import Sequence
from typing import Any


def _restore_excepthook_origin() -> None:
    """Keep torch_npu's excepthook from recursing after a repeated patch.

    ``_apply_all_patches`` installs its excepthook by saving the current hook as
    the origin; a second call saves the already-installed hook, and every
    uncaught exception then dies in a RecursionError, hiding the real error.
    """
    from torch_npu.utils._error_code import _except_handler

    _except_handler._origin_excepthook = sys.__excepthook__


def apply_torch_npu_patches(torch_npu: Any, patches: Sequence[Sequence[Any]]) -> None:
    """Apply torch_npu patches across old and new torch_npu patch APIs."""
    if hasattr(torch_npu, "_apply_patches"):
        torch_npu._apply_patches(patches)
        return

    if hasattr(torch_npu, "_apply_all_patches"):
        torch_npu._apply_all_patches()
        _restore_excepthook_origin()
        return

    raise AttributeError(
        "torch_npu must provide either _apply_patches or _apply_all_patches"
    )
