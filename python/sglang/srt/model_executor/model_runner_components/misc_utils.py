from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

import torch

from sglang.srt.arg_groups.choices import (
    CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS,
)
from sglang.srt.configs.model_config import (
    dsa_layer_skips_topk,
    is_deepseek_dsa,
    is_kimi_k3,
)
from sglang.srt.runtime_context import (
    attention_backends,
    get_context,
    get_observability,
    get_parallel,
    get_schedule,
)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


def maybe_disable_chunked_prefix_cache(
    *, use_mla_backend: bool, is_draft_worker: bool
) -> None:
    # Chunked prefix caching requires an MLA model on a backend whose
    # kernels read that layout. This is a load-time gate, not a
    # resolution-time one: out-of-tree platforms register their supported
    # backends in init_backend(), which runs when this module is imported
    # — after ServerArgs.__post_init__. Target runner only: a draft
    # model's (often non-MLA) config must not flip the shared setting.
    if is_draft_worker:
        return
    # Chunked prefix cache is a prefill feature: the prefill half decides.
    prefill_backend, _ = attention_backends()
    if (
        not use_mla_backend
        or prefill_backend not in CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS
    ):
        if not get_schedule().disable_chunked_prefix_cache:
            get_context().override(
                "model_runner.chunked_prefix_cache_gate",
                disable_chunked_prefix_cache=True,
            )
    if not get_schedule().disable_chunked_prefix_cache:
        logger.info("Chunked prefix cache is turned on.")


def create_msprobe_debugger() -> Optional[Any]:
    if get_observability().msprobe_dump_config is None:
        return None

    try:
        from msprobe.pytorch import PrecisionDebugger, seed_all
    except ImportError:
        logger.warning(
            "Please install msprobe for tensor data dump: pip install mindstudio-probe --pre, "
            "see https://gitcode.com/Ascend/msprobe for details."
        )
        return None

    seed_all(mode=True)
    return PrecisionDebugger(config_path=get_observability().msprobe_dump_config)


def resolve_pp_proxy_topk_size(
    *, model_config: ModelConfig, start_layer: int
) -> Optional[int]:
    hf_config = model_config.hf_text_config
    parallel = get_parallel()
    if (
        parallel.pp_size <= 1
        or parallel.pp_rank == 0
        or not is_deepseek_dsa(hf_config)
        or not dsa_layer_skips_topk(hf_config, start_layer)
    ):
        return None
    return getattr(hf_config, "index_topk", None)


def resolve_pp_proxy_residual_num_blocks(
    *, model_config: ModelConfig, start_layer: int
) -> Optional[int]:
    """Return the inherited Kimi K3 attention-residual bank width."""
    parallel = get_parallel()
    if (
        parallel.pp_size <= 1
        or parallel.pp_rank == 0
        or not is_kimi_k3(model_config.hf_config)
    ):
        return None

    block_size = getattr(model_config.hf_text_config, "attn_res_block_size", None)
    if block_size is None:
        return None
    return (start_layer + block_size - 1) // block_size


def resolve_pp_proxy_hc_pre_width(*, model_config: ModelConfig) -> int:
    """Width of the lagged mHC pre-mix a later PP stage receives."""
    parallel = get_parallel()
    if parallel.pp_size <= 1 or parallel.pp_rank == 0:
        return 0
    hf_config = model_config.hf_text_config
    if not getattr(hf_config, "hc_pre_from_prev_sublayer", False):
        return 0
    return int(getattr(hf_config, "hc_mult", 0))


def validate_pp_proxy_mhc_pre(*, pp_proxy_tensors: Any, width: int) -> None:
    """Reject a live proxy that cannot refresh a captured ``mhc_pre`` buffer.

    The graph runners copy only the keys the live proxy carries, so a missing
    or malformed ``mhc_pre`` (or a missing ``hidden_states``) would silently
    replay the previous batch's rows. ``None`` stays valid: a draft runner
    keeps its own singleton PP group and never consumes the target stage's
    proxy, and a target stage without one fails its own model-level check.
    """
    if pp_proxy_tensors is None or width <= 0:
        return
    tensors = pp_proxy_tensors.tensors
    pre = tensors.get("mhc_pre")
    if pre is None:
        raise ValueError(
            "PP proxy is missing 'mhc_pre' required by this stage; the "
            "previous stage must send its last sublayer's pre-mix"
        )
    if pre.dtype != torch.float32 or pre.dim() != 2 or pre.shape[1] != width:
        raise ValueError(
            "PP proxy 'mhc_pre' has the wrong layout: expected fp32 [T, "
            f"{width}], got {pre.dtype} {tuple(pre.shape)}"
        )
    hidden = tensors.get("hidden_states")
    if hidden is None:
        raise ValueError(
            "PP proxy is missing 'hidden_states' required by this stage; the "
            "graph replay would keep the previous batch's rows"
        )
    if pre.shape[0] != hidden.shape[0]:
        raise ValueError(
            "PP proxy 'mhc_pre' rows must match 'hidden_states': "
            f"{pre.shape[0]} != {hidden.shape[0]}"
        )


@runtime_checkable
class _SupportsDSparkPPProxy(Protocol):
    def get_pp_proxy_dspark_hidden_size(self) -> int: ...


def resolve_pp_proxy_dspark_hidden_size(
    *, model: Any, pp_size: int, pp_rank: int
) -> int:
    if pp_size <= 1 or pp_rank == 0:
        return 0
    if isinstance(model, _SupportsDSparkPPProxy):
        return model.get_pp_proxy_dspark_hidden_size()
    return 0


@runtime_checkable
class _SupportsPackedAuxHiddenStates(Protocol):
    def get_aux_hidden_states_width(self) -> int: ...


def resolve_aux_hidden_states_width(
    *, model: Any, spec_algorithm: Any, is_draft_worker: bool
) -> int:
    # Sharing needs the spec worker to consume target hidden states before
    # the next target forward overwrites them; the DFlash family does.
    if not spec_algorithm.is_dflash_family() or is_draft_worker:
        return 0
    if isinstance(model, _SupportsPackedAuxHiddenStates):
        return model.get_aux_hidden_states_width()
    return 0
