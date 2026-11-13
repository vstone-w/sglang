from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

from sglang.srt.arg_groups.engram_config import (
    EngramConfig,
    EngramLookup,
    EngramReplication,
    cols_per_rank,
)
from sglang.srt.arg_groups.model_override_base import (
    attention_backends_of,
    model_config_of,
    reject_dp_size_under_context_parallel,
    resolved_view,
    resolving_view,
)
from sglang.srt.arg_groups.overrides import (
    _deepseek_v4_kv_cache_dtype,
    declare_resolution,
    run_post_process_pass,
)
from sglang.srt.distributed.utils import (
    compute_pp_indices,
    parse_pp_layer_partition,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.runtime_context import (
    attn_dp_enabled_of,
    derive_attn_tp_size,
    get_platform,
)
from sglang.srt.utils.device import is_gfx95_supported, is_npu

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def _attention_is_tensor_parallel(cfg) -> bool:
    # validate_deepseek_v4_cp sets attn_cp_size to the whole TP group after this runs.
    if cfg.enable_prefill_cp:
        return False
    attn_tp_size = derive_attn_tp_size(
        tp_size=cfg.tp_size,
        attn_cp_size=cfg.attn_cp_size,
        attn_dp_size=cfg.attn_dp_size,
    )
    return attn_tp_size > 1


def _default_dsv4_attention_backend(server_args: ServerArgs) -> str:
    from sglang.srt.utils.device import is_sm100_supported

    cfg = resolving_view(server_args)
    layout = envs.SGLANG_DSV4_KV_LAYOUT
    if (
        model_config_of(server_args).hf_config.model_type != "deepseek_v41"
        or cfg.device != "cuda"
        or not is_sm100_supported()
        or cfg.kv_cache_dtype != "fp8_e4m3"
        # MegaAttention pads each rank to a 64-head tile under attention TP.
        or (cfg.disaggregation_mode == "null" and _attention_is_tensor_parallel(cfg))
        or (layout.is_set() and layout.get().lower() not in ("auto", "v41", "v41_new"))
    ):
        return "flashmla"
    return "flashmla_mega"


def _resolve_engram_config(server_args: ServerArgs) -> None:
    """Validate --engram-config and declare it. A `lookup` of `auto` stays as is:
    the layers resolve it at model construction (`resolve_engram_config`), once the
    parallel layout and the GPUs are known."""
    cfg = resolving_view(server_args)
    raw = cfg.engram_config
    if model_config_of(server_args).hf_config.model_type != "deepseek_v41":
        if raw is not None:
            raise ValueError("--engram-config applies to DeepSeek-V4.1 only.")
        return
    config = EngramConfig() if raw is None else EngramConfig.from_dict(raw)
    if config.lookup == EngramLookup.PEER_READ and not get_platform().is_cuda:
        raise ValueError("--engram-config: lookup 'peer_read' needs CUDA.")
    if config.lookup in (EngramLookup.REDUCE_SCATTER, EngramLookup.ALL_TO_ALL) and (
        cfg.attn_dp_size <= 1 or not get_platform().is_cuda
    ):
        raise ValueError(
            f"--engram-config: lookup {config.lookup!r} applies to DP attention on CUDA."
        )
    if config.lookup == EngramLookup.ALL_TO_ALL:
        text_config = model_config_of(server_args).hf_text_config
        columns = text_config.engram_n_heads * (text_config.engram_max_ngram_size - 1)
        if cols_per_rank(columns, cfg.tp_size) is None:
            raise ValueError(
                f"--engram-config: lookup 'all_to_all' needs the {columns} hash "
                f"columns to divide over --tp-size {cfg.tp_size}."
            )
    # One mapping per TP group, which its ranks open through /proc.
    if config.replication == EngramReplication.PER_NODE and cfg.nnodes > 1:
        raise ValueError(
            "--engram-config: replication 'per_node' needs the TP group on one node."
        )
    if config.replication == EngramReplication.GROUP_SHARDS:
        if config.group_size < cfg.tp_size:
            raise ValueError(
                "--engram-config: replication 'group_shards' needs 'group_size' of "
                "at least --tp-size."
            )
        if cfg.pp_size > 1:
            raise ValueError(
                "--engram-config: replication 'group_shards' does not support "
                "pipeline parallelism."
            )
    declare_resolution(server_args, "_resolve_engram_config", engram_config=config)
    logger.info("DeepSeek-V4.1 Engram tables: %s", config)


def apply_deepseek_v4_defaults(server_args: ServerArgs, model_arch: str) -> None:
    """Apply DeepSeek V4 environment defaults, request limits, and validation."""
    cfg = resolving_view(server_args)

    # FlashMLA sparse prefill (SGLANG_OPT_FLASHMLA_SPARSE_PREFILL, default on)
    # currently returns incorrect output for DeepSeek-V4-Flash on ROCm/HIP
    # (MI355X), which breaks the disaggregation nightly. Keep the previous
    # (dense prefill) behavior on ROCm until the sparse kernel is validated
    # there;
    if get_platform().is_hip and not envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.is_set():
        logger.warning(
            "Disabling SGLANG_OPT_FLASHMLA_SPARSE_PREFILL by default on ROCm/HIP "
            f"for {model_arch}; set it explicitly to override."
        )
        envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.set(False)

    # The kv-cache dtype default moved to the resolution pipeline
    # (arg_groups/overrides.py: _deepseek_v4_kv_cache_dtype), invoked here at
    # its legacy slot.

    run_post_process_pass(server_args, _deepseek_v4_kv_cache_dtype)
    _resolve_engram_config(server_args)

    if cfg.dsv4_attn_backend == "auto":
        declare_resolution(
            server_args,
            "apply_deepseek_v4_defaults",
            dsv4_attn_backend=_default_dsv4_attention_backend(server_args),
        )

    if cfg.dsv4_attn_backend == "flashmla_mega":
        from sglang.srt.utils.device import is_sm100_supported

        assert (
            model_config_of(server_args).hf_config.model_type == "deepseek_v41"
            and cfg.device == "cuda"
            and is_sm100_supported()
        ), "--dsv4-attn-backend flashmla_mega requires DeepSeek-V4.1 on SM100/SM103."

    if cfg.dsv4_attn_backend == "trtllm":
        from sglang.srt.utils.device import is_sm100_supported

        assert cfg.device == "cuda" and is_sm100_supported(), (
            "--dsv4-attn-backend trtllm requires an SM100/SM103 (Blackwell) GPU."
        )
        # The resolution pipeline materializes "auto" as fp8_e4m3 on CUDA.
        assert cfg.kv_cache_dtype in ("auto", "fp8_e4m3"), (
            "--dsv4-attn-backend trtllm requires kv_cache_dtype=fp8_e4m3, "
            f"got {cfg.kv_cache_dtype}."
        )
        assert not cfg.enable_hisparse, (
            "--dsv4-attn-backend trtllm does not support enable_hisparse."
        )
        assert not (
            cfg.attn_cp_size > 1 or cfg.dcp_size > 1 or cfg.enable_prefill_cp
        ), (
            "--dsv4-attn-backend trtllm does not support context parallelism "
            "(prefill CP, attention CP, or decode CP)."
        )
        # The PD handshake rejects peers whose KV layouts differ, but the
        # uniform-FP8 layout has not been run under PD (#37838).
        assert cfg.disaggregation_mode == "null", (
            "--dsv4-attn-backend trtllm does not support PD disaggregation yet "
            "(untested; see https://github.com/sgl-project/sglang/issues/37838)."
        )
        # The trtllm-gen semaphore buffer is sized from the prefill chunk
        # bound; with chunking disabled a single long request has no bound.
        assert cfg.chunked_prefill_size is not None and cfg.chunked_prefill_size > 0, (
            "--dsv4-attn-backend trtllm requires chunked prefill "
            "(--chunked-prefill-size > 0)."
        )
        logger.info(
            "DeepSeek V4 attention: trtllm backend enabled "
            "(uniform-FP8 KV pool, decode + sparse prefill)."
        )

    if cfg.max_running_requests is None:
        declare_resolution(
            server_args,
            "apply_deepseek_v4_defaults",
            max_running_requests=256,
        )
        logger.warning(
            f"Setting max_running_requests to {cfg.max_running_requests} for {model_arch}."
        )

    if cfg.speculative_algorithm is not None:
        assert cfg.speculative_algorithm in (
            "EAGLE",
            "DSPARK",
        ), (
            f"Only EAGLE and DSPARK speculative algorithms are supported for {model_arch}"
        )
        if cfg.speculative_algorithm == "EAGLE":
            assert cfg.speculative_eagle_topk == 1, (
                f"Only EAGLE speculative algorithm with topk == 1 is supported for {model_arch}"
            )


def validate_deepseek_v4_cp(server_args: ServerArgs) -> None:
    """Validate DeepSeek V4 context-parallel configuration."""
    cfg = resolving_view(server_args)
    if not cfg.enable_prefill_cp:
        return

    if cfg.cp_strategy not in ("interleave", "zigzag"):
        raise ValueError(
            f"DeepSeekV4 only supports interleave/zigzag CP strategy, got {cfg.cp_strategy}"
        )

    if cfg.cp_strategy == "zigzag" and not is_npu():
        raise ValueError(
            "DeepSeekV4 zigzag CP requires the NPU backend; CUDA/HIP backends "
            "reindex with interleave order."
        )
    if get_platform().is_hip:
        prefill_backend, decode_backend = attention_backends_of(
            resolved_view(server_args)
        )
        if (prefill_backend, decode_backend) != ("dsv4", "dsv4"):
            raise ValueError(
                "DeepSeekV4 prefill CP on HIP requires the dsv4 attention "
                f"backend for both phases, got prefill={prefill_backend!r}, "
                f"decode={decode_backend!r}."
            )
        unsupported = (
            ("multiple nodes", cfg.nnodes > 1),
            (
                "DeepSeek-V4.1",
                model_config_of(server_args).hf_config.model_type == "deepseek_v41",
            ),
            (
                "decoder SWA bounded replay",
                cfg.enable_decoder_swa_bounded_replay,
            ),
            ("--enable-two-batch-overlap", cfg.enable_two_batch_overlap),
        )
        for feature, enabled in unsupported:
            if enabled:
                raise ValueError(
                    f"DeepSeekV4 prefill CP on HIP does not support {feature} yet."
                )

    reject_dp_size_under_context_parallel(cfg, "DeepSeek-V4 context parallelism")
    declare_resolution(
        server_args,
        "validate_deepseek_v4_cp",
        moe_dense_tp_size=1,
        attn_cp_size=cfg.tp_size // cfg.attn_dp_size,
    )
    if not is_npu():
        assert cfg.attn_dp_size == 1, (
            "For round-robin split mode, dp attention is not supported."
        )
        assert cfg.tp_size <= 8, (
            "Context parallel only supports single machine (tp_size <= 8). Cross-machine CP has precision issues."
        )
    supported_a2a_backends = ("none", "deepep", "megamoe", "mori", "ascend_ops")
    if cfg.moe_a2a_backend not in supported_a2a_backends:
        raise ValueError(
            f"DeepSeekV4 CP supports moe_a2a_backend in {supported_a2a_backends}, "
            f"got {cfg.moe_a2a_backend!r}."
        )
    if model_config_of(server_args).hf_config.model_type != "deepseek_v41":
        # The CP-aware sparse prefill chunk cache is validated on V4.1 only.
        logger.warning(
            "Disabling SGLANG_OPT_FLASHMLA_SPARSE_PREFILL because DeepSeekV4 "
            "context parallelism is enabled."
        )
        envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.set(False)
    logger.warning(
        f"Enable Context Parallel for DeepSeekV4, "
        f"strategy={cfg.cp_strategy}, "
        f"attn_dp_size={cfg.attn_dp_size}, moe_dense_tp_size={cfg.moe_dense_tp_size}, "
        f"attn_cp_size={cfg.attn_cp_size}, ep_size={cfg.ep_size}, tp_size={cfg.tp_size}"
    )


def validate_deepseek_v41_encoder_replay_budget(server_args: ServerArgs) -> None:
    cfg = resolving_view(server_args)
    if not cfg.enable_encoder_swa_bounded_replay:
        return
    # A hit's first chunk also runs up to 128 replay rows and floors its extend to
    # whole pages, so a smaller chunk leaves the hit no page and it never admits.
    min_chunk = 128 + (cfg.page_size or 1)
    if (
        cfg.max_running_requests is None
        or cfg.max_running_requests <= 0
        or not cfg.chunked_prefill_size
        or cfg.chunked_prefill_size < min_chunk
    ):
        raise ValueError(
            "encoder SWA replay requires --max-running-requests > 0 and at least "
            f"{min_chunk} chunked prefill tokens per attention DP rank (128 replay "
            "tokens plus one page)"
        )


_COMPRESSED_RATIOS = (1, 2)


def compute_pp_ranges(
    num_hidden_layers: int,
    pp_size: int,
    partition: Optional[Sequence[int]] = None,
) -> List[Tuple[int, int]]:
    """Per-rank [start, end) ranges from the runtime's own split kernel."""
    if pp_size <= 0:
        raise ValueError(f"pp_size must be positive, got {pp_size}")
    if partition is not None and any(count <= 0 for count in partition):
        raise ValueError(
            f"SGLANG_PP_LAYER_PARTITION has a non-positive stage: {partition}"
        )
    return [
        compute_pp_indices(num_hidden_layers, pp_rank, pp_size, partition)
        for pp_rank in range(pp_size)
    ]


def _model_layer_ratios(hf_config: Any, num_hidden_layers: int) -> List[int]:
    """compress_ratios for the decoder layers only (nextn entries trail)."""
    ratios = list(hf_config.compress_ratios or ())
    if len(ratios) < num_hidden_layers:
        ratios.extend([0] * (num_hidden_layers - len(ratios)))
    return ratios[:num_hidden_layers]


def _source_layers(
    hf_config: Any, field: str, num_hidden_layers: int
) -> List[int]:
    return sorted(
        int(layer)
        for layer in (getattr(hf_config, field, None) or ())
        if 0 <= int(layer) < num_hidden_layers
    )


def _block_source_consumer_groups(
    sources: Sequence[int],
    ratios: Sequence[int],
    num_hidden_layers: int,
    blocked: set,
) -> None:
    """Forbid a stage boundary between a source and its same-ratio readers.

    A compressed layer reads the nearest earlier source of its ratio; every
    same-ratio layer before the next source depends on that source, even when
    other ratios are interleaved.
    """
    for ratio in _COMPRESSED_RATIOS:
        group_sources = [layer for layer in sources if ratios[layer] == ratio]
        for index, source in enumerate(group_sources):
            stop = (
                group_sources[index + 1]
                if index + 1 < len(group_sources)
                else num_hidden_layers
            )
            consumers = [
                layer for layer in range(source + 1, stop) if ratios[layer] == ratio
            ]
            if consumers:
                blocked.update(range(source + 1, consumers[-1] + 1))


def legal_split_points(
    hf_config: Any, num_hidden_layers: int, *, is_dspark: bool = False
) -> List[int]:
    """Layer boundaries a stage may start/end at without splitting a group."""
    ratios = _model_layer_ratios(hf_config, num_hidden_layers)
    blocked = set()
    # kv sources publish the compressed KV; index sources publish the top-k
    # the same-ratio consumers attend with. Both stay per-forward state.
    for field in ("kv_source_layer_ids", "index_source_layer_ids"):
        _block_source_consumer_groups(
            _source_layers(hf_config, field, num_hidden_layers),
            ratios,
            num_hidden_layers,
            blocked,
        )
    candidate = int(hf_config.candidate_source_layer_id)
    if 0 <= candidate < num_hidden_layers:
        blocked.update(range(candidate + 1, num_hidden_layers))
    if is_dspark:
        capture_layers = [
            int(layer)
            for layer in (
                getattr(hf_config, "dspark_target_layer_ids", None) or ()
            )
            if 0 <= int(layer) < num_hidden_layers
        ]
        if capture_layers:
            # The draft lives on the last stage, so every capture layer must
            # start at or before the last stage's first layer.
            blocked.update(range(min(capture_layers) + 1, num_hidden_layers))
    return [
        boundary for boundary in range(num_hidden_layers + 1) if boundary not in blocked
    ]


def suggested_partition(
    hf_config: Any, pp_size: int, *, is_dspark: bool = False
) -> str:
    """A legal split that minimizes the largest stage; "" when none exists."""
    num_hidden_layers = int(hf_config.num_hidden_layers)
    boundaries = legal_split_points(
        hf_config, num_hidden_layers, is_dspark=is_dspark
    )
    inner = [point for point in boundaries if 0 < point < num_hidden_layers]
    if pp_size - 1 > len(inner):
        return ""
    # Bounded dynamic program: best[k][j] is the minimal (largest stage, sum of
    # squares) for k stages ending at cut j, with its per-stage sizes. Both
    # criteria compose across a split, so the lexicographic optimum carries.
    cuts = [0, *inner, num_hidden_layers]
    num_cuts = len(cuts)
    best: List[Optional[Tuple[Tuple[int, int], Tuple[int, ...]]]] = [None] * num_cuts
    for j in range(1, num_cuts):
        size = cuts[j] - cuts[0]
        best[j] = ((size, size * size), (size,))
    for _ in range(2, pp_size + 1):
        nxt: List[Optional[Tuple[Tuple[int, int], Tuple[int, ...]]]] = [
            None
        ] * num_cuts
        for j in range(1, num_cuts):
            for i in range(j):
                prev = best[i]
                if prev is None:
                    continue
                size = cuts[j] - cuts[i]
                score = (max(prev[0][0], size), prev[0][1] + size * size)
                if nxt[j] is None or score < nxt[j][0]:
                    nxt[j] = (score, prev[1] + (size,))
        best = nxt
    final = best[-1]
    return "" if final is None else ",".join(str(size) for size in final[1])


def _pp_split_error(
    hf_config: Any,
    *,
    pp_size: int,
    partition_str: Optional[str],
    detail: str,
    is_dspark: bool,
) -> ValueError:
    num_hidden_layers = int(hf_config.num_hidden_layers)
    boundaries = legal_split_points(hf_config, num_hidden_layers, is_dspark=is_dspark)
    suggestion = suggested_partition(hf_config, pp_size, is_dspark=is_dspark)
    split = (
        f"SGLANG_PP_LAYER_PARTITION={partition_str!r}"
        if partition_str
        else "the default even split"
    )
    hint = (
        f"For --pp-size {pp_size} use SGLANG_PP_LAYER_PARTITION={suggestion}."
        if suggestion
        else f"--pp-size {pp_size} has no legal split for this model."
    )
    return ValueError(
        f"{split} is not a legal DeepSeek-V4.1 pipeline split: {detail} "
        f"Compressed-KV source groups (kv_source_layer_ids), index sources "
        f"(index_source_layer_ids), the candidate source "
        f"({hf_config.candidate_source_layer_id}) and its consumers, and DSpark "
        f"capture layers must stay on one PP stage. Legal stage boundaries "
        f"among {num_hidden_layers} layers: {boundaries}. {hint}"
    )


def validate_dsv41_pp_partition(
    hf_config: Any,
    *,
    pp_size: int,
    partition_str: Optional[str],
    is_dspark: bool,
    language_model_only: bool,
) -> None:
    """Raise when a PP split would break DSV4.1's per-stage shared state."""
    if pp_size <= 1:
        return
    num_hidden_layers = int(hf_config.num_hidden_layers)

    vision_layers = int(hf_config.vision_n_layers or 0)
    if vision_layers > 0 and not language_model_only:
        raise ValueError(
            "DeepSeek-V4.1 pipeline parallelism requires --language-model-only "
            f"today: this checkpoint carries a {vision_layers}-layer vision "
            "tower, and the vision path rejects PP at model build. Use "
            "--language-model-only or a text-only checkpoint."
        )

    def split_error(detail: str) -> ValueError:
        return _pp_split_error(
            hf_config,
            pp_size=pp_size,
            partition_str=partition_str,
            detail=detail,
            is_dspark=is_dspark,
        )

    try:
        partition = (
            parse_pp_layer_partition(partition_str) if partition_str else None
        )
        ranges = compute_pp_ranges(num_hidden_layers, pp_size, partition)
    except ValueError as err:
        raise split_error(str(err)) from err
    legal = legal_split_points(hf_config, num_hidden_layers, is_dspark=is_dspark)
    if pp_size > len(legal) - 1:
        raise split_error(f"pp_size={pp_size} exceeds the number of legal stages.")

    ratios = _model_layer_ratios(hf_config, num_hidden_layers)

    def check_sources(field: str, label: str) -> None:
        sources = _source_layers(hf_config, field, num_hidden_layers)
        for pp_rank, (start, end) in enumerate(ranges):
            for layer in range(start, end):
                ratio = ratios[layer]
                if ratio not in _COMPRESSED_RATIOS:
                    continue
                layer_sources = [
                    source
                    for source in sources
                    if source <= layer and ratios[source] == ratio
                ]
                if not layer_sources or layer_sources[-1] < start:
                    source = layer_sources[-1] if layer_sources else None
                    raise split_error(
                        f"stage {pp_rank} covers layers [{start},{end}); layer "
                        f"{layer} (compress_ratio={ratio}) needs its {label} "
                        f"layer {source} in the same stage."
                    )

    check_sources("kv_source_layer_ids", "kv_source")
    check_sources("index_source_layer_ids", "index source")

    candidate = int(hf_config.candidate_source_layer_id)
    if 0 <= candidate < num_hidden_layers:
        candidate_stage = next(
            (start, end) for start, end in ranges if start <= candidate < end
        )
        if candidate_stage[1] != num_hidden_layers:
            raise split_error(
                f"the candidate source layer {candidate} and its consumers "
                f"layers {candidate + 1}..{num_hidden_layers - 1} are split "
                f"across stages."
            )

    if is_dspark:
        last_start, last_end = ranges[-1]
        for layer in getattr(hf_config, "dspark_target_layer_ids", None) or ():
            layer = int(layer)
            if not (last_start <= layer < last_end):
                raise split_error(
                    f"DSpark capture layer {layer} is not on the last stage "
                    f"[{last_start},{last_end}); the draft runs only there."
                )


def validate_deepseek_v41_features(server_args: ServerArgs) -> None:
    from sglang.srt.layers.attention.dsv4.unified_kv_gate import (
        is_unified_kv_triton,
    )

    cfg = resolving_view(server_args)
    if model_config_of(server_args).hf_config.model_type != "deepseek_v41":
        if cfg.enable_encoder_swa_bounded_replay:
            raise ValueError(
                "--enable-encoder-swa-bounded-replay requires DeepSeek-V4.1"
            )
        return
    validate_dsv41_pp_partition(
        model_config_of(server_args).hf_text_config,
        pp_size=cfg.pp_size,
        partition_str=envs.SGLANG_PP_LAYER_PARTITION.get(),
        is_dspark=(
            cfg.speculative_algorithm is not None
            and str(cfg.speculative_algorithm).upper() == "DSPARK"
        ),
        language_model_only=cfg.language_model_only,
    )

    if (
        cfg.dsv4_attn_backend == "trtllm"
        and cfg.cuda_graph_config.prefill.backend
        not in (Backend.DISABLED, Backend.BREAKABLE)
    ):
        raise ValueError(
            "DeepSeek-V4.1 TRT-LLM requires --cuda-graph-backend-prefill "
            "disabled or breakable"
        )
    if cfg.enable_encoder_swa_bounded_replay:
        incompatible = (
            (
                "hardware other than CUDA or gfx950",
                not (
                    get_platform().is_cuda
                    or (get_platform().is_hip and is_gfx95_supported())
                ),
            ),
            (
                "prefill CUDA graphs other than breakable",
                cfg.cuda_graph_config.prefill.backend
                not in (Backend.DISABLED, Backend.BREAKABLE),
            ),
            # ROCm replays hits in a forward of their own, outside the graph.
            (
                "the prefill CUDA graph on ROCm",
                cfg.cuda_graph_config.prefill.backend != Backend.DISABLED
                and get_platform().is_hip,
            ),
            ("context parallelism", cfg.attn_cp_size > 1),
            # ROCm replays hits in a forward of their own, which only the hitting
            # attention-DP ranks would enter; CUDA folds them into the extend.
            ("DP attention on ROCm", get_platform().is_hip and attn_dp_enabled_of(cfg)),
            ("unified memory", cfg.enable_unified_memory),
            # Decode receives the last SWA window from prefill and never replays.
            ("PD decode", cfg.disaggregation_mode == "decode"),
            # On ROCm the draft keeps paged SWA, so it cannot share the window's pages.
            (
                "PD prefill on ROCm",
                cfg.disaggregation_mode == "prefill" and get_platform().is_hip,
            ),
            ("mixed prefill/decode", cfg.enable_mixed_chunk),
            ("LoRA", cfg.enable_lora),
            ("radix sessions", cfg.enable_session_radix_cache),
        )
        for feature, enabled in incompatible:
            if enabled:
                raise ValueError(
                    f"--enable-encoder-swa-bounded-replay does not support {feature} yet"
                )

    unsupported = (
        (
            "speculative decoding other than DSpark",
            cfg.speculative_algorithm is not None
            and str(cfg.speculative_algorithm).upper() != "DSPARK",
        ),
        ("HiSparse", cfg.enable_hisparse),
        ("the unified KV layout", is_unified_kv_triton()),
        (
            "TRT-LLM with encoder SWA bounded replay",
            cfg.dsv4_attn_backend == "trtllm" and cfg.enable_encoder_swa_bounded_replay,
        ),
        ("two-batch overlap", cfg.enable_two_batch_overlap),
    )
    for feature, enabled in unsupported:
        if enabled:
            raise ValueError(
                f"DeepSeek-V4.1 does not support {feature} yet; disable it to "
                "serve this model."
            )

    if cfg.disaggregation_mode != "null" and cfg.speculative_algorithm is not None:
        from sglang.srt.speculative.ragged_verify import (
            RaggedVerifyMode,
            read_ragged_verify_mode,
        )

        # CP size may still be implicit here; validate_deepseek_v4_cp resolves it.
        supported_prefill_cp = (
            cfg.disaggregation_mode == "prefill"
            and cfg.enable_prefill_cp
            and cfg.cp_strategy == "interleave"
        )
        # Ascend reuses the positional target/draft SWA and request-state
        # transport. Keep its initial DSpark support to CP=1 on both roles;
        # the GPU prefill-CP exception below has not been validated on NPU.
        ascend_transfer = cfg.disaggregation_transfer_backend == "ascend"
        if ascend_transfer and (
            cfg.device != "npu" or cfg.enable_prefill_cp or cfg.attn_cp_size != 1
        ):
            raise ValueError(
                "DeepSeek-V4.1 DSpark PD with Ascend requires --device npu "
                "and CP=1 on both prefill and decode; disable --enable-prefill-cp."
            )
        if (
            read_ragged_verify_mode() is not RaggedVerifyMode.STATIC
            or cfg.disaggregation_transfer_backend
            not in ("mooncake", "nixl", "mori", "ascend")
            or (
                (cfg.enable_prefill_cp or cfg.attn_cp_size != 1)
                and not supported_prefill_cp
            )
            or cfg.dcp_size != 1
        ):
            raise ValueError(
                "DeepSeek-V4.1 DSpark PD requires static verify, Mooncake, NIXL, "
                "MoRI, or Ascend (NPU, CP=1 on both roles), DCP=1, and CP=1 "
                "on decode. GPU prefill CP requires "
                "--enable-prefill-cp with the interleave strategy. "
                "DP attention is supported when "
                "both servers use the same block size and target/draft KV layout."
            )

    incompatible = (
        (
            "prefill CP with the prefill CUDA graph",
            (cfg.enable_prefill_cp or cfg.attn_cp_size != 1)
            and cfg.cuda_graph_config.prefill.backend != Backend.DISABLED,
        ),
        (
            "the prefill CUDA graph",
            cfg.cuda_graph_config.prefill.backend
            not in (Backend.DISABLED, Backend.BREAKABLE),
        ),
        (
            "DP attention other than attention TP 1 with supported MoE backends",
            attn_dp_enabled_of(cfg)
            and not (
                cfg.attn_dp_size == cfg.tp_size
                and cfg.attn_cp_size == 1
                and (
                    (get_platform().is_hip and cfg.moe_a2a_backend == "none")
                    or (
                        get_platform().is_cuda
                        and cfg.moe_a2a_backend in ("none", "megamoe")
                    )
                )
            ),
        ),
        # The bounded-replay tail narrows the residual to tail rows and is
        # scattered back only on the last rank; a non-last PP stage would relay
        # tail-only rows. Its graph is also keyed by the global late-layer start.
        ("pipeline parallelism", cfg.pp_size > 1),
    )
    if cfg.enable_decoder_swa_bounded_replay:
        for feature, unsupported in incompatible:
            if unsupported:
                raise ValueError(
                    "--enable-decoder-swa-bounded-replay cannot be combined with "
                    f"{feature} yet; disable one of them."
                )
        return

    if cfg.enable_decoder_swa_bounded_replay is False:
        return

    enabled = (
        (get_platform().is_cuda or get_platform().is_hip)
        # Both PD roles run it on CUDA; PD on ROCm has not been run with it.
        and (cfg.disaggregation_mode == "null" or get_platform().is_cuda)
        and not (cfg.enable_prefill_cp or cfg.attn_cp_size != 1 or cfg.dcp_size != 1)
        # Returning every hidden state runs every prompt row: nothing to trim.
        and not cfg.enable_return_hidden_states
        and not any(enabled for _, enabled in incompatible)
    )
    declare_resolution(
        server_args,
        "validate_deepseek_v41_features",
        enable_decoder_swa_bounded_replay=enabled,
    )
    logger.info("DeepSeek-V4.1 decoder SWA bounded replay resolved to %s", enabled)
