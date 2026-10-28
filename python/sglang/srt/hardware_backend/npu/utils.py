import contextlib
import functools
import logging
import os
import sys
import threading
from enum import IntEnum
from typing import TYPE_CHECKING, Callable, Optional

import msgspec
import torch

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    resolving_view,
    use_mla_backend,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Phase, with_phase
from sglang.srt.utils import get_npu_memory_capacity, is_npu

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)
_is_npu = is_npu()
indexer_weight_stream = None
gva_is_inited = False


@functools.lru_cache(maxsize=1)
def is_npu_arch35() -> bool:
    """Whether the runtime is on NPU architecture 35."""
    if not is_npu():
        return False

    import acl

    return acl.rt.get_device_info(0, 601) == (3510, 0)


def use_npu_arch35_mxfp8_wo_a(
    quant_config, wo_a_keeps_quant_config: Optional[bool] = None
) -> bool:
    """Whether wo_a runs the native NPU arch35 MXFP8 GEMM.

    Only for serialized DeepSeek block-FP8 checkpoints — those are the ones
    ``Fp8LinearMethod.process_weights_after_loading`` can reinterpret into the
    NPU arch35 MXFP8 scale layout.
    """
    if (
        wo_a_keeps_quant_config is False
        or not _is_npu
        or not is_npu_arch35()
        or quant_config is None
        or not getattr(quant_config, "is_checkpoint_fp8_serialized", False)
    ):
        return False
    weight_block_size = getattr(quant_config, "weight_block_size", None)
    return tuple(weight_block_size or ()) == (128, 128)


@functools.lru_cache(maxsize=1)
def _mqsmla_ops() -> Optional[tuple]:
    """Resolve the official cann_ops_transformer mqsmla op trio.

    DSV4 requires the CANN 9.2 release; a missing package or op surfaces as
    None here and turns into a fail-fast error in require_mqsmla_ops().
    Cached — the answer cannot change within a process.
    """
    try:
        import cann_ops_transformer  # noqa: F401

        ops_ns = torch.ops.cann_ops_transformer
        return (
            ops_ns.mixed_quant_sparse_flash_mla_metadata,
            ops_ns.mixed_quant_sparse_flash_mla,
            ops_ns.kv_compress_epilog,
        )
    except (ImportError, AttributeError, RuntimeError):
        return None


@functools.lru_cache(maxsize=1)
def _log_a5_kv_quant_selection() -> None:
    """Log the A5 pooled-KV row format once."""
    logger.info(
        "A5 DSV4 kv-quant: official cann_ops_transformer "
        "mixed_quant_sparse_flash_mla + kv_compress_epilog, 608B rows with "
        "bf16 group scales"
    )


def require_mqsmla_ops():
    """The official mqsmla op trio; fail fast when the CANN 9.2 package is absent."""
    ops = _mqsmla_ops()
    if ops is None:
        raise RuntimeError(
            "DSV4 requires the CANN 9.2 cann_ops_transformer ops "
            "(mixed_quant_sparse_flash_mla, mixed_quant_sparse_flash_mla_metadata, "
            "kv_compress_epilog), but they cannot be resolved. Install "
            "cann_ops_transformer built from ops-transformer 9.2.0+."
        )
    return ops


class NpuA5KvQuantPath(msgspec.Struct, frozen=True):
    """A5 pooled-KV row format for the official mqsmla ops.

    Row bytes = rope*2 + nope + ceil(nope/group)*bf16 scale, aligned up to
    row_alignment; the reader is cann_ops_transformer
    mixed_quant_sparse_flash_mla.
    """

    quant_group_size: int
    scale_itemsize: int
    row_alignment: int
    reader_quant_mode: int
    reader_layout_kv: str
    writer_quant_mode: str


NPU_A5_KV_QUANT_MQSMLA = NpuA5KvQuantPath(
    quant_group_size=64,
    scale_itemsize=2,  # bf16
    row_alignment=32,
    reader_quant_mode=1,  # quant_mode on the cann_ops_transformer reader
    reader_layout_kv="PA_BBND",
    writer_quant_mode="fp8_bf16",
)


def npu_a5_kv_quant_path() -> NpuA5KvQuantPath:
    _log_a5_kv_quant_selection()
    return NPU_A5_KV_QUANT_MQSMLA


class NPUACLFormat(IntEnum):
    ACL_FORMAT_UNDEFINED = -1
    ACL_FORMAT_ND = 2
    ACL_FORMAT_FRACTAL_NZ = 29


def _call_once(fn: Callable):

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if getattr(fn, "_has_been_called", False):
            logger.debug("Function {} has already been called.", fn.__name__)
            return

        fn._has_been_called = True
        return fn(*args, **kwargs)

    return wrapper


def set_default_server_args(args: "ServerArgs"):
    """
    Set default server arguments for NPU backend.
    """

    cfg = resolving_view(args)

    # NPU only works with "ascend" attention backend for now
    declare_resolution(
        args,
        "set_default_server_args",
        attention_backend="ascend",
    )
    declare_resolution(
        args,
        "set_default_server_args",
        prefill_attention_backend="ascend",
    )
    declare_resolution(
        args,
        "set_default_server_args",
        decode_attention_backend="ascend",
    )
    if cfg.page_size is None:
        declare_resolution(
            args,
            "set_default_server_args",
            page_size=128,
        )

    # NPU memory settings
    npu_mem = get_npu_memory_capacity()
    if npu_mem <= 32 * 1024:
        # Ascend 910B4,910B4_1
        # (chunked_prefill_size 4k, max_bs 16 if tp < 4 else 64)
        if cfg.chunked_prefill_size is None:
            declare_resolution(
                args,
                "set_default_server_args",
                chunked_prefill_size=4 * 1024,
            )
        if cfg.cuda_graph_config.decode.max_bs is None:
            if cfg.tp_size < 4:
                declare_resolution(
                    args,
                    "set_default_server_args",
                    cuda_graph_config=with_phase(
                        cfg.cuda_graph_config,
                        Phase.DECODE,
                        max_bs=16,
                    ),
                )
            else:
                declare_resolution(
                    args,
                    "set_default_server_args",
                    cuda_graph_config=with_phase(
                        cfg.cuda_graph_config,
                        Phase.DECODE,
                        max_bs=64,
                    ),
                )
    elif npu_mem <= 64 * 1024:
        # Ascend 910B1,910B2,910B2C,910B3,910_9391,910_9392,910_9381,910_9382,910_9372,910_9362
        # (chunked_prefill_size 8k, max_bs 64 if tp < 4 else 256)
        if cfg.chunked_prefill_size is None:
            declare_resolution(
                args,
                "set_default_server_args",
                chunked_prefill_size=8 * 1024,
            )
        if cfg.cuda_graph_config.decode.max_bs is None:
            if cfg.tp_size < 4:
                declare_resolution(
                    args,
                    "set_default_server_args",
                    cuda_graph_config=with_phase(
                        cfg.cuda_graph_config,
                        Phase.DECODE,
                        max_bs=64,
                    ),
                )
            else:
                declare_resolution(
                    args,
                    "set_default_server_args",
                    cuda_graph_config=with_phase(
                        cfg.cuda_graph_config,
                        Phase.DECODE,
                        max_bs=256,
                    ),
                )

    # NPU does not support CustomAllReduce
    declare_resolution(
        args,
        "set_default_server_args",
        disable_custom_all_reduce=True,
    )

    # handles hierarchical cache / decode-offload configs
    if (
        cfg.enable_hierarchical_cache
        or cfg.disaggregation_decode_enable_offload_kvcache
    ):
        declare_resolution(
            args,
            "set_default_server_args",
            hicache_io_backend="kernel_ascend",
        )
        if use_mla_backend(args):
            declare_resolution(
                args,
                "set_default_server_args",
                hicache_mem_layout="page_first_kv_split",
            )
        else:
            declare_resolution(
                args,
                "set_default_server_args",
                hicache_mem_layout="page_first_direct",
            )


def _prefer_package_common_headers() -> None:
    """Make CANN op builds use this wheel's aclnn_common.h, not a vendor copy.

    OpBuilder.include_paths() appends the package's common/ dir after every
    vendor op_api/include dir. A site-packages vendor (deep_ep) ships a
    different aclnn_common.h, so the compressor extension JIT-compiles against
    the wrong aclnn ABI and its first call segfaults. Op-specific aclnn_*.h
    headers are unaffected; only the wheel's own helper headers move first.
    """
    try:
        from cann_ops_transformer.op_builder.builder import OpBuilder
    except Exception:
        return
    if getattr(OpBuilder.include_paths, "_sglang_common_first", False):
        return
    original = OpBuilder.include_paths

    def include_paths(self):
        paths = original(self)
        common = os.path.join(self._package_path, "common")
        if os.path.isdir(common):
            paths = [common] + [path for path in paths if path != common]
        return paths

    include_paths._sglang_common_first = True
    OpBuilder.include_paths = include_paths


# aclnn entry points in the CANN "customize" vendor (and similar cust-op
# vendors) shadow the official libopapi_transformer.so implementations of these
# ops -- the vendor copy uses an incompatible cache-row layout, so kv writes
# land in the wrong pages. A cann_ops_transformer JIT extension resolves its
# aclnn symbols once, on its first call, via GetOpApiFuncAddr, which scans
# ASCEND_CUSTOM_OPP_PATH/LD_LIBRARY_PATH before the official libs.
_OFFICIAL_OPAPI_OPS = (
    "mixed_quant_sparse_flash_mla_metadata",
    "mixed_quant_sparse_flash_mla",
    "kv_compress_epilog",
    "compressor",
    "inplace_partial_rotary_mul",
    "quant_lightning_indexer",
    "quant_lightning_indexer_metadata",
    "indexer_quant_cache",
    "mhc_post",
)


@contextlib.contextmanager
def _without_cust_opapi_vendors():
    """Hide cust-op vendors from the env for one op call.

    Other packages in the process (deep_ep) still resolve their own aclnn
    symbols from their vendor, so this is a temporary view, not a removal.
    """
    saved_custom = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
    saved_ld = os.environ.get("LD_LIBRARY_PATH")
    os.environ.pop("ASCEND_CUSTOM_OPP_PATH", None)
    if saved_ld is not None:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            d
            for d in saved_ld.split(os.pathsep)
            if not (d and os.path.exists(os.path.join(d, "libcust_opapi.so")))
        )
    try:
        yield
    finally:
        if saved_custom is not None:
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = saved_custom
        if saved_ld is not None:
            os.environ["LD_LIBRARY_PATH"] = saved_ld


def _guard_official_opapi_resolution(op, name: str):
    """Resolve one extension's aclnn symbols from the official libs, once."""
    done = False
    lock = threading.Lock()

    def guarded(*args, **kwargs):
        nonlocal done
        if done:
            return op(*args, **kwargs)
        with lock:
            if done:
                return op(*args, **kwargs)
            with _without_cust_opapi_vendors():
                result = op(*args, **kwargs)
            done = True
        logger.info(
            "cann_ops_transformer.%s resolved against the official aclnn libs", name
        )
        return result

    guarded._sglang_official_opapi_guard = True
    return guarded


def _install_official_opapi_guards() -> None:
    """Keep CANN cust-op vendors from shadowing the official transformer ops."""
    try:
        import cann_ops_transformer  # noqa: F401  (registers the namespace)
    except Exception as e:
        logger.debug("cann_ops_transformer unavailable; no aclnn guards (%s)", e)
        return
    namespace = torch.ops.cann_ops_transformer
    for name in _OFFICIAL_OPAPI_OPS:
        op = getattr(namespace, name, None)
        if op is None or getattr(op, "_sglang_official_opapi_guard", False):
            continue
        setattr(namespace, name, _guard_official_opapi_resolution(op, name))


@_call_once
def init_npu_backend():
    """
    Initialize NPU backend. This function should be called only once.
    """

    assert _is_npu, "NPU backend initialization called on non-NPU device."

    try:
        import sgl_kernel_npu  # noqa: F401
    except ImportError as e:
        logger.warning("NPU custom kernel packages unavailable: %s", e)

    import torch_npu

    # These imports lead to unpredictable behavior in diffusion models
    # and a significant reduction in performance.
    if "sglang.multimodal_gen" not in sys.modules:
        from torch_npu.contrib import transfer_to_npu  # noqa: F401

        # Re-mock torch.cuda.is_available cuz transfer_to_npu mocks it True
        torch.cuda.is_available = lambda: False
        torch_npu.npu.config.allow_internal_format = True
    torch_npu.npu.set_compile_mode(jit_compile=False)
    _prefer_package_common_headers()
    _install_official_opapi_guards()
    # The op handles are resolved lazily; drop any pre-guard resolution so the
    # cached handles point at the guarded wrappers.
    _mqsmla_ops.cache_clear()

    if os.environ.get("MQSMLA_CAPTURE_DIR"):
        # Debug aid: dump mqsmla/compressor/indexer op args to compare against
        # the verified smoke call. See mqsmla_capture.py for the env knobs.
        from sglang.srt.hardware_backend.npu import mqsmla_capture

        mqsmla_capture.install()
        _mqsmla_ops.cache_clear()


def _is_nz_aligned(tensor: torch.Tensor) -> bool:
    """Check whether the last two dims satisfy FRACTAL_NZ alignment rules.

    A fractal tile is 16 rows by 32 bytes (the C0_32 in the op's error strings),
    so the row rule is always k % 16 and the column rule is 32 // itemsize:

      BF16 / FP16  : k % 16 == 0  and  n % 16 == 0
      INT8 / FP8   : k % 16 == 0  and  n % 32 == 0
      INT4         : k % 16 == 0  and  n % 64 == 0

    Unlisted dtypes fall through to True: this is a cheap pre-filter for known
    bad combinations, not an authority — the op itself is.
    """
    if tensor.dim() < 2:
        return False
    k, n = tensor.shape[-2], tensor.shape[-1]
    if tensor.dtype in (torch.bfloat16, torch.float16):
        return k % 16 == 0 and n % 16 == 0
    if tensor.dtype in (torch.int8, torch.float8_e4m3fn):
        # e4m3 is single-byte like int8, so it shares the column rule. Reached
        # only by the MXFP8 MoE weights; the packed-FP4 callers pass
        # customize_dtype and return before this check.
        return k % 16 == 0 and n % 32 == 0
    if tensor.dtype in (torch.uint8, torch.int32):
        # INT4 is typically packed into uint8/int32; be conservative
        return k % 16 == 0 and n % 64 == 0
    return True


def npu_format_cast(
    tensor: torch.Tensor,
    acl_format: NPUACLFormat = NPUACLFormat.ACL_FORMAT_FRACTAL_NZ,
    *,
    customize_dtype=None,
    input_dtype=None,
) -> torch.Tensor:
    """
    Cast a tensor to a specific NPU ACL format.

    Args:
        tensor (torch.Tensor): The input tensor.
        acl_format (NPUACLFormat): The target NPU ACL format.
        customize_dtype / input_dtype: packed-FP4 unpack kwargs (e.g.
            ``customize_dtype=torch.float8_e4m3fn``,
            ``input_dtype=torch.float4_e2m1fn_x2``). When either is set the unpack
            kwargs are forwarded to the op and the ``_is_nz_aligned`` ND fallback
            is skipped: the FP4 matmul strictly requires FRACTAL_NZ, so a silent
            ND fallback would corrupt results.

    Returns:
        torch.Tensor: The tensor cast to the specified NPU ACL format.
    """

    if not _is_npu:
        return tensor

    if envs.SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT.get():
        return tensor

    if tensor.device == torch.device("cpu"):
        logger.warning_once(
            "Warning: The conversion from 'ND' to 'NZ' does not work on the CPU. "
            "Please disable offloading, otherwise the performance will be "
            "significantly reduced. --dit-cpu-offload false"
        )
        return tensor

    # Skip format cast for meta tensors (used in offloader)
    if tensor.device.type == "meta":
        return tensor

    # Packed-FP4 → FRACTAL_NZ: forward the unpack kwargs to the op, and skip the
    # _is_nz_aligned ND fallback — the FP4 matmul strictly requires NZ, so a
    # silent ND fallback would corrupt results.
    if customize_dtype is not None or input_dtype is not None:
        return torch.ops.npu.npu_format_cast(
            tensor,
            int(acl_format),
            customize_dtype=customize_dtype,
            input_dtype=input_dtype,
        )

    if acl_format == NPUACLFormat.ACL_FORMAT_FRACTAL_NZ and not _is_nz_aligned(tensor):
        k, n = tensor.shape[-2], tensor.shape[-1]
        logger.warning_once(
            "Skipping FRACTAL_NZ format cast: tensor shape (%d, %d) dtype %s "
            "is not aligned to NZ requirements. Falling back to 'ND' format, "
            "which may reduce NPU performance.",
            k,
            n,
            tensor.dtype,
        )
        return tensor

    return torch.ops.npu.npu_format_cast(tensor, acl_format.value)


def get_indexer_weight_stream():
    global indexer_weight_stream
    if indexer_weight_stream is None:
        indexer_weight_stream = torch.npu.Stream()
    return indexer_weight_stream


def init_zbal(world_size, gpu_id, world_rank, do_check=True):
    """
    init zbal, if is mix alloc mode, only register for sma & comm
    """
    zbal_mem_size = envs.SGLANG_ZBAL_LOCAL_MEM_SIZE.get()
    if not zbal_mem_size > 0:
        return 1

    global gva_is_inited
    from zbal import is_mix_alloc, switch_to_allocator, zbal_init

    if is_mix_alloc():
        switch_to_allocator()
        # use lazy init for mix alloc
        return 1
    else:
        if envs.SGLANG_ZBAL_BOOTSTRAP_URL.get():
            ret = zbal_init(
                world_size,
                gpu_id,
                world_rank,
                zbal_mem_size * (1024**2),
                ip_port=envs.SGLANG_ZBAL_BOOTSTRAP_URL.get(),
            )
        else:
            ret = zbal_init(world_size, gpu_id, world_rank, zbal_mem_size * (1024**2))

        gva_is_inited = True

        if do_check and not ret:
            logger.error("[ZBAL] zbal init failed!")
            sys.exit(-1)

        return ret


def lazy_init_zbal_gva_mem(
    device, gpu_id, world_rank, world_size, cpu_group=None, do_check=True
):
    """
    lazy init zbal gva mem, keep weights and kv remains alloc by dma vmm to avoid memory fragment
    """
    from zbal import is_mix_alloc, zbal_init

    if not is_mix_alloc():
        logger.info(
            "lazy init is supported only in mix alloc mode, this action will be passed"
        )
        return 1

    global gva_is_inited
    from sglang.srt.utils.common import get_available_gpu_memory

    # TODO need to use allgather if you want use total_memory stats from mem_get_info as unbalance os
    total_memory = 61.2  # 2.5GB for other (workspace & os) outside torch
    free_gpu_memory = get_available_gpu_memory(
        device,
        gpu_id,
        distributed=world_size > 1,
        cpu_group=cpu_group,
        empty_cache=True,
    )

    used_memory = total_memory - free_gpu_memory

    used_memory_in_mb = int(used_memory * 1024)
    gva_in_mb = envs.SGLANG_ZBAL_LOCAL_MEM_SIZE.get() - used_memory_in_mb
    gva_in_mb = gva_in_mb - gva_in_mb % 128  # align to 128MB
    print(f"[ZBAL] rank {world_rank} allocated {gva_in_mb} MB gva space.")

    assert not gva_is_inited, "zbal gva should be inited only once"
    # zbal_set_logger_level(0)
    if envs.SGLANG_ZBAL_BOOTSTRAP_URL.get():
        res = zbal_init(
            world_size,
            gpu_id,
            world_rank,
            gva_in_mb * (1024**2),
            ip_port=envs.SGLANG_ZBAL_BOOTSTRAP_URL.get(),
        )
    else:
        res = zbal_init(world_size, gpu_id, world_rank, gva_in_mb * (1024**2))

    gva_is_inited = True
    if do_check and not res:
        logger.error("[ZBAL] zbal lazy init failed!")
        sys.exit(-1)
    return res


share_stream = None
routed_stream = None


def get_share_stream():
    global share_stream
    return share_stream


def set_share_stream(stream):
    global share_stream
    share_stream = stream
    # TODO LKL: set stream limit has impact on precision
    # torch.npu.set_stream_limit(share_stream, 8, 16)


def get_routed_stream():
    global routed_stream
    return routed_stream


def set_routed_stream(stream):
    global routed_stream
    routed_stream = stream
    # TODO LKL: set stream limit has impact on precision
    # torch.npu.set_stream_limit(routed_stream, 16, 32)


def wait_share_stream():
    stream = get_share_stream()
    if stream is not None:
        cur_stream = torch.get_device_module().current_stream()
        cur_stream.wait_stream(stream)


def wait_routed_stream():
    stream = get_routed_stream()
    if stream is not None:
        cur_stream = torch.get_device_module().current_stream()
        cur_stream.wait_stream(stream)


def process_shared_expert(hidden_states, forward_func):
    stream = get_share_stream()
    if stream is None:
        stream = torch.get_device_module().Stream()
        set_share_stream(stream)
    stream.wait_stream(torch.get_device_module().current_stream())
    with torch.get_device_module().stream(stream):
        shared_output = forward_func(hidden_states)
    return shared_output


def process_routed_expert(hidden_states, topk_output, forward_func):
    stream = get_routed_stream()
    if stream is None:
        stream = torch.get_device_module().Stream()
        set_routed_stream(stream)
    stream.wait_stream(torch.get_device_module().current_stream())
    with torch.get_device_module().stream(stream):
        shared_output = forward_func(hidden_states, topk_output)
    return shared_output
