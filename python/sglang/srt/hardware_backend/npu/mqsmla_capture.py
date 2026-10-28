"""Argument capture for the A5 DSV4 mqsmla / compressor / indexer ops (debug aid).

Enabled by MQSMLA_CAPTURE_DIR. Hooks are installed from init_npu_backend after
torch_npu is initialized, so every DP/TP worker process is covered. Each call
logs shapes/dtypes, a bounded sample of tensor stats, per-row heads of integer
tensors, and the op return value to ``<op>.<pid>.log`` in that directory:

  MQSMLA_CAPTURE_DIR=/home/w00889861/log/cap
  MQSMLA_CAPTURE_MAX=64      # max captured calls per op per process
  MQSMLA_CAPTURE_PRE=1       # also dump arguments before the op call, so a
                             # native crash still leaves the inputs on disk

Large tensors (KV pools, q) are sampled on device before any host transfer, so
enabling the capture never copies a whole pool. Every 608B packed row logs its
row0/rowlast bf16 scales (an E8M0 0x77 pair read as bf16 is ~5e33) and rope
absmax. Compare the dump against the verified a5_mqsmla_smoke.py call with
mqsmla_capture_report.py; the suspects are c4_topk_indices value space / causal
prefix, the compressor state table and its output scale, seqused values, block
table contents, and pool row ranges.
"""

import itertools
import os

import torch

# Bound every host transfer: pool-sized tensors are sampled, never copied whole.
_SAMPLE_MAX = 1 << 16
_FULL_MAX = 512
_HEAD = 32
_SEQ = itertools.count()
_INSTALLED = False


def _sampled_flat(v: torch.Tensor):
    flat = v.reshape(-1)
    n = flat.numel()
    if n <= _SAMPLE_MAX:
        return flat, None
    step = max(1, n // _SAMPLE_MAX)
    return flat[::step][:_SAMPLE_MAX], step


def _stats(v) -> str:
    if not isinstance(v, torch.Tensor):
        return repr(v)
    if v.numel() == 0:
        return f"{tuple(v.shape)} {v.dtype} empty ptr={v.data_ptr()}"
    ptr = v.data_ptr()
    s, step = _sampled_flat(v)
    tag = f" sampled_step={step}" if step else ""
    if v.is_floating_point():
        f = s.float()
        return (
            f"{tuple(v.shape)} {v.dtype}{tag} "
            f"finite={torch.isfinite(f).all().item()} "
            f"min={f.min().item():.6g} max={f.max().item():.6g} "
            f"absmean={f.abs().mean().item():.6g} ptr={ptr}"
        )
    i = s.cpu()
    return (
        f"{tuple(v.shape)} {v.dtype}{tag} min={i.min().item()} "
        f"max={i.max().item()} neg={int((i < 0).sum().item())} ptr={ptr}"
    )


def _row_views(v: torch.Tensor):
    """The rows that matter for this debug: first/last q row of an index tensor.

    TND indices are (T, N2=1, K); BSND indices are (B, S, N2=1, K). Other
    tensors fall back to their first/last row.
    """
    if v.dim() == 3 and v.shape[-2] == 1:
        return (("row0", v[0]), ("rowlast", v[-1]))
    if v.dim() == 4 and v.shape[-2] == 1:
        return (("row0", v[0, 0]), ("rowlast", v[0, -1]))
    if v.dim() >= 2:
        rows = [("row0", v[0])]
        if v.shape[0] > 1:
            rows.append(("rowlast", v[-1]))
        return tuple(rows)
    return None


def _as_bf16(byte_slice: torch.Tensor) -> torch.Tensor:
    try:
        return byte_slice.view(torch.bfloat16).float()
    except Exception:
        # Some backends cannot dtype-view fp8 -> bf16 directly; go via uint8.
        return byte_slice.view(torch.uint8).view(torch.bfloat16).float()


def _kv_pool_lines(key: str, v) -> list[str]:
    """Row-0/row-last scale bytes of a 608B packed KV pool (uint8 or fp8)."""
    if (
        not isinstance(v, torch.Tensor)
        or v.element_size() != 1
        or v.dim() < 2
        or v.shape[-1] != 608
    ):
        return []
    try:
        rows = v.reshape(-1, 608)
    except Exception:
        return []
    lines = []
    for tag, idx in (("row0", 0), ("rowlast", rows.shape[0] - 1)):
        row = rows[idx]
        try:
            scale = _as_bf16(row[576:590])
            rope = _as_bf16(row[:128])
        except Exception:
            continue
        lines.append(f"  {key} kvscale_{tag}: {scale.cpu().tolist()}")
        lines.append(f"  {key} kvrope_{tag}: [{rope.abs().max().item():.6g}]")
    return lines


def _packed_row_lines(pool, slot: int, key: str) -> list[str]:
    """Bytes of one 608B row: the writer/reader pair is compared slot by slot."""
    try:
        rows = pool.view(torch.uint8).reshape(-1, 608)
        slot = int(slot)
        if slot < 0 or slot >= rows.shape[0]:
            return [f"  {key}[slot={slot}]: out of range ({rows.shape[0]})"]
        row = rows[slot]
        scale = _as_bf16(row[576:590]).cpu().tolist()
        rope = _as_bf16(row[:128]).float()
        nope = row[128:576].view(torch.float8_e4m3fn).float()
        nz = int((row != 0).sum().item())
    except Exception as e:
        return [f"  {key}: dump failed ({e!r})"]
    return [
        f"  {key}[slot={slot}]: nz={nz} "
        f"rope_absmax={rope.abs().max().item():.6g} "
        f"nope_absmax={nope.abs().max().item():.6g} "
        f"scale={scale} nope0={nope[:8].cpu().tolist()}"
    ]


def _indexer_row_lines(cache, scale, slot: int) -> list[str]:
    try:
        rows = cache.reshape(-1, cache.shape[-1])
        slot = int(slot)
        x = rows[slot].float()
        scalars = scale.reshape(-1).float()
        s = scalars[slot].item()
        ctx = scalars[max(0, slot - 2) : slot + 3].cpu().tolist()
    except Exception as e:
        return [f"  indexer_row: dump failed ({e!r})"]
    return [
        f"  indexer_row[slot={slot}]: absmax={x.abs().max().item():.6g} "
        f"absmean={x.abs().mean().item():.6g} scale={s} scale_ctx={ctx}"
    ]


def _special_dump(name: str, args, kwargs) -> list[str]:
    """Write/read the exact pool rows the kernels touch (slot/page aligned)."""
    lines: list[str] = []
    if name in ("kv_compress_epilog", "mixed_quant_sparse_flash_mla"):
        # Writers may be enqueued on a side stream; make the probe see the
        # completed kernel before judging the row bytes.
        try:
            torch.npu.synchronize()
        except Exception:
            pass
    if name == "kv_compress_epilog" and len(args) >= 3:
        pool = args[0]
        try:
            lines.append(
                f"  cache_props: contiguous={pool.is_contiguous()} "
                f"offset={pool.storage_offset()} dtype={pool.dtype} "
                f"ptr_off={pool.data_ptr() - pool.untyped_storage().data_ptr()}"
            )
        except Exception:
            pass
        try:
            x = args[1]
            lines.append(
                f"  x_props: offset={x.storage_offset()} "
                f"storage_elems={x.untyped_storage().nbytes() // x.element_size()} "
                f"strides={tuple(x.stride())} contig={x.is_contiguous()}"
            )
        except Exception:
            pass
        slots = args[2] if isinstance(args[2], torch.Tensor) else None
        if slots is not None:
            for s in slots.reshape(-1)[:8].cpu().tolist():
                lines.extend(_packed_row_lines(args[0], s, "written"))
    elif name == "indexer_quant_cache" and len(args) >= 4:
        cache, scale, slots = args[0], args[1], args[3]
        for s in slots.reshape(-1)[:8].cpu().tolist():
            lines.extend(_indexer_row_lines(cache, scale, s))
    elif name == "mixed_quant_sparse_flash_mla":
        ori = kwargs.get("ori_kv")
        ori_bt = kwargs.get("ori_block_table")
        if ori is not None and ori_bt is not None:
            slot = int(ori_bt.reshape(-1)[0]) * ori.shape[1]
            lines.extend(_packed_row_lines(ori, slot, "read_ori"))
        cmp_kv = kwargs.get("cmp_kv")
        cmp_bt = kwargs.get("cmp_block_table")
        if cmp_kv is not None and cmp_bt is not None:
            slot = int(cmp_bt.reshape(-1)[0]) * cmp_kv.shape[1]
            lines.extend(_packed_row_lines(cmp_kv, slot, "read_cmp"))
    return lines


def _values_lines(key: str, v) -> list[str]:
    lines = _kv_pool_lines(key, v)
    if not isinstance(v, torch.Tensor) or v.numel() == 0:
        return lines
    if v.is_floating_point():
        # First/last row stats separate a valid compressed row from the
        # uninitialised tail rows the compressor leaves past cmpS.
        if v.dim() >= 2 and not (
            v.element_size() == 1 and v.shape[-1] == 608
        ):
            for tag, row in _row_views(v) or ():
                lines.append(f"  {key} {tag}: {_stats(row)}")
        return lines
    rows = _row_views(v)
    if rows is not None:
        for tag, row in rows:
            vals = row.reshape(-1)[:_HEAD].cpu().tolist()
            lines.append(f"  {key} {tag}: {vals}")
        return lines
    if v.numel() <= _FULL_MAX:
        lines.append(f"  {key} values: {v.flatten().cpu().tolist()}")
    else:
        lines.append(f"  {key} values: {v.flatten()[:_HEAD].cpu().tolist()}")
    return lines


def _ret_lines(ret) -> list[str]:
    items = ret if isinstance(ret, (tuple, list)) else (ret,)
    lines = []
    for i, item in enumerate(items):
        key = "ret" if len(items) == 1 else f"ret{i}"
        lines.append(f"  {key}: {_stats(item)}")
        lines.extend(_values_lines(key, item))
    return lines


def _in_graph_capture() -> bool:
    fn = getattr(torch.npu, "is_current_stream_capturing", None)
    if fn is None:
        return False
    try:
        return bool(fn())
    except Exception:
        return False


def _arg_lines(args, kwargs) -> list[str]:
    lines = []
    for i, v in enumerate(args):
        lines.append(f"  arg{i}: {_stats(v)}")
        lines.extend(_values_lines(f"arg{i}", v))
    for key, v in kwargs.items():
        lines.append(f"  {key}: {_stats(v)}")
        lines.extend(_values_lines(key, v))
    return lines


def _append(path: str, lines: list[str]) -> None:
    """Append and fsync so a native crash leaves the pre-call dump on disk."""
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _wrap(ns, name: str, path: str, max_calls: int) -> bool:
    try:
        orig = getattr(ns, name)
    except Exception as e:
        print(f"mqsmla_capture: skip {name} ({e!r})")
        return False
    seen = {"n": 0}
    pre = os.environ.get("MQSMLA_CAPTURE_PRE") == "1"

    def hooked(*args, **kwargs):
        n = seen["n"]
        seen["n"] += 1
        capture = n < max_calls and not _in_graph_capture()
        seq = next(_SEQ) if capture else 0
        if capture and pre:
            try:
                _append(
                    path,
                    [
                        f"\n=== {name} call#{n} seq#{seq} ===",
                        "  phase: PRE",
                        *_arg_lines(args, kwargs),
                    ],
                )
            except Exception as e:  # a debug hook must never break the model
                print(f"mqsmla_capture: pre-dump {name} call#{n} failed: {e!r}")
        ret = orig(*args, **kwargs)
        if capture:
            try:
                _append(
                    path,
                    [
                        f"\n=== {name} call#{n} seq#{seq} ===",
                        "  phase: POST",
                        *_arg_lines(args, kwargs),
                        *_ret_lines(ret),
                        *_special_dump(name, args, kwargs),
                    ],
                )
            except Exception as e:  # a debug hook must never break the model
                print(f"mqsmla_capture: dump {name} call#{n} failed: {e!r}")
        return ret

    try:
        setattr(ns, name, hooked)
    except Exception as e:  # pragma: no cover - surface hook failures loudly
        print(f"mqsmla_capture: cannot hook {name}: {e!r}")
        return False
    return True


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    outdir = os.environ.get("MQSMLA_CAPTURE_DIR")
    if not outdir:
        return
    try:
        import cann_ops_transformer  # noqa: F401  (registers the namespace)
    except ImportError:
        print("mqsmla_capture: cann_ops_transformer unavailable; no hooks")
        return
    except Exception as e:  # never take the service down for a debug hook
        print(f"mqsmla_capture: import cann_ops_transformer failed: {e!r}")
        return
    try:
        max_calls = int(os.environ.get("MQSMLA_CAPTURE_MAX", "64"))
    except ValueError:
        max_calls = 64
    try:
        os.makedirs(outdir, exist_ok=True)
    except OSError as e:
        print(f"mqsmla_capture: cannot create {outdir}: {e!r}")
        return
    pid = os.getpid()

    targets = [
        (torch.ops.cann_ops_transformer, "mixed_quant_sparse_flash_mla"),
        (torch.ops.cann_ops_transformer, "mixed_quant_sparse_flash_mla_metadata"),
        (torch.ops.cann_ops_transformer, "kv_compress_epilog"),
        # A5 uses the officially published unfused compressor + partial RoPE.
        (torch.ops.cann_ops_transformer, "compressor"),
        (torch.ops.cann_ops_transformer, "inplace_partial_rotary_mul"),
        (torch.ops.cann_ops_transformer, "quant_lightning_indexer"),
        (torch.ops.cann_ops_transformer, "quant_lightning_indexer_metadata"),
        (torch.ops.cann_ops_transformer, "indexer_quant_cache"),
        (torch.ops.cann_ops_transformer, "mhc_post"),
    ]

    installed = []
    for ns, name in targets:
        path = os.path.join(outdir, f"{name}.{pid}.log")
        if _wrap(ns, name, path, max_calls):
            installed.append(name)
    _INSTALLED = True
    print(f"mqsmla_capture: pid {pid} -> {outdir}; hooks={installed}")
