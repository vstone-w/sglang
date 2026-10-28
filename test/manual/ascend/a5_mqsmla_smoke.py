"""A5 eager smoke for the official mqsmla op pair (CANN 9.2 image).

Verifies, without sglang, the three vendor ops the env switch routes to:
  torch.ops.cann_ops_transformer.kv_compress_epilog          (writer)
  torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla_metadata
  torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla (reader)

Run on an A5 (arch35) host inside the CANN 9.2 image:

  python test/manual/ascend/a5_mqsmla_smoke.py

Sections:
  1. Writer row layout, byte-exact: synthetic probes (constant rows expose
     field landing spots) + a candidate matrix on random rows. Expected
     [rope bf16 128B @0, verbatim from x[:, 448:]][nope fp8-e4m3 448B @128,
     group 64][scale 7 x bf16 @576][pad -> 608B].
  2. Reader numerics against the vendor golden semantics
     (tests/pytest/mixed_quant_sparse_flash_mla_golden.py:478-498):
     thr = S2 - S1 + t + 1; mode 3 attends [0, thr); mode 4 attends
     [thr - win_left - 1, thr + win_right). A TND/PA_BBND dialect check plus a
     S2=512 mask matrix (win 15/31/63/127 x mode 3/4) so mask variants are
     geometrically distinguishable, and a c4 (cmp_ratio=4 + topk) oracle.
  3. slot_mapping is IDENTITY and block tables are arange: logical position i
     maps to physical row i. Any permutation here scrambles the logical order
     the reader sees and invalidates every oracle below.
"""

import math
import sys

import torch
import torch_npu  # noqa: F401  (registers the npu device)

try:
    # Op registration is an import side effect; without it the
    # torch.ops.cann_ops_transformer namespace is empty.
    import cann_ops_transformer  # noqa: F401
except ImportError as e:
    print(
        "cannot import cann_ops_transformer; source the CANN env first, e.g.\n"
        "  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
    )
    raise SystemExit(1) from e

torch.npu.set_device(0)
torch.manual_seed(0)

BLOCK_SIZE = 16
NOPE_DIM, ROPE_DIM, Q_DIM, Q_HEADS, GROUP = 448, 64, 512, 64, 64
KV_DIM = 608  # 448 + 128 + 7*2, aligned to 32
S1 = 8
SCALE = 1.0 / math.sqrt(Q_DIM)

OPS = torch.ops.cann_ops_transformer
# The op pair is exercised with zero sinks; the golden softmax must include the
# sink term in its denominator (V row is absent) to match the vendor golden.
SINKS = torch.zeros(Q_HEADS, dtype=torch.float32, device="npu")

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        failures.append(name)


def write_cache(n_tokens: int, x: torch.Tensor) -> torch.Tensor:
    block_num = (n_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    cache = torch.zeros(
        block_num, BLOCK_SIZE, 1, KV_DIM, dtype=torch.uint8, device="npu"
    )
    # Identity slot mapping: token i lands in physical row i.
    OPS.kv_compress_epilog(
        cache,
        x,
        torch.arange(n_tokens, dtype=torch.int64, device="npu"),
        quant_group_size=GROUP,
        quant_mode="fp8_bf16",
        round_scale=True,
        x_scale=1.0,
    )
    torch.npu.synchronize()
    return cache


def dequant(cache: torch.Tensor, n: int) -> torch.Tensor:
    """Rows -> float, arranged in q order [nope 448 | rope 64]."""
    rows = cache.view(-1, KV_DIM)[:n]
    rope = rows[:, 0 : ROPE_DIM * 2].view(torch.bfloat16).float()
    nope_bytes = rows[:, 128 : 128 + NOPE_DIM].view(torch.float8_e4m3fn)
    scales = rows[:, 576 : 576 + 14].view(torch.bfloat16).float()
    nope = (nope_bytes.float().view(n, 7, GROUP) * scales.unsqueeze(-1)).view(
        n, NOPE_DIM
    )
    return torch.cat([nope, rope], dim=-1)


# ============ section 1: writer row layout ==================================

N_SMALL = 64
x_small = torch.randn(N_SMALL, Q_DIM, dtype=torch.bfloat16, device="npu")
cache_small = write_cache(N_SMALL, x_small)

check(
    "pad bytes [590:608] are zero",
    bool((cache_small.view(-1, KV_DIM)[0, 590:608] == 0).all()),
)


def _probe(name: str, fill_rope: bool):
    px = torch.zeros(1, Q_DIM, dtype=torch.bfloat16, device="npu")
    if fill_rope:
        px[:, NOPE_DIM:] = 1.0
    else:
        px[:, :NOPE_DIM] = 1.0
    pc = torch.zeros(1, 1, 1, KV_DIM, dtype=torch.uint8, device="npu")
    OPS.kv_compress_epilog(
        pc,
        px,
        torch.zeros(1, dtype=torch.int64, device="npu"),
        quant_group_size=GROUP,
        quant_mode="fp8_bf16",
        round_scale=True,
        x_scale=1.0,
    )
    torch.npu.synchronize()
    prow = pc.view(-1, KV_DIM)[0].cpu()
    nz = (prow != 0).nonzero().flatten()
    span = (int(nz.min()), int(nz.max())) if nz.numel() else None
    print(f"probe {name}: nonzero byte span={span}")
    for lo, hi in ((0, 32), (440, 480), (560, 608)):
        print(f"  [{lo}:{hi}]  {prow[lo:hi].numpy().tobytes().hex(' ', 4)}")


_probe("rope=1.0", fill_rope=True)
_probe("nope=1.0", fill_rope=False)

rows = cache_small.view(-1, KV_DIM)
written = rows[:N_SMALL]  # identity slot mapping
rope_bit_rate = (
    (
        written[:, 0 : ROPE_DIM * 2].view(torch.int16)
        == x_small[:, NOPE_DIM:].contiguous().view(torch.int16)
    )
    .float()
    .mean()
    .item()
)
deq = dequant(cache_small, N_SMALL)
nope_cos = (
    torch.nn.functional.cosine_similarity(
        deq[:, :NOPE_DIM], x_small[:, :NOPE_DIM].float(), dim=-1
    )
    .mean()
    .item()
)
check(
    "writer rope bf16 bit-exact @0",
    rope_bit_rate == 1.0,
    f"rate={rope_bit_rate:.4f}",
)
check("writer nope dequant (g64 bf16 scale)", nope_cos > 0.98, f"cos={nope_cos:.4f}")

# ============ section 2: reader vs golden semantics =========================


def run_reader(
    kv_cache,
    block_table,
    q,
    *,
    mode,
    win_l,
    win_r,
    cmp_cache=None,
    cmp_block_table=None,
    cmp_indices=None,
    seqused_cmp=None,
    cmp_residual=None,
    cmp_ratio=1,
    layout_q="BSND",
    cu_q=None,
):
    """One metadata + main-op call; every attr mirrored between the two."""
    b = q.shape[0] if layout_q == "BSND" else 1
    s1 = q.shape[1] if layout_q == "BSND" else q.shape[0]
    n_kv = kv_cache.view(-1, KV_DIM).shape[0]
    seqused_q = torch.full((b,), s1, dtype=torch.int32, device="npu")
    seqused_ori = torch.full((b,), n_kv, dtype=torch.int32, device="npu")
    md_kwargs = dict(
        num_heads_q=Q_HEADS,
        num_heads_kv=1,
        head_dim=Q_DIM,
        quant_mode=1,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori,
        batch_size=b,
        max_seqlen_q=s1,
        max_seqlen_ori_kv=n_kv,
        ori_topk=0,
        rope_head_dim=ROPE_DIM,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=mode,
        cmp_mask_mode=3 if cmp_cache is not None else 0,
        ori_win_left=win_l,
        ori_win_right=win_r,
        layout_q=layout_q,
        layout_kv="PA_BBND",
        has_ori_kv=True,
        has_cmp_kv=cmp_cache is not None,
    )
    if cmp_cache is not None:
        md_kwargs.update(
            seqused_cmp_kv=seqused_cmp,
            cmp_residual_kv=cmp_residual,
            cmp_topk=cmp_indices.shape[-1],
            max_seqlen_cmp_kv=int(seqused_cmp.max().item()),
        )
    if cu_q is not None:
        md_kwargs["cu_seqlens_q"] = cu_q
    metadata = OPS.mixed_quant_sparse_flash_mla_metadata(**md_kwargs)

    read_kwargs = dict(
        quant_mode=1,
        ori_kv=kv_cache.view(torch.float8_e4m3fn),
        ori_block_table=block_table,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori,
        sinks=SINKS,
        metadata=metadata,
        rope_head_dim=ROPE_DIM,
        softmax_scale=SCALE,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=mode,
        cmp_mask_mode=3 if cmp_cache is not None else 0,
        ori_win_left=win_l,
        ori_win_right=win_r,
        layout_q=layout_q,
        layout_kv="PA_BBND",
        topk_value_mode=1,
        return_softmax_lse=False,
    )
    if cmp_cache is not None:
        read_kwargs.update(
            cmp_kv=cmp_cache.view(torch.float8_e4m3fn),
            cmp_block_table=cmp_block_table,
            seqused_cmp_kv=seqused_cmp,
            cmp_residual_kv=cmp_residual,
            cmp_sparse_indices=cmp_indices,
        )
    if cu_q is not None:
        read_kwargs["cu_seqlens_q"] = cu_q
    out, _ = OPS.mixed_quant_sparse_flash_mla(q, **read_kwargs)
    torch.npu.synchronize()
    return out


def golden_set(t: int, s2: int, mode: int, win_l: int, win_r: int):
    """golden.py:478-498 — thr = S2 - S1 + t + 1 (exclusive end)."""
    thr = s2 - S1 + t + 1
    if mode == 3:
        # Mode 3 ignores the window attrs (checker requires -1/-1).
        return 0, min(max(thr, 0), s2)
    start = 0 if win_l == -1 else max(thr - win_l - 1, 0)
    end = s2 if win_r == -1 else min(max(thr + win_r, 0), s2)
    return start, end


def golden_out(
    kv: torch.Tensor,
    qf: torch.Tensor,
    s2: int,
    mode: int,
    win_l: int,
    win_r: int,
    cmp=None,
    cmp_valid=None,
):
    """Sink-aware softmax over the golden ori window + causal cmp prefix.

    Mirrors mixed_quant_sparse_flash_mla_golden.py::sinks_softmax: the sink
    adds exp(sink - max) to the denominator and has no V row.
    """
    ref = torch.empty_like(qf)
    for t in range(S1):
        start, end = golden_set(t, s2, mode, win_l, win_r)
        keys = [kv[start:end]]
        if cmp is not None:
            keys.append(cmp[: cmp_valid[t]])
        k = torch.cat(keys, dim=0)
        logits = k @ qf[t].transpose(0, 1) * SCALE  # (kv, H)
        x_max = torch.maximum(logits.max(dim=0).values, SINKS)
        weights = torch.exp(logits - x_max.unsqueeze(0))
        denom = weights.sum(dim=0) + torch.exp(SINKS - x_max)
        ref[t] = (weights / denom.unsqueeze(0)).transpose(0, 1) @ k
    return ref


def cos_vs(out, ref):
    return torch.nn.functional.cosine_similarity(
        ref, out.view(S1, Q_HEADS, Q_DIM).float(), dim=-1
    )


q = torch.randn(1, S1, Q_HEADS, Q_DIM, dtype=torch.bfloat16, device="npu")
qf = q.view(S1, Q_HEADS, Q_DIM).float()
q_tnd = q.view(S1, Q_HEADS, Q_DIM)

# --- TND dialect (sglang's actual call shape), S2=64 -------------------------
cu_q = torch.tensor([0, S1], dtype=torch.int32, device="npu")
out_tnd = run_reader(
    cache_small,
    torch.arange(4, dtype=torch.int32, device="npu").unsqueeze(0),
    q_tnd,
    mode=4,
    win_l=63,
    win_r=0,
    layout_q="TND",
    cu_q=cu_q,
)
ref = golden_out(deq, qf, N_SMALL, 4, 63, 0)
per_tok = cos_vs(out_tnd, ref).mean(-1)
print(
    "TND per-token cos vs golden (mode4 win63): "
    f"{[round(v, 4) for v in per_tok.tolist()]}"
)
check(
    "TND dialect matches golden (S2=64, win=63)",
    per_tok.min().item() > 0.98,
    f"min={per_tok.min().item():.4f}",
)

# --- mask matrix: S2=512 so window sizes are geometrically distinguishable ---
N_BIG = 512
x_big = torch.randn(N_BIG, Q_DIM, dtype=torch.bfloat16, device="npu")
cache_big = write_cache(N_BIG, x_big)
kv_big = dequant(cache_big, N_BIG)
bt_big = torch.arange(N_BIG // BLOCK_SIZE, dtype=torch.int32, device="npu").unsqueeze(0)

# Checker rule (EZ0027): when ori_mask_mode != 4, win_left AND win_right must
# both be -1; non-negative offsets are only legal with mode 4.
MASK_CASES = ((3, -1, -1), (4, 15, 0), (4, 31, 0), (4, 63, 0), (4, 127, 0))
for layout in ("BSND", "TND"):
    q_arg = q if layout == "BSND" else q_tnd
    for mode, win_l, win_r in MASK_CASES:
        out = run_reader(
            cache_big,
            bt_big,
            q_arg,
            mode=mode,
            win_l=win_l,
            win_r=win_r,
            layout_q=layout,
            cu_q=cu_q if layout == "TND" else None,
        )
        ref = golden_out(kv_big, qf, N_BIG, mode, win_l, win_r)
        per_tok = cos_vs(out, ref).mean(-1)
        print(
            f"mask matrix ({layout}, S2=512) mode={mode} "
            f"win=({win_l},{win_r}): min={per_tok.min().item():.4f} "
            f"mean={per_tok.mean().item():.4f}"
        )
        check(
            f"golden match: {layout} mode={mode} win=({win_l},{win_r})",
            per_tok.min().item() > 0.98,
            f"min={per_tok.min().item():.4f}",
        )

# --- c4 (CSA) oracle: ratio=4, topk indices as a causal prefix ---------------
CMP_RATIO = 4
CMP_LEN = N_BIG // CMP_RATIO  # 128 compressed tokens, residual 0
TOPK = 8
x_cmp = torch.randn(CMP_LEN, Q_DIM, dtype=torch.bfloat16, device="npu")
cache_cmp = write_cache(CMP_LEN, x_cmp)
cmp_deq = dequant(cache_cmp, CMP_LEN)
bt_cmp = torch.arange(CMP_LEN // BLOCK_SIZE, dtype=torch.int32, device="npu").unsqueeze(
    0
)

seqused_cmp = torch.tensor([CMP_LEN], dtype=torch.int32, device="npu")
cmp_residual = torch.tensor([0], dtype=torch.int32, device="npu")
# Doc example: each q row carries floor(thr_t / ratio) valid indices up front,
# -1 after; the kernel consumes exactly that prefix without re-checking.
cmp_indices = torch.full((1, S1, 1, TOPK), -1, dtype=torch.int32, device="npu")
cmp_valid = []
for t in range(S1):
    thr = N_BIG - S1 + t + 1
    n_valid = min(thr // CMP_RATIO, TOPK)
    cmp_indices[0, t, 0, :n_valid] = torch.arange(
        n_valid, dtype=torch.int32, device="npu"
    )
    cmp_valid.append(n_valid)

# SGLang calls c4a with TND (3-D indices); keep the BSND 4-D dialect too.
for layout, q_arg, cmp_idx in (
    ("BSND", q, cmp_indices),
    ("TND", q_tnd, cmp_indices.view(S1, 1, TOPK)),
):
    out_csa = run_reader(
        cache_big,
        bt_big,
        q_arg,
        mode=4,
        win_l=63,
        win_r=0,
        layout_q=layout,
        cu_q=cu_q if layout == "TND" else None,
        cmp_cache=cache_cmp,
        cmp_block_table=bt_cmp,
        cmp_indices=cmp_idx,
        seqused_cmp=seqused_cmp,
        cmp_residual=cmp_residual,
        cmp_ratio=CMP_RATIO,
    )
    ref_csa = golden_out(kv_big, qf, N_BIG, 4, 63, 0, cmp=cmp_deq, cmp_valid=cmp_valid)
    per_tok = cos_vs(out_csa, ref_csa).mean(-1)
    print(
        f"c4 oracle ({layout}) per-token cos: {[round(v, 4) for v in per_tok.tolist()]}"
    )
    check(
        f"c4 ({layout}, ratio=4 + topk prefix) matches golden",
        per_tok.min().item() > 0.98,
        f"min={per_tok.min().item():.4f}",
    )

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all mqsmla smoke checks passed")
