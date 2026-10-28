"""Standalone A5 smoke for the official CANN compressor op.

Runs the op outside SGLang so a crash can be attributed to the op or to the
arguments it is given. cache_mode=1 mirrors cann-recipes-infer's
compressor_prolog: 2-D x + cu_seqlens, state table [B, ceil(Smax/ring)] whose
non-zero ids are request banks (+1) and where block 0 is "unassigned".

  python3 test/manual/ascend/a5_compressor_smoke.py --ratio 4 --cache-mode 1
  python3 test/manual/ascend/a5_compressor_smoke.py --ratio 4 --cache-mode 2

A finite output with absmax O(1) after RMSNorm/RoPE means the op path is
healthy; a segfault points at the op build or the state-table ABI.
"""

import argparse
import math

import torch
import torch_npu  # noqa: F401  (registers the npu device)


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--seq", type=int, default=256, help="tokens per request")
    ap.add_argument("--ring", type=int, default=8, help="state block size")
    ap.add_argument("--cache-mode", type=int, default=1)
    ap.add_argument("--device", default="npu:0")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    import cann_ops_transformer  # noqa: F401  (registers the torch namespace)

    compressor = torch.ops.cann_ops_transformer.compressor
    rope_op = torch.ops.cann_ops_transformer.inplace_partial_rotary_mul
    device = torch.device(args.device)
    torch.manual_seed(0)

    ratio = args.ratio
    head_dim = args.head_dim
    coff = 2 if ratio == 4 else 1
    bs = args.bs
    ring = args.ring
    state_dim = 2 * coff * head_dim

    def run(chunk_start: int, chunk_len: int, label: str):
        cu = torch.arange(
            0, bs * chunk_len + 1, chunk_len, dtype=torch.int32, device=device
        )
        seqused = torch.full((bs,), chunk_len, dtype=torch.int32, device=device)
        start_pos = torch.full((bs,), chunk_start, dtype=torch.int32, device=device)
        x = (torch.randn(bs * chunk_len, args.hidden, device=device) * 0.02).to(
            torch.bfloat16
        )
        smax = chunk_start + chunk_len
        width = max(1, math.ceil(smax / ring))
        if args.cache_mode == 1:
            # Block 0 is the unassigned sentinel; request r owns block r+1.
            banks = torch.arange(1, bs + 1, dtype=torch.int32, device=device)
            table = (
                banks.view(-1, 1).expand(-1, width).contiguous().to(torch.int32)
            )
        else:
            # Ring mode: one bank id per request, 0 is a valid physical block.
            table = torch.arange(bs, dtype=torch.int32, device=device)
        cmp_kv = compressor(
            x,
            wkv,
            wgate,
            state_cache,
            ape,
            ratio,
            state_block_table=table,
            cu_seqlens=cu,
            seqused=seqused,
            start_pos=start_pos,
            coff=coff,
            cache_mode=args.cache_mode,
        )
        raw = cmp_kv.float()
        print(
            f"{label} raw: rows={cmp_kv.shape[0]} "
            f"finite={torch.isfinite(raw).all().item()} "
            f"absmax={raw.abs().max().item():.6g} "
            f"absmean={raw.abs().mean().item():.6g}"
        )
        for tag, row in (("row0", cmp_kv[0]), ("rowlast", cmp_kv[-1])):
            rf = row.float()
            print(
                f"{label} raw_{tag}: finite={torch.isfinite(rf).all().item()} "
                f"absmax={rf.abs().max().item():.6g} "
                f"absmean={rf.abs().mean().item():.6g}"
            )
        cmp_kv = torch_npu.npu_rms_norm(cmp_kv, norm_w.to(cmp_kv.dtype), 1e-6)[0]
        cos = torch.ones(cmp_kv.shape[0], 64, device=device)
        sin = torch.zeros(cmp_kv.shape[0], 64, device=device)
        rope_op(
            cmp_kv.view(-1, 1, 1, head_dim),
            cos.view(-1, 1, 1, 64),
            sin.view(-1, 1, 1, 64),
            rotary_mode="interleave",
            partial_slice=[head_dim - 64, head_dim],
        )
        stats = cmp_kv.float()
        print(
            f"{label}: rows={cmp_kv.shape[0]} table={tuple(table.shape)} "
            f"finite={torch.isfinite(stats).all().item()} "
            f"absmax={stats.abs().max().item():.6g} "
            f"absmean={stats.abs().mean().item():.6g}"
        )
        assert torch.isfinite(stats).all(), f"{label}: non-finite compressor output"
        assert stats.abs().max().item() < 100, f"{label}: output not O(1)"

    wkv = (
        torch.randn(coff * head_dim, args.hidden, device=device) / args.hidden**0.5
    ).to(torch.bfloat16)
    wgate = (
        torch.randn(coff * head_dim, args.hidden, device=device) / args.hidden**0.5
    ).to(torch.bfloat16)
    ape = torch.zeros(ratio, coff * head_dim, dtype=torch.float32, device=device)
    norm_w = torch.ones(head_dim, dtype=torch.float32, device=device)
    # One dummy block plus one block per request, matching the runtime pool.
    state_cache = torch.zeros(
        bs + 2, ring, state_dim, dtype=torch.float32, device=device
    )

    # Two chunks: the second reads the state written by the first.
    first = args.seq // 2
    run(0, first, "chunk0")
    run(first, args.seq - first, "chunk1")
    print("PASS")


if __name__ == "__main__":
    main()
