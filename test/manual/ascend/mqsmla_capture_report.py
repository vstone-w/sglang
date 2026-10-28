"""Parse mqsmla_capture logs (MQSMLA_CAPTURE_DIR) and flag suspect arguments.

Run on the A5 host after exporting MQSMLA_CAPTURE_DIR / MQSMLA_CAPTURE_MAX,
launching the service, and sending a request:

  python3 test/manual/ascend/mqsmla_capture_report.py --dir /home/w00889861/log/cap

Checks per main-op call:
  R1  cmp_sparse_indices causal prefix: for the captured first/last q row,
      valid = (restored_kv - seqused_q + 1 + t) // cmp_ratio entries must be
      compressed-token positions in [0, seqused_cmp); the kernel takes exactly
      that prefix and never checks it (block-table lookup is idx // block_size,
      so the page count is NOT the bound).
  R2  block table entries < pool page count (ori and cmp).
  R3  seqused_* <= pool capacity.
  R4  sum(seqused_q) <= q token count (padding is legal; only over-count flags).
  R5  q/sinks/attention output finite and bounded (a huge attention output is
      the final-norm overflow fingerprint).
  R6  printed per-row summary of the cmp index prefix (first/last q row).
  R7  metadata scalar attrs match the main-op call: paired by the metadata
      tensor's data_ptr, falling back to "same cmp_ratio, latest seq".
  R9  packed-row bf16 scales of row0/rowlast of every 608B pool: a scale near
      5e33 is the E8M0-as-bf16 fingerprint (5e33 * fp8 payload -> 1e36).
The compressor / indexer / writer calls get a first-call summary; a non-finite
or huge compressor output is flagged (compressed-KV corruption suspect).
"""

import argparse
import glob
import math
import os
import re
import sys

CALL_RE = re.compile(r"^=== (\S+) call#(\d+)(?: seq#(\d+))? ===")
KV_RE = re.compile(r"^  (\S+): (.*)$")
VALS_RE = re.compile(
    r"^  (\S+) (values|row0|rowlast|kvscale_row0|kvscale_rowlast|"
    r"kvrope_row0|kvrope_rowlast): \[(.*)\]$"
)
SHAPE_RE = re.compile(r"^\(([\d, ]*)\) (torch\.[\w]+)")
PTR_RE = re.compile(r"ptr=(\d+)")
NUM_RE = r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
HUGE = 1e6
HEAD = 32

SCALAR_KEYS = (
    "quant_mode",
    "cmp_ratio",
    "ori_mask_mode",
    "cmp_mask_mode",
    "ori_win_left",
    "ori_win_right",
    "layout_q",
    "layout_kv",
)


def parse_args(s: str) -> list:
    vals = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        try:
            vals.append(int(x))
        except ValueError:
            try:
                vals.append(float(x))
            except ValueError:
                vals.append(x)
    return vals


def parse_file(path):
    calls = []
    cur = None
    try:
        fh = open(path)
    except OSError:
        return calls
    with fh:
        for line in fh:
            line = line.rstrip("\n")
            m = CALL_RE.match(line)
            if m:
                cur = {
                    "name": m.group(1),
                    "n": int(m.group(2)),
                    "seq": int(m.group(3) or 0),
                    "items": {},
                    "vals": {},
                }
                calls.append(cur)
                continue
            if cur is None:
                continue
            m = VALS_RE.match(line)
            if m:
                cur["vals"].setdefault(m.group(1), {})[m.group(2)] = parse_args(
                    m.group(3)
                )
                continue
            m = KV_RE.match(line)
            if m:
                cur["items"][m.group(1)] = m.group(2)
    return calls


def ptr(call, key):
    m = PTR_RE.search(call["items"].get(key, ""))
    return int(m.group(1)) if m else None


def shape_of(call, key):
    m = SHAPE_RE.match(call["items"].get(key, ""))
    if not m:
        return None
    return tuple(int(d) for d in m.group(1).split(",") if d.strip())


def num(call, key, field):
    m = re.search(field + r"=" + NUM_RE, call["items"].get(key, ""))
    return float(m.group(1)) if m else None


def val(call, key, tag="values"):
    return call["vals"].get(key, {}).get(tag)


def vals(call, key, tag="values"):
    return val(call, key, tag) or []


def text(call, key):
    s = call["items"].get(key)
    return s.strip().strip("'\"") if s is not None else None


def ratio_of(call):
    s = text(call, "cmp_ratio")
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return 1


def is_tnd(call):
    layout = text(call, "layout_q") or ""
    return "TND" in layout


def pool_tokens(call, key):
    dims = shape_of(call, key)
    layout = text(call, "layout_kv") or ""
    if not dims:
        return None
    if "PA_" in layout and len(dims) >= 2:
        return dims[0] * dims[1]
    return dims[0]


def q_key(call):
    # SGLang calls the main op as attn_op(q, **kwargs), so q is arg0.
    return "q" if "q" in call["items"] else "arg0"


def q_tokens(call):
    dims = shape_of(call, q_key(call))
    if not dims:
        return None
    return dims[0] if is_tnd(call) else (dims[0] * dims[1] if len(dims) > 1 else None)


def pair_metadata(main, md_calls):
    md_ptr = ptr(main, "metadata")
    if md_ptr is not None:
        # The caching allocator can reuse a freed metadata tensor's address in a
        # later step, so prefer the latest matching return before this call.
        matches = [
            md
            for md in md_calls
            if ptr(md, "ret") == md_ptr and md["seq"] <= main["seq"]
        ]
        if matches:
            return max(matches, key=lambda c: c["seq"])
    ratio = ratio_of(main)
    cands = [
        md for md in md_calls if ratio_of(md) == ratio and md["seq"] <= main["seq"]
    ]
    return max(cands, key=lambda c: c["seq"]) if cands else None


def check_rows(call, flags, warns, notes):
    """R1/R6: per-row causal-prefix check of cmp_sparse_indices."""
    idx = call["vals"].get("cmp_sparse_indices")
    if not idx or ratio_of(call) <= 1:
        return
    ratio = ratio_of(call)
    tnd = is_tnd(call)
    sq = vals(call, "seqused_q")
    so = vals(call, "seqused_ori_kv")
    sc = vals(call, "seqused_cmp_kv")
    res = vals(call, "cmp_residual_kv")
    if not sq:
        warns.append(("R6", "no seqused_q captured; cannot compute the row bound"))
        return
    for tag in ("row0", "rowlast"):
        row = idx.get(tag)
        if not row:
            continue
        if tag == "row0":
            b, t = 0, 0
        else:
            b = 0 if not tnd else len(sq) - 1
            t = sq[b] - 1 if b < len(sq) else -1
        if b >= len(sq) or t < 0:
            continue
        restored = (
            sc[b] * ratio + (res[b] if b < len(res) else 0)
            if sc and b < len(sc)
            else (so[b] if b < len(so) else None)
        )
        if restored is None:
            continue
        valid = (restored - sq[b] + 1 + t) // ratio
        cmp_len = sc[b] if b < len(sc) else None
        prefix = row[: min(max(valid, 0), len(row))]
        notes.append(
            f"    R6 {tag} b={b} t={t} valid={valid} cmp_len={cmp_len} prefix={prefix}"
        )
        if valid < 0:
            flags.append(("R1", f"{tag} valid={valid} is negative"))
            continue
        if cmp_len is not None and any(x < 0 or x >= cmp_len for x in prefix):
            flags.append(
                (
                    "R1",
                    f"{tag} b={b} t={t}: prefix {prefix} not in "
                    f"[0, {cmp_len}) (compressed positions)",
                )
            )
        if valid > len(row):
            warns.append(
                (
                    "R6",
                    f"{tag} valid={valid} > captured head {len(row)}; "
                    "checked the captured prefix only",
                )
            )
            continue
        tail = row[valid:]
        if any(x != -1 for x in tail):
            warns.append(
                (
                    "R6",
                    f"{tag} padding after valid={valid} is not all -1: {tail[:8]}",
                )
            )


def check_kvscales(call, flags):
    # A misread E8M0 0x77 pair shows up as a bf16 scale near 5e33.
    for key, tags in call["vals"].items():
        for tag, v in tags.items():
            if tag.startswith("kvscale_") and any(
                (not math.isfinite(x)) or abs(x) > 1e2 for x in v
            ):
                flags.append(("R9", f"{key} {tag} bf16 scales {v} suspicious"))


def check_main(call, md_calls, flags, warns, notes):
    # R2 block tables
    for bt, pool in (
        ("ori_block_table", "ori_kv"),
        ("cmp_block_table", "cmp_kv"),
    ):
        dims = shape_of(call, pool)
        tmax = num(call, bt, "max")
        if dims and tmax is not None and tmax >= dims[0]:
            flags.append(("R2", f"{bt} max={tmax:.0f} >= pool pages={dims[0]}"))

    # R3 seqused vs capacity
    for key, pool in (
        ("seqused_ori_kv", "ori_kv"),
        ("seqused_cmp_kv", "cmp_kv"),
    ):
        v = vals(call, key)
        cap = pool_tokens(call, pool)
        if v and cap is not None and max(v) > cap:
            flags.append(("R3", f"{key} max={max(v)} > pool tokens={cap}"))

    # R4 q token count
    sq = vals(call, "seqused_q")
    toks = q_tokens(call)
    if sq and toks is not None:
        if sum(sq) > toks:
            flags.append(("R4", f"sum(seqused_q)={sum(sq)} > q tokens={toks}"))
        elif sum(sq) < toks:
            warns.append(
                ("R4", f"sum(seqused_q)={sum(sq)} < q tokens={toks} (padding?)")
            )

    # R5 q / sinks / attention output (the main op returns (out, lse)).
    for key in (q_key(call), "sinks"):
        if "finite=False" in call["items"].get(key, ""):
            flags.append(("R5", f"{key} contains non-finite values"))
        for field in ("max", "min"):
            v = num(call, key, field)
            if v is not None and abs(v) > HUGE:
                flags.append(("R5", f"{key} {field}={v:.3g} suspiciously large"))
    for key in [k for k in call["items"] if k == "ret" or k.startswith("ret")]:
        if "finite=False" in call["items"][key]:
            flags.append(("R5", f"attention {key} contains non-finite values"))
        for field in ("max", "min"):
            v = num(call, key, field)
            if v is not None and abs(v) > HUGE:
                flags.append(("R5", f"attention {key} {field}={v:.3g} huge"))

    check_kvscales(call, flags)
    check_rows(call, flags, warns, notes)

    # R7 metadata/main scalar agreement
    md = pair_metadata(call, md_calls)
    if md is None:
        warns.append(("R7", "no metadata call found for this main call"))
        return
    md_ptr = ptr(call, "metadata")
    if md_ptr is not None and ptr(md, "ret") != md_ptr:
        warns.append(
            (
                "R7",
                f"metadata tensor ptr={md_ptr} not in metadata returns; paired by seq",
            )
        )
    for key in SCALAR_KEYS:
        a, b = text(call, key), text(md, key)
        if a is not None and b is not None and a != b:
            flags.append(("R7", f"metadata/main mismatch on {key}: {b} vs {a}"))


def summarize_other(call, flags, warns):
    """Flag-only checks; the caller prints items/values for call#0 and flagged calls."""
    check_kvscales(call, flags)
    if call["name"] == "compressor":
        # Input/state sanity: a huge x or state means the fault is upstream of
        # (or inside) the compressor, not in its output interpretation.
        for key in ("x", "arg0", "state_cache", "arg3"):
            stats = call["items"].get(key)
            if not stats:
                continue
            if "finite=False" in stats:
                flags.append(("R5", f"compressor {key} contains non-finite values"))
            for field in ("max", "min"):
                v = num(call, key, field)
                if v is not None and abs(v) > HUGE:
                    flags.append(("R5", f"compressor {key} {field}={v:.3g} huge"))
        sb = vals(call, "state_block_table")
        # state_cache is positional arg3 in forward_compress (x, wkv, wgate, ...).
        cache_dims = shape_of(call, "state_cache") or shape_of(call, "arg3")
        if sb and cache_dims and max(sb) >= cache_dims[0]:
            flags.append(
                (
                    "R8",
                    f"state_block_table max={max(sb)} >= state_cache banks={cache_dims[0]}",
                )
            )
        if "ret" in call["items"]:
            if "finite=False" in call["items"]["ret"]:
                flags.append(("R5", "compressor ret contains non-finite values"))
            for field in ("max", "min"):
                v = num(call, "ret", field)
                if v is not None and abs(v) > HUGE:
                    flags.append(("R5", f"compressor ret {field}={v:.3g} huge"))


def analyze_file(path, calls, flags, warns):
    name = os.path.basename(path)
    md_calls = []
    if "mixed_quant_sparse_flash_mla_metadata" in name:
        md_calls = calls
    md_by_pid = {}
    if md_calls:
        md_by_pid[os.path.basename(path)] = md_calls

    saw_c4 = False
    for call in calls:
        local_flags, local_warns, local_notes = [], [], []
        is_main = "mixed_quant_sparse_flash_mla" in name and "metadata" not in name
        if is_main:
            md_path = path.replace(
                "mixed_quant_sparse_flash_mla.",
                "mixed_quant_sparse_flash_mla_metadata.",
            )
            mds = md_by_pid.get(os.path.basename(md_path))
            if mds is None:
                mds = parse_file(md_path)
                md_by_pid[os.path.basename(md_path)] = mds
            check_main(call, mds, local_flags, local_warns, local_notes)
        elif call["name"] in (
            "compressor",
            "npu_quant_lightning_indexer",
            "kv_compress_epilog",
        ):
            summarize_other(call, local_flags, local_warns)

        is_c4 = is_main and "cmp_sparse_indices" in call["vals"] and ratio_of(call) > 1
        show = (
            call["n"] == 0
            or bool(local_flags)
            or bool(local_warns)
            or (is_c4 and not saw_c4)
        )
        if show:
            print(f"  --- {call['name']} call#{call['n']} seq#{call['seq']}")
            for key, stats in call["items"].items():
                print(f"    {key}: {stats}")
                for tag, v in call["vals"].get(key, {}).items():
                    print(f"    {key} {tag}: {v}")
            for line in local_notes:
                print(line)
            for code, msg in local_flags:
                print(f"    [FLAG {code}] {msg}")
            for code, msg in local_warns:
                print(f"    [WARN {code}] {msg}")
        if is_c4:
            saw_c4 = True
        flags.extend(local_flags)
        warns.extend(local_warns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        default=os.environ.get("MQSMLA_CAPTURE_DIR", "/home/w00889861/log/cap"),
    )
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.log")))
    if not files:
        print(f"no *.log under {args.dir}")
        return 1

    flags, warns = [], []
    for path in files:
        calls = parse_file(path)
        if not calls:
            continue
        print(f"\n########## {os.path.basename(path)} ({len(calls)} captured calls)")
        analyze_file(path, calls, flags, warns)

    print(f"\n==== total flags: {len(flags)}  warnings: {len(warns)} ====")
    for code, msg in flags:
        print(f"  [FLAG {code}] {msg}")
    for code, msg in warns:
        print(f"  [WARN {code}] {msg}")
    return 2 if flags else 0


if __name__ == "__main__":
    sys.exit(main())
