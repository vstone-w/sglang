"""Build the validation report from a validate.sh run directory.

Usage: summarize_replay.py <out_dir> <stamp>
Reads env: GSM8K_MIN (default 0.95), GSM8K_DELTA (default 0.02),
MIN_CACHE_HIT_PCT (default 30).
"""

import json
import os
import re
import sys

OUT_DIR = sys.argv[1]
STAMP = sys.argv[2]

GSM8K_MIN = float(os.environ.get("GSM8K_MIN", "0.95"))
GSM8K_DELTA = float(os.environ.get("GSM8K_DELTA", "0.02"))
MIN_HIT = float(os.environ.get("MIN_CACHE_HIT_PCT", "30"))


def read(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def find(pattern, text):
    matches = re.findall(pattern, text)
    return matches[-1] if matches else None


def gsm8k_score(phase):
    return find(r"sgl-eval score=([0-9.]+)", read(f"{OUT_DIR}/{phase}/gsm8k.log"))


def accept_lengths(phase):
    return find(
        r"avg_spec_accept_length per dp: (\[[^\]]*\])",
        read(f"{OUT_DIR}/{phase}/gsm8k.log"),
    )


def prefix_hit(phase):
    return find(r"cache hit rate: ([0-9.]+)%", read(f"{OUT_DIR}/{phase}/prefix.log"))


def probe(phase):
    try:
        with open(f"{OUT_DIR}/{phase}/probe.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def logprob_diff(baseline, replay):
    if not baseline or not replay:
        return None
    a = baseline.get("long_prompt_logprob", {}).get("input_token_logprobs")
    b = replay.get("long_prompt_logprob", {}).get("input_token_logprobs")
    if a is None or b is None or len(a) != len(b):
        return None
    max_diff = 0.0
    compared = 0
    for left, right in zip(a, b):
        if left is None or right is None or left[0] is None or right[0] is None:
            continue
        compared += 1
        max_diff = max(max_diff, abs(float(left[0]) - float(right[0])))
    return {"max_diff": max_diff, "compared": compared}


def radix_delta(baseline, replay):
    if not baseline or not replay:
        return None
    b = baseline.get("radix_reprefill", {}).get("max_cached")
    r = replay.get("radix_reprefill", {}).get("max_cached")
    return None if b is None or r is None else b - r


def fmt(value, suffix=""):
    return "missing" if value is None else f"{value}{suffix}"


def mark(passed):
    return "PASS" if passed else "FAIL"


def main():
    base_score = gsm8k_score("baseline")
    replay_score = gsm8k_score("replay")
    base_hit = prefix_hit("baseline")
    replay_hit = prefix_hit("replay")
    base_probe = probe("baseline")
    replay_probe = probe("replay")
    lp = logprob_diff(base_probe, replay_probe)
    delta = radix_delta(base_probe, replay_probe)

    score_check = None
    if base_score is not None and replay_score is not None:
        base_f, replay_f = float(base_score), float(replay_score)
        score_check = (
            replay_f >= GSM8K_MIN
            and abs(base_f - replay_f) <= GSM8K_DELTA
        )

    hit_check = None
    if replay_hit is not None:
        hit_check = float(replay_hit) >= MIN_HIT

    lp_check = None if lp is None else lp["max_diff"] == 0.0
    radix_check = None if delta is None else 120 <= abs(delta) <= 136
    checks = [c for c in (score_check, hit_check, lp_check, radix_check) if c is not None]

    print(f"# DSV4.1 bounded replay validation — {STAMP}\n")
    print("## Environment\n")
    for line in read(f"{OUT_DIR}/env.txt").strip().splitlines():
        print(f"- {line}")
    print()
    print("## Results\n")
    print("| 指标 | baseline (replay off) | replay on | 判定 |")
    print("|---|---|---|---|")
    print(
        f"| GSM8K score (问法/采样见日志) | {fmt(base_score)} | "
        f"{fmt(replay_score)} | "
        f"{'SKIP' if score_check is None else mark(score_check)} |"
    )
    print(
        f"| prefix cache hit | {fmt(base_hit, '%')} | {fmt(replay_hit, '%')} | "
        f"{'SKIP' if hit_check is None else mark(hit_check)} |"
    )
    print(
        f"| DSpark accept length | {fmt(accept_lengths('baseline'))} | "
        f"{fmt(accept_lengths('replay'))} | - |"
    )
    print(
        f"| prompt logprob max abs diff | - | "
        f"{'missing' if lp is None else format(lp['max_diff'], '.3e')} | "
        f"{'SKIP' if lp_check is None else mark(lp_check)} |"
    )
    print(
        f"| radix cached delta (baseline-replay) | - | {fmt(delta)} | "
        f"{'SKIP' if radix_check is None else mark(radix_check)} |"
    )
    print()
    print("## 判定\n")
    if not checks:
        print("- 没有可判定的数据（服务未跑完或测试被跳过）。")
    elif all(checks):
        print("- 全部已测项 PASS。")
    else:
        print("- 存在 FAIL 项，见上表；先看对应 phase 的日志。")
    print()
    print("## 产物\n")
    print(f"- `{OUT_DIR}/baseline/`：server / probe / prefix / gsm8k 日志")
    print(f"- `{OUT_DIR}/replay/`：同上")
    print(f"- `{OUT_DIR}/compare.txt`：probe 逐项对比")


if __name__ == "__main__":
    main()
