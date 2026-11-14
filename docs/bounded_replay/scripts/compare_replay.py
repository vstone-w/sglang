"""Compare baseline (replay off) and replay-on probe outputs.

Usage: compare_replay.py <baseline.json> <replay.json>
"""

import json
import sys

baseline = json.load(open(sys.argv[1]))
replay = json.load(open(sys.argv[2]))


def logprob_stats(a, b):
    if a is None or b is None:
        return "missing"
    if len(a) != len(b):
        return f"length mismatch {len(a)} vs {len(b)}"
    max_diff = 0.0
    exact = 0
    compared = 0
    for left, right in zip(a, b):
        if left is None or right is None or left[0] is None or right[0] is None:
            continue
        compared += 1
        diff = abs(float(left[0]) - float(right[0]))
        max_diff = max(max_diff, diff)
        if diff == 0.0:
            exact += 1
    return f"compared={compared} exact={exact} max_abs_diff={max_diff:.3e}"


def main():
    for key in sorted(set(baseline) & set(replay)):
        if key == "base_url":
            continue
        b = baseline[key]
        r = replay[key]
        if key == "long_prompt_logprob":
            print(
                f"{key}: prompt_tokens {b.get('prompt_tokens')} vs "
                f"{r.get('prompt_tokens')}; "
                + logprob_stats(
                    b.get("input_token_logprobs"), r.get("input_token_logprobs")
                )
            )
        elif key == "radix_reprefill":
            b_max = b.get("max_cached")
            r_max = r.get("max_cached")
            print(
                f"{key}: prompt_tokens {b.get('prompt_tokens')} vs "
                f"{r.get('prompt_tokens')}; max_cached {b_max} vs {r_max}"
            )
            if b_max is not None and r_max is not None:
                print(f"{key}: max cached delta (baseline-replay) = {b_max - r_max}")
        else:
            same = b.get("output_text") == r.get("output_text")
            print(
                f"{key}: prompt_tokens {b.get('prompt_tokens')} vs "
                f"{r.get('prompt_tokens')}; text_equal={same}"
            )
            if not same:
                print(f"  baseline: {b.get('output_text')!r}")
                print(f"  replay  : {r.get('output_text')!r}")


if __name__ == "__main__":
    main()
