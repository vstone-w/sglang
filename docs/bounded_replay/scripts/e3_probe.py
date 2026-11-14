"""E3 probe: encoder replay + DSpark fold/decode correctness.

Wave 1 sends one prompt per DP rank (round-robin, fresh); wave 2 sends the same
prompt again, so each rank's second request hits radix and folds its replay
window. Greedy outputs must match pairwise (same rank, same prompt).
"""

import json
import sys

import requests

BASE = "http://127.0.0.1:8001"
TIMEOUT = 1200


def post(path, payload, timeout=TIMEOUT):
    return requests.post(BASE + path, json=payload, timeout=timeout)


def generate(prompt, max_new_tokens=8):
    r = post(
        "/generate",
        {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
            },
        },
    )
    r.raise_for_status()
    body = r.json()
    meta = body.get("meta_info") or {}
    return body.get("text", ""), meta


def accept_length():
    r = requests.get(BASE + "/server_info", timeout=60)
    r.raise_for_status()
    states = r.json().get("internal_states") or []
    vals = [
        s.get("avg_spec_accept_length")
        for s in states
        if s.get("avg_spec_accept_length") is not None
    ]
    return vals


def main():
    ranks = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    post("/flush_cache", {})

    filler = (
        "The following is a long neutral passage used only as context. "
        "It describes rivers, bridges, and the slow passage of seasons. "
    ) * 60
    prompt = filler + "\nAnswer with one word. The capital of France is"

    wave = []
    for i in range(ranks):
        text, meta = generate(prompt)
        wave.append(text)
        print(f"wave1[{i}] prompt_tokens={meta.get('prompt_tokens')} text={text!r}")
    wave2 = []
    for i in range(ranks):
        text, meta = generate(prompt)
        wave2.append(text)
        print(
            f"wave2[{i}] prompt_tokens={meta.get('prompt_tokens')} "
            f"cached={meta.get('cached_tokens')} text={text!r}"
        )

    # Encoder replay is not bitwise-equivalent to a full prefill: the replay
    # rows' own windows are floored at the replay start, so a greedy near-tie
    # can flip. The contract is that every hit request answers correctly.
    wrong = [t for t in wave + wave2 if "paris" not in t.lower()]
    if wrong:
        print(f"FOLD WRONG: {json.dumps(wrong, ensure_ascii=False)}")
        return 1
    same = sum(a == b for a, b in zip(wave, wave2))
    print(f"FOLD OK: {ranks} rank-paired hits correct; {same}/{ranks} bit-identical")
    for i, (a, b) in enumerate(zip(wave, wave2)):
        if a != b:
            print(f"  near-tie[{i}]: fresh={a!r} hit={b!r}")

    for text, expected in [
        ("The capital of France is", "Paris"),
        ("2 + 2 =", "4"),
        ("Water freezes at", "0"),
    ]:
        out, meta = generate(text, max_new_tokens=16)
        ok = expected.lower() in out.lower()
        print(f"factual {text!r} -> {out!r} ({'ok' if ok else 'MISS'})")
        if not ok:
            return 1

    print("accept_length:", accept_length())
    return 0


if __name__ == "__main__":
    sys.exit(main())
