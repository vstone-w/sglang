"""Probe the NPU decoder tail row-bucket graph.

Two waves of one fresh >128-token request per DP rank: wave 1 captures the
bucket graph, wave 2 replays it with refreshed metadata. Every answer must be
"Paris".
"""

import concurrent.futures
import sys

import requests

BASE = "http://127.0.0.1:8001"
RANKS = int(sys.argv[1]) if len(sys.argv) > 1 else 16
FILLER_REPS = int(__import__("os").environ.get("M2_FILLER_REPS", "40"))
FILLER = (
    "The river crossed the bridge while the engineer checked the cache. "
) * FILLER_REPS


def one(i: int, wave: int):
    prompt = f"Probe {wave}-{i} unique {i * 7919}: {FILLER}\nAnswer with one word. The capital of France is"
    r = requests.post(
        BASE + "/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": 8,
                "temperature": 0.0,
                "top_k": 1,
            },
        },
        timeout=1800,
    )
    r.raise_for_status()
    body = r.json()
    meta = body.get("meta_info") or {}
    return (
        body.get("text", ""),
        meta.get("prompt_tokens"),
        meta.get("cached_tokens"),
        meta.get("finish_reason"),
    )


def wave(n: int):
    with concurrent.futures.ThreadPoolExecutor(RANKS) as pool:
        return list(pool.map(lambda i: one(i, n), range(RANKS)))


def main() -> int:
    requests.get(BASE + "/flush_cache", timeout=120).raise_for_status()
    rc = 0
    for n in (1, 2):
        outs = wave(n)
        ok = sum("paris" in t.lower() for t, *_ in outs)
        print(
            f"wave{n}: {ok}/{RANKS} paris; "
            f"prompt_tokens={[p for _, p, _, _ in outs]}; "
            f"cached={[c for _, _, c, _ in outs]}"
        )
        for i, (text, _, _, finish) in enumerate(outs):
            print(f"  wave{n}[{i}] finish={finish} text={text!r}")
        if ok != RANKS:
            rc = 1
    info = requests.get(BASE + "/server_info", timeout=60).json()
    accept = [
        s.get("avg_spec_accept_length")
        for s in (info.get("internal_states") or [])
        if s.get("avg_spec_accept_length") is not None
    ]
    print(f"accept_length: {accept}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
