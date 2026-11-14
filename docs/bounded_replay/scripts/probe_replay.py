"""Functional probe for a DSV4.1 bounded replay server.

Records a fixed request battery so a baseline (replay off) and a replay run can
be compared afterwards:

- same prompt twice: the second request's ``cached_tokens`` exposes the radix
  reprefill contract (decoder replay trims the last 128 matched tokens);
- prompt-logprob request (``logprob_start_len=0``) on a long prompt: forces
  whole-extend and must be bit-identical between feature off/on;
- greedy generations on short prompts around the 128-token window boundary;
- one long-prompt greedy generation, where bounded replay may differ.

Usage: probe_replay.py <base_url> <out.json>
"""

import json
import sys

import requests

BASE_URL = sys.argv[1].rstrip("/")
OUT = sys.argv[2]

SENTENCE = "The quick brown fox jumps over the lazy dog while the engineer checks the cache. "


def repeat_to(target_words: int) -> str:
    words = SENTENCE.split()
    reps = max(1, target_words // len(words))
    return " ".join(words * reps) + " Explain why 17 times 23 equals 391."


def post(path: str, payload: dict) -> dict:
    response = requests.post(BASE_URL + path, json=payload, timeout=1800)
    response.raise_for_status()
    return response.json()


def logprob_probe(prompt: str) -> dict:
    data = post(
        "/generate",
        {
            "text": prompt,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
            "return_logprob": True,
            "logprob_start_len": 0,
            "top_logprobs_num": 0,
            "stream": False,
        },
    )
    meta = data["meta_info"]
    return {
        "prompt_tokens": meta.get("prompt_tokens"),
        "input_token_logprobs": meta.get("input_token_logprobs"),
        "output_text": data.get("text"),
        "finish_reason": meta.get("finish_reason"),
    }


def greedy_probe(prompt: str, max_new_tokens: int = 16) -> dict:
    data = post(
        "/generate",
        {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
                "top_k": 1,
            },
            "stream": False,
        },
    )
    return {
        "prompt_tokens": data["meta_info"].get("prompt_tokens"),
        "cached_tokens": data["meta_info"].get("cached_tokens"),
        "output_text": data.get("text"),
        "finish_reason": data["meta_info"].get("finish_reason"),
    }


def radix_probe(prompt: str, repeats: int = 40) -> dict:
    """DP attention distributes requests round-robin across ranks, so a single
    duplicate pair usually lands on two different radix caches. Repeat enough
    times that every rank sees the prompt twice and report the best hit."""
    samples = [greedy_probe(prompt, max_new_tokens=1) for _ in range(repeats)]
    cached = [sample.get("cached_tokens") or 0 for sample in samples]
    return {
        "prompt_tokens": samples[0].get("prompt_tokens"),
        "max_cached": max(cached),
        "cached_tokens": cached,
    }


def main() -> None:
    requests.get(BASE_URL + "/flush_cache", timeout=120).raise_for_status()
    results = {"base_url": BASE_URL}
    results["radix_reprefill"] = radix_probe(repeat_to(1500))
    # The radix probe's prompt shares its prefix with the logprob prompt; a hit
    # would change the prefill shape between baseline and replay. Flush first.
    requests.get(BASE_URL + "/flush_cache", timeout=120).raise_for_status()
    results["long_prompt_logprob"] = logprob_probe(repeat_to(4000))
    for label, words in (
        ("short_16", 16),
        ("short_100", 100),
        ("short_160", 160),
        ("short_400", 400),
    ):
        results[f"greedy_{label}"] = greedy_probe(repeat_to(words))
    results["greedy_long"] = greedy_probe(repeat_to(2000), max_new_tokens=32)
    with open(OUT, "w") as f:
        json.dump(results, f)
    print(f"wrote {OUT}")
    print(
        "prompt_tokens:",
        {
            key: value.get("prompt_tokens")
            for key, value in results.items()
            if isinstance(value, dict)
        },
    )
    radix = results["radix_reprefill"]
    print(
        "radix max cached_tokens:",
        radix.get("max_cached"),
        "of",
        radix.get("prompt_tokens"),
    )


if __name__ == "__main__":
    main()
