"""Hit-path accuracy: a cached GSM8K question must still answer correctly.

The first wave (one request per DP rank) misses; the second wave hits radix,
folds the replay window and runs DSpark verify against the rebuilt draft ring.
Encoder replay is an approximation (the replay rows' windows are floored at
the replay start), so this checks correctness, not bitwise equality.
"""

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

BASE = "http://127.0.0.1:8001"
DATA = "/home/w00889861/dsv4/dataset/test.jsonl"
RANKS = int(sys.argv[1]) if len(sys.argv) > 1 else 16
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 512


def load_question():
    with open(DATA) as f:
        ex = json.loads(f.readline())
    answer = ex["answer"].split("####")[-1].strip()
    return ex["question"].strip(), answer


def ask(question):
    r = requests.post(
        BASE + "/v1/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": question}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "chat_template_kwargs": {"thinking": False},
        },
        timeout=1800,
    )
    r.raise_for_status()
    body = r.json()
    text = body["choices"][0]["message"].get("content") or ""
    usage = body.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return text, details.get("cached_tokens")


def answer_of(text):
    nums = re.findall(r"-?\d[\d,]*", text)
    return nums[-1].replace(",", "") if nums else None


def main():
    question, expected = load_question()
    # A short question caches no page-aligned prefix, so pad it past a few
    # pages; the radix hit (and fold) then happens before the question.
    filler = (
        "The following is a long neutral passage used only as context. "
        "It describes rivers, bridges, and the slow passage of seasons. "
    ) * 60
    prompt = filler + "\n" + question
    requests.post(BASE + "/flush_cache", json={}, timeout=60)

    for wave in (1, 2):
        with ThreadPoolExecutor(RANKS) as pool:
            outs = list(pool.map(ask, [prompt] * RANKS))
        digits = [answer_of(t) for t, _ in outs]
        cached = [c for _, c in outs]
        ok = sum(d == expected for d in digits)
        print(
            f"wave{wave}: {ok}/{RANKS} correct, expected={expected}, "
            f"answers={digits}, cached={cached[:4]}"
        )
        if wave == 1:
            first = outs
        else:
            bad = [
                (t[-120:], d)
                for (t, _), d in zip(outs, digits)
                if d != expected
            ]
            if bad:
                for tail, d in bad[:4]:
                    print(f"  WRONG answer={d}: ...{tail!r}")
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
