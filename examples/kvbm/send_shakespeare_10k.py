#!/usr/bin/env python3
import argparse
import json
import sys
import urllib.error
import urllib.request


QUOTE = (
    "To be, or not to be: that is the question: "
    "Whether 'tis nobler in the mind to suffer "
    "The slings and arrows of outrageous fortune, "
    "Or to take arms against a sea of troubles, "
    "And by opposing end them."
)


def build_prompt(target_tokens: int) -> str:
    # Rough English estimate: 1 token ~= 0.75 words.
    target_words = max(1, int(target_tokens * 0.75))
    quote_words = QUOTE.split()
    repeats = max(1, (target_words + len(quote_words) - 1) // len(quote_words))
    body = " ".join([QUOTE] * repeats)
    prompt = (
        "Repeat and analyze the following Shakespeare passage. "
        "Then provide a short summary in plain English.\n\n"
        f"{body}"
    )
    return prompt


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def clear_cpu_cache(mgmt_url: str) -> None:
    result = post_json(f"{mgmt_url.rstrip('/')}/v1/cache/clear", {"pool": "cpu"})
    print("CPU cache clear:", json.dumps(result))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send a roughly 10k-token Shakespeare prompt and clear CPU cache after."
    )
    parser.add_argument("--url", default="http://localhost:19000", help="vLLM base URL")
    parser.add_argument(
        "--mgmt",
        default="http://localhost:19881",
        help="KVBM management base URL",
    )
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="model name")
    parser.add_argument(
        "--target-isl",
        type=int,
        default=10_000,
        help="approximate target input sequence length in tokens",
    )
    parser.add_argument("--max-tokens", type=int, default=128, help="max output tokens")
    args = parser.parse_args()

    prompt = build_prompt(args.target_isl)
    prompt_words = len(prompt.split())
    prompt_chars = len(prompt)
    print(
        f"Sending prompt: ~{args.target_isl} target tokens, "
        f"{prompt_words} words, {prompt_chars} chars",
        file=sys.stderr,
    )

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": args.max_tokens,
    }

    try:
        response = post_json(f"{args.url.rstrip('/')}/v1/chat/completions", payload)
        print(json.dumps(response, indent=2))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"Request failed: HTTP {exc.code}\n{body}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            clear_cpu_cache(args.mgmt)
        except Exception as exc:  # best-effort cleanup
            print(f"CPU cache clear failed: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
