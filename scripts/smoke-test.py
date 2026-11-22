#!/usr/bin/env python3
"""Standalone smoke check for Inkling 2 Mini on Ciru Strix Halo (gfx1151)."""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


def stream_chat(base_url: str, body: dict, timeout: int = 180) -> dict:
    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    first_t = None
    usage = None
    content = []
    reasoning = []
    tool_name = ""
    tool_args = ""
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if first_t is None and (
                    delta.get("content")
                    or delta.get("reasoning")
                    or delta.get("reasoning_content")
                    or delta.get("tool_calls")
                ):
                    first_t = time.perf_counter()
                if delta.get("content"):
                    content.append(delta["content"])
                if delta.get("reasoning"):
                    reasoning.append(delta["reasoning"])
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                for tc in delta.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        tool_name += fn["name"]
                    if fn.get("arguments"):
                        tool_args += fn["arguments"]
    t1 = time.perf_counter()
    pt = usage["prompt_tokens"] if usage else 0
    ct = usage["completion_tokens"] if usage else 0
    ttft = (first_t - t0) if first_t is not None else (t1 - t0)
    dec_t = max(t1 - (first_t or t0), 1e-6)
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "ttft_s": ttft,
        "wall_s": t1 - t0,
        "pp_tok_s": pt / max(ttft, 1e-6),
        "dec_tok_s": max(ct - 1, 0) / dec_t,
        "content": "".join(content),
        "reasoning_chars": len("".join(reasoning)),
        "tool_name": tool_name,
        "tool_args": tool_args,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:18081/v1")
    ap.add_argument("--model", default="inkling-2-mini")
    ap.add_argument("--check-tools", action="store_true")
    args = ap.parse_args()

    with urllib.request.urlopen(f"{args.base_url.rstrip('/')}/models", timeout=15) as r:
        models = json.loads(r.read().decode("utf-8"))
    ids = [m["id"] for m in models.get("data", [])]
    if args.model not in ids:
        print(f"ERROR: model {args.model!r} not in {ids}", file=sys.stderr)
        return 1
    print(f"Models endpoint OK: {ids}")

    # Warmup
    stream_chat(
        args.base_url,
        {
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "temperature": 0,
            "max_tokens": 16,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )

    checks = [
        (
            "json",
            "Return strictly valid JSON with keys ok (boolean true), items (array of 3 strings: alpha, beta, gamma), and total (integer 3). No markdown.",
            256,
        ),
        (
            "math",
            "Compute the number of integer solutions (x, y) to x^2 + y^2 = 65^2 with x > 0 and y > 0. Give the final integer.",
            512,
        ),
        (
            "pp2k",
            (
                "Technical specification excerpt detailing memory bandwidth, cache tiling, "
                "wave32 scheduling, and speculative verification on RDNA 3.5 integrated graphics. "
            )
            * 90
            + "\nSummarize the core theme in one sentence.",
            48,
        ),
    ]
    for name, prompt, max_tokens in checks:
        res = stream_chat(
            args.base_url,
            {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        print(
            f"[{name}] pt={res['prompt_tokens']} ct={res['completion_tokens']} "
            f"ttft={res['ttft_s']:.3f}s pp={res['pp_tok_s']:.1f} tok/s "
            f"tg={res['dec_tok_s']:.2f} tok/s reasoning_chars={res['reasoning_chars']}"
        )
        if name == "json":
            parsed = json.loads(res["content"].strip())
            assert parsed.get("ok") is True and parsed.get("total") == 3, parsed
            print("  JSON check PASS")
        elif name == "math":
            assert "14" in res["content"] or "28" in res["content"], res["content"][-120:]
            print(f"  Math excerpt: {res['content'].strip()[-80:]!r}")

    if args.check_tools:
        tr = stream_chat(
            args.base_url,
            {
                "model": args.model,
                "messages": [{"role": "user", "content": "What is the weather in Austin, TX?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get current weather for a city",
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                                "required": ["location"],
                            },
                        },
                    }
                ],
                "temperature": 0,
                "max_tokens": 128,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        print(
            f"[tool] name={tr['tool_name']!r} args={tr['tool_args']!r} "
            f"ct={tr['completion_tokens']} tg={tr['dec_tok_s']:.2f} tok/s"
        )
        assert tr["tool_name"] == "get_weather", tr
        print("  Tool-call check PASS")

    print("All smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
