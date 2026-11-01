#!/usr/bin/env python3
"""Capture actual local OpenAI-compatible SSE delivery times (stdlib only)."""
import argparse
import datetime
import json
from pathlib import Path
import time
import urllib.request

PROMPT = (
    "Write a Python function that merges overlapping intervals. Include type hints, "
    "comments, a short explanation, and three example calls. Return the code first."
)


def request(base_url, model, prompt, cap, stream):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": cap,
        "stream": stream,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "thinking": {"type": "disabled"},
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    return urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--quant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with urllib.request.urlopen(args.base_url.rstrip("/") + "/models", timeout=10) as r:
        model = json.load(r)["data"][0]["id"]
    # Different prompt: initialize kernels without caching the measured prompt.
    with urllib.request.urlopen(request(args.base_url, model, "Reply with OK.", 8, False), timeout=1200) as r:
        r.read()
    result = {
        "label": args.label, "quant": args.quant, "model": model,
        "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hardware": "Apple M5 Max, 40-core GPU, 128 GB unified memory",
        "prompt": PROMPT, "max_tokens": 512, "temperature": 0,
        "thinking": "disabled", "warmup": "Reply with OK. (8-token cap)",
        "timing": "Client-observed SSE chunks; not individual GPU token timestamps",
        "events": [],
    }
    start = time.monotonic()
    with urllib.request.urlopen(request(args.base_url, model, PROMPT, 512, True), timeout=1200) as r:
        for line in r:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            payload = json.loads(data)
            if payload.get("error"):
                raise RuntimeError(payload["error"])
            if payload.get("usage"):
                result["usage"] = payload["usage"]
            for choice in payload.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("reasoning_content") or delta.get("reasoning"):
                    raise RuntimeError("Reasoning was emitted despite requesting thinking off")
                if delta.get("content"):
                    result["events"].append({"t": round(time.monotonic() - start, 6), "text": delta["content"]})
                if choice.get("finish_reason"):
                    result["finish_reason"] = choice["finish_reason"]
    result["total_s"] = round(time.monotonic() - start, 6)
    events = result["events"]
    if len(events) < 2 or not result.get("usage", {}).get("completion_tokens"):
        raise RuntimeError("Missing streamed text or token usage")
    result["time_to_first_text_s"] = events[0]["t"]
    tokens = result["usage"]["completion_tokens"]
    result["streamed_tokens_per_s"] = round((tokens - 1) / (events[-1]["t"] - events[0]["t"]), 2)
    result["end_to_end_tokens_per_s"] = round(tokens / result["total_s"], 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ["events", "prompt"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
