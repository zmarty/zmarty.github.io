---
layout: post
title: "Running DeepSeek-V4-Flash at 700 tokens/s on 2x RTX Pro 6000"
date: 2026-06-20 09:00:00 -0700
categories: [AI]
tags: [llm, vllm, local-inference, deepseek]
description: "Run DeepSeek-V4-Flash on a 2x RTX Pro 6000 (96GB each) workstation using the voipmonitor/vllm:lucifer Docker image, a Blackwell-targeted vLLM fork with sm_120 kernels, FP8 KV cache, and MTP speculative decoding."
---

On a single 2-GPU workstation (2x RTX Pro 6000, TP2), [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) serves at about **210 tokens/sec** on a single stream and scales to roughly **700 tokens/sec aggregate** across 10 concurrent requests, with **sub-second time-to-first-token** all the way to 10 streams and prefill saturating near **10,000 tokens/sec**. This is the original DeepSeek-V4-Flash weights, not a quantized version. The full benchmark is at the end of this article.

DeepSeek-V4-Flash is an open weights (MIT-licensed) model from the Chinese lab [DeepSeek](https://www.deepseek.com/), released in April 2026. It is a Mixture-of-Experts model with 284B total parameters but only 13B active per token, and it supports a 1M-token context. Despite the small active footprint, it punches well above its weight class: it scores around 40 on the [Artificial Analysis Intelligence Index](https://artificialanalysis.ai/models/deepseek-v4-flash), versus a median of about 24 for comparable open weights models, which puts it near the top of its size class for reasoning and coding.

I got these numbers with [`voipmonitor/vllm:lucifer`](https://hub.docker.com/r/voipmonitor/vllm/tags), a Blackwell-targeted fork of vLLM built from the [`lucifer` branch of `local-inference-lab/vllm`](https://github.com/local-inference-lab/vllm/tree/lucifer). The image ships a CUDA 13.2 / PyTorch 2.12 stack with pinned FlashInfer, DeepGEMM, CUTLASS, and a patched NCCL, all compiled for `sm_120a`, the compute capability of the RTX Pro 6000 (Blackwell).

The steps below are based on the original reference notes in [local-inference-lab/rtx6kpro](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/ds4-flash-v4.md), specifically its Standard Lucifer Cutlass path. That page also documents a faster B12X build and full benchmark numbers if you want to go deeper.

## What the fork changes for the RTX Pro 6000

- **sm_120 kernels.** `SPARSE_MLA_SM120` attention and a FlashInfer/CUTLASS MoE path (`flashinfer_cutlass`) are compiled for Blackwell rather than running from generic PTX.
- **FP8 where it matters.** FP8 KV cache and DeepGEMM UE8M0 FP8 GEMMs fit the 262K context and improve throughput within 2x 96 GB.
- **MTP speculative decoding.** DeepSeek's multi-token-prediction head gives a single-stream decode speedup.
- **PCIe-aware TP2.** NCCL is tuned (`NCCL_P2P_LEVEL=SYS`) for the no-NVLink, PCIe-host-bridge topology of these workstation cards.
- **Cached JIT/autotune.** FlashInfer autotune, DeepGEMM warmup, and CUDA-graph capture are written to a `/cache` volume, so the ~5-minute first-run warmup happens once.

## Hardware and assumptions

```text
GPU: 2x NVIDIA RTX Pro 6000 (96 GB VRAM each, sm_120, no NVLink)
NVIDIA driver: recent, CUDA 13 capable
Docker: with the NVIDIA container runtime
Disk: ~160 GB free
```

The server runs tensor-parallel across both GPUs (TP2).

## 1. Pull the runtime image

```bash
docker pull voipmonitor/vllm:lucifer
```

## 2. Download the model (~160 GB)

```bash
pip install -U "huggingface_hub[cli]"
hf download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir /models/DeepSeek-V4-Flash
```

## 3. Start the server

This runs in the foreground. **Ctrl+C** stops and removes the container. Port 8000 is exposed on the host.

```bash
docker run --rm -it --init --name ds4 \
  --gpus all --runtime nvidia --ipc host --shm-size 32g --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /models:/models \
  -v /root/.cache/lucifer:/cache \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e CUTE_DSL_ARCH=sm_120a \
  -e HF_HUB_OFFLINE=1 \
  -e NCCL_P2P_LEVEL=SYS -e NCCL_PROTO=LL,LL128,Simple -e NCCL_IB_DISABLE=1 \
  voipmonitor/vllm:lucifer \
  /bin/bash -lc 'unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_CACHE_DIR; \
  exec vllm serve /models/DeepSeek-V4-Flash \
    --served-model-name DeepSeek-V4-Flash --trust-remote-code \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 2 \
    --kv-cache-dtype fp8 --block-size 256 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 262144 --max-num-seqs 64 --max-num-batched-tokens 8192 \
    --max-cudagraph-capture-size 192 \
    --compilation-config="{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"]}" \
    --async-scheduling --no-scheduler-reserve-full-isl \
    --enable-chunked-prefill --enable-prefix-caching --enable-flashinfer-autotune \
    --attention-backend SPARSE_MLA_SM120 \
    --kernel-config.moe_backend flashinfer_cutlass \
    --tokenizer-mode deepseek_v4 \
    --reasoning-parser deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
    --default-chat-template-kwargs.thinking=true \
    --default-chat-template-kwargs.reasoning_effort=high \
    --speculative-config.method mtp \
    --speculative-config.num_speculative_tokens 2 \
    --speculative-config.draft_sample_method probabilistic'
```

The first launch JIT-compiles kernels (~5-6 min) into `/cache`. Reuse the same `/cache` volume so restarts are fast.

## 4. Test

```bash
curl -s http://localhost:8000/v1/models | jq
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V4-Flash","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}' | jq
```

The endpoint is OpenAI-compatible at `http://<host>:8000/v1`.

## Benchmark on 2x RTX Pro 6000 (TP2)

I benchmarked the live server through the OpenAI streaming API, not the raw engine. Each request uses a unique ~889-token prompt to defeat prefix caching, forces 256 generated tokens with `ignore_eos`, runs with MTP speculative decoding on, FP8 KV cache, and thinking off. Streaming lets me separate time-to-first-token (prefill) from the decode rate. I warm up every concurrency level first so the CUDA-graph capture cost is not counted in the measured numbers, then sweep 1 to 10 concurrent streams.

The chart below plots aggregate throughput as concurrency rises. Prefill and decode use separate Y axes because their scales differ by more than 10x.

<svg viewBox="0 0 760 430" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Aggregate throughput vs concurrent streams" style="width:100%;height:auto;font-family:sans-serif;font-size:13px;">
  <!-- horizontal gridlines + shared tick levels -->
  <g stroke="#9993" stroke-width="1">
    <line x1="70" y1="30" x2="700" y2="30"/>
    <line x1="70" y1="112.5" x2="700" y2="112.5"/>
    <line x1="70" y1="195" x2="700" y2="195"/>
    <line x1="70" y1="277.5" x2="700" y2="277.5"/>
    <line x1="70" y1="360" x2="700" y2="360"/>
  </g>
  <!-- axes -->
  <g stroke="currentColor" stroke-width="1.5">
    <line x1="70" y1="30" x2="70" y2="360"/>
    <line x1="700" y1="30" x2="700" y2="360"/>
    <line x1="70" y1="360" x2="700" y2="360"/>
  </g>
  <!-- left axis labels (prefill, 0-12000) -->
  <g fill="#2563eb" text-anchor="end">
    <text x="62" y="364">0</text>
    <text x="62" y="281.5">3k</text>
    <text x="62" y="199">6k</text>
    <text x="62" y="116.5">9k</text>
    <text x="62" y="34">12k</text>
  </g>
  <!-- right axis labels (decode, 0-800) -->
  <g fill="#ea580c" text-anchor="start">
    <text x="708" y="364">0</text>
    <text x="708" y="281.5">200</text>
    <text x="708" y="199">400</text>
    <text x="708" y="116.5">600</text>
    <text x="708" y="34">800</text>
  </g>
  <!-- x axis labels (streams 1-10) -->
  <g fill="currentColor" text-anchor="middle">
    <text x="70" y="380">1</text>
    <text x="140" y="380">2</text>
    <text x="210" y="380">3</text>
    <text x="280" y="380">4</text>
    <text x="350" y="380">5</text>
    <text x="420" y="380">6</text>
    <text x="490" y="380">7</text>
    <text x="560" y="380">8</text>
    <text x="630" y="380">9</text>
    <text x="700" y="380">10</text>
  </g>
  <!-- axis titles -->
  <text x="385" y="406" fill="currentColor" text-anchor="middle">Concurrent streams</text>
  <text x="18" y="195" fill="#2563eb" text-anchor="middle" transform="rotate(-90 18 195)">Prefill aggregate (tok/s)</text>
  <text x="748" y="195" fill="#ea580c" text-anchor="middle" transform="rotate(90 748 195)">Decode aggregate (tok/s)</text>
  <!-- prefill line (left axis) -->
  <polyline fill="none" stroke="#2563eb" stroke-width="2.5"
    points="70,127.1 140,92.9 210,116.6 280,106.3 350,79.1 420,86.4 490,80.9 560,75.2 630,73.7 700,72.2"/>
  <g fill="#2563eb">
    <circle cx="70" cy="127.1" r="3.5"/><circle cx="140" cy="92.9" r="3.5"/><circle cx="210" cy="116.6" r="3.5"/>
    <circle cx="280" cy="106.3" r="3.5"/><circle cx="350" cy="79.1" r="3.5"/><circle cx="420" cy="86.4" r="3.5"/>
    <circle cx="490" cy="80.9" r="3.5"/><circle cx="560" cy="75.2" r="3.5"/><circle cx="630" cy="73.7" r="3.5"/>
    <circle cx="700" cy="72.2" r="3.5"/>
  </g>
  <!-- decode line (right axis) -->
  <polyline fill="none" stroke="#ea580c" stroke-width="2.5"
    points="70,273.4 140,230.7 210,205.4 280,200.1 350,168.1 420,120.5 490,126.4 560,84.2 630,97.9 700,72.4"/>
  <g fill="#ea580c">
    <circle cx="70" cy="273.4" r="3.5"/><circle cx="140" cy="230.7" r="3.5"/><circle cx="210" cy="205.4" r="3.5"/>
    <circle cx="280" cy="200.1" r="3.5"/><circle cx="350" cy="168.1" r="3.5"/><circle cx="420" cy="120.5" r="3.5"/>
    <circle cx="490" cy="126.4" r="3.5"/><circle cx="560" cy="84.2" r="3.5"/><circle cx="630" cy="97.9" r="3.5"/>
    <circle cx="700" cy="72.4" r="3.5"/>
  </g>
  <!-- legend -->
  <g transform="translate(110,18)">
    <line x1="0" y1="0" x2="24" y2="0" stroke="#2563eb" stroke-width="2.5"/>
    <text x="30" y="4" fill="currentColor">Prefill aggregate</text>
    <line x1="180" y1="0" x2="204" y2="0" stroke="#ea580c" stroke-width="2.5"/>
    <text x="210" y="4" fill="currentColor">Decode aggregate</text>
  </g>
</svg>

| Streams | TTFT (s) | Prefill /stream (tok/s) | Prefill aggregate (tok/s) | Decode /stream (tok/s) | Decode aggregate (tok/s) |
|---:|---:|---:|---:|---:|---:|
| 1  | 0.10 | 8,471 | 8,471 | 209.9 | 209.9 |
| 2  | 0.19 | 4,862 | 9,713 | 167.5 | 313.5 |
| 3  | 0.21 | 5,162 | 8,852 | 139.5 | 374.7 |
| 4  | 0.27 | 4,264 | 9,224 | 105.9 | 387.6 |
| 5  | 0.37 | 3,189 | 10,213 | 108.4 | 465.2 |
| 6  | 0.42 | 3,026 | 9,948 | 113.2 | 580.7 |
| 7  | 0.49 | 2,742 | 10,149 | 95.9 | 566.4 |
| 8  | 0.62 | 2,092 | 10,357 | 104.0 | 668.7 |
| 9  | 0.70 | 1,850 | 10,411 | 87.5 | 635.3 |
| 10 | 0.71 | 2,035 | 10,465 | 87.7 | 697.3 |

Prompt ~889 tokens/request, 256 generated tokens/request, thinking off, `ignore_eos`.

### Takeaways

- Prefill saturates the two GPUs around **~10,000 tok/s aggregate** from about 5 concurrent streams onward, and TTFT stays under a second through 10 streams.
- Decode scales sub-linearly: **~210 tok/s single-stream up to ~697 tok/s aggregate at 10 streams** (about 3.3x). Per-stream decode degrades gracefully from 210 to ~88 tok/s as load increases.
- Single-stream decode (210 tok/s) lines up with the reference Lucifer TP2 MTP-probabilistic figure (~207 tok/s), so the box is performing as expected.

These numbers are end-to-end over the streaming API, so they are slightly conservative compared to the raw engine, generation-only benchmarks in the [reference notes](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/ds4-flash-v4.md). Those go up to 64 concurrent streams and include the faster B12X build, so the headline aggregate figures there are higher and not directly comparable to this 1-to-10 stream, Lucifer-image run. Decode aggregate also has mild run-to-run noise (about 5%).

<details markdown="1">
<summary markdown="span">Benchmark script `bench_ds4.py` (click to expand)</summary>

```python
import json, time, threading, random, string, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "DeepSeek-V4-Flash"
GEN_TOKENS = 256          # forced generation length per stream
PROMPT_WORDS = 700        # ~900-1000 prompt tokens, unique per request
WORDS = ["alpha","river","copper","lunar","quartz","meadow","cipher","tangent","ember","willow",
         "harbor","nimbus","pixel","cobalt","syntax","fathom","zephyr","granite","oracle","velvet"]

def make_prompt():
    nonce = "".join(random.choices(string.ascii_lowercase, k=12))
    body = " ".join(random.choice(WORDS) for _ in range(PROMPT_WORDS))
    return f"[{nonce}] Read this token list then write a long neutral description. {body}"

def one_request():
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": make_prompt()}],
        "max_tokens": GEN_TOKENS,
        "temperature": 0.7,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = None; t_last = None
    prompt_tokens = gen_tokens = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            try:
                d = json.loads(p)
            except Exception:
                continue
            ch = d.get("choices") or []
            if ch:
                delta = ch[0].get("delta", {})
                txt = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")
                if txt:
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
            u = d.get("usage")
            if u:
                prompt_tokens = u.get("prompt_tokens", prompt_tokens)
                gen_tokens = u.get("completion_tokens", gen_tokens)
    if t_first is None:
        t_first = t_last = time.perf_counter()
    return {
        "ttft": t_first - t0,
        "decode_time": max(t_last - t_first, 1e-6),
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "t_first": t_first,
        "t_last": t_last,
    }

def run_level(n):
    with ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(lambda _: one_request(), range(n)))
    mean_ttft = sum(r["ttft"] for r in results) / n
    mean_prefill = sum(r["prompt_tokens"] / r["ttft"] for r in results) / n
    mean_decode = sum(r["gen_tokens"] / r["decode_time"] for r in results) / n
    win = max(r["t_last"] for r in results) - min(r["t_first"] for r in results)
    win = max(win, 1e-6)
    agg_decode = sum(r["gen_tokens"] for r in results) / win
    pf_win = max(r["t_first"] for r in results) - min(r["t_first"] - r["ttft"] for r in results)
    agg_prefill = sum(r["prompt_tokens"] for r in results) / max(pf_win, 1e-6)
    return mean_ttft, mean_prefill, agg_prefill, mean_decode, agg_decode, results[0]["prompt_tokens"]

print("Warming up all concurrency levels 1..10 (capturing CUDA graphs)...", flush=True)
for n in range(1, 11):
    run_level(n)
print(f"{'N':>2} | {'TTFT(s)':>8} | {'prefill/str':>11} | {'prefill agg':>11} | {'decode/str':>10} | {'decode agg':>10}")
print("-" * 74)
for n in range(1, 11):
    mt, mp, ap, md, ad, ptok = run_level(n)
    print(f"{n:>2} | {mt:8.2f} | {mp:9.0f}   | {ap:9.0f}   | {md:8.1f}   | {ad:8.1f}")
print(f"\n(prompt ~{ptok} tokens/req, {GEN_TOKENS} generated tokens/req, thinking off, ignore_eos)")
```

</details>
