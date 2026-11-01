---
layout: post
title: "The best models to run on an Apple M5 Max 128GB Macbook Pro"
date: 2026-10-31 12:00:00 -0700
categories: [AI]
tags: [llm, local-inference, apple-silicon, mlx, qwen, deepseek]
description: "Three models on my 128GB M5 Max: Qwen3.8-Flash-Next for speed, Qwen3.8-27B at 8-bit for a smaller memory footprint, and DeepSeek-V4.1-Flash streamed from SSD. Local speed tests included."
---

My [dual RTX Pro 6000 workstation]({% post_url 2025-12-25-dual-rtx-pro-6000-llm-guide %}) is still where I run the bigger models, but I also want a useful coding model on the laptop itself. On my M5 Max MacBook Pro with 128 GB of unified memory, my first pick from the models I've installed is Qwen3.8-Flash-Next through MTPLX. It generates around 72–86 tokens/sec in my short-prompt tests. There's also a smaller 8-bit Qwen worth keeping, and a much larger DeepSeek that runs with substantial help from the SSD.

My configuration has an 18-core CPU, 40-core GPU and 128 GB of unified memory, with [614 GB/s of memory bandwidth according to Apple](https://support.apple.com/en-us/126318). That's the rated bandwidth, not a measurement from these tests. It matters because single-user LLM generation often spends much of its time reading weights from memory. Once a model needs SSD streaming, the SSD and cache behavior matter too; 614 GB/s doesn't describe that path.

## The two Qwens I'd start with

[Qwen3.8-Flash-Next Optimized Speed](https://huggingface.co/Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed) is a 125B mixture-of-experts model with 6B active parameters per token. This pack uses mostly 4-bit weights, with 8-bit attention projections. The download is about 115 GB, but its n-gram table streams from SSD instead of staying resident. The model card reports an 87 GiB serving peak, which makes this a reasonable fit for 128 GB, provided you leave room for context and macOS.

[Qwen3.8-27B Optimized Quality](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality) is the less memory-hungry option: a 29.4 GB download, 8-bit weight matrices, and a model-card peak of about 33 GB. It leaves much more room for the rest of my applications. “Quality” here means less quantization loss relative to the original 27B checkpoint, not that I've established it answers better than Flash-Next.

Both run through [MTPLX](https://github.com/youssofal/mtplx), an Apple Silicon engine built on MLX with native multi-token prediction. It drafts several tokens and verifies them together, the same general idea as [speculative decoding]({% post_url 2025-10-23-speculative-decoding %}). Both packs advertise a 262,144-token context window; that isn't a promise that filling it will be fast.

## Local speed tests

These are results from my local server logs, not the model cards. The installed MTPLX version is 2.12.2. Each Qwen request hit a 2,048-token output cap, including reasoning tokens, on three similar tasks:

| Task | Flash-Next 4-bit | 27B 8-bit |
| :--- | ---: | ---: |
| Explain B-tree indexes | 85.8 tok/s | 43.5 tok/s |
| Implement a Python LRU cache with TTL | 83.1 tok/s | 45.3 tok/s |
| Design an offline-first notes app | 71.6 tok/s | 38.1 tok/s |

Those are the engine's reported generation rates. Including prompt processing and request overhead, the ranges were 71–85 and 38–45 tok/s respectively. Prompts were short, 48–93 tokens depending on the model and task. This is a practical spot check, not a controlled quality evaluation or a long-context benchmark.

## DeepSeek: possible, but bring patience

I also have DeepSeek-V4.1-Flash Q4 running through [antirez's ds4](https://github.com/antirez/ds4). The GGUF is about 483 GiB. It does **not** fit in 128 GB: `ds4` uses SSD streaming and an automatically sized expert cache.

Separate 128-output-token tests measured 8.3 tok/s after a 39-token prompt and 7.1 tok/s after a 1,844-token prompt. Time to first token was roughly 6–7 seconds and 35 seconds respectively. A separate retrieval check correctly recalled a code buried in a 54,045-token prompt in 193 seconds total, with the server configured for 128K context. That doesn't establish reliable retrieval across the full window.

I'm glad it runs, but Flash-Next is the one I'd start with for an interactive coding session.

## Run it

Get the [MTPLX Mac app or CLI](https://mtplx.com), or build [ds4 from source](https://github.com/antirez/ds4) using its [Apple Silicon installation guide](https://github.com/antirez/ds4/blob/main/docs/METAL.md). The Qwen model downloads are linked above; DeepSeek's [model download guide](https://github.com/antirez/ds4/blob/main/docs/MODELS.md#deepseek-v41-flash) lists the supported GGUFs.

These are simplified foreground versions of my launch scripts. Run one model at a time and use Ctrl+C to stop it.

<details markdown="1">
<summary markdown="span">Start either Qwen with MTPLX (click to expand)</summary>

Install once, using Python 3.11 or newer:

```bash
mkdir -p ~/models
python3 -m venv ~/models/.mtplx-venv
~/models/.mtplx-venv/bin/pip install mtplx==2.12.2
```

Save as `~/models/start-qwen.sh`. MTPLX downloads the selected pack on first use.

```bash
#!/bin/bash
set -euo pipefail

case "${1:-flash}" in
  flash) MODEL="Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed" ;;
  quality) MODEL="Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality" ;;
  *) echo "Usage: $0 [flash|quality]" >&2; exit 1 ;;
esac

exec "$HOME/models/.mtplx-venv/bin/mtplx" serve \
  --model "$MODEL" --host 127.0.0.1 --port 8000
```

Run `bash ~/models/start-qwen.sh flash` or `bash ~/models/start-qwen.sh quality`.

</details>

<details markdown="1">
<summary markdown="span">Start DeepSeek-V4.1-Flash Q4 with ds4 (click to expand)</summary>

Install Apple's Command Line Tools and the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) first. Build and download once; allow space for the download parts as well as the assembled 483 GiB GGUF:

```bash
mkdir -p ~/models
cd ~/models
git clone https://github.com/antirez/ds4.git
cd ds4
# Revision used for my local tests:
git checkout c97964b8aa77034a8e17afb10ca7a686c40ab511
make -j12
./download_model.sh ds41f-q4
```

Save as `~/models/start-deepseek.sh`. This keeps my 128K context allocation and 8 GiB disk KV-cache budget; it does not load the entire model into RAM.

```bash
#!/bin/bash
set -euo pipefail
ROOT="$HOME/models"
KV="$ROOT/deepseek-v4.1-flash/cache/kv"
mkdir -p "$KV"
chmod 700 "$KV"

exec "$ROOT/ds4/ds4-server" --chdir "$ROOT/ds4" \
  -m "$ROOT/ds4/gguf/DeepSeek-V4.1-Flash-Q4.gguf" \
  --ssd-streaming --ctx 131072 \
  --host 127.0.0.1 --port 8001 \
  --kv-disk-dir "$KV" --kv-disk-space-mb 8192
```

Run `bash ~/models/start-deepseek.sh`. The KV cache contains conversation data, so keep it private. Stop the server before deleting it.

</details>

The OpenAI-compatible API is at `http://127.0.0.1:8000/v1` for MTPLX or `http://127.0.0.1:8001/v1` for ds4, so I can connect either to [my existing agent setup]({% post_url 2026-08-27-my-personal-llm-toolchain %}). I'd use Flash-Next when the laptop is mainly serving the model, and 27B when I want to keep working without giving most of the machine's memory to inference.
