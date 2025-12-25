---
layout: post
title: "Guide on installing and running the best models on a dual RTX Pro 6000 rig with vLLM"
date: 2025-12-25 10:30:00 -0800
categories: [AI]
tags: [llm, vllm]
description: "Step-by-step vLLM stable/nightly install on Ubuntu 24.04 for a dual RTX Pro 6000 (96GB x2), model download workflow, and a fix for tp=2 hangs (IOMMU). Includes tested serve commands for Devstral 123B, GLM-4.5/4.6V, Qwen3 235B, MiniMax-M2, and gpt-oss-120b."
---

## How to install vLLM stable

Prerequisite: [Ubuntu 24.04 and the proper NVIDIA drivers](https://forum.level1techs.com/t/wip-blackwell-rtx-6000-pro-max-q-quickie-setup-guide-on-ubuntu-24-04-lts-25-04/230521)

```console
mkdir vllm
cd vllm
uv venv --python 3.12 --seed
source .venv/bin/activate

uv pip install vllm --torch-backend=auto
```

## How to install vLLM nightly

Prerequisite: [Ubuntu 24.04 and the proper NVIDIA drivers](https://forum.level1techs.com/t/wip-blackwell-rtx-6000-pro-max-q-quickie-setup-guide-on-ubuntu-24-04-lts-25-04/230521)

```console
mkdir vllm-nightly
cd vllm-nightly
uv venv --python 3.12 --seed
source .venv/bin/activate

uv pip install -U vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly
```

## How to download models

```console
mkdir /models
cd /models
uv venv --python 3.12 --seed
source .venv/bin/activate

pip install huggingface_hub

# To download a model after going to /models and running source .venv/bin/activate
mkdir /models/awq
hf download cyankiwi/Devstral-2-123B-Instruct-2512-AWQ-4bit --local-dir /models/awq/cyankiwi-Devstral-2-123B-Instruct-2512-AWQ-4bit
```

## If setting tensor-parallel-size 2 fails in vLLM

I spent two months debugging why I cannot start vLLM with tp 2 (--tensor-parallel-size 2). It was always hanging because the two GPUs could not communicate with each other. I would only see this output in the terminal:

```console
[shm_broadcast.py:501] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
```

Here is my hardware:

```
CPU: AMD Ryzen 9 7950X3D 16-Core Processor
Motherboard: ROG CROSSHAIR X670E HERO
GPU: Dual NVIDIA RTX Pro 6000 (each at 96 GB VRAM)
RAM: 192 GB DDR5 5200
```

And here was the solution:

```console
sudo vi /etc/default/grub
At the end of GRUB_CMDLINE_LINUX_DEFAULT add md_iommu=on iommu=pt like so:
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash md_iommu=on iommu=pt"
sudo update-grub
```

## Devstral 2 123B

Model: [cyankiwi/Devstral-2-123B-Instruct-2512-AWQ-4bit](https://huggingface.co/cyankiwi/Devstral-2-123B-Instruct-2512-AWQ-4bit)

vLLM version tested: vllm-nightly on December 25th, 2025

```console
hf download cyankiwi/Devstral-2-123B-Instruct-2512-AWQ-4bit --local-dir /models/awq/cyankiwi-Devstral-2-123B-Instruct-2512-AWQ-4bit

vllm serve \
    /models/awq/cyankiwi-Devstral-2-123B-Instruct-2512-AWQ-4bit \
    --served-model-name Devstral-2-123B-Instruct-2512-AWQ-4bit \
    --enable-auto-tool-choice \
    --tool-call-parser mistral \
    --max-num-seqs 4 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8000
```

## zai-org/GLM-4.5-Air-FP8

Model: [zai-org/GLM-4.5-Air-FP8](https://huggingface.co/zai-org/GLM-4.5-Air-FP8)

vLLM version tested: 0.12.0

```console
vllm serve \
    /models/original/GLM-4.5-Air-FP8 \
    --served-model-name GLM-4.5-Air-FP8 \
    --max-num-seqs 10 \
    --max-model-len 128000 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --host 0.0.0.0 \
    --port 8000
```

## zai-org/GLM-4.6V-FP8

Model: [zai-org/GLM-4.6V-FP8](https://huggingface.co/zai-org/GLM-4.6V-FP8)

vLLM version tested: 0.12.0

```console
vllm serve \
    /models/original/GLM-4.6V-FP8/ \
    --served-model-name GLM-4.6V-FP8 \
    --tensor-parallel-size 2 \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --max-num-seqs 10 \
    --max-model-len 131072 \
    --mm-encoder-tp-mode data \
    --mm_processor_cache_type shm \
    --allowed-local-media-path / \
    --host 0.0.0.0 \
    --port 8000
```

## QuantTrio/MiniMax-M2-AWQ

Model: [QuantTrio/MiniMax-M2-AWQ](https://huggingface.co/QuantTrio/MiniMax-M2-AWQ)

vLLM version tested: 0.12.0

```console
vllm serve \
    /models/awq/QuantTrio-MiniMax-M2-AWQ \
    --served-model-name MiniMax-M2-AWQ \
    --max-num-seqs 10 \
    --max-model-len 128000 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 1 \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think \
    --host 0.0.0.0 \
    --port 8000
```

## OpenAI gpt-oss-120b

Model: [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)

vLLM version tested: 0.12.0

Note: We are running this on a single GPU

```console
vllm serve \
  /models/original/openai-gpt-oss-120b \
  --served-model-name gpt-oss-120b \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 2 \
  --max_num_seqs 20 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.85 \
  --tool-call-parser openai \
  --reasoning-parser openai_gptoss \
  --enable-auto-tool-choice \
  --host 0.0.0.0 \
  --port 8000
```

## Qwen/Qwen3-235B-A22B

Model: [Qwen/Qwen3-235B-A22B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen3-235B-A22B-GPTQ-Int4)

vLLM version tested: 0.12.0

```console
vllm serve \
    /models/gptq/Qwen-Qwen3-235B-A22B-GPTQ-Int4 \
    --served-model-name Qwen3-235B-A22B-GPTQ-Int4 \
    --reasoning-parser deepseek_r1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --swap-space 16 \
    --max-num-seqs 10 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8000
```

## QuantTrio/Qwen3-235B-A22B-Thinking-2507-AWQ

Model: [QuantTrio/Qwen3-235B-A22B-Thinking-2507-AWQ](https://huggingface.co/QuantTrio/Qwen3-235B-A22B-Thinking-2507-AWQ)

vLLM version tested: 0.12.0

```console
vllm serve \
    /models/awq/QuantTrio-Qwen3-235B-A22B-Thinking-2507-AWQ \
    --served-model-name Qwen3-235B-A22B-Thinking-2507-AWQ \
    --reasoning-parser deepseek_r1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --swap-space 16 \
    --max-num-seqs 10 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8000
```

## nvidia/Qwen3-235B-A22B-NVFP4

Model: [nvidia/Qwen3-235B-A22B-NVFP4](https://huggingface.co/nvidia/Qwen3-235B-A22B-NVFP4)

vLLM version tested: 0.12.0

Note: NVFP4 is slow on vLLM and RTX Pro 6000 (sm120)

```console
hf download nvidia/Qwen3-235B-A22B-NVFP4 --local-dir /models/nvfp4/nvidia/Qwen3-235B-A22B-NVFP4

vllm serve \
    /models/nvfp4/nvidia/Qwen3-235B-A22B-NVFP4 \
    --served-model-name Qwen3-235B-A22B-NVFP4 \
    --reasoning-parser deepseek_r1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --swap-space 16 \
    --max-num-seqs 10 \
    --max-model-len 40960 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8000
```

## QuantTrio/Qwen3-VL-235B-A22B-Thinking-AWQ

Model: [Qwen3-VL-235B-A22B-Thinking-AWQ](https://huggingface.co/QuantTrio/Qwen3-VL-235B-A22B-Thinking-AWQ)

vLLM version tested: 0.12.0

```console
vllm serve \
    /models/awq/QuantTrio-Qwen3-VL-235B-A22B-Thinking-AWQ \
    --served-model-name Qwen3-VL-235B-A22B-Thinking-AWQ \
    --reasoning-parser deepseek_r1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --swap-space 16 \
    --max-num-seqs 1 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8000
```
