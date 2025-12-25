---
layout: post
title: "Guide on installing and running best models on a dual RTX Pro 6000 rig"
date: 2025-12-25 10:30:00 -0800
categories: [AI]
tags: [llm, vllm]
description: "."
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



## Devstral 2 123B

Model: [Devstral-2-123B-Instruct-2512-AWQ-4bit](https://huggingface.co/cyankiwi/Devstral-2-123B-Instruct-2512-AWQ-4bit)
vLLM version: vllm-nightly on December 25th, 2025

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
