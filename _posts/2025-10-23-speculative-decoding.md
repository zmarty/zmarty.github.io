---
layout: post
title: "Speculative decoding"
date: 2025-10-23 22:15:00 -0700
categories: [AI, Development]
tags: [gpt-oss-120b, llm, vllm]
description: "Speculative decoding!"
---

mkdir -p /models/original/

pip install huggingface-hub
huggingface-cli download nvidia/gpt-oss-120b-Eagle3 --local-dir /models/original/gpt-oss-120b-Eagle3
huggingface-cli download openai/gpt-oss-120b --local-dir /models/original/gpt-oss-120b


docker run --rm --ipc=host -it \
  --ulimit stack=67108864 \
  --ulimit memlock=-1 \
  --gpus all \
  -p 8000:8000 \
  -e TRTLLM_ENABLE_PDL=1 \
  -v /models:/models:rw \
  nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc1 \
  /bin/bash



mpirun -n 1 --oversubscribe --allow-run-as-root \
trtllm-serve 
  --model openai/gpt-oss-120b \
  --model_path /models/original/gpt-oss-120b \
  --host 0.0.0.0 \
  --port 8000 \
  --backend pytorch \
  --tp_size 8 \
  --ep_size 8 \
  --max_batch_size 640 \
  --trust_remote_code \
  --extra_llm_api_options max_throughput.yaml \
  --kv_cache_free_gpu_memory_fraction 0.9
