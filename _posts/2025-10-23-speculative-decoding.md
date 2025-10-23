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

cd /models

https://github.com/NVIDIA/TensorRT-LLM/issues/8474

cat <<EOF > low_latency.yaml
enable_attention_dp: false
cuda_graph_config:
    max_batch_size: 1
    enable_padding: true
moe_config:
    backend: CUTLASS
EOF


trtllm-serve \
  /models/original/gpt-oss-120b \
  --host 0.0.0.0 \
  --port 8000 \
  --backend pytorch \
  --tp_size 1 \
  --ep_size 1 \
  --max_batch_size 1 \
  --trust_remote_code \
  --extra_llm_api_options low_latency.yaml \
  --kv_cache_free_gpu_memory_fraction 0.9

--

sudo iptables -I INPUT -i docker0 -p tcp --dport 8000 -j ACCEPT
sudo iptables -I DOCKER-USER -i docker0 -j ACCEPT

--

https://developer.nvidia.com/blog/tensorrt-llm-speculative-decoding-boosts-inference-throughput-by-up-to-3-6x/

low_latency_speculative.yaml:

enable_attention_dp: false
disable_overlap_scheduler: true
enable_autotuner: false
cuda_graph_config:
    max_batch_size: 1
    enable_padding: true
moe_config:
    backend: CUTLASS
speculative_config:
    decoding_type: Eagle
    max_draft_len: 3
    speculative_model_dir: /models/original/gpt-oss-120b-Eagle3/
kv_cache_config:
    enable_block_reuse: false



trtllm-serve \
  /models/original/gpt-oss-120b \
  --host 0.0.0.0 \
  --port 8000 \
  --backend pytorch \
  --tp_size 1 \
  --ep_size 1 \
  --max_batch_size 1 \
  --trust_remote_code \
  --extra_llm_api_options low_latency_speculative.yaml \
  --kv_cache_free_gpu_memory_fraction 0.9
