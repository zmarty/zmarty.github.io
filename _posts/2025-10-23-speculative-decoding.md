---
layout: post
title: "Speeding up local LLM inference 2x with Speculative Decoding"
date: 2025-10-23 22:15:00 -0700
categories: [AI, Development]
tags: [gpt-oss-120b, llm, vllm]
description: "Speculative decoding!"
---

https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release?version=1.2.0rc1

mkdir -p /models/original/

pip install huggingface-hub
huggingface-cli download openai/gpt-oss-120b --local-dir /models/original/gpt-oss-120b
huggingface-cli download nvidia/gpt-oss-120b-Eagle3 --local-dir /models/original/gpt-oss-120b-Eagle3
huggingface-cli download nvidia/gpt-oss-120b-Eagle3-v2 --local-dir /models/original/gpt-oss-120b-Eagle3-v2


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
  --kv_cache_free_gpu_memory_fraction 0.95

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
    speculative_model_dir: /models/original/gpt-oss-120b-Eagle3-v2/
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
  --kv_cache_free_gpu_memory_fraction 0.95


--log_level debug



-----
-----

git clone https://github.com/ray-project/llmperf
cd llmperf
python3 -m venv venv
source venv/bin/activate
pip3 install -e .

----

git clone https://github.com/coder543/llm-speed-benchmark
cd llm-speed-benchmark
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

export OPENAI_API_KEY="None"
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
python benchmark.py --models "gpt-oss-120b" -n 100 --plot no

----

vllm serve /models/original/gpt-oss-120b \
  --tensor-parallel-size 1 \
  --max_num_seqs 1 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.97 \
  --tool-call-parser openai \
  --reasoning-parser openai_gptoss \
  --enable-auto-tool-choice \
  --host 0.0.0.0 \
  --port 8000
