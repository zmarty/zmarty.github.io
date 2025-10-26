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

venv etc!!!!!!!!!!!!!
source venv/bin/activate !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
cd /models/original/

pip install huggingface-hub
huggingface-cli download openai/gpt-oss-120b --local-dir /models/original/gpt-oss-120b
huggingface-cli download nvidia/gpt-oss-120b-Eagle3 --local-dir /models/original/gpt-oss-120b-Eagle3
huggingface-cli download nvidia/gpt-oss-120b-Eagle3-v2 --local-dir /models/original/gpt-oss-120b-Eagle3-v2

huggingface-cli download Snowflake/Arctic-LSTM-Speculator-gpt-oss-120b --local-dir /models/original/Arctic-LSTM-Speculator-gpt-oss-120b


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

----

https://github.com/snowflakedb/ArcticInference/tree/main

pip3 install arctic-inference[vllm]

ARCTIC_INFERENCE_ENABLED=1 vllm serve /models/original/gpt-oss-120b \
  --tensor-parallel-size 1 \
  --max_num_seqs 1 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.97 \
  --reasoning-parser GptOss \
  --speculative-config '{ "method": "arctic", "model":"/models/original/Arctic-LSTM-Speculator-gpt-oss-120b", "num_speculative_tokens": 3, "enable_suffix_decoding": true }' \
  --host 0.0.0.0 \
  --port 8000

----

git clone https://github.com/coder543/llm-speed-benchmark
cd llm-speed-benchmark
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

export OPENAI_API_KEY="None"
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
python benchmark.py --models "/models/original/gpt-oss-120b" -n 10 --plot no

----


VLLM_TORCH_BACKEND=auto uv pip install -U \
  --prerelease=allow \
  --extra-index-url https://wheels.vllm.ai/nightly \
  "triton-kernels @ git+https://github.com/triton-lang/triton.git@v3.5.0#subdirectory=python/triton_kernels" \
  vllm

----

----

vllm serve /models/original/gpt-oss-120b \
  --tensor-parallel-size 1 \
  --max_num_seqs 1 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.97 \
  --tool-call-parser openai \
  --reasoning-parser openai_gptoss \
  --enable-auto-tool-choice \
  --speculative-config '{"model": "/models/original/gpt-oss-120b-Eagle3", "num_speculative_tokens": 3, "method":"eagle3", "draft_tensor_parallel_size":1}' \
  --host 0.0.0.0 \
  --port 8000

----

uv pip install flashinfer-python

----

# 1. Create a virtual environment
     python3 -m venv .venv
     
     # 2. Activate the virtual environment and upgrade pip
     source .venv/bin/activate
     pip install --upgrade pip
     
     # 3. Install nightly Triton first
     pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton
     
     # 4. Install nightly vLLM
     pip install vllm --pre
     
     # 5. Force reinstall nightly Triton (vLLM will have downgraded it to stable 3.4.0)
     pip install --force-reinstall --no-deps --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton
     
     # 6. Verify installation
     pip list | grep -E "(vllm|triton)"

   Key insight: vLLM's dependencies will pull in the stable Triton 3.4.0, so you
   must force reinstall nightly Triton afterward using --force-reinstall --no-deps
   to keep the nightly version.

----

Using flash attention

uv pip install flashinfer-python

https://huggingface.co/RedHatAI/Qwen3-32B-speculator.eagle3

vllm serve /models/original/Qwen-Qwen3-32B \
  --served-model-name "Qwen3-32B" \
  --reasoning-parser deepseek_r1 \
  --speculative-config '{ "model": "/models/original/RedHatAI-Qwen3-32B-speculator.eagle3", "num_speculative_tokens": 3, "method": "eagle3" }'

python3 benchmark.py --models "Qwen3-32B" -n 10 --plot no

```python
# Default prompt used for benchmarking. Long enough to hopefully get a good sense of prompt processing speed, and generate enough response tokens to get a reasonable measure there too.
DEFAULT_PROMPT = ("Imagine you are planning a week-long vacation to a place you've never visited before. "
                  "Describe the destination, including its main attractions and cultural highlights. "
                  "What activities would you prioritize during your visit? Additionally, explain how you would prepare for the trip, "
                  "including any specific items you would pack and any research you would conduct beforehand. "
                  "Finally, discuss how you would balance relaxation and adventure during your vacation.")
```

```console
(APIServer pid=722408) INFO 10-26 12:36:16 [metrics.py:96] SpecDecoding metrics: Mean acceptance length: 2.01, Accepted throughput: 20.60 tokens/s, Drafted throughput: 60.90 tokens/s, Accepted: 206 tokens, Drafted: 609 tokens, Per-position acceptance rate: 0.601, 0.286, 0.128, Avg Draft acceptance rate: 33.8%
(APIServer pid=722408) INFO 10-26 12:36:26 [loggers.py:127] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 47.4 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.9%, Prefix cache hit rate: 37.2%
```

