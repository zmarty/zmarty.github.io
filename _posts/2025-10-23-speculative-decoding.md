---
layout: post
title: "Speeding up local LLM inference 2x with Speculative Decoding"
date: 2025-10-26 13:15:00 -0700
categories: [AI, Development]
tags: [gpt-oss-120b, llm, vllm]
description: "Speculative decoding!"
---

### Install VLLM

```bash
cd /git/vllm
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
uv pip install flashinfer-python
```

### Hardware

I ran everything locally on my Workstation

```
CPU: AMD Ryzen 9 7950X3D 16-Core Processor
GPU: NVIDIA RTX Pro 6000 (96 GB VRAM)
OS: Ubuntu 25.04
```

``nvidia-smi`` output:

```bash
(vllm) zmarty@zmarty-aorus:/git/vllm$ nvidia-smi
Sun Oct 26 13:03:58 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX PRO 6000 Blac...    On  |   00000000:01:00.0 Off |                  Off |
| 30%   31C    P0            106W /  600W |      20MiB /  97887MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            5635      G   /usr/bin/gnome-shell                      8MiB |
+-----------------------------------------------------------------------------------------+
```

### Install benchmark

We will use the lighweight [llm-speed-benchmark](https://github.com/coder543/llm-speed-benchmark) to test speed.

```bash
git clone https://github.com/coder543/llm-speed-benchmark
cd llm-speed-benchmark
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### Download models

We need to download both the original ``Qwen-Qwen3-32B`` model, as well as the draft model ``RedHatAI-Qwen3-32B-speculator.eagle3``.

```bash
cd /models/original/
python3 -m venv venv
source venv/bin/activate

pip3 install huggingface-hub
huggingface-cli download Qwen/Qwen3-32B  --local-dir /models/original/Qwen-Qwen3-32B
huggingface-cli download RedHatAI/Qwen3-32B-speculator.eagle3  --local-dir /models/original/RedHatAI-Qwen3-32B-speculator.eagle3
```

## Run and benchmark model variants

First start the model without speculative decoding:

```bash
vllm serve /models/original/Qwen-Qwen3-32B \
  --served-model-name "Qwen3-32B" \
  --reasoning-parser deepseek_r1
```

Then after benchmark is done, start it with speculative decoding:

```bash
vllm serve /models/original/Qwen-Qwen3-32B \
  --served-model-name "Qwen3-32B" \
  --reasoning-parser deepseek_r1 \
  --speculative-config '{ "model": "/models/original/RedHatAI-Qwen3-32B-speculator.eagle3", "num_speculative_tokens": 3, "method": "eagle3" }'
```

Below is an example of the output that one can see in the console when the speculative version is running. Note the ``Avg Draft acceptance rate``.

```bash
(APIServer pid=722408) INFO 10-26 12:36:16 [metrics.py:96] SpecDecoding metrics: Mean acceptance length: 2.01, Accepted throughput: 20.60 tokens/s, Drafted throughput: 60.90 tokens/s, Accepted: 206 tokens, Drafted: 609 tokens, Per-position acceptance rate: 0.601, 0.286, 0.128, Avg Draft acceptance rate: 33.8%
```

For each model I ran the benchmark from inside its folder and saved the results from output.csv:

```bash
export OPENAI_API_KEY="None"
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
python3 benchmark.py --models "Qwen3-32B" -n 10 --plot no
```

Here's the prompt used by the benchmark tool:

```python
# Default prompt used for benchmarking. Long enough to hopefully get a good sense of prompt processing speed, and generate enough response tokens to get a reasonable measure there too.
DEFAULT_PROMPT = ("Imagine you are planning a week-long vacation to a place you've never visited before. "
                  "Describe the destination, including its main attractions and cultural highlights. "
                  "What activities would you prioritize during your visit? Additionally, explain how you would prepare for the trip, "
                  "including any specific items you would pack and any research you would conduct beforehand. "
                  "Finally, discuss how you would balance relaxation and adventure during your vacation.")
```

### Results

<img width="1184" height="1236" alt="output" src="https://github.com/user-attachments/assets/8ad9398b-f08e-4d7e-8996-93b35366c38d" />

Note I am running the full and unquantized ``Qwen3-32B`` model!

Here's a Markdown table summarizing the **Qwen3-32B speculative decoding speedup**:

| Metric                      | Normal | Speculative |      Speedup (×) | Notes                              |
| :-------------------------- | -----: | ----------: | ---------------: | :--------------------------------- |
| **Response Tok/s**          |  22.96 |       41.88 | **1.82× faster** | Biggest improvement in throughput  |
| **Time To First Token (s)** |  25.62 |       15.98 | **1.60× faster** | Tokens start streaming much sooner |
| **Prompt Tok/s**            |   3.05 |        5.22 | **1.71× faster** | Input encoding also speeds up      |

