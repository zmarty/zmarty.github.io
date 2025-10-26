---
layout: post
title: "Speeding up local LLM inference 2x with Speculative Decoding"
date: 2025-10-26 13:15:00 -0700
categories: [AI, Development]
tags: [gpt-oss-120b, llm, vllm]
description: "How a small draft model can speed up LLM inference by 1.82× without sacrificing quality - benchmarking Qwen3-32B with speculative decoding"
---

Speculative decoding is a technique that can speed up LLM inference at a small cost of extra compute and VRAM usage. In this post I explore this technique by benchmarking the Qwen3-32B model with and without speculative decoding, achieving a **1.82× speedup** in token generation throughput while maintaining identical output quality.

<img width="800" height="836" alt="output" src="https://github.com/user-attachments/assets/8ad9398b-f08e-4d7e-8996-93b35366c38d" />

### How Does Speculative Decoding Work?

Speculative decoding uses two models working together: a small, fast "draft" model that proposes candidate tokens, and the full large model that verifies them. The draft model quickly generates a few consecutive token suggestions, and then the large model checks whether those suggestions align with what it would have generated.

At first, I had trouble understanding how this could possibly be faster. Here's what confused me initially - if the large model still has to process those tokens, how is that cheaper than just having the large model generate them in the first place?

The breakthrough came when I realized that **the large model can verify multiple tokens in a single forward pass**. With speculative decoding:

1. A small, fast draft model proposes N tokens (for example, 3 tokens: "The", "cat", "sat")
2. The large model runs **one forward pass** to check all N tokens at once
3. In that single pass, it computes the probabilities for each: p("The"|context), p("cat"|context, "The"), p("sat"|context, "The", "cat")
4. The large model accepts the prefix of tokens that match its own probability distribution
5. If any token is rejected, the large model resumes generation from that point

In normal decoding, generating N tokens would require N separate expensive forward passes through the large model. With speculative decoding, you only need 1 large model forward pass to verify all N tokens. Modern GPUs can compute activations for multiple tokens in parallel efficiently, so verifying N draft tokens in one pass is only marginally more expensive than generating 1 token - but you save N-1 expensive forward passes.

**Speculative decoding is not an approximation** as the output quality is exactly the same as if the large model generated everything itself. The draft model's suggestions are just proposals that get verified and corrected by the large model.

The technique works best when the draft model is well-aligned with the large model, resulting in high acceptance rates. In my tests, I observed an average draft acceptance rate of 33.8%.

### Models Used

For this benchmark, I used the [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) model as the main model - a 32 billion parameter model in BF16 format, weighing in at 65.5GB. For the draft model, I used [Qwen3-32B-speculator.eagle3](https://huggingface.co/RedHatAI/Qwen3-32B-speculator.eagle3) from RedHat AI, which is a 2 billion parameter model also in BF16 format at just 3.12GB. This 16:1 parameter ratio between the models allows the draft model to run very quickly while still producing reasonable suggestions that the large model can verify efficiently.

### Hardware

I ran everything locally on my Workstation

```
CPU: AMD Ryzen 9 7950X3D 16-Core Processor
GPU: NVIDIA RTX Pro 6000 (96 GB VRAM)
OS: Ubuntu 25.04
vllm: version 0.11.0
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

### Install VLLM

```bash
cd /git/vllm
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
uv pip install flashinfer-python
```

### Install benchmark

We will use the lightweight [llm-speed-benchmark](https://github.com/coder543/llm-speed-benchmark) to test speed.

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

The results show impressive speedups across all metrics. Speculative decoding nearly doubled the token generation throughput, achieving **1.82× faster** response generation (from 22.96 to 41.88 tokens/s). The time to first token also improved by **1.60×**, meaning responses start streaming much sooner. Even prompt processing saw a **1.71× speedup**.

What's remarkable is that these gains come with zero quality loss - the output is mathematically identical to what the large model would generate alone. The tradeoff is modest: an additional ~3GB of VRAM to load the draft model (3.12GB on-disk) and slightly higher compute overhead, but the benefits far outweigh the costs.

**Note I am running the full and unquantized ``Qwen3-32B`` model!**

Here's a Markdown table summarizing the **Qwen3-32B speculative decoding speedup**:

| Metric                      | Normal | Speculative |      Speedup (×) | Notes                              |
| :-------------------------- | -----: | ----------: | ---------------: | :--------------------------------- |
| **Response Tok/s**          |  22.96 |       41.88 | **1.82× faster** | Biggest improvement in throughput  |
| **Time To First Token (s)** |  25.62 |       15.98 | **1.60× faster** | Tokens start streaming much sooner |
| **Prompt Tok/s**            |   3.05 |        5.22 | **1.71× faster** | Input encoding also speeds up      |

