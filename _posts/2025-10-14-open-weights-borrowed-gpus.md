---
layout: post
title: "Open Weights, Borrowed GPUs"
date: 2025-10-14 23:00:00 -0700
categories: [AI, Development]
tags: [gpt-oss, glm-4.6, llm, vllm]
description: "A practical guide to renting GPUs for running open-weight LLM models with control, privacy, and flexibility."
---

## Introduction

The AI revolution has brought us powerful language models, but there's a catch: the best models often require expensive GPUs to run. While companies like OpenAI and Anthropic offer convenient API access, what if you want more control over your infrastructure, need to keep your data private, or simply want to learn how modern LLMs actually work under the hood?

Enter the world of **open-weight models** and **GPU rentals**. "Open weights" refers to models whose parameters (weights) are publicly available—think Meta's Llama, Mistral's models, or Zhipu's GLM series. Unlike closed models from OpenAI, you can download these and run them yourself. The challenge? These models need serious hardware.

This is where GPU rental services come in. Rather than buying a $10,000 workstation with high-end GPUs (and dealing with the power bills), you can rent GPU time by the minute. A cottage industry has sprung up around this need, with platforms like [Vast.ai](https://vast.ai/), [RunPod](https://www.runpod.io/), [Lambda Labs](https://lambda.ai/), and others offering access to everything from consumer GPUs to datacenter-grade hardware.

**Why rent GPUs for inference?** It's true that for simple API calls, services like OpenAI might be more cost-effective. But renting GPUs offers unique advantages:

- **Full control** over the software stack and model configurations
- **Privacy** - your data never leaves the server you're renting
- **Learning opportunity** - understand what it takes to run and fine-tune models
- **Flexibility** - experiment with different models, quantizations, and serving frameworks
- **Fine-tuning** - train custom adaptors or fine-tune models on your own data

In this post, I'll walk through the practical steps of renting GPUs and running open-weight models using [Vast.ai](https://vast.ai/) as an example. We'll set up a [vLLM](https://github.com/vllm-project/vllm) inference server running the GLM-4 series models, but the principles apply to any platform and model.


<img width="732" height="440" alt="image" src="https://github.com/user-attachments/assets/23468865-098e-4184-b6aa-40fb56190181" />

<img width="735" height="174" alt="image" src="https://github.com/user-attachments/assets/7c19f373-5109-40ad-99fe-37dbcae1c7c8" />

<img width="400" height="441" alt="image" src="https://github.com/user-attachments/assets/6b9fbb84-571a-4456-b767-54c2ce160466" />

<img width="350" height="505" alt="image" src="https://github.com/user-attachments/assets/2e76b9dc-b789-4e68-97e6-1f1cd14e014f" />

<img width="804" height="463" alt="image" src="https://github.com/user-attachments/assets/0e6223ac-6754-4f79-aa13-35e1d7e11aaa" />

<img width="714" height="89" alt="image" src="https://github.com/user-attachments/assets/be228d64-4c7e-4da5-bf4e-62237c1f6107" />

<img width="876" height="480" alt="image" src="https://github.com/user-attachments/assets/51e8f2fe-5113-47f2-8d0b-bb81fab24b9c" />

<img width="726" height="461" alt="image" src="https://github.com/user-attachments/assets/c3a3064b-6534-4404-81d8-00a57318a994" />

"--max-model-len 8192 --enforce-eager --download-dir /workspace/models --host 127.0.0.1 --port 18000 --enable-auto-tool-choice --tool-call-parser glm45 --reasoning-parser glm45 --swap-space 16 --max-num-seqs 64 --gpu-memory-utilization 0.9 --tensor-parallel-size 2 --enable-expert-parallel --trust-remote-code"



Create an SSH key BEFORE you create an instance, otherwise you will need to go under Instances and set a SSH key that will get propagated to the machine.

<img width="788" height="400" alt="image" src="https://github.com/user-attachments/assets/2b293a20-0e2b-4224-8b66-d18f8bfebcf4" />

<img width="450" height="268" alt="image" src="https://github.com/user-attachments/assets/61151006-c43a-465c-966c-a70778c1bf88" />

<img width="532" height="336" alt="image" src="https://github.com/user-attachments/assets/a0e1de83-0b0b-4204-8fa7-b9060da7e93c" />

<img width="450" height="225" alt="image" src="https://github.com/user-attachments/assets/78978d0c-8a10-4b34-b5e0-ed98fd377a22" />

<img width="802" height="281" alt="image" src="https://github.com/user-attachments/assets/991b3905-2eb2-4abc-acde-333a9e57ea53" />

<img width="500" height="352" alt="image" src="https://github.com/user-attachments/assets/f4d72a67-56d6-4e10-af37-d5f4ab1e9230" />

<img width="832" height="514" alt="image" src="https://github.com/user-attachments/assets/754793e6-1d01-424b-a1ac-3f0e4b81807d" />

<img width="500" height="198" alt="image" src="https://github.com/user-attachments/assets/9cb3f803-4dea-4963-8691-8489b1920936" />

<img width="678" height="640" alt="image" src="https://github.com/user-attachments/assets/deff81ab-1788-4e76-9d88-18007e83449c" />

<img width="678" height="640" alt="image" src="https://github.com/user-attachments/assets/9e9c3215-016e-46fe-bfa2-8600bc85411b" />

<img width="944" height="478" alt="image" src="https://github.com/user-attachments/assets/7c272700-08a6-4e7e-9a3d-8618aadd8d8a" />

<img width="685" height="642" alt="image" src="https://github.com/user-attachments/assets/5917c985-2ceb-4590-8777-7a91d5e3847b" />

<img width="685" height="642" alt="image" src="https://github.com/user-attachments/assets/5b45c3fd-f2a2-4033-875c-96d49e1fcbbf" />

<img width="1188" height="655" alt="image" src="https://github.com/user-attachments/assets/9393198c-da40-4eca-b9ea-6555b8e128c6" />

<img width="1188" height="655" alt="image" src="https://github.com/user-attachments/assets/c4ab908d-e758-4584-9ef2-911098b1c0d1" />

<img width="684" height="634" alt="image" src="https://github.com/user-attachments/assets/ea90d150-17a7-4b0e-9fe2-76ebecf0f64a" />

<img width="917" height="486" alt="image" src="https://github.com/user-attachments/assets/31832b1f-3888-4dea-9794-796b7750bba1" />

---

GLM-4.5 Air:
"--max-model-len 65536 --download-dir /workspace/models --host 127.0.0.1 --port 18000 --enable-auto-tool-choice --tool-call-parser glm45 --reasoning-parser glm45 --gpu-memory-utilization 0.95 --tensor-parallel-size 2 "

<img width="554" height="132" alt="image" src="https://github.com/user-attachments/assets/c8fda289-7895-4b59-90c7-28f4420f918d" />





