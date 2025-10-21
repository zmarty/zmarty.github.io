---
layout: post
title: "Open Weights, Borrowed GPUs"
date: 2025-10-20 23:00:00 -0700
categories: [AI, Development]
tags: [glm-4.6, llm, vllm]
description: "A practical guide to renting GPUs for running open-weight LLM models with control, privacy, and flexibility."
---

There's a gap between using ChatGPT through an API and running your own GPU cluster. That gap is where **power users and tinkerers** live - people who want to understand how LLMs really work, need privacy for their data, or want full control over their AI stack.

A cottage industry has emerged to serve this need: GPU rental platforms like [Vast.ai](https://vast.ai/), [RunPod](https://www.runpod.io/), and [Lambda Labs](https://lambda.ai/) where you can rent datacenter-grade hardware by the minute for inference or fine-tuning. People aren't doing this to save money. They're paying for knowledge, privacy, and control that APIs can't provide. 

Join me in exploring this nascent industry. Along the way we will run an open weights model which is competitive with closed models such as Anthropic Claude 4 Sonnet and OpenAI o3.

## What We'll Cover

In this post, I'll walk through the practical steps of renting four [NVIDIA RTX Pro 6000](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/) GPUs for a total of 384 GB of VRAM (!) and running models. We'll:

1. Choose and configure GPU hardware
2. Set up SSH access securely
3. Deploy a [vLLM](https://github.com/vllm-project/vllm) inference server
4. Run the full Zhipu AI [GLM-4.6](https://z.ai/blog/glm-4.6) open weights model (the principles apply to any open-weight model)
5. Understand the key parameters and trade-offs
6. Use the [Jan](https://www.jan.ai/) UI to access and use the model remotely.

## My Setup: Why I Needed More Than My Home Rig

I run a dual RTX 3090 setup at home . This handles most models fine, but I hit limitations when:
- Testing models that need more than the 48GB VRAM that I have available
- Running long context windows (65k+ tokens)
- Using quantization methods like [NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) supported natively by newer GPU architectures such as Blackwell.
- Fine-tuning models

## About GLM-4.6

[GLM-4.6](https://z.ai/blog/glm-4.6) is Zhipu AI's latest flagship model, representing a significant upgrade over GLM-4.5. Built with a Mixture-of-Experts architecture, it brings several key improvements: a **200K token context window** (up from 128K), superior coding performance approaching Claude Sonnet 4 levels, and enhanced reasoning capabilities with tool use during inference. In real-world evaluations using CC-Bench, GLM-4.6 achieves near parity with Claude Sonnet 4 (48.6% win rate) while clearly outperforming other open-source models like DeepSeek-V3.2-Exp. The model also demonstrates 15% better token efficiency than GLM-4.5, completing tasks with fewer tokens while maintaining higher quality. Like its predecessor, GLM-4.6 offers hybrid reasoning modes - thinking mode for complex tasks and non-thinking mode for instant responses - all while being available as open weights.

<img width="800" height="626" alt="image" src="https://github.com/user-attachments/assets/8d487191-6338-4fd3-b94e-f9a1f6436349" />

## Step-by-Step: Renting GPUs and Running vLLM

Let's walk through the entire process. I'll use Vast.ai, but the concepts apply to RunPod, Lambda Labs, or any other provider.

After I created an account, I filtered down to the RTX Pro 6000 GPUs that I was interested in. I am experimenting with them because I plan on buying one for my home workstation. These rental platforms have many other GPUs available, including gaming GPUs such as 5090, as well as datacenter-grade GPUs such as H100 or H200.

<img width="2054" height="1251" alt="image" src="https://github.com/user-attachments/assets/b63d9cb5-ff2b-4fe0-babf-eb5247e81e9c" />

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





