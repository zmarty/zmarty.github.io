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

Join me in exploring this nascent industry. Along the way I'll run an open weights model which is competitive with closed models such as Anthropic Claude 4 Sonnet and OpenAI o3.

## What We'll Cover

In this post, I'll walk through the practical steps of renting four [NVIDIA RTX Pro 6000](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/) GPUs for a total of 384 GB of VRAM (!) and running models. We'll:

1. Choose and configure GPU hardware
2. Set up SSH access securely
3. Deploy a [vLLM](https://github.com/vllm-project/vllm) inference server
4. Run the full Zhipu AI [GLM-4.6](https://z.ai/blog/glm-4.6) open weights model (the principles apply to any open-weight model)
5. Understand the key parameters and trade-offs
6. Use the [Jan](https://www.jan.ai/) UI to access and use the model remotely.

## My Setup: Why I Needed More Than My Home Rig

I run a dual RTX 3090 setup at home. This handles most models fine, but I hit limitations when:
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

<img width="1000" height="609" alt="image" src="https://github.com/user-attachments/assets/b63d9cb5-ff2b-4fe0-babf-eb5247e81e9c" />

<br>

The interface also allows selecting which prebuilt Docker container we want to run on the rented machine, which arguments to send to the container, and how much disk space we want to allocate. The example below shows the template that I used to run vLLM on the four GPUs.

<img width="350" height="505" alt="image" src="https://github.com/user-attachments/assets/2e76b9dc-b789-4e68-97e6-1f1cd14e014f" />

<br>

And this is the configuration screen to set the variables:

<img width="804" height="463" alt="image" src="https://github.com/user-attachments/assets/0e6223ac-6754-4f79-aa13-35e1d7e11aaa" />

<br>

Scrolling down this config page we get to the place where we specify which model it should run, in our case it's [this AWQ quant of the model](https://huggingface.co/QuantTrio/GLM-4.6-AWQ), which is 197 GB in size. 

<img width="500" height="352" alt="image" src="https://github.com/user-attachments/assets/f4d72a67-56d6-4e10-af37-d5f4ab1e9230" />

<br>

Under VLLM args I specified the following parameters to load the model:

```text
--max-model-len 8192 --enforce-eager --download-dir /workspace/models --host 127.0.0.1 --port 18000 --enable-auto-tool-choice --tool-call-parser glm45 --reasoning-parser glm45 --swap-space 16 --max-num-seqs 64 --gpu-memory-utilization 0.9 --tensor-parallel-size 4 --enable-expert-parallel --trust-remote-code
```

Here's what these mean:

- `--max-model-len 8192`: Max context length (tokens). Limits prompt + output length to save GPU memory. Here I could have set it much higher since I had the VRAM available.
- `--enforce-eager`: Forces PyTorch eager mode (disables CUDA graphs). More stable, slightly slower.
- `--download-dir /workspace/models`: Folder where model weights are stored/downloaded.
- `--host 127.0.0.1 --port 18000`: Local API server address and port.
- `--enable-auto-tool-choice`: Lets the model decide automatically when to call tools/functions.
- `--tool-call-parser glm45`: Parser for interpreting GLM-4.5/4.6 tool-call outputs.
- `--reasoning-parser glm45`: Parser for structured reasoning output (chain-of-thought style).
- `--swap-space 16`: Allows 16 GiB of CPU offload per GPU for overflow memory.
- `--max-num-seqs 64`: Max number of concurrent sequences/batch per iteration.
- `--gpu-memory-utilization 0.9`: Uses up to 90% of GPU memory. vLLM needs headroom because GPU memory usage spikes during initialization, kernel setup, and dynamic batching. Setting utilization to 0.9 prevents startup out-of-memory and keeps things stable during runtime.
- `--tensor-parallel-size 4`: Splits model across 4 GPUs for inference.
- `--enable-expert-parallel`: Enables expert parallelism for MoE (Mixture-of-Experts) models.
- `--trust-remote-code`: Allows loading custom model code from remote repositories (e.g., Hugging Face).

<img width="714" height="89" alt="image" src="https://github.com/user-attachments/assets/be228d64-4c7e-4da5-bf4e-62237c1f6107" />

<br>

Once the rental has started, you can view the status of the machine:

<img width="900" height="213" alt="image" src="https://github.com/user-attachments/assets/7c19f373-5109-40ad-99fe-37dbcae1c7c8" />

<br>

This particular platform also provides a dashboard to view the running applications and their logs, as well as the forwarded ports - in this case port 44095 on the public IP was mapped to the internal vLLM server running on port 8080.

<img width="876" height="480" alt="image" src="https://github.com/user-attachments/assets/51e8f2fe-5113-47f2-8d0b-bb81fab24b9c" />

<br>

You can see in the logs that it was loading the GLM model:

<img width="802" height="281" alt="image" src="https://github.com/user-attachments/assets/991b3905-2eb2-4abc-acde-333a9e57ea53" />

To access the machine over SSH, first **create an SSH key BEFORE you create an instance**, otherwise you will need to go under Instances and set a SSH key that will get propagated to the machine.

1. Generate a SSH key pair in your terminal:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

2. Copy your public key, then paste it into the vast.ai interface under Keys:

```bash
# Print the public key
cat ~/.ssh/id_ed25519.pub
# Output will look like:
# ssh-ed25519 AAAAC3NzaC1lZ9DdI1NTE5AAAAIHWGYlMT8CxcILI/i3DsRvX74HNChkm4JSNFu0wmcv0a your_email@example.com
```

<img width="788" height="400" alt="image" src="https://github.com/user-attachments/assets/2b293a20-0e2b-4224-8b66-d18f8bfebcf4" />

<br>

Once you are logged in, you can run nvidia-smi to view the available GPUs:

<img width="732" height="440" alt="image" src="https://github.com/user-attachments/assets/23468865-098e-4184-b6aa-40fb56190181" />

<br>

We can also check the size of the model on disk as it is being downloaded from Huggingface:

<img width="500" height="198" alt="image" src="https://github.com/user-attachments/assets/9cb3f803-4dea-4963-8691-8489b1920936" />

<br>

Once the model is fully loaded the vLLM log in the dashboard will show that it is ready to accept requests:

<img width="832" height="514" alt="image" src="https://github.com/user-attachments/assets/754793e6-1d01-424b-a1ac-3f0e4b81807d" />

Finally, we can connect a UI such as [Jan](https://www.jan.ai/) to the server remotely to access the LLM:






<img width="532" height="336" alt="image" src="https://github.com/user-attachments/assets/a0e1de83-0b0b-4204-8fa7-b9060da7e93c" />

<img width="450" height="225" alt="image" src="https://github.com/user-attachments/assets/78978d0c-8a10-4b34-b5e0-ed98fd377a22" />

<br>

You can see at the bottom here we are getting about 8 tokens per second: 

<img width="685" height="642" alt="image" src="https://github.com/user-attachments/assets/5917c985-2ceb-4590-8777-7a91d5e3847b" />

<br>

And here is the output of `nvtop`, which shows how the 4 GPUs are being exercised during inference:

<img width="1188" height="655" alt="image" src="https://github.com/user-attachments/assets/9393198c-da40-4eca-b9ea-6555b8e128c6" />

## Pricing

As you can see from one of the screenshots above, I was paying $4.084 per hour for 4x RTX Pro 6000 GPUs. These cards retail at around $8,000 each ($32,000 total for four), not including the server itself. That's a relatively small hourly cost for accessing 384GB of VRAM and such powerful GPUs.

At the moment of this writing (October 20th, 2025), here are some typical hourly rental prices that I see per GPU type:

| GPU Model | Price/Hour | VRAM |
|-----------|------------|------|
| H200 | $2.59 | 141 GB HBM3e |
| H200 NVL | $2.27 | 141 GB HBM3e |
| H100 NVL | $2.06 | 94 GB HBM3e |
| H100 SXM | $1.87 | 80 GB HBM3e |
| RTX Pro 6000 (Blackwell) | $1.01 | 96 GB GDDR7 |
| A100 SXM4 | $0.88 | 80 GB HBM2e |
| RTX 6000 Ada | $0.63 | 48 GB GDDR6 |
| RTX 5090 | $0.60 | 32 GB GDDR7 |
| L40S | $0.58 | 48 GB GDDR6 |
| L40 | $0.57 | 48 GB GDDR6 |
| RTX 4090 | $0.40 | 24 GB GDDR6X |
| RTX 5880 Ada | $0.36 | 48 GB GDDR6 |
| RTX 5060 Ti | $0.19 | 16 GB GDDR6 |
| RTX 4080 | $0.18 | 16 GB GDDR6X |
| RTX 3090 | $0.18 | 24 GB GDDR6X |
| RTX 5070 | $0.13 | 16 GB GDDR6 |

## Key Takeaways and Tips

After spending time experimenting with GPU rentals, here are the practical lessons I learned:

**Download Speed Matters More Than You Think**
Pick a machine with at least 5 Gbps internet access. On a 1 Gbps connection, it took over 30 minutes just to download the 197 GB model and get vLLM started - wasting precious rental time and money. Faster connections can reduce this to under 10 minutes.

**Setting up SSH Keys Beforehand**
Set up your SSH keys in the Keys section before creating any instances. If you forget, you'll need to manually configure them in the Instances tab after the fact, which adds unnecessary friction to your workflow.

**Wide Range of Hardware Options**
The marketplace offers everything from budget-friendly RTX 3090s at $0.18/hour to cutting-edge H200s at $2.59/hour. This variety means you can match your GPU choice to your specific needs - whether that's experimenting with quantization methods on newer architectures like Blackwell, maximizing VRAM for large context windows, or testing your code on different hardware before making a purchase decision.

**Understanding the Cost Trade-offs**
Don't just look at the hourly rate—consider what you're actually getting. My 4x RTX Pro 6000 setup at $4.08/hour gave me 384GB VRAM to run an AWQ-quantized GLM-4.6 model. Compare that to the much cheaper cloud inference APIs: you're not paying per token here, you're paying for dedicated access to test architectures, experiment with context windows, and understand deployment before committing to hardware purchases. A few hours of experimentation costs less than a nice dinner, but the knowledge gained about VRAM requirements, quantization trade-offs, and performance characteristics is invaluable if you're considering buying your own GPUs.

**It's More Accessible Than You'd Expect**
Setting up a full inference server with vLLM is straightforward once you understand the basic parameters. The barrier to entry is lower than most people think, and the cost is reasonable - especially for experimentation or fine-tuning workloads where you need the compute for limited time periods.

**Who's Behind These Rentals?**
The GPU rental marketplace is surprisingly diverse. You're renting from:
- Enthusiasts with high-end gaming rigs or workstations. Yes could be renting a computer in somebody's basement
- Small businesses with spare compute capacity
- Small datacenter operators looking to monetize idle hardware

This peer-to-peer model keeps costs competitive while providing access to hardware that would otherwise sit unused. The downside is that availability is not always guaranteed.
