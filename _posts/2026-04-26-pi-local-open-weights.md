---
layout: post
title: "Coding locally with Pi Coding agent and Qwen3.6-27B, Gemma 4 31B, and MiniMax M2.7 open weights models (April 2026 edition)"
date: 2026-04-26 10:00:00 -0700
categories: [AI]
tags: [llm, local-inference, qwen, gemma, minimax, coding-agents]
description: "Run Qwen3.6-27B, Gemma 4 31B, and MiniMax M2.7 locally, then connect them to the Pi coding agent for local coding."
---

# Coding with Pi Coding agent and the latest local models

In April 2026, cloud coding assistants suddenly look a lot less predictable. Anthropic spent the month explaining [Claude Code regressions](https://www.anthropic.com/engineering/april-23-postmortem) caused by changes to reasoning defaults, context handling, and prompt tuning, while also [testing confusing pricing changes](https://arstechnica.com/ai/2026/04/anthropic-tested-removing-claude-code-from-the-pro-plan/) and [restricting flat-rate subscriptions from powering third-party agent frameworks](https://thenextweb.com/news/anthropic-openclaw-claude-subscription-ban-cost) like OpenClaw. GitHub, meanwhile, [tightened Copilot Individual plans](https://github.blog/news-insights/company-news/changes-to-github-copilot-individual-plans/) with stricter usage limits, reduced model availability, and a pause on new sign-ups as agentic workflows pushed costs beyond what consumer subscriptions were built to handle (see references at the bottom). Against that backdrop, it makes sense to revisit local coding with open weights models: models like Qwen3.6-27B, Gemma 4 31B, and MiniMax M2.7 are now strong enough to be worth wiring into a real coding agent.

In this article, we will set up these models locally on a [machine with NVIDIA RTX Pro 6000 GPUs](https://www.ovidiudan.com/2025/12/27/running-claude-code-with-minimax-m2-1.html), serve them with a local inference stack, and then connect them to the [Pi Coding agent](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent) so they can be used for real local coding workflows instead of relying on a cloud-hosted model.

<img width="700" height="346" alt="image" src="https://github.com/user-attachments/assets/26c73d7e-9319-411c-aa60-e3c92b829f0e" />

## Open models are catching up

These three models are close enough in practical coding quality to compare directly, but they make different tradeoffs. Qwen3.6-27B and Gemma 4 31B are multimodal dense models, while MiniMax M2.7 is a text-only MoE model with much higher total parameter count but far lower active compute per token. More importantly, the gap between these open models and Claude Sonnet 4.6 is now small enough that local coding is starting to look practical again.

| Model | Modality | Architecture | Total Parameters | Active Parameters | Context Length |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen3.6-27B** | Multimodal (text, image, video) | Dense (Hybrid Attention) | ~27B | ~27B | 256K-262K (extendable to ~1M) |
| **Gemma 4 31B** | Multimodal (text, image) | Dense | ~31.3B | ~31.3B | 256K |
| **MiniMax M2.7** | Text-only | Mixture of Experts (MoE) | ~230B | ~10B | ~200K |

The chart below makes the point quickly: MiniMax M2.7 is already brushing up against Claude Sonnet 4.6 on coding benchmarks, while Gemma 4 31B and Qwen3.6-27B are close enough to take seriously. A year ago, open weights models this small being in the same conversation as Sonnet for coding would have been unusual. With the right guidance, tools, and agent harness, they can now be very powerful local coding models.

<img width="1000" height="557" alt="Artificial Analysis Coding Index bar chart. GPT-5.5 C leads at 59, Claude Sonnet and Opus variants score in the 43 to 53 range, MiniMax-M2.7 scores 42, and the open models highlighted here include Gemma 4 31B at 39 and 34 and Qwen3.6 27B at 37 and 27." src="https://github.com/user-attachments/assets/3ee1e3f4-18b4-44b4-ada9-efe7abe1c134" />

## Shared setup

We'll run the models on [vllm nightly](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#install-the-latest-code). For a more detailed basic setup guide, please see my article [here](https://www.ovidiudan.com/2025/12/27/running-claude-code-with-minimax-m2-1.html). You can run the same models on Apple Silicon as well, [with mlx](https://huggingface.co/collections/mlx-community/gemma-4).

```bash
uv pip install -U vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly # add variant subdirectory here if needed
```

## Running Qwen3.6-27B locally

```bash
hf download Qwen/Qwen3.6-27B --local-dir /models/original/Qwen-Qwen3.6-27B

vllm serve \
    /models/original/Qwen-Qwen3.6-27B \
    --served-model-name Qwen3.6-27B \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
    --host 0.0.0.0 \
    --port 8000
```

The ``speculative-config`` line enables [Multi-Token Prediction](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/) [speculative decoding](https://www.ovidiudan.com/2025/10/26/speculative-decoding.html) which speeds up the model when the tokens are "easy".

## Running Gemma 4 31B locally

```bash
hf download google/gemma-4-31B-it --local-dir /models/original/google-gemma-4-31B-it
```

Gemma 4 supports structured thinking, where the model can reason step-by-step before producing a final answer. The reasoning process is exposed via the reasoning field in the API response. To enable thinking, first download [this](https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_gemma4.jinja) file and store it with the model so you can load it with vllm.

Then reference it when you load the model:

```bash
vllm serve \
    /models/original/google-gemma-4-31B-it \
    --served-model-name gemma-4-31B \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --chat-template /models/original/google-gemma-4-31B-it/tool_chat_template_gemma4.jinja \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --host 0.0.0.0 \
    --port 8000
```

## Running MiniMax M2.7 locally

I can't fit the entire model in my 192 GB VRAM so I have to use a slightly quantized version.

```bash
hf download lukealonso/MiniMax-M2.7-NVFP4 --local-dir /models/nvfp4/lukealonso-MiniMax-M2.7-NVFP4

export SAFETENSORS_FAST_GPU=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export VLLM_LOGGING_LEVEL=INFO

vllm serve \
    /models/nvfp4/lukealonso-MiniMax-M2.7-NVFP4 \
    --served-model-name MiniMax-M2.7-NVFP4 \
    --max-model-len 196608 \
    --max-num-seqs 2 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think \
    --trust-remote-code \
    --enable_expert_parallel \
    --disable-custom-all-reduce \
    --host 0.0.0.0 \
    --port 8000
```

Here are a few quantized versions of this model that I know of:
- [lukealonso/MiniMax-M2.7-NVFP4](https://huggingface.co/lukealonso/MiniMax-M2.7-NVFP4)
- [cyankiwi/MiniMax-M2.7-AWQ-4bit](https://huggingface.co/cyankiwi/MiniMax-M2.7-AWQ-4bit)
- [QuantTrio/MiniMax-M2.7-AWQ](https://huggingface.co/QuantTrio/MiniMax-M2.7-AWQ)
- [NinjaBoffin/MiniMax-M2.7-NVFP4](https://huggingface.co/NinjaBoffin/MiniMax-M2.7-NVFP4)

## Connecting the Pi coding agent

Pi Coding agent is opinionated in a very specific way: it keeps the core deliberately small and expects you to extend it with extensions, skills, plain files, README-driven CLI tools, tmux sessions, or third-party packages. That means no built-in MCP support, no dedicated sub-agents, no permission popups, no plan mode, no internal to-do system, and no background bash process manager. The point is not that these workflows are impossible, but that Pi wants them to stay explicit and observable instead of hiding them behind extra orchestration.

That design is especially useful when you are pairing Pi with local open weights models. Smaller prompts, fewer baked-in abstractions, and a minimal tool surface leave more context budget for your actual project and make the agent's behavior easier to understand when a local model goes off course. If you want the full rationale behind that philosophy, [Mario Zechner's post](https://mariozechner.at/posts/2025-11-30-pi-coding-agent/) on building an opinionated minimal coding agent is worth reading.

```bash
npm install -g @mariozechner/pi-coding-agent
nano ~/.pi/agent/models.json
```

```json
{
  "providers": {
    "vllm": {
      "baseUrl": "http://localhost:8000/v1",
      "api": "openai-completions",
      "apiKey": "vllm",
      "models": [
        {
            "id": "Qwen3.6-27B",
            "input": ["text", "image"],
            "contextWindow": 262144
        },
        {
            "id": "gemma-4-31B",
            "input": ["text", "image"],
            "contextWindow": 262144
        },
        {
            "id": "MiniMax-M2.7-NVFP4",
            "input": ["text"],
            "contextWindow": 196608
        }
      ]
    }
  }
}
```

In pi select /model and scroll down to the bottom of the list to select your local model(s), otherwise it will use a free cloud model:

<img width="700" height="498" alt="image" src="https://github.com/user-attachments/assets/ac611b1e-d729-4a5b-af40-cf92abdb8f2c" />

## References:
- [Anthropic tested removing Claude Code from the Pro plan](https://arstechnica.com/ai/2026/04/anthropic-tested-removing-claude-code-from-the-pro-plan/)
- [An update on recent Claude Code quality reports](https://www.anthropic.com/engineering/april-23-postmortem)
- [Anthropic cuts Claude subscribers off from OpenClaw in cost crackdown](https://thenextweb.com/news/anthropic-openclaw-claude-subscription-ban-cost)
- [Changes to GitHub Copilot Individual plans](https://github.blog/news-insights/company-news/changes-to-github-copilot-individual-plans/)
- [Anthropic's Claude Code pricing pain is Sam Altman's pleasure](https://www.businessinsider.com/anthropic-claude-code-price-confusion-sam-altman-2026-4)
