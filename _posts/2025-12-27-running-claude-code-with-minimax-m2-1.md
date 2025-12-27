---
layout: post
title: "Running MiniMax-M2.1 Locally with Claude Code on Dual RTX Pro 6000"
date: 2025-12-27 13:15:00 -0800
categories: [AI]
tags: [llm, vllm]
description: "Run Claude Code with your own local MiniMax-M2.1 model using vLLM's native Anthropic API endpoint support."
---

# Running MiniMax-M2.1 Locally with Claude Code

Run Claude Code with your own local MiniMax-M2.1 model using vLLM's native Anthropic API endpoint support.

## Hardware Used

| Component | Specification |
|-----------|---------------|
| CPU | AMD Ryzen 9 7950X3D 16-Core Processor |
| Motherboard | ROG CROSSHAIR X670E HERO |
| GPU | Dual NVIDIA RTX Pro 6000 (96 GB VRAM each) |
| RAM | 192 GB DDR5 5200 (note the model does not use the RAM, it fits into VRAM entirely to make it fast enough to run) |

---

## Install vLLM Nightly

**Prerequisite:** [Ubuntu 24.04 and the proper NVIDIA drivers](https://forum.level1techs.com/t/wip-blackwell-rtx-6000-pro-max-q-quickie-setup-guide-on-ubuntu-24-04-lts-25-04/230521)

```bash
mkdir vllm-nightly
cd vllm-nightly
uv venv --python 3.12 --seed
source .venv/bin/activate

uv pip install -U vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly
```

---

## Download MiniMax-M2.1

Set up a separate environment for downloading models:

```bash
mkdir /models
cd /models
uv venv --python 3.12 --seed
source .venv/bin/activate

pip install huggingface_hub
```

Download the AWQ-quantized MiniMax-M2.1 model:

```bash
mkdir /models/awq
huggingface-cli download cyankiwi/MiniMax-M2.1-AWQ-4bit \
    --local-dir /models/awq/cyankiwi-MiniMax-M2.1-AWQ-4bit
```

---

## Start vLLM Server

From your vLLM environment, launch the server with the Anthropic-compatible endpoint:

```bash
cd ~/vllm-nightly
source .venv/bin/activate

vllm serve \
    /models/awq/cyankiwi-MiniMax-M2.1-AWQ-4bit \
    --served-model-name MiniMax-M2.1-AWQ \
    --max-num-seqs 10 \
    --max-model-len 128000 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 1 \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000
```

**Key flags explained:**

| Flag | Purpose |
|------|---------|
| `--tensor-parallel-size 2` | Splits model across 2 GPUs |
| `--enable-auto-tool-choice` | Enables tool/function calling |
| `--tool-call-parser minimax_m2` | Uses MiniMax-specific tool parsing |
| `--reasoning-parser minimax_m2_append_think` | Enables thinking/reasoning output |

The server exposes `/v1/messages` (Anthropic-compatible) at `http://localhost:8000`.

---

## Install Claude Code

Install Claude Code on macOS, Linux, or WSL:

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

See the [official Claude Code documentation](https://code.claude.com/docs/en/overview) for more details.

---

## Configure Claude Code

### Create settings.json

Create or edit `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8000",
    "ANTHROPIC_AUTH_TOKEN": "dummy",
    "API_TIMEOUT_MS": "3000000",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "ANTHROPIC_MODEL": "MiniMax-M2.1-AWQ",
    "ANTHROPIC_SMALL_FAST_MODEL": "MiniMax-M2.1-AWQ",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "MiniMax-M2.1-AWQ",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "MiniMax-M2.1-AWQ",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "MiniMax-M2.1-AWQ"
  }
}
```

### Skip Onboarding (Workaround for Bug)

Due to a [known bug in Claude Code 2.0.65+](https://github.com/anthropics/claude-code/issues/13827), fresh installs may ignore `settings.json` during onboarding. Add `hasCompletedOnboarding` to `~/.claude.json`:

```bash
# If ~/.claude.json doesn't exist, create it:
echo '{"hasCompletedOnboarding": true}' > ~/.claude.json

# If it exists, add the field manually or use jq:
jq '. + {"hasCompletedOnboarding": true}' ~/.claude.json > tmp.json && mv tmp.json ~/.claude.json
```

---

## Run Claude Code

With vLLM running in one terminal, open another and run:

```bash
claude
```

Claude Code will now use your local MiniMax-M2.1 model! If you also want to configure the Claude Code VSCode extension, see [here](https://platform.minimax.io/docs/guides/text-ai-coding-tools#use-m2-1-in-claude-code-extension-for-vs-code).

---

## References

- [vLLM Anthropic API Support (GitHub Issue #21313)](https://github.com/vllm-project/vllm/issues/21313)
- [MiniMax M2.1 for AI Coding Tools](https://platform.minimax.io/docs/guides/text-ai-coding-tools)
- [cyankiwi/MiniMax-M2.1-AWQ-4bit on Hugging Face](https://huggingface.co/cyankiwi/MiniMax-M2.1-AWQ-4bit)
