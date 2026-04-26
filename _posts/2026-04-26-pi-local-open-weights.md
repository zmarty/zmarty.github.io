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

## Recent models

These three models are close enough in practical coding quality to compare directly, but they make different tradeoffs. Qwen3.6-27B is the leanest dense model here and stretches furthest on context, Gemma 4 31B is the larger dense multimodal option, and MiniMax M2.7 is the MoE outlier with much higher total parameter count but far lower active compute per token. More importantly, the gap between these open models and Claude Sonnet 4.6 is now small enough that local coding is starting to look practical again.

| Model | Architecture | Total Parameters | Active Parameters | Context Length |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen3.6-27B** | Dense (Hybrid Attention) | ~27B | ~27B | 256K-262K (extendable to ~1M) |
| **Gemma 4 31B** | Dense | ~31.3B | ~31.3B | 256K |
| **MiniMax M2.7** | Mixture of Experts (MoE) | ~230B | ~10B | ~200K |

<img width="1000" height="557" alt="Artificial Analysis Coding Index bar chart. GPT-5.5 C leads at 59, Claude Sonnet and Opus variants score in the 43 to 53 range, MiniMax-M2.7 scores 42, and the open models highlighted here include Gemma 4 31B at 39 and 34 and Qwen3.6 27B at 37 and 27." src="https://github.com/user-attachments/assets/3ee1e3f4-18b4-44b4-ada9-efe7abe1c134" />

The chart makes the point quickly: MiniMax M2.7 is already brushing up against Claude Sonnet 4.6 on coding benchmarks, while Gemma 4 31B and Qwen3.6-27B are close enough to take seriously. A year ago, open weights models this small being in the same conversation as Sonnet for coding would have been unusual. With the right guidance, tools, and agent harness, they can now be very powerful local coding models.


## Hardware assumptions




## Why this update



## Shared setup

## Running Qwen3.6-27B locally

## Running Gemma 4 31B locally

## Running MiniMax M2.7 locally

## Connecting the Pi coding agent

Pi Coding agent is a very opinionated coding agent

```bash
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
        }
      ]
    }
  }
}
```

In pi select /model and scroll down to the bottom of the list to select your local model(s), otherwise it will use a free cloud model.

## Model-by-model notes

## Troubleshooting

## References:
- [Anthropic tested removing Claude Code from the Pro plan](https://arstechnica.com/ai/2026/04/anthropic-tested-removing-claude-code-from-the-pro-plan/)
- [An update on recent Claude Code quality reports](https://www.anthropic.com/engineering/april-23-postmortem)
- [Anthropic cuts Claude subscribers off from OpenClaw in cost crackdown](https://thenextweb.com/news/anthropic-openclaw-claude-subscription-ban-cost)
- [Changes to GitHub Copilot Individual plans](https://github.blog/news-insights/company-news/changes-to-github-copilot-individual-plans/)
- [Anthropic's Claude Code pricing pain is Sam Altman's pleasure](https://www.businessinsider.com/anthropic-claude-code-price-confusion-sam-altman-2026-4)