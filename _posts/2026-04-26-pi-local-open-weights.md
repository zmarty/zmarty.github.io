---
layout: post
title: "Coding locally with Pi and Qwen3.6-27B, Gemma 4 31B, and MiniMax M2.7 open weights models (April 2026 edition)"
date: 2026-04-26 10:00:00 -0700
categories: [AI]
tags: [llm, local-inference, qwen, gemma, minimax, coding-agents]
description: "Run Qwen3.6-27B, Gemma 4 31B, and MiniMax M2.7 locally, then connect them to the Pi coding agent for local coding."
---

# Coding locally with open weights models (April 2026 edition)

Working draft.

## Why this update

<img width="3236" height="1804" alt="Coding Index (26 Apr &#39;26) (1)" src="https://github.com/user-attachments/assets/3ee1e3f4-18b4-44b4-ada9-efe7abe1c134" />


## Hardware assumptions

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

## References
