---
layout: post
title: "My personal LLM toolchain: switch between models and harnesses, control them from your phone"
date: 2026-08-27 13:30:00 -0700
categories: [AI]
tags: [llm, coding-agents, pi, local-inference, agents, paseo]
description: "Paseo to run and watch agents from any device, Pi as the harness, and four extensions (web search via Gemini, Chrome, sudo, sub-agents) that make local models useful."
---

My personal LLM toolchain lets me switch between models and harnesses and talk to all of them from a single mobile app. This makes it possible to code on the go on my own machine back home. My setup is made up of [Paseo](https://github.com/getpaseo/paseo), which puts one interface in front of Claude Code, Codex, Copilot, OpenCode and the rest, and [Pi](https://pi.dev/), a minimal and extensible harness. I have four extensions installed in Pi, and the most important is `pi-web-access`, configured to search the web through Gemini / Google (5,000 free searches a month). This article shows you how to set up your own version.

## Paseo

A daemon on your machine owns the agents. Every client (phone, desktop, web, terminal) connects to it, so an agent I start at my desk is still mine to check from the couch. From the phone I can reach my agents over the relay or over Tailscale, and both are encrypted. The relay is zero setup because the daemon makes an outbound connection, the phone meets it there, and it's encrypted end to end so the relay never sees my code. Tailscale is just an encrypted tunnel from the phone to the daemon machine. Neither one needs port forwarding.

It runs on macOS, Linux and Windows, there's a Docker image if you'd rather containerize it, and the desktop app starts a daemon by itself. Tailscale sets up an encrypted tunnel between my phone and the computer where the harnesses run, so  there's no port forwarding to configure.

Paseo speaks to Claude Code, Codex, Copilot, OpenCode and Pi through the same interface, and I pick the provider and the model per agent. Today the other four are switched off in `~/.paseo/config.json` and I only run Pi, but this flexibility allows me to use another harness on the future.

<img width="400" height="398" alt="image" src="https://github.com/user-attachments/assets/0f870418-6b52-4b2c-ae75-9e0cfbea46f5" />

## Pi

Pi [installs as one npm package](https://pi.dev/docs/latest/quickstart) and its core is small on purpose: read, bash, edit, write, plus extensions.

Models come from `~/.pi/agent/models.json`, and a local server is just another provider, so anything OpenAI-compatible drops in. vLLM, SGLang, LM Studio Bionic, a box on my LAN or a server somewhere else:

```json
{
  "providers": {
    "vllm": {
      "baseUrl": "http://10.0.10.213:8000/v1",
      "api": "openai-completions",
      "apiKey": "vllm",
      "models": [ { "id": "qwen3.8-flash-next", "contextWindow": 262144 } ]
    }
  }
}
```

`enabledModels` in `~/.pi/agent/settings.json` lists what I choose from. Mine holds nine, five served on my own hardware and four from cloud providers, and Ctrl+P cycles them inside a running session. When a local model starts re-reading a file it already read, I change models in place instead of restarting the task somewhere else.

<img width="300" height="667" alt="image" src="https://github.com/user-attachments/assets/7ee471eb-9f17-4844-af5f-899a4f801d0c" />

## Four extensions give Pi super powers

On its own a local model writes plausible code against APIs from two years ago. Each of these is `pi install npm:<name>`.

- `pi-web-access` gives Pi `web_search`, `fetch_content` and video understanding. I run it on a Gemini API key, and Google's free allowance is 5,000 grounded prompts a month, then $14 per 1,000 queries, which is well past what I burn. This is the super power: it's what lets a model on my own hardware answer "what's the current flag for this" instead of inventing one.
- `pi-mcp-adapter` puts MCP servers behind one proxy tool, about 200 tokens of tool surface instead of hundreds, and starts them lazily. I use it for Chrome through `chrome-devtools-mcp`, so "load this at 390 px and tell me where the table overflows" ends with the agent driving a real browser and reading its own screenshot. A model that can't see what it rendered is guessing at front-end work.
- `@xynogen/pix-sudo` adds a `sudo_run` tool that asks me every time, with a 60-second auto-deny if I don't answer. The password goes to `sudo -S` on stdin and is never stored, and the tool refuses outright when nobody's watching the terminal. Giving an agent root was the decision I sat on longest.
- `pi-subagents` spawns child Pi sessions with their own context, in parallel. It's mostly a context-budget trick: a reviewer's 40,000 tokens of file reading don't belong in the session where I'm editing. I run one for correctness, one for tests, one for needless complexity.

<img width="300" height="667" alt="image" src="https://github.com/user-attachments/assets/ede3c7ac-b3f9-4f15-afe6-12d185b5a86e" />

## References

- [Paseo](https://github.com/getpaseo/paseo), [voice docs](https://github.com/getpaseo/paseo/blob/main/public-docs/voice.md), [connectivity docs](https://github.com/getpaseo/paseo/blob/main/public-docs/connectivity.md)
- [Pi](https://pi.dev/) and [source](https://github.com/earendil-works/pi)
- [pi-web-access](https://github.com/nicobailon/pi-web-access), [pi-mcp-adapter](https://github.com/nicobailon/pi-mcp-adapter), [pi-subagents](https://github.com/nicobailon/pi-subagents)
- [pix-mono (@xynogen/pix-sudo)](https://github.com/xynogen/pix-mono)
- [Gemini API pricing: Grounding with Google Search](https://ai.google.dev/gemini-api/docs/pricing)
