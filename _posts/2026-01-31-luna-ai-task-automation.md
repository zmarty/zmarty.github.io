---
layout: post
title: "Luna: An AI Assistant That Works While I Sleep"
date: 2026-01-31 12:00:00 -0800
categories: [AI]
tags: [llm, vllm, agents]
description: "Luna monitors, follows up, and takes action on her own—powered by a local LLM and a team of background agents."
---

Luna is an always-on AI assistant built for one user: me. She keeps her own memory updated, searches the web, analyzes YouTube videos, runs Python code in a sandbox, manages a team of background agents that check on things while I'm away, and texts me via Signal when something needs my attention. She runs entirely locally on the [MiniMax M2.1](https://huggingface.co/mratsim/MiniMax-M2.1-BF16-INT4-AWQ) model on [my AI workstation](https://www.ovidiudan.com/2025/12/25/dual-rtx-pro-6000-llm-guide.html).

Luna builds up knowledge about me over time as we talk. She stores documents like "User Profile", "Whidbey Island Favorite Places", and "Entertainment Preferences". She knows my family members' names and birthdays, my favorite restaurant, what video games I like. When I mention something new, she decides whether to create a new memory or update an existing one. That's how she knew which restaurant I meant and could look up its current rating.

<img width="500" height="121" alt="image" src="https://github.com/user-attachments/assets/6c9003eb-30db-4097-88ba-8e700222dfad" alt="I asked about my favorite restaurant. Luna remembered which one, looked up its current rating, and answered - without me having to specify." />

## Background Tasks

Ask Luna to "check something every month" and she creates a task that runs on its own:

```python
schedule_background_task(
    interval="30 days",
    task="Search for new/upcoming theater musicals at Village Theatre, 
          Paramount Theatre, and 5th Avenue Theatre in the Seattle area. 
          Compile a report with show titles, dates, ticket links, 
          and brief descriptions."
)
```

Tasks persist across restarts. If Luna goes down and comes back up, she picks up where she left off. If a task was supposed to run while she was offline, it runs once on startup—no spam of catch-up notifications.

Each task agent remembers what it already told me. When it runs again, it compares new findings against that history and only reports the diff—genuinely new information. If nothing changed, it stays quiet. Nobody wants "I checked and found nothing" every hour.

## Tools

Luna has access to several tools, and background tasks can use them too.

**Web Search** — Uses Gemini's grounding to search the web and return answers with source URLs.

**YouTube Analysis** — Gemini ingests video audio and visuals. Ask Luna to summarize a video, answer questions about specific moments, or extract names and products mentioned.

**Maps Search** — Location-aware queries. "Best ramen near me" returns places with descriptions and Google Maps links.

**Python Sandbox** — Sandboxed code execution via Anthropic's runtime. Numpy, pandas, matplotlib, scipy available. No network access.

**Persistent Memory** — Stores and updates documents about me over time, as described above.

## Signal Integration

Luna can text me:

```bash
python main.py --signal --signal-phone +14155551234
```

When I create a task via Signal, Luna saves my phone number. Hours later when the task finds something, she texts me directly.

## Example: Theater Monitoring

Every month, Luna checks what's coming up at my local theaters and sends me a summary via Signal:

```
Luna: Here's your theater report! A few highlights:

Hot right now:
• 9 to 5 at Village Theatre (Everett) - running until Feb 8
• Spamalot at 5th Avenue - Feb 4-15
• A Beautiful Noise (Neil Diamond) at Paramount - Jan 13-18

Coming soon:
• The Wiz at Paramount (Feb 10-15) - the new touring production
• Grease at Village Theatre Issaquah (Mar 24 - May 3)

Big ones to watch:
• Les Misérables at Paramount (Apr 7-19)
• Phantom of the Opera at Paramount (May 13-24)
```

The task ran in the background, searched the web, compiled the report, and texted me. I didn't have to remember to check.

## The Stack

- **FastAPI** — Python web framework exposing OpenAI-compatible endpoints
- **[vLLM](https://github.com/vllm-project/vllm)** — Local LLM inference backend running MiniMax M2.1
- **[Gemini APIs](https://ai.google.dev/gemini-api/docs/google-search)** — Powers web search, Maps search, and YouTube analysis via grounding
- **[Anthropic Sandbox Runtime](https://github.com/anthropic-experimental/sandbox-runtime)** — Isolated Python execution with numpy, pandas, matplotlib, scipy
- **[Signal CLI REST API](https://github.com/bbernhard/signal-cli-rest-api)** — Mobile messaging via Docker
- **MCP servers** — Extensible tool access (filesystem, etc.)
- **JSON files** — Simple persistence for tasks and memory

## Why Local?

Running against a local LLM means background tasks can burn through millions of tokens without API costs adding up. A task checking something every 5 minutes for a month is a lot of inference. With [local hardware](https://www.ovidiudan.com/2025/12/25/dual-rtx-pro-6000-llm-guide.html), the marginal cost is zero.

More importantly, I control everything. The orchestrator logic, task scheduling, memory system, tool integrations—all of it is code I can change. When I want Luna to behave differently, I edit the prompts or add tools. No waiting for a provider to ship a feature.

The goal is an assistant that handles ongoing responsibilities, not just one-off questions.
