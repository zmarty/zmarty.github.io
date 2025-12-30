---
layout: post
title: "The age of hyper-personalized software"
date: 2025-12-29 21:00:00 -0800
categories: [AI]
tags: [llm]
description: "Why I run local LLMs to power a multimodal event crawler"
---

People often ask me: why run local LLMs when cloud providers are so cheap? That's true, until you're generating millions of tokens regularly. Local models can also be faster. Claude and GPT sometimes grind to a halt during peak loads, while [my home server](https://www.ovidiudan.com/2025/12/25/dual-rtx-pro-6000-llm-guide.html) delivers consistent throughput on dedicated hardware.

I use Claude Opus to write hyper-personalized software. These applications often call local LLMs as part of their pipelines. This is the kind of software that would have been impractical to build a few years ago. Now I can throw together a custom tool over a weekend that does exactly what I need.

Case in point: I've been building a multimodal agent that crawls local websites to find events happening around me. The crawler calls GLM-4.6V running on vLLM to process text and images.

## What the LLM actually does

The crawler uses [GLM-4.6V](https://huggingface.co/zai-org/GLM-4.6V), a 106 billion parameter model, for five distinct tasks:

### 1. Extracting information from event flyers

This is where multimodal models shine. [Here](https://whidbeycamanoislands.com/event/the-dead-guise-new-years-eve/) is an event where the text description doesn't mention the price, but the flyer image above it does. The LLM reads the flyer and extracts "$25" into a structured field.

<img width="800" height="744" alt="image" src="https://github.com/user-attachments/assets/2e66f3ab-ee5d-42a4-b903-01ad901879d0" />

The model also extracts venue names, performer lineups, age restrictions, and registration requirements from a combination of the raw HTML description and the accompanying image.

### 2. Rewriting messy descriptions

Scraped event descriptions are often a mess: HTML artifacts, escaped characters, inconsistent formatting. The LLM rewrites these into clean, readable paragraphs while preserving the essential information. The prompt instructs it to omit details shown elsewhere in the UI (venue, date, price) to avoid redundancy.

### 3. Link classification

Rather than using fragile regex patterns to find ticket links, the LLM analyzes all links on a page and identifies:
- The primary registration URL (not the "Buy Tickets" link for a different event in the sidebar)
- Relevant supplementary links (venue website, accessibility info, parking)

This works better than pattern matching because the model understands context.

### 4. Cross-source deduplication

The same event often appears on multiple websites. When a new event comes in, the LLM compares it against existing events with similar dates and determines whether it's a duplicate. It understands that "NYE Party at The Clyde" and "New Year's Eve Celebration - Clyde Theatre" are the same event, while two different concerts on different nights are not.

### 5. Multi-event extraction

Some sources publish newsletter images or event roundups containing multiple events. The LLM extracts each event separately, parsing dates, venues, and descriptions from a single composite image.

## Running it locally

nvtop showing GPU utilization while the model processes events:

<img width="1000" height="502" alt="GPU usage during inference" src="https://github.com/user-attachments/assets/04e36e0c-2e53-4118-8beb-a06e3af92742" />

vLLM server logs showing request throughput:

<img width="1000" height="502" alt="vLLM server logs" src="https://github.com/user-attachments/assets/7069a788-085a-4b97-a4f6-a97781d1946b" />
