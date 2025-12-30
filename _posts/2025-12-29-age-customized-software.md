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

Case in point: I've been building a multimodal agent that crawls local websites to find events happening around me. The crawler calls GLM-4.6V running on vLLM to process text and images, stores the results in a Firebase backend, and surfaces them in a Flutter mobile app. The entire stack (crawler, backend, mobile app) was vibe coded with Claude. I didn't write a single line myself, just brainstormed specs and refined the output.

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

## The backend: Firebase and Firestore

After the crawler enriches events, they need to go somewhere. I spec'd out a Firebase backend with Claude that handles:

- **Firestore database**: Events are stored with all their extracted metadata: dates, venues, prices, tags, image URLs, occurrences for recurring events, and geo coordinates for map integration.
- **Cloud Functions (Gen2)**: TypeScript functions handle queries, pagination, and a scheduled job that updates `nextOccurrenceDate` fields so past occurrences drop off the feed automatically.
- **Cloud Storage**: Event flyers are re-uploaded to Firebase Storage so the app doesn't hotlink to external sites that might go down.

The data model supports recurring events properly, which most event systems get wrong. An event can have multiple discrete occurrences (like a band playing Friday and Saturday) or a continuous date range (like a gallery exhibition running for two months). The crawler detects which pattern applies and stores it accordingly.

## The mobile app: Flutter

The Flutter app consumes this data and presents it in a clean feed. Features include:

- **Tag-based filtering**: Events are auto-tagged by the LLM (family-friendly, 21+, free, outdoors, etc.) and users can filter the feed.
- **Infinite scroll**: Firestore pagination keeps the initial load fast while letting users scroll through hundreds of events.
- **Full flyer images**: Tapping an event shows the original flyer at full resolution with pinch-to-zoom, because sometimes you just need to read the fine print.
- **Map integration**: Events with geo coordinates can open in your maps app for directions.
- **YouTube embeds**: Some events have promo videos that play inline.

The entire app (screens, widgets, services, data models) was generated through iterative prompting. I'd describe what I wanted ("add a tag filter bar below the header that shows pills for each tag, selected tags should be highlighted"), Claude would generate the code, and I'd refine from there.

<img width="864" height="1922" alt="image" src="https://github.com/user-attachments/assets/f9f3dbd8-3519-42f4-8d35-aa773e7e6408" />

<img width="864" height="1922" alt="image" src="https://github.com/user-attachments/assets/c3b9ee19-6f8a-4736-8cc9-3b4e7f6240cb" />

<img width="864" height="1922" alt="image" src="https://github.com/user-attachments/assets/328791b7-9134-4ef4-bf33-8a65eda542b0" />

<img width="864" height="1922" alt="image" src="https://github.com/user-attachments/assets/fbd6a745-0cd6-484e-b8fa-ba6cd76100f1" />


## The split

There are two kinds of AI work happening here. Claude Opus in the cloud handles the coding: I describe what I want, iterate on the spec, and it writes the crawler, the backend, the mobile app. A heavier model that's good at creative problem-solving.

The local LLM handles the inference: parsing flyers, classifying links, deduplicating events. Tasks that need to run millions of times and require reasoning over unstructured data. A few years ago, some of this would have been practically impossible. OCR can read text from an image, but it can't understand that "$25" on a psychedelic Grateful Dead flyer is the ticket price and not a date or an address. That requires a model that actually understands what it's looking at.

Tools built for an audience of one, running on hardware you control.
