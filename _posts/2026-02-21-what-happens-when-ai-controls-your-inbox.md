---
layout: post
title: "What Happened When I Gave Luna Access to My Email"
date: 2026-02-21 12:00:00 -0800
categories: [AI]
tags: [llm, agents, email]
description: "I gave Luna access to my email. Here's what happened in a single day."
---

Luna, [my AI assistant](https://www.ovidiudan.com/2026/01/31/luna-ai-task-automation.html), could already search the web, run background tasks, and text me via Signal. I recently gave her the ability to search, read, and send emails on my behalf. We had a family vacation to Mexico coming up, and it turned into a good stress test.

## Planning the resort stay

We booked a resort through Costco Travel. I asked Luna which building would be best for a family with young kids—close to the pool, restaurants, activities. She searched for resort maps and forum posts and came back with a recommendation.

Then I told her to email the concierge and request that building. She found the concierge's email address on the resort website, pulled our reservation number and dates from the Costco Travel confirmation in my inbox, and sent the request. I didn't have to look up any of it.

## Restaurant reservations

The resort recommended making dinner reservations in advance. Luna sent emails for each night. She looked at our flight itinerary and picked the most casual restaurant for the first night as we'd be arriving late and wouldn't want anything formal.

She added allergy information for one of my kids on her own. I never mentioned it in this conversation, she knew from her long-term memory.

My birthday fell during the trip. She picked the fanciest restaurant for that night and also helped inquire about babysitting.

## Events and activities

Luna searched online for resort events during our dates and sent emails to reserve spots.

## Postponing a delivery

I had a recurring product delivery that would've arrived while we were gone. Luna found the upcoming shipment notification in my email and helped me push it back.

## Resort research

I asked Luna to find reviews for the resort and summarize them. She also looked up tips—which pool is quietest, whether to bring cash for tips, best time to get lounge chairs. Stuff that's scattered across Reddit threads and travel blogs.

## Newsletter cleanup

Separately from the trip, I asked Luna to go through my email and find newsletters I don't open. She looked at my history, flagged the ones I hadn't touched in months, and helped me unsubscribe. That had been on my to-do list for years.

## How it works

Luna connects to Gmail via OAuth2 with read, send, and modify scopes. The token is stored locally and refreshes silently.

When I ask Luna something email-related, the orchestrator doesn't handle it directly. It spawns a dedicated email subagent - a separate LLM agent loop running on the same local vLLM server with its own system prompt. The subagent has six tools: `list_emails`, `read_email`, `search_emails`, `send_email`, `reply_to_email`, and `save_email_to_file`. It also gets access to Python execution and the filesystem.

```
User message
  → Orchestrator (local vLLM)
    → email_assistant(request)
      → Email subagent (separate agent loop, same vLLM)
        → Gmail API calls
      → Returns summary to orchestrator
    → Orchestrator relays to user
```

The orchestrator never sees raw email content. The subagent processes everything and returns a natural-language summary. This keeps the orchestrator's context window clean and avoids dumping entire email threads into the conversation.

Luna runs on [my hardware](https://www.ovidiudan.com/2025/12/25/dual-rtx-pro-6000-llm-guide.html) and because she's been building up [memory about my family](https://www.ovidiudan.com/2026/01/31/luna-ai-task-automation.html) over time—names, birthdays, allergies, preferences—she can act on things I don't explicitly tell her.