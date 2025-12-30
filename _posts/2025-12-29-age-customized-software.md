---
layout: post
title: "The age of hyper-personalized software"
date: 2025-12-29 21:00:00 -0800
categories: [AI]
tags: [llm]
description: ""
---

People often ask me: why do you run local LLMs, the cloud providers are so cheap! That's all fine, until you get to generating millions upon millions of tokens regularly. Local LLMs can also be faster. Sometimes Claude and GPT grind to a halt during peak loads, whereas at home I have dedicated hardware / same throughput.

I use Claude Opus to write hyper-personalized software, which often requires (lesser) LLMs in their software. Case in point, I have been writing a multimodal agent which crawls and parses several local websites to find the events around me. For the intelligence part I use GLM 4.6V, which allows processing both text, images, and videos.

[Here](https://whidbeycamanoislands.com/event/the-dead-guise-new-years-eve/) is an example of an event where the text description does not contain the price for the event, but through the magic of multimodal LLMs my agent parses the price from the image flyer just above it.

[I will insert a screenshot here]

nvtop showing GPU usage while the model runs:

<img width="3687" height="1851" alt="image" src="https://github.com/user-attachments/assets/04e36e0c-2e53-4118-8beb-a06e3af92742" />

vLLM logs example: 

<img width="3687" height="1851" alt="image" src="https://github.com/user-attachments/assets/7069a788-085a-4b97-a4f6-a97781d1946b" />
