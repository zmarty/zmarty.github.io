---
layout: post
title: "Playing games across Puget Sound"
date: 2026-10-27 22:05:00 -0700
categories: [Hardware, Gaming]
tags: [game-streaming, pyrowave, punktfunk, gpu, networking, wireguard]
description: "Streaming games from my AI workstation 40 miles away at 4K60 and around 558 Mbps, across two ISPs and an encrypted WireGuard tunnel."
---

I am playing a game rendered by a GPU 40 miles away and streamed to me as live video at around 558 Mbps. The fiber travels underneath the Puget Sound waterways, and the route eventually includes coaxial links, across two ISPs. All of this to get a game onto the screen in front of me. At 4K UHD and 60 Hz.

I finally found a use case for having low latency multi-gigabit Internet :)

<video controls muted loop playsinline preload="none" poster="/assets/img/game-streaming/uncharted-intro.jpg" width="1920" height="1080" style="width: 100%; height: auto;" aria-label="Uncharted streamed from my remote workstation, with streaming statistics visible">
  <source src="/assets/video/game-streaming/uncharted-intro.webm" type="video/webm">
  <a href="/assets/video/game-streaming/uncharted-intro.webm">Watch the streamed game intro.</a>
</video>

The gaming machine is my [dual RTX Pro 6000 AI workstation](/2025/12/25/dual-rtx-pro-6000-llm-guide.html). Hardware prices have gone completely insane, and I already own these GPUs, so buying another machine for gaming doesn't make much sense. The AI machine gets to do dual duty. I free up the streaming GPU before starting the game.

I've tried Sunshine and Polaris, and I'm currently playing with [Punktfunk](https://docs.punktfunk.unom.io/docs/pyrowave) and [PyroWave](https://github.com/Themaister/pyrowave), the codec Valve recently [added to Steam Remote Play](https://steamcommunity.com/groups/homestream/discussions/0/564794422009744473). It is intra-only, meaning every frame is encoded independently, and uses GPU compute for very low encode and decode latency. It supports HDR and full 4:4:4 color. The trade-off is bandwidth, which I happen to have. Punktfunk accepts bitrates up to 8 Gbps, well beyond what I'm sending here.

The two sites are connected through a WireGuard VPN, so the traffic is encrypted even at these speeds. Here's the stats overlay from a gameplay clip:

![Streaming statistics showing 3840x2160 at 60 Hz with PyroWave and 4:4:4 color](/assets/img/game-streaming/uncharted-gameplay-stats.gif)

And a few more clips of Uncharted, including driving around after I switched the client to SDR. These screen recordings are scaled to 1080p; the game stream itself is 4K.

<video controls muted loop playsinline preload="none" poster="/assets/img/game-streaming/uncharted-menu.jpg" width="1920" height="1080" style="width: 100%; height: auto;" aria-label="Two seconds of the streamed Uncharted menu">
  <source src="/assets/video/game-streaming/uncharted-menu.webm" type="video/webm">
  <a href="/assets/video/game-streaming/uncharted-menu.webm">Watch the streamed game menu.</a>
</video>

<video controls muted loop playsinline preload="none" poster="/assets/img/game-streaming/uncharted-driving.jpg" width="1920" height="1080" style="width: 100%; height: auto;" aria-label="Two seconds of driving in Uncharted streamed in SDR">
  <source src="/assets/video/game-streaming/uncharted-driving.webm" type="video/webm">
  <a href="/assets/video/game-streaming/uncharted-driving.webm">Watch the driving clip.</a>
</video>

<video controls muted loop playsinline preload="none" poster="/assets/img/game-streaming/uncharted-gameplay.jpg" width="1920" height="1080" style="width: 100%; height: auto;" aria-label="Five seconds of Uncharted gameplay streamed across Puget Sound">
  <source src="/assets/video/game-streaming/uncharted-gameplay.webm" type="video/webm">
  <a href="/assets/video/game-streaming/uncharted-gameplay.webm">Watch the gameplay clip.</a>
</video>
