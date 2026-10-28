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

<video controls muted loop playsinline preload="none" poster="/assets/img/game-streaming/uncharted-intro.jpg" width="1920" height="1080" style="width: 100%; height: auto;" aria-label="Uncharted intro streamed from my remote workstation">
  <source src="/assets/video/game-streaming/uncharted-intro.webm" type="video/webm">
  <a href="/assets/video/game-streaming/uncharted-intro.webm">Watch the video.</a>
</video>

<div class="streaming-video-preview">
  <a href="/assets/video/game-streaming/uncharted-intro.webm"><img loading="lazy" src="/assets/img/game-streaming/uncharted-intro.gif" alt="Uncharted intro streamed from my remote workstation" /></a>
</div>

The gaming machine is my [dual RTX Pro 6000 AI workstation](/2025/12/25/dual-rtx-pro-6000-llm-guide.html). Hardware prices have gone completely insane, and I already own these GPUs, so buying another machine for gaming doesn't make much sense. The AI machine gets to do dual duty. I free up the streaming GPU before starting the game.

I've tried [Sunshine](https://github.com/LizardByte/Sunshine) and [Polaris](https://github.com/papi-ux/polaris), and I'm currently playing with [Punktfunk](https://docs.punktfunk.unom.io/docs/pyrowave) and [PyroWave](https://github.com/Themaister/pyrowave), the codec Valve recently [added to Steam Remote Play](https://steamcommunity.com/groups/homestream/discussions/0/564794422009744473). It is intra-only, meaning every frame is encoded independently, and uses GPU compute for very low encode and decode latency. It supports HDR and full 4:4:4 color. The trade-off is bandwidth, which I happen to have. Punktfunk accepts bitrates up to 8 Gbps, well beyond what I'm sending here.

The two sites are connected through a WireGuard VPN, so the traffic is encrypted even at these speeds. Here's the stats overlay from a gameplay clip:

<img src="/assets/img/game-streaming/uncharted-driving-stats.gif" width="500" height="125" style="max-width: 100%; height: auto;" alt="PyroWave stream at 3840x2160 and 60 Hz, with 60 received, decoded and presented fps, 0.0% lost and 0.0% skipped frames" />

Here are a few more clips of Uncharted, including driving around after I switched the client to SDR. These screen recordings are 1080p at 30 fps; the game stream itself is 4K at 60 fps.

<video controls muted loop playsinline preload="none" poster="/assets/img/game-streaming/uncharted-menu.jpg" width="1920" height="1080" style="width: 100%; height: auto;" aria-label="Streamed Uncharted menu">
  <source src="/assets/video/game-streaming/uncharted-menu.webm" type="video/webm">
  <a href="/assets/video/game-streaming/uncharted-menu.webm">Watch the video.</a>
</video>

<div class="streaming-video-preview">
  <a href="/assets/video/game-streaming/uncharted-menu.webm"><img loading="lazy" src="/assets/img/game-streaming/uncharted-menu.gif" alt="Streamed Uncharted menu" /></a>
</div>

<video controls muted loop playsinline preload="none" poster="/assets/img/game-streaming/uncharted-driving.jpg" width="1920" height="1080" style="width: 100%; height: auto;" aria-label="Driving in Uncharted over the remote stream">
  <source src="/assets/video/game-streaming/uncharted-driving.webm" type="video/webm">
  <a href="/assets/video/game-streaming/uncharted-driving.webm">Watch the video.</a>
</video>

<div class="streaming-video-preview">
  <a href="/assets/video/game-streaming/uncharted-driving.webm"><img loading="lazy" src="/assets/img/game-streaming/uncharted-driving.gif" alt="Driving in Uncharted over the remote stream" /></a>
</div>

<video controls muted loop playsinline preload="none" poster="/assets/img/game-streaming/uncharted-gameplay.jpg" width="1920" height="1080" style="width: 100%; height: auto;" aria-label="Uncharted gameplay streamed across Puget Sound">
  <source src="/assets/video/game-streaming/uncharted-gameplay.webm" type="video/webm">
  <a href="/assets/video/game-streaming/uncharted-gameplay.webm">Watch the video.</a>
</video>

<div class="streaming-video-preview">
  <a href="/assets/video/game-streaming/uncharted-gameplay.webm"><img loading="lazy" src="/assets/img/game-streaming/uncharted-gameplay.gif" alt="Uncharted gameplay streamed across Puget Sound" /></a>
</div>

## Getting past the frozen video

Getting here involved a bit of debugging. With Punktfunk 0.45.0, I initially got audio with no picture, or a few frames followed by a frozen image while the audio kept going. An AI coding assistant helped trace two separate problems in Punktfunk's PyroWave integration.

The first was in the startup bandwidth test. Incomplete test frames could hold all eight of the decoder's payload buffers, leaving none for the actual video. A version-specific local client hotfix routed those probes to a separate heap-backed pool. After that, video could start reliably in the short tests.

The second showed up when larger 4K frames arrived. Our connection negotiated 1,322-byte packet windows. That size isn't divisible by four, so alternating packet bodies needed alignment copies into a fixed 128 KiB scratch buffer. Larger frames exhausted it, and the client kept showing the last good picture. On the host, we set `PUNKTFUNK_WIRE_MTU=1394`, which produced 1,320-byte windows in the tested session and avoided that alignment-copy pressure.

A source patch to grow the scratch buffer with the frame capacity passed an isolated parser test, but we didn't rebuild and install the full client with it. The working setup uses the local startup hotfix plus the packet-size workaround. These were integration bugs exposed by this setup; we didn't find a defect in the PyroWave codec itself.
