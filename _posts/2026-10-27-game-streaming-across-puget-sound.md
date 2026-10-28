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

I am surprised by how well it works. I move the controller, and the input goes over Bluetooth to my local Mini PC, through the switches on my local network. A UDM Pro encrypts the traffic into a VPN tunnel across the Internet. The UDM Pro at the other end decrypts it and forwards it through the switches at the remote site to the gaming server. The character moves, the GPU renders the next frames, and the encoded video makes the trip back, including encryption and decryption again, before my Mini PC decodes and displays it. Yet the game feels very responsive.

The overlay shows around 15 ms of network round-trip time through that tunnel, including the gateway and VPN processing along the way. That isn't the full controller-to-screen latency: Bluetooth, game processing, rendering, decoding and display timing add to it. I haven't measured that full response time separately.

<video controls muted loop playsinline preload="none" poster="/assets/img/game-streaming/uncharted-intro.jpg" width="1920" height="1080" style="width: 100%; height: auto;" aria-label="Uncharted intro streamed from my remote workstation">
  <source src="/assets/video/game-streaming/uncharted-intro.webm" type="video/webm">
  <a href="/assets/video/game-streaming/uncharted-intro.webm">Watch the video.</a>
</video>

<div class="streaming-video-preview">
  <a href="/assets/video/game-streaming/uncharted-intro.webm"><img loading="lazy" src="/assets/img/game-streaming/uncharted-intro.gif" alt="Uncharted intro streamed from my remote workstation" /></a>
</div>

The gaming machine is my [dual RTX Pro 6000 AI workstation](/2025/12/25/dual-rtx-pro-6000-llm-guide.html). With hardware prices where they are, buying another machine for gaming doesn't make much sense when I already own these GPUs. The AI machine gets to do dual duty. I free up the streaming GPU before starting the game.

I've tried [Sunshine](https://github.com/LizardByte/Sunshine) and [Polaris](https://github.com/papi-ux/polaris), and I'm currently playing with [Punktfunk](https://docs.punktfunk.unom.io/docs/pyrowave) and [PyroWave](https://github.com/Themaister/pyrowave), the codec Valve recently [added to Steam Remote Play](https://steamcommunity.com/groups/homestream/discussions/0/564794422009744473). It is intra-only, meaning every frame is encoded independently, and uses GPU compute for very low encode and decode latency. It supports HDR and full 4:4:4 color. The trade-off is bandwidth, which I happen to have. Punktfunk accepts bitrates up to 8 Gbps, well beyond what I'm sending here.

The two UDM Pros use [UniFi Site Magic](https://www.ui.com/trust-center), which builds [WireGuard tunnels over UDP](https://www.wireguard.com/). The traffic stays encrypted between the gateways even at these speeds. Here's the stats overlay from a gameplay clip:

<img src="/assets/img/game-streaming/uncharted-driving-stats.gif" width="500" height="125" style="max-width: 100%; height: auto;" alt="PyroWave stream at 3840x2160 and 60 Hz, with 60 received, decoded and presented fps, 0.0% lost and 0.0% skipped frames" />

Here is a gameplay example from Uncharted after I switched the client to SDR. The screen recording is 1080p at 30 fps; the game stream itself is 4K at 60 fps.

<video controls muted loop playsinline preload="none" poster="/assets/img/game-streaming/uncharted-gameplay.jpg" width="1920" height="1080" style="width: 100%; height: auto;" aria-label="Uncharted gameplay streamed across Puget Sound">
  <source src="/assets/video/game-streaming/uncharted-gameplay.webm" type="video/webm">
  <a href="/assets/video/game-streaming/uncharted-gameplay.webm">Watch the video.</a>
</video>

<div class="streaming-video-preview">
  <a href="/assets/video/game-streaming/uncharted-gameplay.webm"><img loading="lazy" src="/assets/img/game-streaming/uncharted-gameplay.gif" width="960" height="540" alt="Uncharted gameplay streamed across Puget Sound" /></a>
</div>

## Getting past the frozen video

It took some debugging to get here. Punktfunk 0.45.0 initially gave me audio with no picture, or a frozen image while the sound kept going. An AI coding assistant helped trace two buffer problems: the startup bandwidth test could use up the decoder's buffers, and larger 4K frames could overflow a temporary buffer when packets needed extra alignment copies.

We applied a local client hotfix to keep bandwidth tests out of the decoder's buffer pool, and changed the host's packet-size setting to avoid the alignment problem. A broader buffer-sizing correction was tested separately but wasn't installed. The working setup uses the hotfix and packet-size workaround; the bugs were in Punktfunk's integration with PyroWave.
