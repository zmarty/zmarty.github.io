---
layout: post
title: "Fixing RTX Pro 6000 Blackwell shutdowns with custom fan control"
date: 2026-01-17 08:40:00 -0800
categories: [AI]
tags: [llm, hardware]
description: "Unexpected shutdowns under sustained load on RTX Pro 6000 Blackwell: fix with a small NVML fan control daemon + systemd."
---

My dual RTX Pro 6000 Blackwell workstation would hard power off under sustained inference load. The fix ended up being simple: take control of the fan curve.

## Hardware

```text
CPU: AMD Ryzen 9 7950X3D 16-Core Processor
Motherboard: ROG CROSSHAIR X670E HERO
GPU: Dual NVIDIA RTX Pro 6000 (96 GB VRAM each)
RAM: 192 GB DDR5 5200
```

## Symptom

During long-running jobs the machine would abruptly lose power (no graceful shutdown). The default fan curve is tuned for low noise and is slow to ramp under sustained high power draw.

## What I changed

I run a small Python daemon that uses NVML (NVIDIA Management Library) to set fan speed based on temperature:

- Poll temperature once per second
- Apply an aggressive curve that still idles quietly
- Run as a `systemd` service so it starts on boot

The curve I’m using:

| Temperature | Fan speed |
|-------------|-----------|
| ≤40°C       | 30%       |
| 50°C        | 55%       |
| 55°C        | 75%       |
| 60°C        | 90%       |
| ≥65°C       | 100%      |

Once the GPUs approach 60–65°C the fans are already high, which keeps them away from the thermal cliff.

## Install

Tested on Ubuntu 24.04.

```console
git clone https://github.com/zmarty/nvidia-fan-control
cd nvidia-fan-control

sudo apt install python3-pynvml

sudo mkdir -p /opt/nvidia-fan-control
sudo cp nvidia-fan-control.py /opt/nvidia-fan-control/
sudo chmod +x /opt/nvidia-fan-control/nvidia-fan-control.py

sudo cp nvidia-fan-control.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable nvidia-fan-control.service
sudo systemctl start nvidia-fan-control.service
```

Logs:

```console
journalctl -u nvidia-fan-control -f
```

Code: **[github.com/zmarty/nvidia-fan-control](https://github.com/zmarty/nvidia-fan-control)**

## Validation

I load-tested with a local vLLM server and a small client that sends 50 concurrent requests in a loop. With the custom curve in place, the workstation stays up under sustained load.
