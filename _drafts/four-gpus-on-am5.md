---
layout: post
title: "Four GPUs on AM5, and the PCIe traffic that took the long way"
categories: [AI, Hardware]
tags: [gpu, pcie, switch, pex88096, p2p, nccl, llm, vllm, am5]
description: "I built the four-GPU expansion box. Peer copies worked on every pair, but a PCIe routing change took four-GPU all-reduce from 5.53 to 21.50 GB/s."
---

In September I wrote about [connecting four GPUs to my AM5 workstation]({% post_url 2026-09-17-gpu-expansion-box-pcie-switch %}) with a PCIe switch. The box now has four RTX PRO 6000 Blackwell cards, with 96 GB of VRAM each. Every pair of cards could move data at around 27 GB/s. Then I ran an all-reduce across all four and got 5.53 GB/s of effective bus bandwidth.

I had bought the switch so the GPUs could talk to each other without going back through the motherboard. It turns out the switch could still send their traffic there. After changing two routing settings on its ports, the same all-reduce reached 21.50 GB/s, and GLM-5.3-Flash processed a 32K-token prompt in 6.4 seconds instead of 15.2.

## What I ended up with

I kept the Ryzen 9 7950X3D and the ASUS ROG Crosshair X670E Hero. The host adapter sits in the motherboard's top x16 slot and connects to the Broadcom PEX88096 backplane in the expansion box. The two cables carry eight lanes apiece and together form one Gen4 x16 uplink. Each GPU has its own Gen4 x16 link into the switch fabric. I verified those links under load because the GPU links can downshift at idle.

![The AM5 root port connects through a host adapter and two eight-lane cables to a PEX88096 backplane. The cables form one Gen4 x16 uplink shared by four GPUs, each with its own Gen4 x16 link.]({{ '/assets/img/pcie-switch-follow-up/topology.svg' | relative_url }})

_The GPU-side links are simplified here. The backplane exposes several PCIe bridges. Diagram style inspired by [Doug's RTX 3090 splitter post](https://doug.sh/posts/pcie-splitter-rtx-3090/)._

Before the box, I had two cards in the motherboard's two graphics slots, which are wired for Gen5 x8 each in that configuration. Gen5 x8 and Gen4 x16 both work out to about 31.5 GB/s per direction before packet overhead. Each of the four cards now has that link budget to the switch. When they all need to talk to the host, they share one link with the same 31.5 GB/s budget. The reason to build this at all was to keep the repeated GPU-to-GPU transfers inside the box.

Getting all four GPUs to show up took some work. An experiment with the cards' compute profile exposed a 128 GiB framebuffer aperture and additional virtual-function reservations per card, and I ran into PCIe resource allocation problems. I got all four running with Above 4G Decoding on, the motherboard's Resizable BAR toggle off, and the cards in the 8 GiB graphics profile. That 8 GiB is the size of BAR1, a PCIe address window into GPU memory. Each card still has its 96 GB of VRAM.

## Peer-to-peer was already working

I checked all six GPU pairs in both directions with CUDA copies and kernels that read and wrote peer memory directly. All 36 data checks passed. The copies did 27.1–27.6 GB/s, one pair at a time. Had I stopped there, I would have thought I was done.

For the workload I care about, a single pair is only part of the picture. In tensor parallel inference, the GPUs each compute part of the model and repeatedly exchange partial results. An all-reduce combines those results and makes the combined result available to each GPU. When four GPUs do that together, their transfers can compete for a shared link even if a copy between two GPUs has plenty of bandwidth to itself.

The clue was in the PCIe Access Control Services (ACS) settings. The switch ports had both peer request and completion redirection enabled. Those controls can send peer transactions upstream to the CPU's PCIe root complex before they come back down to the destination GPU. NCCL reported the same `P2P/CUMEM` transport before and after I changed the settings. That told me how NCCL was accessing peer memory. The port settings explained why those accesses could still take the longer route. [NVIDIA's NCCL troubleshooting guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html#pci-access-control-services-acs)

![Two simplified routes between GPUs on the same backplane. With ACS redirection on, peer traffic can go up to the CPU root complex and back down. With redirection off, the switch can forward it locally.]({{ '/assets/img/pcie-switch-follow-up/peer-routing.svg' | relative_url }})

_The arrows show what the routing controls allow. I inferred the detour from the port settings and the benchmark, without tracing individual PCIe packets._

For the test I turned off just request and completion redirection on seven ports in the GPU switch fabric. I left the CPU root port and the 8 GiB BAR1 configuration alone. The change was made at runtime and was never set up to run at boot.

## What changed in vLLM

I ran the NCCL test immediately before and after the switch change, with the model unloaded. Then I compared inference runs from the same day with the same GLM-5.3-Flash NVFP4 Spark checkpoint, vLLM image, and four-way tensor parallel configuration. The prefill tests used fresh prompts, so the server had to process every input token.

| Test | Redirection on | Redirection off |
| :--- | ---: | ---: |
| Four-GPU NCCL all-reduce, 16 MiB per GPU | 5.53 GB/s | 21.50 GB/s |
| Prefill, 8,192 input tokens | 2,899 tok/s | 5,151 tok/s |
| Prefill, 32,768 input tokens | 2,159 tok/s | 5,099 tok/s |
| Generation, one request, 2,048 output tokens | 218 tok/s | 288 tok/s |
| Generation, four concurrent requests, 2,048 output tokens each | 425.3 tok/s total | 599.5 tok/s total |

The NCCL numbers are medians from three runs per setting. Its effective bus bandwidth accounts for the amount of data exchanged by the four ranks, so it measures something different from the speed of a copy between two cards. The collective went from 4.55 ms to 1.17 ms. At 16 KiB, the time barely moved, from 14.3 to 13.4 microseconds.

On a 32K prompt, prefill went from 15.2 seconds to 6.4 seconds. That's server-measured time spent processing the prompt. Generation improved by about 32% on the single-request 2,048-token test. The inference runs were several hours apart, with three samples per prefill length and per single-request generation test. The generation tests used short prompts and timed the full request. Output text and speculative decoding acceptance can vary between runs, so I put more weight on the immediately repeated NCCL result.

## One reboot later

On a later normal reboot, Linux could still see the switch, but all four GPUs were missing from the PCIe device list. The uplink had come up at Gen1 x4. A complete shutdown and restart brought back all four cards and the Gen4 x16 uplink. I still don't know what caused that boot failure. The recovery confirmed that the cards were visible again. I didn't verify the runtime ACS settings or rerun the benchmarks after that restart.

In later tests, increasing vLLM's prefill chunk budget and gather limit got the same 32K workload to about 9,818 input tokens a second. That took a separate round of testing, which I'll cover in a later post.
