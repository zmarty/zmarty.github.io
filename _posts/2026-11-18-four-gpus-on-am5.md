---
layout: post
title: "How I fit four RTX PRO 6000s in one AM5 machine"
categories: [AI, Hardware]
tags: [gpu, pcie, switch, pex88096, mcio, p2p, nccl, am5]
description: "The four-GPU AM5 build I actually assembled, from an MCIO switch card and separate GPU adapters to the PCIe settings that got the cards talking."
---

[The dream]({% post_url 2026-09-17-gpu-expansion-box-pcie-switch %}) is alive! After four months of iterating on the design and waiting around for packages, I finally built a four-GPU AI server with a whopping 384 GB of VRAM. The model I built it for is the original DeepSeek-V4.1-Flash checkpoint. It has 552 billion backbone parameters and is one of the most capable open-weight models I wanted to run at home. With the Engram lookup table on NVMe, it processed uncached 32K-token prompts at about 15,350 input tokens per second. It generated 235 output tokens per second on a single request and 678 tokens per second in total across eight concurrent requests. [DeepSeek model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)

My AM5 motherboard has two graphics slots, and a regular PC case doesn't have room to mount four cards this size with all their cables. I ordered custom cases from China, got a PCIe switch, and moved to a 240 V power supply. That last part is why I ran a dedicated circuit to the rack when I [upgraded my house electrical system]({% post_url 2026-08-08-upgrading-house-electrical-for-ai %}).

The original plan called for a separate box with all four GPUs plugged into a single switch backplane. I ended up with a switch card in the motherboard and a separate slot board for each GPU, connected over MCIO. That let me fit the whole machine in one chassis.

![Open rackmount case with four RTX PRO 6000s in pairs around the CPU cooler and two rows of fans on the front panel.]({{ '/assets/img/pcie-switch-follow-up/four-gpus-in-case.jpg' | relative_url }})

## The switch card and the GPU slots

I kept the Ryzen 9 7950X3D and the ASUS ROG Crosshair X670E Hero. The Broadcom PEX88096 switch card sits in the motherboard's top x16 slot. Its gold fingers are the upstream connection to the CPU. The connectors around the card take the MCIO cables that go out to the GPUs. The little fan on the switch card cools the switch chip itself.

![The blue PCIe switch card on its packing foam, with a gold edge connector, a small blower, and MCIO sockets beside a bundle of cables.]({{ '/assets/img/pcie-switch-follow-up/mcio-switch-card.jpg' | relative_url }})

At the other end, each pair of eight-lane MCIO cables goes to a small board with a full-size PCIe x16 slot. The GPU plugs into that slot. The slot board also has a power input, and the GPU still has its own power lead. The two MCIO cables to a GPU carry the two halves of its x16 data link.

![One of the green GPU slot boards, with a long PCIe x16 socket, two MCIO connectors, and a slot power input.]({{ '/assets/img/pcie-switch-follow-up/mcio-gpu-adapter.jpg' | relative_url }})

![An RTX PRO 6000 plugged into its slot adapter, with two MCIO cables attached beneath the GPU.]({{ '/assets/img/pcie-switch-follow-up/gpu-on-slot-adapter.jpg' | relative_url }})

## Fitting it in the case

I bought a [dual-chamber rackmount case from AliExpress](https://www.aliexpress.us/item/3256812027515666.html). Even with two chambers, fitting everything was difficult. A single backplane would have fixed the spacing between the GPUs. With separate slot boards I could mount the cards in pairs on either side of the motherboard area. The trade-off was finding room for eight MCIO runs, the GPU power leads, and enough clearance around the fans.

![Bundled MCIO cables laid over the front of the case during assembly, before connecting the GPU slot boards.]({{ '/assets/img/pcie-switch-follow-up/routing-mcio-cables.jpg' | relative_url }})

The overhead view shows why the extra chamber didn't solve everything. The power leads bend over the ends of the cards, and the MCIO bundles still have to come back to the switch. The space between the GPUs also has the CPU cooler and RAM in it.

![Overhead view of the two GPU pairs surrounding the CPU cooler and RAM, with power cables and MCIO bundles routed through the remaining space.]({{ '/assets/img/pcie-switch-follow-up/top-view.jpg' | relative_url }})

![The assembled chassis in the rack, with three large front fan grilles above four smaller ones.]({{ '/assets/img/pcie-switch-follow-up/case-in-rack.jpg' | relative_url }})

## What changed electrically

With two GPUs plugged directly into this motherboard, the two slots are wired for Gen5 x8 each. Each card has its own eight-lane path to the CPU, and traffic between the two goes through the CPU's PCIe root complex. With the switch installed, the motherboard gives the switch the full x16 link from the top slot. I leave the other CPU-connected graphics slot empty so the top slot keeps all 16 lanes. The switch supplies four downstream x16 links of its own. The MCIO cabling starts on the downstream side of the switch.

![One Gen4 x16 link from the AM5 root port to the PCIe switch card, followed by four Gen4 x16 links, each carried by two x8 MCIO cables to a separate GPU slot adapter.]({{ '/assets/img/pcie-switch-follow-up/topology.svg' | relative_url }})

_The drawing omits the extra PCIe bridges Linux sees inside the switch fabric. Diagram style inspired by [Doug's RTX 3090 splitter post](https://doug.sh/posts/pcie-splitter-rtx-3090/)._

Gen5 x8 and Gen4 x16 both have a theoretical bandwidth of about 31.5 GB/s per direction before packet overhead. Each GPU now has that budget to the switch, and I verified Gen4 x16 on the GPU links under load. The four cards share one Gen4 x16 uplink for transfers to and from the CPU, including loading model weights from host memory. The switch can also forward peer traffic between its downstream ports, which is why I wanted it for a model that stays in GPU memory.

## Getting all four GPUs to enumerate

With the switch in place, I checked the PCIe tree before testing CUDA. `lspci -Dtv` showed the bridges and GPU endpoints below the CPU's root port. I used `sudo lspci -Dvv -s <address>` to inspect individual devices and ports. On a working boot the root port and the first switch upstream port both reported 16 GT/s at x16. The GPU-facing links sometimes showed a lower speed at idle, so I checked them again while the cards were moving data.

Getting to that working boot took some experimenting with GPU firmware profiles. My working setup used **Above 4G Decoding on**, the motherboard's **Resizable BAR toggle off**, and the **8 GiB graphics profile** on all four GPUs. Linux also had `pci=realloc=on` on its command line. The 8 GiB is BAR1, a PCIe address window into GPU memory. Each card still has its 96 GB of VRAM. In compute mode the cards exposed a much larger framebuffer aperture and additional virtual-function reservations, and I ran into resource allocation problems with multiple cards.

I also had one normal reboot where Linux could still see the switch, but all four GPUs were missing from the PCIe tree and the uplink had come up at Gen1 x4. A complete shutdown and restart brought back all four GPUs and Gen4 x16. I still don't know which part of the initialization failed. Having the device tree and link speeds from a good boot gave me something concrete to compare.

## Getting peer traffic to stay on the switch

`nvidia-smi topo -p2p r` and `nvidia-smi topo -p2p w` reported peer access between every GPU pair. I checked all six pairs in both directions with CUDA copies and kernels that read and wrote peer memory directly. All 36 data checks passed, and the copies ran at 27.1–27.6 GB/s with one pair active at a time.

Then I ran a four-GPU NCCL all-reduce, which has the GPUs exchange and combine partial results. At 16 MiB per GPU, the effective bus bandwidth was only 5.53 GB/s. The switch ports showed `ReqRedir+ CmpltRedir+` under PCIe Access Control Services (ACS). Those settings can send peer requests and completions up the shared link to the root complex. The transfers can still use peer memory the whole time. [NVIDIA's NCCL troubleshooting guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html#pci-access-control-services-acs)

![With ACS redirection on, peer traffic can travel up to the CPU root complex and back down. With redirection off, the switch can forward it locally.]({{ '/assets/img/pcie-switch-follow-up/peer-routing.svg' | relative_url }})

I stopped the inference server, mapped the ports in the GPU subtree, and saved their original ACS settings. For the test I cleared only the peer request and completion redirect bits on seven switch ports, then read them back to confirm `ReqRedir- CmpltRedir-`. The CPU root port's ACS settings stayed as they were. It was a runtime change on this machine, with no boot hook. I didn't verify or reapply it after the later restart.

The same four-GPU all-reduce reached **21.50 GB/s** after the change, up from **5.53 GB/s**. That's the median of three runs per setting with the model unloaded. NCCL reported the same `P2P/CUMEM` transport in both cases. The port settings and the change in collective speed point to the upstream detour shown in the diagram. I didn't trace individual packets.

In a same-day comparison with the same four-way vLLM configuration, GLM-5.3-Flash processed an uncached 32K-token prompt in 6.4 seconds after the change, down from 15.2 seconds. The collective was the check that showed me what the switch configuration was doing. I'll get into the model setup and the later vLLM tuning in another post.
