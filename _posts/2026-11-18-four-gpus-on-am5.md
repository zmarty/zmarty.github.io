---
layout: post
title: "How I fit four RTX PRO 6000s in one AM5 machine"
categories: [AI, Hardware]
tags: [gpu, pcie, switch, pex88096, mcio, p2p, nccl, am5]
description: "The four-GPU AM5 build I actually assembled, from an MCIO switch card and separate GPU adapters to the PCIe settings that got the cards talking."
---

[The dream]({% post_url 2026-09-17-gpu-expansion-box-pcie-switch %}) is alive! After four months of iterating on the design and waiting around for packages, I finally built a four-GPU AI server with a whopping 384 GB of VRAM. The model I built it for is the original DeepSeek-V4.1-Flash checkpoint. It has 552 billion backbone parameters and is one of the most capable open-weight models I wanted to run at home. With the Engram lookup table on NVMe, it processed uncached 32K-token prompts at about 15,350 input tokens per second. It generated 235 output tokens per second on a single request and 678 tokens per second in total across eight concurrent requests. [DeepSeek model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)

![One Gen4 x16 link from the AM5 root port to the PCIe switch card, followed by four Gen4 x16 links, each carried by two x8 MCIO cables to a separate GPU slot adapter.]({{ '/assets/img/pcie-switch-follow-up/topology.svg' | relative_url }}){: width="800" height="605" style="width: 800px; max-width: 100%; height: auto;" }

_The drawing omits the extra PCIe bridges Linux sees inside the switch fabric. Diagram style inspired by [Doug's RTX 3090 splitter post](https://doug.sh/posts/pcie-splitter-rtx-3090/)._

## Why I used a PCIe switch

When I split a model across four GPUs, they exchange partial results as they work. Peer-to-peer access (P2P) lets one GPU read or write another GPU's memory directly. The switch can forward those transfers between its own ports, leaving the shared link to the CPU available for host traffic. That matters most when several cards are exchanging data at once.

With two GPUs plugged directly into this motherboard, the two slots are wired for Gen5 x8 each. Each card has its own eight-lane path to the CPU, and traffic between the two goes through the CPU's PCIe root complex. With the switch installed, the motherboard gives the switch the full x16 link from the top slot. I leave the other CPU-connected graphics slot empty so the top slot keeps all 16 lanes. The switch supplies four downstream x16 links of its own. The MCIO cabling starts on the downstream side of the switch.

Gen5 x8 and Gen4 x16 both have a theoretical bandwidth of about 31.5 GB/s per direction before packet overhead. Each GPU now has that budget to the switch, and I verified Gen4 x16 on the GPU links under load. The four cards share one Gen4 x16 uplink for transfers to and from the CPU, including loading model weights from host memory.

My AM5 motherboard has two graphics slots, and a regular PC case doesn't have room to mount four cards this size with all their cables. I ordered a custom case from China, got a PCIe switch, and moved to a 240 V power supply. That last part is why I ran a dedicated circuit to the rack when I [upgraded my house electrical system]({% post_url 2026-08-08-upgrading-house-electrical-for-ai %}).

The original plan called for a separate box with all four GPUs plugged into a single switch backplane. I ended up with a switch card in the motherboard and a separate slot board for each GPU, connected over MCIO. That let me fit the whole machine in one chassis.

![Open rackmount case with four RTX PRO 6000s in pairs around the CPU cooler and two rows of fans on the front panel.]({{ '/assets/img/pcie-switch-follow-up/four-gpus-in-case.jpg' | relative_url }}){: width="800" height="602" style="width: 800px; max-width: 100%; height: auto;" }

## The switch card and the GPU slots

I kept the Ryzen 9 7950X3D and the ASUS ROG Crosshair X670E Hero. The Broadcom PEX88096 switch card sits in the motherboard's top x16 slot. That slot connects the switch directly to the CPU. The connectors around the card take the MCIO cables that go out to the GPUs. The little fan on the switch card cools the switch chip itself.

![The blue PCIe switch card on its packing foam, with a gold edge connector, a small blower, and MCIO sockets beside a bundle of cables.]({{ '/assets/img/pcie-switch-follow-up/mcio-switch-card.jpg' | relative_url }}){: width="800" height="1062" style="width: 800px; max-width: 100%; height: auto;" }

At the other end, each pair of eight-lane MCIO cables goes to a small board with a full-size PCIe x16 slot. The GPU plugs into that slot. The slot board also has a power input, and the GPU still has its own power lead. The two MCIO cables to a GPU carry the two halves of its x16 data link.

![One of the green GPU slot boards, with a long PCIe x16 socket, two MCIO connectors, and a slot power input.]({{ '/assets/img/pcie-switch-follow-up/mcio-gpu-adapter.jpg' | relative_url }}){: width="800" height="1062" style="width: 800px; max-width: 100%; height: auto;" }

![An RTX PRO 6000 plugged into its slot adapter, with two MCIO cables attached beneath the GPU.]({{ '/assets/img/pcie-switch-follow-up/gpu-on-slot-adapter.jpg' | relative_url }}){: width="800" height="602" style="width: 800px; max-width: 100%; height: auto;" }

## Fitting it in the case

I bought a [dual-chamber rackmount case from AliExpress](https://www.aliexpress.us/item/3256812027515666.html). Even with two chambers, fitting everything was difficult. A single backplane would have fixed the spacing between the GPUs. With separate slot boards I could mount the cards in pairs on either side of the motherboard area. The trade-off was finding room for eight MCIO runs, the GPU power leads, and enough clearance around the fans.

![Bundled MCIO cables laid over the front of the case during assembly, before connecting the GPU slot boards.]({{ '/assets/img/pcie-switch-follow-up/routing-mcio-cables.jpg' | relative_url }}){: width="800" height="602" style="width: 800px; max-width: 100%; height: auto;" }

The overhead view shows why the extra chamber didn't solve everything. The power leads bend over the ends of the cards, and the MCIO bundles still have to come back to the switch. The space between the GPUs also has the CPU cooler and RAM in it.

![Overhead view of the two GPU pairs surrounding the CPU cooler and RAM, with power cables and MCIO bundles routed through the remaining space.]({{ '/assets/img/pcie-switch-follow-up/top-view.jpg' | relative_url }}){: width="800" height="1062" style="width: 800px; max-width: 100%; height: auto;" }

![The assembled chassis in the rack, with three large front fan grilles above four smaller ones.]({{ '/assets/img/pcie-switch-follow-up/case-in-rack.jpg' | relative_url }}){: width="800" height="602" style="width: 800px; max-width: 100%; height: auto;" }

## Getting all four GPUs to enumerate

With the switch in place, I checked the PCIe tree before testing CUDA. `lspci -Dtv` showed the bridges and GPU endpoints below the CPU's root port. I used `sudo lspci -Dvv -s <address>` to inspect individual devices and ports. On a working boot the root port and the first switch upstream port both reported 16 GT/s at x16. The GPU-facing links sometimes showed a lower speed at idle, so I checked them again while the cards were moving data.

Getting to that working boot took some experimenting with GPU firmware profiles. My working setup used **Above 4G Decoding on**, the motherboard's **Resizable BAR toggle off**, and the **8 GiB graphics profile** on all four GPUs. Linux also had `pci=realloc=on` on its command line. The 8 GiB is BAR1, a PCIe address window into GPU memory. Each card still has its 96 GB of VRAM. In compute mode the cards exposed a much larger framebuffer aperture and additional virtual-function reservations, and I ran into resource allocation problems with multiple cards.

I used **NVIDIA Display Mode Selector 1.77** for Linux to set `physical_display_enabled_8GB_bar1` on each card. I changed the cards in stages, then checked `nvidia-smi -q` after reboot. All four reported **8,192 MiB** under BAR1 Memory Usage. [NVIDIA Display Mode Selector download](https://developer.nvidia.com/displaymodeselector)

## Getting peer traffic to stay on the switch

`nvidia-smi topo -p2p r` and `nvidia-smi topo -p2p w` reported peer access between every GPU pair. I checked all six pairs in both directions with CUDA copies and kernels that read and wrote peer memory directly. All 36 data checks passed, and the copies ran at 27.1–27.6 GB/s with one pair active at a time.

Then I ran a four-GPU NCCL all-reduce, which has the GPUs exchange and combine partial results. At 16 MiB per GPU, the effective bus bandwidth was only 5.53 GB/s. The switch ports showed `ReqRedir+ CmpltRedir+` under PCIe Access Control Services (ACS). Those settings can send peer requests and completions up the shared link to the root complex. The transfers can still use peer memory the whole time. [NVIDIA's NCCL troubleshooting guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html#pci-access-control-services-acs)

![With ACS redirection on, peer traffic can travel up to the CPU root complex and back down. With redirection off, the switch can forward it locally.]({{ '/assets/img/pcie-switch-follow-up/peer-routing.svg' | relative_url }}){: width="800" height="446" style="width: 800px; max-width: 100%; height: auto;" }

I stopped the inference server, mapped the ports in the GPU subtree, and saved their original ACS settings. For the test I cleared only the peer request and completion redirect bits on seven switch ports, then read them back to confirm `ReqRedir- CmpltRedir-`. The CPU root port's ACS settings stayed as they were.

The same four-GPU all-reduce reached **21.50 GB/s** after the change, up from **5.53 GB/s**. That's the median of three runs per setting with the model unloaded. NCCL reported the same `P2P/CUMEM` transport in both cases. The port settings and the change in collective speed point to the upstream detour shown in the diagram. I didn't trace individual packets.

In a same-day comparison with the same four-way vLLM configuration, GLM-5.3-Flash processed an uncached 32K-token prompt in 6.4 seconds after the change, down from 15.2 seconds. The collective was the check that showed me what the switch configuration was doing.

## What happens after a reboot?

The GPUs already enumerated and ran inference before I changed ACS. The change was for peer traffic performance, and I applied it at runtime. A later privileged read confirmed that all seven switch ports were back to `ReqRedir+ CmpltRedir+` after reboot. The BIOS settings and the GPUs' saved 8 GiB profiles are separate from those runtime bits.

I also had one normal reboot where Linux could still see the switch, but all four GPUs were missing from the PCIe tree and the uplink had come up at Gen1 x4. A complete shutdown and restart brought back all four GPUs and Gen4 x16. I still don't know which part of the initialization failed. Having the device tree and link speeds from a good boot gave me something concrete to compare.

The DeepSeek-V4.1-Flash scores at the top came from a run after I cleared redirection and before the reboot. The timeline points to redirection being off during that run. I didn't read the bits during the benchmark or run DeepSeek with both settings. The before-and-after comparison above used NCCL and GLM-5.3-Flash.

### Reapplying the ACS change at Linux boot

A systemd oneshot service can run the same masked write after Linux enumerates the switch. I tested the register change by hand. The service below is how I'd make it persistent. I haven't installed it on this machine. Linux also documents `pci=config_acs`, but upstream 6.11 can fall back to the firmware value for bits marked `0`. The script below reads the hardware back to confirm the two bits actually cleared. [Linux 6.11 PCI implementation](https://github.com/torvalds/linux/blob/v6.11/drivers/pci/pci.c#L957-L1035)

First, use `lspci -Dtv` and `lspci -DPP -d 1000:c010` to identify the switch ports in the paths to the GPUs. The addresses below are from my machine. Replace them with the verified ports on your system. Clearing these ACS bits affects PCIe peer isolation, so scope the change to the GPU switch ports you intend to share.

Save this as `/usr/local/sbin/gpu-switch-acs`. It checks that all seven devices are Broadcom switch bridges below my GPU root port and have readable ACS controls before it writes anything. The `000c` mask selects only request and completion redirection.

```bash
#!/bin/bash
set -euo pipefail

root=$(readlink -e /sys/bus/pci/devices/0000:00:01.1)
ports=(
    0000:02:00.0 0000:02:04.0 0000:02:08.0 0000:04:10.0
    0000:07:00.0 0000:07:10.0 0000:0b:00.0
)

for port in "${ports[@]}"; do
    device=$(readlink -e "/sys/bus/pci/devices/$port")
    if [[ $device != "$root/"* || $(cat "$device/vendor") != 0x1000 ||
          $(cat "$device/device") != 0xc010 || $(cat "$device/class") != 0x060400 ]]; then
        echo "Unexpected PCIe topology or device at $port" >&2
        exit 1
    fi
    value=$(setpci -s "$port" ECAP_ACS+6.w)
    [[ $value =~ ^[0-9a-fA-F]{4}$ && ${value,,} != ffff ]] || exit 1
done

for port in "${ports[@]}"; do
    setpci -s "$port" ECAP_ACS+6.w=0000:000c
    value=$(setpci -s "$port" ECAP_ACS+6.w)
    if [[ ! $value =~ ^[0-9a-fA-F]{4}$ ]] || (( (16#$value & 0x000c) != 0 )); then
        echo "ACS redirection is still enabled or unreadable at $port" >&2
        exit 1
    fi
done
```

Save the unit as `/etc/systemd/system/gpu-switch-acs.service`.

```ini
[Unit]
Description=Clear peer redirection on GPU PCIe switch ports

[Service]
Type=oneshot
ExecStart=/bin/bash /usr/local/sbin/gpu-switch-acs
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

Run `sudo systemctl daemon-reload` and `sudo systemctl enable gpu-switch-acs.service` to schedule it for the next boot. If inference also starts from a systemd unit, give that unit both `Requires=gpu-switch-acs.service` and `After=gpu-switch-acs.service` so inference waits for the check to succeed.

After reboot, check `systemctl status gpu-switch-acs.service` and confirm that `sudo lspci -Dvv -s <port>` reports `ReqRedir- CmpltRedir-` under `ACSCtl` on each targeted port. Then repeat the four-GPU all-reduce to check performance. To go back to the defaults, disable the service and reboot.

For DeepSeek-V4.1-Flash I used the Local Inference Lab serving profile with disk-backed Engram. I'll get into the model setup and the later vLLM tuning in another post. [The DeepSeek-V4.1-Flash serving guide I used](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/deepseek-v4.1-flash.md)
