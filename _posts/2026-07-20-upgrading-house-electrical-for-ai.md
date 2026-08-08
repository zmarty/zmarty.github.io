---
layout: post
title: "Upgrading my house electrical system for AI workloads"
date: 2026-07-20 09:00:00 -0700
categories: [AI, Hardware]
tags: [hardware, infrastructure]
description: "From 100A to 200A service: running dedicated 120V and 240V circuits for a multi-GPU AI rack, with NEMA L6-30R twist-lock receptacles, a Seasonic PRIME PX-2200 PSU, and a 240V PDU."
---

I upgraded my house electrical system to support a multi-GPU AI rack. The house went from 100A to 200A service, and I ran two dedicated circuits to the rack: a 120V / 20A for networking gear, and a 240V / 30A on a NEMA L6-30R twist-lock for the GPU workstations. That 30A outlet gives me 5,760W of continuous power to work with.

Here is why I needed it and how the pieces fit together.

## Why a normal outlet runs out

My dual RTX Pro 6000 workstation pulls over 1,200W from the wall under sustained inference load, and that is before transient spikes. A single RTX Pro 6000 is rated at 600W TDP, and GPUs happily spike above their TDP for short periods. Two of them plus a 7950X3D, DDR5, storage and fans puts one workstation at roughly 1,300-1,400W under full load.

A 120V / 15A circuit gives you 1,800W on paper, but code derates continuous loads to 80%, so the real number is about 1,440W. Two GPUs already push right up against that, and going to four is not a matter of shaving watts somewhere. Four RTX Pro 6000 plus the rest of the machine lands north of 2,600W, which is over 21A continuous at 120V. A four-GPU node needs 240V, full stop. And all of that is before anything else in the rack, which was sharing the same 120V circuit.

## The panel was the actual blocker

My house had 100A service and the main panel was full. There were no free slots for new breakers at all.

I also needed an EV charger around the same time, which is a 40-50A continuous load on its own and usually wants a 240V / 60A circuit. So even if I had found panel space, 100A service was not going to carry an EV charger plus the rack.

## Going to 200A

I upgraded the house service from 100A to 200A. That meant a new 200A main panel with more breaker slots, and an electrician doing the panel swap, grounding and inspection. The existing service drop from the utility was fine as it was, so nothing had to change on the utility side.

The EV charger is what made this necessary. The AI rack alone would not have justified it. But once the electrician is already there and the panel has empty slots, adding a couple of dedicated circuits costs very little extra.

<!-- TODO: picture of old vs new panel -->

## The two circuits I ran to the rack

### 120V / 20A

A NEMA 5-20R receptacle on its own breaker, for networking gear, smaller machines and accessories. The point is just that the rack does not share a circuit with whatever else is running in the house.

### 240V / 30A

This is the one that matters. I had a [Leviton 2620](https://www.amazon.com/dp/B00002NAT9) installed, a 30A 250V flush-mount locking receptacle. It is a NEMA L6-30R, which is what nearly every 240V rack PDU expects.

At 240V and 30A you get 7,200W on paper, or 5,760W continuous after the 80% derate. That covers two dual-GPU workstations with plenty of headroom left for spikes.

<!-- TODO: picture of the L6-30R receptacle on the wall -->

## The PDU

Almost all 240V rack PDUs use an L6-30P twist-lock plug, so once you have the L6-30R on the wall the PDU choice is easy. I went with a Tripp Lite PDUMH30HV: 30A, 208/240V, L6-30P input, and a mix of C13 and C19 outlets. It is metered, so there is a digital current display on the front. It is not networked or managed. I mostly wanted the readout so I can see total draw at a glance.

The twist-lock connection is also nice because it will not slowly work its way loose the way a straight-blade plug can.

<!-- TODO: picture of the PDU mounted in the rack -->

## The PSU only runs on 240V

The [Seasonic PRIME PX-2200](https://seasonic.com/atx3-prime-px-2200/) is a 2,200W, 80 PLUS Platinum, fully modular unit. Most consumer PSUs accept 100-240V. This one does not. It only operates on 200-240V AC input and will not turn on at 120V at all.

Because of the wattage it also uses a C20 inlet (rated for 16A) instead of the C14 you see on lower-wattage PSUs. So the chain from the wall looks like this:

```
Wall (L6-30R) -> PDU (L6-30P input, C19 outlet) -> C20-to-C19 jumper -> PSU (C20 inlet)
```

The part that trips people up is the cable. You need a C20 to C19 jumper: the C20 end goes into the PDU's C19 outlet, and the C19 end goes into the PSU's C20 inlet. It is not the same as the common C13/C14 cable you use for servers and monitors.

I use a StarTech 2 ft C20 to C19 cable, 14 AWG, 15A / 250V. Two feet is deliberately short because the PSU sits right next to the PDU and I did not want loose loops of cable in the rack. If you need slack, the Tripp Lite 6 ft version works fine. Look for 12 or 14 AWG rated at least 15A at 250V.

## Why 240V and not just another 120V circuit

For a build in this class 240V is the only real option, because the PX-2200 will not start on 120V. Even if you found a 120V PSU rated for 1,600W+, you would be pulling over 13A continuous on a circuit rated for 12A. You could go to a 120V / 20A circuit instead, but at that point you are already pulling new wire, so you may as well go 240V.

The headroom is the other reason. 1,400W at 240V is only 5.8A. On a 30A circuit derated to 24A continuous, that leaves a lot of room for transients, inrush and whatever I add later. And one L6-30 circuit handles two workstations in this class. On 120V I would need a separate dedicated circuit per machine.

## Cost

I am not going to post numbers, because mine came in well above what you will find quoted online and they would not be a useful guide for anyone else. Get a few quotes. The one thing worth knowing going in is that the service upgrade dominates everything else by a wide margin. The receptacle, the PDU and the cables are rounding errors next to it.

If you already have 200A service and a free slot or two in the panel, you skip the expensive part entirely and running a 240V circuit to a rack is a much smaller job.

## Where it ended up

The rack has its own 120V and 240V feeds, each on a dedicated breaker. The dual RTX Pro 6000 workstation runs off the 240V PDU through the PX-2200. The second machine and the networking gear are on the 120V circuit. The PDU meter shows I am well under the continuous limit even at full inference load, with enough left over to add a second 240V workstation later.