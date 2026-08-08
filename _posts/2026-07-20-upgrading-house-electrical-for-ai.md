---
layout: post
title: "Upgrading my house electrical system for AI workloads"
date: 2026-07-20 09:00:00 -0700
categories: [AI, Hardware]
tags: [hardware, infrastructure]
description: "From 100A to 200A service: running dedicated 120V and 240V circuits for a multi-GPU AI rack, with NEMA L6-30R twist-lock receptacles, a Seasonic PRIME PX-2200 PSU, and a 240V PDU."
---

I upgraded my house electrical system to support a local AI rack, meaning multiple multi-GPU machines running large models on my own hardware instead of in somebody else's data center. The house went from 100A to 200A service, and I ran two dedicated circuits to the rack: a 120V / 20A for networking gear, and a 240V / 30A on a NEMA L6-30R twist-lock for the GPU workstations. That 30A outlet gives me 5,760W of continuous power to work with, which is more than I need today, and that is the point. Local models keep getting bigger and every GPU generation pulls more than the last one, but wiring is the slowest and most expensive part of the stack to change later. I sized the electrical for where I expect to be in a few years, not for what is sitting in the rack right now.

Here is why I needed it and how the pieces fit together.

## A full panel on 100A service

My house had 100A service and the main panel was full. There were no free slots for new breakers at all, so the rack was blocked before I even got to the question of how many watts I needed.

I also needed an EV charger around the same time, which is a 40-50A continuous load on its own and needed a 240V / 60A breaker. So even if I had found panel space, 100A service was not going to carry an EV charger plus the rack.

## Going to 200A

I upgraded the house service from 100A to 200A. That meant a new 200A main panel with more breaker slots, and an electrician doing the panel swap, grounding and inspection. The drop coming in from the street was reused as is, so the utility never had to pull new wire.

The EV charger is what made this necessary. The AI rack alone would not have justified it. But once the electrician is already there and the panel has empty slots, adding a couple of dedicated circuits costs very little extra.

This is a bigger job than it sounds like. Most of the first day was trenching along the side of the house:

<img width="650" height="490" alt="Open trench running along the side of the house past the gas meter, with excavated soil and rocks piled on plastic sheeting and two traffic cones in the foreground" src="https://github.com/user-attachments/assets/f6d413a3-2608-4af7-b45b-2d27b09789e2" />

<img width="650" height="914" alt="Two electricians standing on a plank laid across the open trench, working on the new service equipment mounted on the exterior wall" src="https://github.com/user-attachments/assets/4c04e782-a26e-49fd-9a5f-64a3d203176f" />

The new meter main went on the outside wall. The utility came out and did a disconnect and reconnect in the span of 20 minutes. During that window the electrician moved the drop coming in from the street off the old conduit and onto the new one.

<img width="650" height="863" alt="New gray meter main mounted on the exterior wall with conduit running down into the still-open trench, and a cover over the meter socket reading LEAVE THIS SHIELD IN PLACE UNTIL METER IS INSTALLED" src="https://github.com/user-attachments/assets/44d4f125-6169-4765-8363-a949e90f9c3b" />

<img width="650" height="863" alt="The finished exterior installation with the meter set and the trench backfilled with fresh soil, air conditioner condenser to the right" src="https://github.com/user-attachments/assets/56ef4d0a-1521-49b8-84e0-153117a7ef0d" />

Inside, the new Siemens panel. The 200A main breaker is at the top, and the important part for me was the row of empty slots below the populated ones:

<img width="650" height="863" alt="Interior of the new Siemens load center with the cover off, showing the 200A main breaker at the top, two columns of branch breakers, and neutral bars with circuit conductors landed on both sides" src="https://github.com/user-attachments/assets/f99b8dfa-5162-47eb-a472-fd36851caa0b" />

<img width="650" height="863" alt="The same Siemens panel with the dead front cover installed, mounted in an open stud bay with fiberglass insulation and labeled conduits entering from above" src="https://github.com/user-attachments/assets/3feffbfb-6854-420c-b0f6-bc1ea2e4e15a" />

## The two circuits I ran to the rack

### 120V / 20A

A NEMA 5-20R receptacle on its own breaker, for networking gear, smaller machines and accessories. The point is just that the rack does not share a circuit with whatever else is running in the house.

### 240V / 30A

This is the one that matters. I had a [Leviton 2620](https://www.amazon.com/dp/B00002NAT9) installed, a 30A 250V flush-mount locking receptacle. It is a NEMA L6-30R, which is what nearly every 240V rack power distribution unit (PDU) expects.

At 240V and 30A you get 7,200W on paper, or 5,760W continuous after the 80% derate. That covers two dual-GPU workstations with plenty of headroom left for spikes.

<img width="400" height="531" alt="Hand holding the Leviton 2620 twist-lock receptacle before installation, stamped 30A-250V and L6-30 around the black face, with a blue locking collar in the center" src="https://github.com/user-attachments/assets/1762b1c4-a49e-415a-8f5b-1b05ab8f4e87" />

## Picking a PDU

Almost all 240V rack PDUs take an L6-30P twist-lock plug, so once the L6-30R is on the wall the choice gets easy. Two things need to line up: the input plug has to be L6-30P to match the receptacle, and the outlets have to include C19 if you are feeding a high-wattage PSU, because that is what the other end of the cable needs.

As an example of something that fits, the [Tripp Lite 5800K](https://www.amazon.com/Tripp-Lite-Outlets-Rack-Mount-PDUH30HV19/dp/B00ZPOIPO4/) (PDUH30HV19) is a 1U basic PDU with a 12 ft L6-30P input cable and four C19 outlets, rated 208/240V at 24A, with two 20A breakers covering two outlets each.

<img width="650" height="117" alt="Tripp Lite PDUH30HV19 basic PDU, a 1U black rack-mount strip with four C19 outlets and two circuit breaker buttons" src="https://github.com/user-attachments/assets/c20585af-f202-4035-a63f-1a2748a41096" />

That 24A works out to 5,760W at 240V, which is the same as the continuous rating of the circuit feeding it, so the PDU is not the thing that limits you. Note that all four outlets are C19, so anything low-power in the rack needs a C20 to C14 cable rather than an ordinary C13/C14 cord.

The twist-lock connection on the input is also nice because it will not slowly work its way loose the way a straight-blade plug can.

<!-- TODO: picture of the PDU mounted in the rack -->

## PSUs in this class are 240V only

The one I used is the [Seasonic PRIME PX-2200](https://seasonic.com/atx3-prime-px-2200/), a 2,200W 80 PLUS Platinum fully modular unit. It is just one option at this wattage and nothing below depends on that particular choice. Most consumer PSUs accept 100-240V. This one does not. It only operates on 200-240V AC input and will not turn on at 120V at all, which is normal once you get into this power class.

<img width="300" height="222" alt="Seasonic PRIME PX-2200 power supply, a black fully modular ATX unit" src="https://github.com/user-attachments/assets/1f2b3b64-c924-4c78-ba73-3d7a8acc217d" />

Because of the wattage it also uses a C20 inlet (rated for 16A) instead of the C14 you see on lower-wattage PSUs. So the chain from the wall looks like this:

```
Wall (L6-30R) -> PDU (L6-30P input, C19 outlet) -> C20-to-C19 jumper -> PSU (C20 inlet)
```

The part that trips people up is the cable. You need a C20 to C19 jumper: the C20 end goes into the PDU's C19 outlet, and the C19 end goes into the PSU's C20 inlet. It is not the same as the common C13/C14 cable you use for servers and monitors.

I use a StarTech 2 ft C20 to C19 cable, 14 AWG, 15A / 250V. Two feet is deliberately short because the PSU sits right next to the PDU and I did not want loose loops of cable in the rack. If you need slack, the Tripp Lite 6 ft version works fine. Look for 12 or 14 AWG rated at least 15A at 250V.

## Why 240V and not just another 120V circuit

Start with what one machine actually draws. My dual RTX Pro 6000 workstation pulls over 1,200W from the wall under sustained inference load, and that is before transient spikes. A single RTX Pro 6000 is rated at 600W TDP, and GPUs happily spike above their TDP for short periods. Two of them plus a 7950X3D, DDR5, storage and fans puts one workstation at roughly 1,300-1,400W under full load.

A 120V / 15A circuit gives you 1,800W on paper, but code derates continuous loads to 80%, so the real number is about 1,440W. Two GPUs already push right up against that, and going to four is not a matter of shaving watts somewhere. Four RTX Pro 6000 plus the rest of the machine lands north of 2,600W, which is over 21A continuous at 120V. A four-GPU node needs 240V, full stop. And all of that is before anything else in the rack, which was sharing the same circuit.

A PSU in this class will not start on 120V anyway. Even if you found a 120V unit rated for 1,600W+, you would be pulling over 13A continuous on a circuit rated for 12A. You could pull a 120V / 20A circuit instead, but at that point you are already running new wire, so you may as well go 240V.

Most of this is a North American problem. Europe and most of the rest of the world run 230V to ordinary wall sockets, so a PSU that refuses to start on 120V is a non-issue there and a normal outlet already delivers something like 3,700W. If you are reading this from that side of the Atlantic, you get for free the part that cost me a panel upgrade. You would still want a dedicated circuit once you are running several machines, but you would not be doing it just to reach 240V.

The better reason is headroom. One workstation at 1,400W draws 5.8A at 240V, on a circuit good for 24A continuous. That is not headroom for one machine, it is room for several. A single L6-30 run carries multiple nodes plus whatever else ends up in the rack, which is the whole point: I can keep adding machines without calling an electrician again. On 120V every new node is another dedicated circuit and eventually another panel slot, and panel slots are what I ran out of in the first place.

## Where it ended up

The rack has its own 120V and 240V feeds, each on a dedicated breaker. The dual RTX Pro 6000 workstation runs on the 240V side through the PX-2200. The second machine and the networking gear are on the 120V circuit. Even at full inference load the 240V circuit sits nowhere near its continuous limit, which was the point of running 30A rather than the minimum that would have worked. There is room for several more machines on that one circuit.
