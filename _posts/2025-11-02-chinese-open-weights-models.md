---
layout: post
title: "Silicon Valley's New Secret: Chinese Base Models"
date: 2025-11-02 11:05:00 -0700
categories: [AI, Development]
tags: [llm]
description: "From fine-tunes to founder stacks, the center of gravity is moving east."
---

When entrepreneurs walk into the offices of Andreessen Horowitz (a16z), one of Silicon Valley's premier venture capital firms, the odds are high their startups are running on AI models made in China. ["I'd say there's an 80% chance they're using a Chinese open-source model"](https://www.economist.com/business/2025/08/21/china-is-quietly-upstaging-america-with-its-open-models), reveals Martin Casado, a partner at a16z. This quiet migration from expensive closed-source models to cheaper open-source alternatives is reshaping the AI landscape.

The evidence arrived just this week. On October 29th, both [Cognition launched SWE-1.5](https://cognition.ai/blog/swe-1-5), the coding assistant powering Windsurf, and [Cursor unveiled their new Composer agent](https://cursor.com/blog/composer). Within hours, developers [noticed something striking](https://x.com/deedydas/status/1984092103358738846): SWE-1.5 appears to be a customized version of Zhipu's GLM 4.6 model running on Cerebras infrastructure, while Cursor's Composer occasionally reveals Chinese reasoning traces in its outputs - telltale signs of their base model origins. Even prominent investors are making the switch: [Chamath Palihapitiya announced on the All-In Podcast](https://x.com/CodeByPoonam/status/1982763665314267435) that he's migrating significant workloads from OpenAI and Anthropic to Kimi K2, despite being a top-tier Amazon Bedrock customer. His verdict? "It's way more performant and a ton cheaper."

This isn't just about cost savings. Percy Liang, co-founder of Together AI, [notes that open-weight models enable "different forms of adoption than proprietary technology"](https://www.economist.com/business/2025/08/21/china-is-quietly-upstaging-america-with-its-open-models)- they can be more easily adapted to specific use cases and run on-premises rather than relying on cloud services. While American labs bet big on pushing the frontiers of intelligence with closed models, their Chinese rivals are focused on encouraging widespread AI adoption through openness. [As Ali Farhadi of the Allen Institute for AI admits](https://www.economist.com/business/2025/08/21/china-is-quietly-upstaging-america-with-its-open-models): "As hard as it is for us all to swallow, I think we're behind [on open weights] now."

It all started when DeepSeek burst out with V3 in Dec 2024 and the R1 reasoning model in Jan 2025, touting frontier-level performance trained in ~2 months for under $6M on H800s. The app rocketed to #1 on the U.S. App Store, triggering Wall Street jitters and a flurry of "AI price war" headlines as DeepSeek slashed off-peak API rates and rivals followed. In the weeks after, competitors rushed out reasoning upgrades, Chinese labs accelerated releases, and the narrative flipped from compute-as-moat to "efficient scaling", cementing DeepSeek as the spark for a global reset on costs and pace.

The pressure became undeniable when OpenAI, in their [recent partnership announcement with Microsoft](https://openai.com/index/next-chapter-of-microsoft-openai-partnership/), quietly acknowledged the shift with a line tucked away as the last bullet point: "OpenAI is now able to release open weight models that meet requisite capability criteria." Even the company that pioneered and then abandoned the open approach is being forced back to the table.

## The Rise of Chinese Models

Looking at the top-performing open weights models by [intelligence ratings](https://artificialanalysis.ai/models/open-source), Chinese models like MiniMax-2 and Qwen-3-235B now match or exceed their American counterparts. The bar chart shows a striking pattern: of the 10 highest-scoring models, China claims 6 spots while the US holds 4. What's remarkable isn't just the quantity - it's that Chinese models are competing at the very top of the intelligence scale.

<img width="1000" height="432" alt="image" src="https://github.com/user-attachments/assets/d2e3564b-c3f0-4325-8f5b-b00be41a080e" />

The aggregate performance intelligence ratings chart reveals the inflection point. Starting from roughly equal footing in April 2024, China's red line climbs relentlessly while the US blue line begins to plateau. The crossover happened around April 2025—and the gap has only widened since. Europe's trajectory flatlined, a stark reminder that AI leadership requires more than regulation.

<img width="1000" height="667" alt="image" src="https://github.com/user-attachments/assets/562b1e3c-60f7-4dc2-91d7-eba0bded04af" />

Perhaps the most significant inflection point: in August 2025, cumulative downloads of Chinese models surpassed those from the US for the first time. The crossover wasn't close—China's trajectory is steeper, suggesting the gap will only widen. This chart captures the moment when developer preference fundamentally shifted.

<img width="1000" height="710" alt="4863b0bb-31e2-4bb5-b909-7122c11b964e" src="https://github.com/user-attachments/assets/bcbacd38-336e-4528-afdb-660c39ed845d" />

The growth trajectories tell an even more dramatic story. Meta's Llama maintained a steady lead through 2024, but Qwen's explosive acceleration in 2025 changed everything - the red line shoots nearly vertical, reaching 400M downloads while Llama approaches 350M. Mistral and DeepSeek trail at 100M and 80M respectively. This isn't gradual adoption; it's a developer exodus to Chinese models.

<img width="1000" height="797" alt="15f8f279-680a-4f73-aee3-842655210c30" src="https://github.com/user-attachments/assets/fc52d0b9-cecc-4b5b-9ee1-3812e1a8ef33" />

The downstream impact is even more striking. When developers build custom models through fine-tuning, they increasingly start with Chinese base models. By late 2025, Chinese models account for over 50% of all fine-tuned derivatives—a majority that keeps growing. These aren't just being downloaded; they're becoming the foundation of the next generation of AI applications.

<img width="1000" height="708" alt="image" src="https://github.com/user-attachments/assets/23163e25-e12d-4a0c-ad34-debb9855402d" />

Perhaps the most striking trend: the gap between frontier closed models and open-weight models is rapidly closing. In July 2023, frontier models like GPT-4 dominated with scores in the 40s while open models barely reached 20%. Just seven months later, o1-mini achieved 65%—but open models were already catching up. By mid-2025, top open models like Qwen3-32B and EXAONE-4.0 are tracking closely behind Grok 4, with the performance delta shrinking to single digits. The chart suggests consumer GPUs (RTX 5090, 6000) can now run models approaching frontier intelligence.

<img width="1000" height="510" alt="c4c2ac11-6da5-412a-b565-9653eac2411f" src="https://github.com/user-attachments/assets/9c19ba05-7154-4673-824b-d3f0d17f777b" />

The efficiency advantage becomes clear when plotting intelligence against active parameters. The "most attractive quadrant"—high intelligence, lower parameter count—is highlighted in green. OpenAI's gpt-oss models (12B and 20B) sit squarely in this zone, but Chinese models are pushing into it from the right. MiniMax-M2 achieves top-tier intelligence around the 10B mark, while models like DeepSeek V3.2, Qwen3-235B, and GLM-4.6 cluster in the 40-60B range with competitive scores. The trend is clear: newer releases are climbing toward that upper-left corner, delivering more intelligence per parameter.

<img width="1000" height="476" alt="image" src="https://github.com/user-attachments/assets/207cd4ba-38b9-4b5b-b1b5-985ef85460b5" />

Behind these numbers is an entire ecosystem of organizations competing at every tier. The chart categorizes players from frontier models down to honorable mentions. Since this visualization was created for Nathan Lambert's October talk, [MiniMax released M2](https://huggingface.co/MiniMaxAI/MiniMax-M2), which now claims the top spot in Artificial Analysis intelligence rankings. The pace of releases is so rapid that static snapshots become obsolete within weeks.

<img width="1000" height="615" alt="image" src="https://github.com/user-attachments/assets/73a5adc7-e213-4336-aa44-44b1bd3a9a7e" />

This post is based HEAVILY on a [talk](https://www.interconnects.ai/p/state-of-open-models-2025) and [slides](https://docs.google.com/presentation/d/1f1Et0Mz8zb1yVCnCgdYSy4tAa0Kv_gKT4wPEg1XPdUA/edit?slide=id.p#slide=id.p) presented by Nathan Lambert on The State of Open Models. Further augmented with more recent developments, notable quotes, and graphs from [Artificial Analysis](https://artificialanalysis.ai/).