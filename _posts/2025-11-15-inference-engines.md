---
layout: post
title: "LLM inference Engines and UI Interfaces"
date: 2025-11-15 12:00:00 -0700
categories: [AI]
tags: [llm, vllm]
description: ""
---

| Name                                                                       | GitHub stars* | Type                                                                               | Scale                                     |
| -------------------------------------------------------------------------- | ------------- | ---------------------------------------------------------------------------------- | ----------------------------------------- |
| [vLLM](https://docs.vllm.ai/)                                              | 63.1k         | Open Source (Apache-2.0)                                                           | High-throughput production                |
| [SGLang](https://docs.sglang.ai/)                                          | 20.2k         | Open Source (Apache-2.0)                                                           | High-throughput production                |
| [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/)                     | 12.1k         | Open Source (Apache-2.0)                                                        | High-throughput production                |
| [llama.cpp](https://github.com/ggml-org/llama.cpp)                         | 89.8k         | Open Source (MIT)                                                                  | Enthusiast / local-first                  |
| [exllamav3](https://github.com/turboderp-org/exllamav3)                    | 571           | Open Source (MIT)                                                                  | Enthusiast / local-first                  |
| [MLX (Apple MLX + MLX-LM)](https://ml-explore.github.io/mlx/)              | 22.8k         | Open Source (MIT)                                                                  | Enthusiast / Apple-silicon dev & research |
| [Modular (MAX engine)](https://www.modular.com/max/solutions/ai-inference) | 25.2k         | **Proprietary / source-available** (Apache-2.0 code under a community use license) | High-throughput production                |

Tensor Parallelism and batch inference
https://medium.com/@himanshushukla.shukla3/stop-using-llama-cpp-for-multi-gpu-setups-use-vllm-or-exllamav2-instead-73992cf1a1ad

### Production level, high-throughput inference engines

These engines are used in production by large companies to serve text and multimodal LLMs at scale. They primarily focus on serving models that fit within the VRAM of the GPUs.

- [vLLM](https://github.com/vllm-project/vllm)
  - Originally developed in the Sky Computing Lab at UC Berkeley

### Enthusiast inference engines

These engines often allow using a combination of 


Where to start

If you are new to the LLM Inference world, I would recommend starting with LMStudio.

