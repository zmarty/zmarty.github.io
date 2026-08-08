---
layout: post
title: "DeepSeek-V4-Flash-0731 on 2x RTX Pro 6000: DSpark now ships in the weights"
date: 2026-08-03 09:00:00 -0700
categories: [AI]
tags: [llm, vllm, local-inference, deepseek, speculative-decoding, dspark]
description: "Update on running the new deepseek-ai/DeepSeek-V4-Flash-0731 checkpoint locally. The DSpark speculative-decoding module is now part of the weights, and native DSpark serving through the Gilded Gnosis r24 image is noticeably faster than the MTP path from my June post."
---

DeepSeek-V4-Flash-0731 is a post-training refresh of the same 284B-total / 13B-active architecture I [benchmarked in June](https://www.ovidiudan.com/posts/deepseek-v4-flash-rtx-pro-6000-vllm/), and the change is not subtle. Released July 31 as the official upgrade that supersedes the preview, the checkpoint nearly doubled DeepSeek's agentic scores in several places: Terminal-Bench 2.1 rose from 61.8 to 82.7, Cybergym from 38.7 to 76.7, and DeepSWE from a weak 7.3 to 54.4. On every agent benchmark DeepSeek published, the Flash tier with just 13B active parameters now outscores its own 1.6T V4-Pro (Preview), and lands within reach of the strongest proprietary models. Those numbers came from DeepSeek's own unreleased harness with no independent reproduction, so I treated them as vendor talk until the MIT-licensed weights showed up on Hugging Face and I could run them myself.

The result on my bench is what makes this post worth writing. The DSpark speculative-decoding module ships inside the new weights, native DSpark serving decodes at 307 tokens/s against the ~210/s I got from the MTP path in June, and the model behind those agentic numbers no longer feels like a preview. On this box the upgrade is a straight win.

## DSpark is in the new weights

DeepSeek-V4-Flash-DSpark was released as a variant with an attached speculative-decoding module. The 0731 checkpoint has the same structure, so DSpark is folded into the release weights. That is the single biggest practical change versus my old setup, where speculative decoding meant running `--speculative-config.method mtp` against the Lucifer fork.

With vLLM, DSpark is one flag away:

```bash
--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}'
```

The draft and target weights come from the same checkpoint, so there is nothing extra to download. The model card is explicit that structure and size are unchanged from the preview; every gain in the opening numbers comes from a re-post-training pass, not a new architecture. That, plus [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) being the release checkpoint, is what makes the whole article about a weights swap rather than a workflow change.

`reasoning_effort` also grew a third level. It was `low`/`high`; it is now `low`, `high`, and `max`, and for the `high`/`max` levels DeepSeek recommends allowing up to a 384K token output. That is a lot of decode on a 2-GPU workstation, but the model stays usable thanks to DSpark.

## The runtime: Gilded Gnosis r24

Native DSpark serving for this checkpoint is the domain of the [Gilded Gnosis r24 image](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/ds4dspark-v20.md) from the local-inference-lab folks. It is the follow-up to the Lucifer image I used before, but it is a different stack: instead of the MTP/flashinfer_cutlass path, it runs DSpark on SparkInfer with the B12X W4A8 target (`BACKEND=b12x-a8`).

The image is self-contained, launcher included, so you do not need to mount or run an outside script:

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout build/gilded-gnosis-r21-ds4-runtime-20260802

GPUS=0,1 \
  docker compose -f examples/docker-compose-ds4-v20-r24.yml up -d
```

By default the image runs fixed K5 draft depth (5 speculative tokens), which the runbook recommends over K7: matched tests favored K5 at 217.8 tok/s sustained decode and 289.4 tok/s coding median, versus 192.1 and 281.2 for K7, and there are still open quality questions around long-context behavior with K7. The release defaults are TP2/DCP1, `MAX_NUM_SEQS=16`, `MAX_MODEL_LEN=131072`, 0.975 GPU memory utilization, and `LOAD_FORMAT=instanttensor`.

## The one blocker on this box: InstantTensor vs cudaHostRegister

The default `instanttensor` loader with the BUFFERED backend crashed at weight load on this host with `cudaHostRegister → invalid argument`. It is not a fluke of one backend: [the report in issue #52](https://github.com/local-inference-lab/rtx6kpro/issues/52) covers the same hardware, and all InstantTensor backends (AIO, MMAP, URING, CUFILE) go through the same registration path and fail identically. The host in question is 2x RTX PRO 6000 Blackwell over PCIe with no NVLink, driver 610.43.02, CUDA UMD 13.3, kernel 6.11.0-29-generic.

The fix is a one-liner, and it is what the `LOAD_FORMAT=safetensors` line in my script does. That fallback also produced the interesting side effect of better numbers than advertised:

- decode: 307 tok/s (advertised 218)
- coding with thinking: 284 tok/s (advertised 289)
- 121K-token prefill: 10.7K tok/s, 11.3 s
- needle-in-haystack at 109K context: pass

The decode number is about 40% over the advertised sustained figure and comfortably past what the MTP path gave me in June (~210 tok/s single stream). I read that as the safetensors load path being less intrusive for the graph-captured spec-decode loop than the fast loader on this driver, not as a magic quality improvement, but the measured result is what it is.

## How I run it

I run it detached in the background with the r24 image, TP2, and my own envelope. Two deliberate differences from the compose defaults: `MAX_MODEL_LEN=262144` to keep the 256K context the original post ran at, and `MAX_NUM_SEQS=8` with `MAX_NUM_BATCHED_TOKENS=2048` to fit that context in 96 GB per card with the b12x-a8 backend. The full script lives on the box at `/models/start_ds4_flash_0731.sh`:

<details markdown="1">
<summary markdown="span">Startup script `start_ds4_flash_0731.sh` (click to expand)</summary>

```bash
#!/usr/bin/env bash
# Start DeepSeek-V4-Flash-0731 on 2× RTX PRO 6000 Blackwell (TP2) via the Gilded Gnosis r24 image.
# Uses DSpark K5 (fixed depth, 5 draft tokens) speculative decoding.
# Runs detached in the background; you start it manually (no auto-restart).
# Re-run this script to (re)start.
set -euo pipefail

IMAGE="voipmonitor/vllm:gilded-gnosis-v20-vllmf5981f1-si2b9bf2a-fi801d57a-cu132-20260803-r24"
NAME="ds4-0731-r24"
MODEL_DIR="/models/DeepSeek-V4-Flash-0731"
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/ds4-0731-r24"   # warm JIT/autotune cache; keep this
PORT="8000"

mkdir -p "$CACHE_DIR"
docker rm -f "$NAME" >/dev/null 2>&1 || true

# ---- Docker runtime flags -------------------------------------------------
DOCKER_ARGS=(
  -d                             # detached: run in the background
  --name "$NAME"                 # container name (used to stop / inspect / log)
  --init                         # real init as PID 1 for clean signal handling
  --restart no                   # never auto-restart; you launch this manually
  --gpus all                     # expose GPUs to the container
  --runtime nvidia               # via the NVIDIA container runtime
  --privileged                   # required by the Gilded Gnosis image (GPU/NUMA access)
  --ipc host                     # share host IPC namespace (large shared-mem tensors)
  --shm-size 32g                 # /dev/shm size for NCCL and worker comms
  --network host                 # host networking; PORT is a host port
  --ulimit memlock=-1            # unlimited locked memory (pinned / RDMA buffers)
  --ulimit nofile=1048576:1048576  # high fd limit for model files
  --ulimit stack=67108864        # 64 MiB thread stack
  -v /models:/models:ro          # mount the model tree (read-only)
  -v "$CACHE_DIR":/cache         # persist JIT / autotune cache across runs

  # ---- GPU & NCCL ----------------------------------------------------------
  -e CUDA_VISIBLE_DEVICES=0,1    # use the first two GPUs (TP2)
  -e CUTE_DSL_ARCH=sm_120a       # target Blackwell sm_120a kernels
  -e NCCL_P2P_LEVEL=SYS          # permit GPU-to-GPU P2P across the PCIe system
  -e NCCL_PROTO=LL,LL128,Simple  # NCCL protocols allowed
  -e NCCL_IB_DISABLE=1           # no InfiniBand on this host

  # ---- HuggingFace ---------------------------------------------------------
  -e HF_HUB_OFFLINE=1            # never contact Hugging Face; local weights only
  -e HF_HUB_ENABLE_FILE_HASHING=0 # skip hash checks on local files

  # ---- Model & serving config (DSpark K5 release defaults) -----------------
  -e MODEL_PATH="$MODEL_DIR"     # local path to the 0731 checkpoint (bypasses HF Hub)
  -e SERVED_MODEL_NAME=DeepSeek-V4-Flash  # match the old model name for client compatibility
  -e PORT="$PORT"                # server port
  -e MODE=dspark                 # native DSpark serving for the 0731 checkpoint
  -e BACKEND=b12x-a8             # SparkInfer/B12X W4A8 target path
  -e TP_SIZE=2                   # tensor-parallel across 2 GPUs
  -e DCP_SIZE=1                  # no data-copy parallelism
  -e DSPARK_DEPTH_MODE=fixed     # fixed draft depth (dynamic confidence control is opt-in)
  -e DSPARK_TOKENS=5             # K5 profile (13.3% faster decode than K7)
  -e MAX_NUM_SEQS=8              # scheduler concurrency
  -e MAX_MODEL_LEN=262144        # extended to 256K
  -e MAX_NUM_BATCHED_TOKENS=2048 # prefill scheduler budget (reduced for memory)
  -e GPU_MEMORY_UTILIZATION=0.98  # GPU memory target (leave ~1 GiB headroom for prefill activation)
  -e LOAD_FORMAT=safetensors     # instanttensor cudaHostRegister still fails on this host (driver 610.43.02)
  -e KV_OFFLOADING_SIZE=0        # native CPU KV offload disabled (set to e.g. 48.5 to enable)
  -e DSPARK_MODEL="$MODEL_DIR"    # DSpark target model (used for spec config when MODEL_PATH is set)
)

docker run "${DOCKER_ARGS[@]}" \
  --entrypoint /bin/bash \
  "$IMAGE" \
  -lc '
    # The image may ship PCIe / fused-all-reduce tunables that hurt this 2-GPU box.
    # Clear them so we fall back to the plain NCCL path, then launch the serve script.
    unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE \
          VLLM_ENABLE_PCIE_ALLREDUCE VLLM_PCIE_ALLREDUCE_BACKEND \
          VLLM_CPP_AR_1STAGE_NCCL_CUTOFF VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS \
          VLLM_RTX6K_FUSED_ALLREDUCE_ADD VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER \
          VLLM_CACHE_DIR
    exec /usr/local/bin/serve-ds4-flash.sh
  '

echo "Started '$NAME'. Follow startup with:  docker logs -f $NAME"
echo "First launch after an image change warms the cache (~5 min); reuses $CACHE_DIR otherwise."
```

</details>

Notes on the pieces that were not obvious:

- The model lives locally as `/models/DeepSeek-V4-Flash-0731` and `HF_HUB_OFFLINE=1` keeps the container from ever touching the hub. Get it with the same `hf download deepseek-ai/DeepSeek-V4-Flash-0731 --local-dir /models/DeepSeek-V4-Flash-0731` step as before, the checkpoint is around 160 GB again.
- `SERVED_MODEL_NAME=DeepSeek-V4-Flash` keeps the OpenAI-compatible endpoint model ID unchanged, so anything already pointed at the old server keeps working.
- The first launch JIT-compiles SparkInfer, TileLang, and CUDA graph artifacts into `/cache`; it takes a few minutes once, then it is warm. The runner uses `--privileged` because the Gilded Gnosis image documents it as required for its GPU/NUMA access.
- The `unset` block drops PCIe fused-all-reduce tunables that ship with the Gilded Gnosis image; on this 2-GPU, no-NVLink host the plain NCCL path wins, which is why the script clears them before launching the serve helper.
- `LOAD_FORMAT=safetensors` sidesteps the InstantTensor crash above. The `KV_OFFLOADING_SIZE=0` line keeps native CPU KV offload off, which is the way to go for wall-clock decode on this box now that there is headroom.
- If the model itself is loaded from the 0731 directory, `DSPARK_MODEL` gets set to the same path so the DSpark spec config finds the draft weights in the same checkpoint.

## The numbers in practice

The honest summary is that the new checkpoint is good enough that I have not gone back to the old MTP setup. Single-stream decode measured at 307 tok/s with the safetensors fix versus the 218 tok/s the runbook advertises and the ~210 tok/s I got from the Lucifer build in June. Prefill at 121K context ran at 10.7K tok/s (11.3 s for a 121K prompt), and the 109K needle-in-haystack test passed. Coding with thinking on sits around the advertised 289 tok/s.

Two caveats from the runbook that are worth carrying over. K7 draft depth still has an open long-context output-quality investigation, so fixed K5 is the safe default (which is what I run). And native CPU KV offload is the qualified host-cache path in r24; LMCache's long-context correctness is explicitly not closed for DS4, so I leave `KV_OFFLOADING_SIZE=0` for now. If I need more effective context than what fits in the 2x 96 GB, I would enable native offload (set it to the host capacity in GiB, e.g. 48.5) and re-benchmark, rather than reaching for LMCache.

The whole thing stays inside the same "two GPUs, no NVLink, no datacenter networking" envelope as my earlier posts. The difference is that the model got dramatically better at agentic tasks and the official weights now bring their own speculative decoder. Swapping the checkpoint and the runtime gave me a faster local server and a model that no longer feels like a preview.
