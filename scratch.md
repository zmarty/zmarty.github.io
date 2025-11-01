

```code
https://huggingface.co/QuantTrio/GLM-4.5-Air-AWQ-FP16Mix

huggingface-cli download QuantTrio/GLM-4.5-Air-AWQ-FP16Mix --local-dir /models/original/QuantTrio-GLM-4.5-Air-AWQ-FP16Mix

vllm serve \
    "/models/original/QuantTrio-GLM-4.5-Air-AWQ-FP16Mix/" \
    --served-model-name GLM-4.5-Air-AWQ-FP16Mix \
    --enable-expert-parallel \
    --swap-space 16 \
    --max-num-seqs 512 \
    --gpu-memory-utilization 0.97 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --host 0.0.0.0 \
    --port 8000

---

vllm serve \
    /models/awq/QuantTrio-Qwen3-235B-A22B-Instruct-2507-AWQ \
    --served-model-name Qwen3-235B-A22B-Instruct-2507 \
    --enable-expert-parallel \
    --swap-space 16 \
    --max-num-seqs 512 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.97 \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8000

---

vllm 0.11.0 error hanging on nccl when starting vllm serve...
Tried installing nightly - also fails....

mkdir vllm-nightly
cd vllm-nightly
uv venv --python 3.12 --seed
source .venv/bin/activate

# https://github.com/vllm-project/vllm/issues/27880 - [Installation]: [HELP]How to install the latest main version of vllm #27880
uv pip install -U vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly \
    --extra-index-url https://download.pytorch.org/whl/cu1290 \
    --index-strategy unsafe-best-match \
    --prerelease=allow

pip install flashinfer-python

---

https://www.reddit.com/r/LocalLLaMA/comments/1my3why/rtx_pro_6000_maxq_blackwell_for_llm/
https://www.reddit.com/r/LocalLLaMA/comments/1nj5igv/help_running_2_rtx_pro_6000_blackwell_with_vllm/

This solves it but maybe lower performance??
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

Read more:
https://github.com/vllm-project/vllm/issues/5484
https://github.com/NVIDIA/nccl/issues/631

FlashInfer error:
export VLLM_DISABLE_FLASHINFER=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

https://www.reddit.com/r/LocalLLaMA/comments/1o4m71e/help_with_rtx6000_pros_and_vllm/

```

```console
# Preconditions
# Driver 580.95.05 + CUDA 13.0 (great).
# Two RTX PRO 6000 Blackwell (96 GB) on the same CPU root complex if possible.

mkdir /git/vllm-nightly/
cd /git/vllm-nightly/
git clone https://github.com/vllm-project/vllm.git .

# fresh venv
python3 -m venv .venv && source .venv/bin/activate

# compile kernels for Blackwell (SM_120)
export TORCH_CUDA_ARCH_LIST="12.0"

# PyTorch nightly for CUDA 13.0
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130

# Fix the NCCL version (this unblocks the TP=2 hang)
# Find the latest NCCL and install it, in my case it was 2.28.7
pip install -U nvidia-nccl-cu12

# Verify version:
python - <<'PY'
from importlib.metadata import version, PackageNotFoundError
try:
    print("nvidia-nccl-cu12:", version("nvidia-nccl-cu12"))
except PackageNotFoundError as e:
    print("Not installed:", e)
PY



```
