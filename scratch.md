

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
# Driver 580.95.05 + CUDA 13.0
# Two RTX PRO 6000 Blackwell (96 GB) on the same CPU root complex if possible.

mkdir /git/vllm-nightly/
cd /git/vllm-nightly/
git clone https://github.com/vllm-project/vllm.git .

# Fresh venv. Note that Python 3.12 is important for vllm compatibility, do not attempt with newer Python.
uv venv --python 3.12 --seed
source .venv/bin/activate

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

# Install/Compile vllm
pip install -U pip setuptools wheel

pip install -U flashinfer-python

# Install vLLM from source but **skip dependency resolution** to avoid the xformers pin
# Compile and keep a build log
export VLLM_USE_PRECOMPILED=0
pip install --no-deps -e .
pip install --no-deps -e . -v 2>&1 | tee build.log

While vllm is compiling you will see a lot of warnings. It does not mean anything is wrong. Just check the CPU usage and monitor the steps. From time to time you will see something lke [321/489] which tells you it is at step 321 out of 489.

At the end you will see something like `Successfully installed vllm-0.11.1rc6.dev45+gc2ed069b3.cu130`

# Let's pin Torch back to a 2.9 nightly for cu130 (this is the combo most folks have working on Blackwell):
pip uninstall -y torch torchvision torchaudio


# Add the minimal runtime deps (no xformers)
pip install -U "transformers>=4.49" safetensors sentencepiece \
  fastapi uvicorn "pydantic<3" numpy packaging psutil einops \
  huggingface_hub hf_transfer grpcio

pip install -U \
  aiohttp cloudpickle diskcache msgspec pillow protobuf pyzmq \
  prometheus_client prometheus-fastapi-instrumentator \
  "ray[cgraph]==2.48.0" scipy setproctitle tiktoken gguf \
  "lark==1.2.2" "outlines_core==0.2.11" partial-json-parser \
  "lm-format-enforcer==0.11.3" \
  openai openai-harmony blake3 cachetools cbor2 py-cpuinfo pybase64 six \
  watchfiles "compressed-tensors==0.12.2" "depyf==0.20.0" \
  opencv-python-headless mistral_common[audio,image] xgrammar

# exact versions vLLM warned about:
pip install \
  "anthropic==0.71.0" \
  "lark==1.2.2" \
  "outlines_core==0.2.11" \
  "lm-format-enforcer==0.11.3" \
  "xgrammar==0.1.25" \
  python-json-logger

python -m pip install --pre "torch>=2.9.0.dev0,<2.10" \
  --index-url https://download.pytorch.org/whl/nightly/cu130

# Sanity check
python - <<'PY'
import torch, importlib.metadata as im
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "CUDA ok?", torch.cuda.is_available())
try: print("NCCL:", im.version("nvidia-nccl-cu12"))
except: print("NCCL wheel not found in this venv")
print("CC:", torch.cuda.get_device_capability(0))
PY

# Above you want to see
# Torch: 2.9.0.dev20250909+cu130 CUDA: 13.0 CUDA ok? True
# NCCL: 2.28.7
# CC: (12, 0)

# Running vllm server failed with
# Applying https://github.com/vllm-project/vllm/pull/26844
git fetch origin pull/26844/head:pr-26844
git merge pr-26844 --no-edit

pip install llguidance
pip install uvloop
pip install python-multipart
pip install numba

# Test vllm serve
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_ATTENTION_BACKEND=FLASHINFER
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1 #Absolutely required !!!!!
export VLLM_SLEEP_WHEN_IDLE=1

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
```

```console
mkdir -p /models/gguf/Unsloth/GLM-4.5-Air-GGUF/UD-Q8_K_XL

hf download unsloth/GLM-4.5-Air-GGUF \
  "UD-Q8_K_XL/GLM-4.5-Air-UD-Q8_K_XL-00001-of-00003.gguf" \
  "UD-Q8_K_XL/GLM-4.5-Air-UD-Q8_K_XL-00002-of-00003.gguf" \
  "UD-Q8_K_XL/GLM-4.5-Air-UD-Q8_K_XL-00003-of-00003.gguf" \
  --local-dir /models/gguf/Unsloth/GLM-4.5-Air-GGUF
```

```console

docker pull danucore/vllm-cu128-sm120:latest

docker run --rm --name vllm-qwen3 \
  --gpus all \
  --ipc=host --shm-size=16g \
  -p 8000:8000 -v /models:/models:ro \
  -e NVIDIA_VISIBLE_DEVICES=0,1 \
  -e VLLM_ATTENTION_BACKEND=FLASHINFER \
  -e NCCL_DEBUG=INFO -e NCCL_IB_DISABLE=1 -e NCCL_P2P_DISABLE=1 \
  -e VLLM_SLEEP_WHEN_IDLE=1 \
  danucore/vllm-cu128-sm120:latest \
  /models/awq/QuantTrio-Qwen3-235B-A22B-Instruct-2507-AWQ \
    --served-model-name Qwen3-235B-A22B-Instruct-2507 \
    --enable-expert-parallel \
    --swap-space 16 \
    --max-num-seqs 512 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.97 \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 --port 8000
```

```console
cd /git/llama/llama-b6919-bin-ubuntu-x64/build/bin
CUDA_VISIBLE_DEVICES=0,1 \
./llama-server \
  --model  /models/gguf/Unsloth/Qwen3-VL-235B-A22B-Instruct-GGUF/Qwen3-VL-235B-A22B-Instruct-UD-Q5_K_XL-00001-of-00004.gguf \
  --mmproj /models/gguf/Unsloth/Qwen3-VL-235B-A22B-Instruct-GGUF/mmproj-F32.gguf \
  --alias "Qwen3-VL-235B-A22B-Instruct" \
  --n-gpu-layers -1 \
  --tensor-split 0.5,0.5 \
  --main-gpu 0 \
  --host 0.0.0.0 --port 8000

FAIL - CPU ONLY@
```

```console
cd /git
git clone https://github.com/ggml-org/llama.cpp

sudo apt install cmake
sudo apt install libcurl4-openssl-dev

cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)

cd /git/llama.cpp/build/bin
CUDA_VISIBLE_DEVICES=0,1 \
./llama-server \
  --model  /models/gguf/Unsloth/Qwen3-VL-235B-A22B-Instruct-GGUF/Qwen3-VL-235B-A22B-Instruct-UD-Q5_K_XL-00001-of-00004.gguf \
  --mmproj /models/gguf/Unsloth/Qwen3-VL-235B-A22B-Instruct-GGUF/mmproj-F32.gguf \
  --alias "Qwen3-VL-235B-A22B-Instruct" \
  --jinja \
  --ctx-size 262144 \
  --n-gpu-layers -1 \
  --tensor-split 0.5,0.5 \
  --main-gpu 0 \
  --host 0.0.0.0 --port 8000

We suggest using Temperature=0.7, TopP=0.8, TopK=20, and MinP=0.
```
