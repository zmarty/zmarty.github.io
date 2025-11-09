

```code
https://huggingface.co/QuantTrio/GLM-4.5-Air-AWQ-FP16Mix

huggingface-cli download QuantTrio/GLM-4.5-Air-AWQ-FP16Mix --local-dir /models/original/QuantTrio-GLM-4.5-Air-AWQ-FP16Mix

export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="12.0"

vllm serve \
    "/models/awq/QuantTrio-GLM-4.5-Air-AWQ-FP16Mix/" \
    --served-model-name GLM-4.5-Air-AWQ-FP16Mix \
    --enable-expert-parallel \
    --max-model-len 128000 \
    --swap-space 16 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.97 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
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

cd /git/llama.cpp/build/bin
CUDA_VISIBLE_DEVICES=0,1 \
./llama-server \
  --model  /models/gguf/Unsloth/Qwen3-VL-235B-A22B-Instruct-GGUF/Qwen3-VL-235B-A22B-Instruct-UD-Q5_K_XL-00001-of-00004.gguf \
  --mmproj /models/gguf/Unsloth/Qwen3-VL-235B-A22B-Instruct-GGUF/mmproj-F32.gguf \
  --alias "Qwen3-VL-235B-A22B-Instruct" \
  --jinja \
  --ctx-size 128000 \
  --n-gpu-layers -1 \
  --tensor-split 0.5,0.5 \
  --main-gpu 0 \
  --host 0.0.0.0 --port 8000

We suggest using Temperature=0.7, TopP=0.8, TopK=20, and MinP=0.
```

```console
cd /git/llama.cpp/build/bin
CUDA_VISIBLE_DEVICES=0,1 \
./llama-server \
  --model /models/gguf/Unsloth/GLM-4.5-Air-GGUF/UD-Q8_K_XL/GLM-4.5-Air-UD-Q8_K_XL-00001-of-00003.gguf \
  --alias "GLM-4.5-Air" \
  --jinja \
  --ctx-size 128000 \
  --n-gpu-layers -1 \
  --tensor-split 0.5,0.5 \
  --main-gpu 0 \
  --host 0.0.0.0 --port 8000
```

```console
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
    --max-num-seqs 512 \
    --max-model-len 128000 \
    --gpu-memory-utilization 0.97 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8000

FAIL - nccl
```

```console
https://github.com/flashinfer-ai/flashinfer/issues/1353

ONLY VERY PARTIALLY WORKS - dies after second query :(

docker pull danucore/vllm-cu128-sm120:latest

docker run -it --rm --name vllm-qwen3 \
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
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --swap-space 16 \
    --max-num-seqs 4 \
    --max-model-len 128000 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 2 \
    --host 0.0.0.0 --port 8000
```

https://www.reddit.com/r/LocalLLaMA/comments/1o387tc/benchmarking_llm_inference_on_rtx_4090_rtx_5090/

```code
docker run -it --rm --name vllm-qwen3 \
  --gpus all \
  --ipc=host --shm-size=16g \
  -p 8000:8000 -v /models:/models:ro \
  -e NVIDIA_VISIBLE_DEVICES=0,1 \
  -e VLLM_ATTENTION_BACKEND=FLASHINFER \
  -e NCCL_DEBUG=INFO -e NCCL_IB_DISABLE=1 -e NCCL_P2P_DISABLE=1 \
  -e VLLM_SLEEP_WHEN_IDLE=1 \
  danucore/vllm-cu128-sm120:latest \
  /models/original/GLM-4.5-Air-FP8/ \
    --served-model-name GLM-4.5-Air-FP8 \
    --enable-expert-parallel \
    --enable-auto-tool-choice \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --swap-space 16 \
    --max-num-seqs 4 \
    --max-model-len 128000 \
    --gpu-memory-utilization 0.97 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 2 \
    --host 0.0.0.0 --port 8000

Disable FlashInfer:

docker run -it --rm --name vllm-qwen3 \
  --gpus all \
  --ipc=host --shm-size=16g \
  -p 8000:8000 -v /models:/models:ro \
  -e NVIDIA_VISIBLE_DEVICES=0,1 \
  -e VLLM_DISABLE_FLASHINFER=1 \
  -e VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  -e NCCL_DEBUG=INFO -e NCCL_IB_DISABLE=1 -e NCCL_P2P_DISABLE=1 \
  -e VLLM_SLEEP_WHEN_IDLE=1 \
  danucore/vllm-cu128-sm120:latest \
  /models/original/GLM-4.5-Air-FP8/ \
    --served-model-name GLM-4.5-Air-FP8 \
    --enable-expert-parallel \
    --enable-auto-tool-choice \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --swap-space 16 \
    --max-num-seqs 4 \
    --max-model-len 128000 \
    --gpu-memory-utilization 0.97 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 2 \
    --host 0.0.0.0 --port 8000
```

```console
ComfyUI:

git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# 2) Create & enter a venv (Python 3.13 is recommended by ComfyUI)
python3 -m venv .venv
source .venv/bin/activate
python -V      # should show 3.13.x on Ubuntu 25.04 (ok if 3.12/3.13)

# 3) Upgrade pip/wheel
pip install --upgrade pip wheel

# 4) Install PyTorch with the CUDA 13.0 wheels (includes CUDA runtime)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
# (Nightly, if you want bleeding edge instead of stable)
# pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# 5) Install ComfyUI Python deps
pip install -r requirements.txt

https://comfyanonymous.github.io/ComfyUI_examples/qwen_image/?utm_source=chatgpt.com

Qwen-Image
wget https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors -> ComfyUI/models/diffusion_models
wget https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors -> ComfyUI/models/text_encoders
wget https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors -> ComfyUI/models/vae/

Qwen-Image-Edit-2509
wget https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors -> ComfyUI/models/diffusion_models

python main.py --listen 0.0.0.0 --port 8188

sudo ufw allow 8188/tcp
```

```console
https://github.com/mcmonkeyprojects/SwarmUI
https://github.com/invoke-ai/InvokeAI?tab=readme-ov-file
```

```console
WORKING WORKING WORKING WORKING WORKING

python3 -m venv .venv

source .venv/bin/activate

uv pip install --pre \
  --index-strategy unsafe-best-match \
  'triton-kernels @ git+https://github.com/triton-lang/triton.git@v3.5.0#subdirectory=python/triton_kernels' \
  vllm \
  --extra-index-url https://wheels.vllm.ai/nightly \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu130 \
  --extra-index-url https://download.pytorch.org/whl/xformers/

--

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_ATTENTION_BACKEND=FLASHINFER
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1 #Absolutely required !!!!!
export VLLM_SLEEP_WHEN_IDLE=1

# Pipeline parallel 2. Tensor parallel 2 will fail with vLLM and NCCL hanging forever

vllm serve \
    /models/original/GLM-4.5-Air-FP8 \
    --served-model-name GLM-4.5-Air-FP8 \
    --enable-expert-parallel \
    --max-num-seqs 8 \
    --max-model-len 128000 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 2 \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --host 0.0.0.0 \
    --port 8000

Note I set max-num-seqs low due to FlashInfer 256MB hardcoded limit - for now - see https://github.com/vllm-project/vllm/issues/25342?utm_source=chatgpt.com


--

huggingface-cli download QuantTrio/Qwen3-VL-235B-A22B-Thinking-AWQ --local-dir /models/awq/QuantTrio-Qwen3-VL-235B-A22B-Thinking-AWQ

# Install Qwen-VL utility library (recommended for offline inference)
uv pip install qwen-vl-utils==0.0.14

# ModuleNotFoundError: No module named 'torchvision'
uv pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu130 torchvision

# Qwen3-VL does not support _Backend.FLASHINFER backend now.
export VLLM_DISABLE_FLASHINFER=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1 #Absolutely required !!!!!
export VLLM_SLEEP_WHEN_IDLE=1

vllm serve \
    /models/awq/QuantTrio-Qwen3-VL-235B-A22B-Thinking-AWQ \
    --served-model-name Qwen3-VL-235B-A22B-Thinking-AWQ \
    --enable-expert-parallel \
    --limit-mm-per-prompt.video 0 \
    --mm-encoder-tp-mode data \
    --max-num-seqs 8 \
    --max-model-len 128000 \
    --gpu-memory-utilization 0.97 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8000

---

huggingface-cli download QuantTrio/MiniMax-M2-AWQ --local-dir /models/awq/QuantTrio-MiniMax-M2-AWQ

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_ATTENTION_BACKEND=FLASHINFER
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1 #Absolutely required !!!!!
export VLLM_SLEEP_WHEN_IDLE=1

vllm serve \
    /models/awq/QuantTrio-MiniMax-M2-AWQ \
    --served-model-name MiniMax-M2-AWQ \
    --max-num-seqs 8 \
    --max-model-len 128000 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 2 \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think \
    --host 0.0.0.0 \
    --port 8000

---

# Only working if I disable triton and I use Marlin

export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="12.0"
export VLLM_MXFP4_USE_MARLIN=1

vllm serve \
  /models/original/openai-gpt-oss-120b \
  --served-model-name openai-gpt-oss-120b \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --max_num_seqs 8 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.85 \
  --tool-call-parser openai \
  --reasoning-parser openai_gptoss \
  --enable-auto-tool-choice \
  --host 0.0.0.0 \
  --port 8000

```

```console
GLM 4.5 Air LM Studio Jinja template fix for OpenAI style function calling

https://pastebin.com/CfMw7hFS
```

```console
huggingface-cli download lukealonso/MiniMax-M2-NVFP4 --local-dir /models/nvfp4/lukealonso-MiniMax-M2-NVFP4

docker run --rm \
  --name inference \
  --gpus all \
  --shm-size=32g \
  --ipc=host \
  -p 0.0.0.0:8000:8000 \
  --ulimit memlock=-1 \
  --ulimit nofile=1048576 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_NVLS_ENABLE=0 \
  -e NCCL_P2P_DISABLE=0 \
  -e NCCL_SHM_DISABLE=0 \
  -e VLLM_USE_V1=1 \
  -e VLLM_USE_FLASHINFER_MOE_FP4=1 \
  -e OMP_NUM_THREADS=8 \
  -e SAFETENSORS_FAST_GPU=1 \
  -v /dev/shm:/dev/shm \
  -v /models:/models:ro \
  vllm/vllm-openai:nightly \
  --model /models/nvfp4/lukealonso-MiniMax-M2-NVFP4 \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think \
  --all2all-backend pplx \
  --enable-expert-parallel \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --served-model-name "MiniMax-M2" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-num-batched-tokens 16384 \
  --dtype auto \
  --max-num-seqs 8 \
  --kv-cache-dtype fp8 \
  --host 0.0.0.0 \
  --port 8000

--

export NCCL_DEBUG=INFO
export VLLM_MXFP4_USE_MARLIN=1
export TORCH_CUDA_ARCH_LIST="12.0"
# !!! --disable-custom-all-reduce

# NCCL_P2P_DISABLE=1 !
docker run --rm \
  --name inference \
  --gpus all \
  --shm-size=32g \
  --ipc=host \
  -p 0.0.0.0:8000:8000 \
  --ulimit memlock=-1 \
  --ulimit nofile=1048576 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_NVLS_ENABLE=0 \
  -e NCCL_P2P_DISABLE=1 \
  -e NCCL_SHM_DISABLE=0 \
  -e VLLM_MXFP4_USE_MARLIN=1 \
  -e VLLM_USE_V1=1 \
  -e TORCH_CUDA_ARCH_LIST="12.0" \
  -e VLLM_USE_FLASHINFER_MOE_FP4=1 \
  -e OMP_NUM_THREADS=8 \
  -e SAFETENSORS_FAST_GPU=1 \
  -v /dev/shm:/dev/shm \
  -v /models:/models:ro \
  vllm/vllm-openai:nightly \
  --model /models/nvfp4/lukealonso-MiniMax-M2-NVFP4 \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think \
  --all2all-backend pplx \
  --enable-expert-parallel \
  --disable-custom-all-reduce \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --served-model-name "MiniMax-M2" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-num-batched-tokens 16384 \
  --dtype auto \
  --max-num-seqs 8 \
  --kv-cache-dtype fp8 \
  --host 0.0.0.0 \
  --port 8000

---

# Qwen3-VL does not support _Backend.FLASHINFER backend now.
export VLLM_DISABLE_FLASHINFER=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1 #Absolutely required !!!!!
export VLLM_SLEEP_WHEN_IDLE=1

vllm serve \
    /models/awq/QuantTrio-Qwen3-VL-235B-A22B-Thinking-AWQ \
    --served-model-name Qwen3-VL-235B-A22B-Thinking-AWQ \
    --enable-expert-parallel \
    --swap-space 16 \
    --max-num-seqs 8 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.97 \
    --disable-custom-all-reduce \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8000

-> Fails with This flash attention build does not support headdim not being a multiple of 32
https://github.com/vllm-project/vllm/issues/27562

--limit-mm-per-prompt.video 0
--max-model-len 128000
--async-scheduling
--mm-encoder-tp-mode data
--enable-expert-parallel

---

WORKS 100%

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_ATTENTION_BACKEND=FLASHINFER
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1 #Absolutely required !!!!!
export VLLM_SLEEP_WHEN_IDLE=1

# Pipeline parallel 2. Tensor parallel 2 will fail with vLLM and NCCL hanging forever

vllm serve \
    /models/original/GLM-4.5-Air-FP8 \
    --served-model-name GLM-4.5-Air-FP8 \
    --enable-expert-parallel \
    --max-num-seqs 8 \
    --max-model-len 128000 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 2 \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --host 0.0.0.0 \
    --port 8000

---

# Now try tp 2 but with P2P disable:

export NCCL_P2P_DISABLE=1

DOES NOT WORK ! DOES NOT WORK ! DOES NOT WORK ! DOES NOT WORK ! DOES NOT WORK ! 

vllm serve \
    /models/original/GLM-4.5-Air-FP8 \
    --served-model-name GLM-4.5-Air-FP8 \
    --enable-expert-parallel \
    --max-num-seqs 8 \
    --max-model-len 128000 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --host 0.0.0.0 \
    --port 8000

```

```console
TensortRT-LLM Qwen3-235B-A22B-Thinking-2507-FP4

WORKS WORKS WORKS WORKS WORKS WORKSWORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKSWORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKSWORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKSWORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKSWORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKSWORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKSWORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKSWORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKSWORKS WORKS WORKS WORKS WORKS WORKS 
But needs thinking parsing template?

https://chatgpt.com/share/69110274-c930-8000-ade4-4b37833fee2b

huggingface-cli download NVFP4/Qwen3-235B-A22B-Thinking-2507-FP4 --local-dir /models/nvfp4/NVFP4-Qwen3-235B-A22B-Thinking-2507-FP4

docker run --rm -it \
  -p 8000:8000 \
  -v /models:/models:ro \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --gpus=all \
  nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc2

# To fallback to pytorch instead of TensortRT C++, add: --backend pytorch

export CUDA_VISIBLE_DEVICES=0,1
export TLLM_LOG_LEVEL=DEBUG # or TRACE
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH
export NCCL_IB_DISABLE=1          # you likely have no InfiniBand
export NCCL_NET_GDR_LEVEL=0
export NCCL_PXN_DISABLE=1         # avoid cross-NIC/complex paths
export NCCL_P2P_LEVEL=PIX         # or PHB if GPUs are under different root complexes
export NCCL_SOCKET_IFNAME=^lo,docker0  # keep NCCL off loopback/docker

trtllm-serve "/models/nvfp4/NVFP4-Qwen3-235B-A22B-Thinking-2507-FP4" \
  --host 0.0.0.0 \
  --port 8000 \
  --tp_size 2 \
  --gpus_per_node 2 \
  --max_seq_len 8192 \
  --max_num_tokens 1024 \
  --max_batch_size 1 \
  --log_level debug

Result: 70 tokens/sec
```

```console
TensortRT-LLM openai-gpt-oss-120b

WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS WORKS
Also has thinking template working fine

docker run --rm -it \
  -p 8000:8000 \
  -v /models:/models:ro \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --gpus=all \
  nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc2

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_IB_DISABLE=1          # you likely have no InfiniBand
export NCCL_NET_GDR_LEVEL=0
export NCCL_PXN_DISABLE=1         # avoid cross-NIC/complex paths
export NCCL_P2P_LEVEL=PIX         # or PHB if GPUs are under different root complexes
export NCCL_SOCKET_IFNAME=^lo,docker0  # keep NCCL off loopback/docker

trtllm-serve "/models/original/openai-gpt-oss-120b/" \
  --host 0.0.0.0 \
  --port 8000 \
  --tp_size 2 \
  --gpus_per_node 2 \
  --max_seq_len 8192 \
  --max_num_tokens 1024 \
  --max_batch_size 1 \
  --log_level debug

Result: 185 tokens/sec
```

```console
Tensor RT LLM - GLM-4.5-Air-FP8

DOES NOT WORK DOES NOT WORK DOES NOT WORK DOES NOT WORK DOES NOT WORK DOES NOT WORK DOES NOT WORK DOES NOT WORK DOES NOT WORK DOES NOT WORK
Unsupported quantization error!

docker run --rm -it \
  -p 8000:8000 \
  -v /models:/models:ro \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --gpus=all \
  nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc2

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_IB_DISABLE=1          # you likely have no InfiniBand
export NCCL_NET_GDR_LEVEL=0
export NCCL_PXN_DISABLE=1         # avoid cross-NIC/complex paths
export NCCL_P2P_LEVEL=PIX         # or PHB if GPUs are under different root complexes
export NCCL_SOCKET_IFNAME=^lo,docker0  # keep NCCL off loopback/docker

trtllm-serve "/models/original/GLM-4.5-Air-FP8/" \
  --host 0.0.0.0 \
  --port 8000 \
  --tp_size 2 \
  --gpus_per_node 2 \
  --max_seq_len 8192 \
  --max_num_tokens 1024 \
  --max_batch_size 1


```

```console
MAJOR FAIL
https://github.com/vllm-project/vllm/issues/17676

vllm - GLM-4.5-Air-FP8

cd /git/vllm-nightly/
source .venv/bin/activate

# === GPU selection ===
export CUDA_VISIBLE_DEVICES=0,1

# === NCCL: maximum verbosity + useful extras ===
export NCCL_DEBUG=TRACE                         # TRACE > INFO > WARN
export NCCL_DEBUG_SUBSYS=ALL                    # or: INIT,ENV,GRAPH,COLL,P2P,SHM,NET
# Write one log file per rank: hostname, pid, rank in filename
export NCCL_DEBUG_FILE=/tmp/nccl_%h_%p_%r.log
export NCCL_ASYNC_ERROR_HANDLING=1              # surface async errors sooner

# Keep your existing NCCL env (tweak as needed)
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_PXN_DISABLE=1
export NCCL_P2P_LEVEL=PIX                       # use PHB if needed
export NCCL_SOCKET_IFNAME=^lo,docker0

# === PyTorch distributed: richer diagnostics ===
# DETAIL shows collective shapes, store ops, timeouts, etc. Use INFO for less noise.
export TORCH_DISTRIBUTED_DEBUG=DETAIL
# Optional: make CUDA ops sync to get proper Python stack traces on kernel errors
export CUDA_LAUNCH_BLOCKING=1
# Helpful when debugging OOMs / allocator behavior (noisy):
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

# === vLLM logging ===
# vLLM uses Python logging; this env bumps its internal loggers.
export VLLM_LOGGING_LEVEL=DEBUG

vllm serve \
    /models/original/GLM-4.5-Air-FP8 \
    --served-model-name GLM-4.5-Air-FP8 \
    --enable-expert-parallel \
    --enforce-eager \
    --max-num-seqs 8 \
    --max-model-len 128000 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --host 0.0.0.0 \
    --port 8000
```

```console
vllm - NVFP4-Qwen3-235B-A22B-Thinking-2507-FP4

DOES NOT WORK - needs more investigation! Out of memory?!

cd /git/vllm-nightly/
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_IB_DISABLE=1          # you likely have no InfiniBand
export NCCL_NET_GDR_LEVEL=0
export NCCL_PXN_DISABLE=1         # avoid cross-NIC/complex paths
export NCCL_P2P_LEVEL=PIX         # or PHB if GPUs are under different root complexes
export NCCL_SOCKET_IFNAME=^lo,docker0  # keep NCCL off loopback/docker

vllm serve \
    /models/nvfp4/NVFP4-Qwen3-235B-A22B-Thinking-2507-FP4 \
    --served-model-name Qwen3-235B-A22B-Thinking-2507 \
    --enable-expert-parallel \
    --swap-space 16 \
    --max-num-seqs 8 \
    --max-model-len 128000 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8000
```
