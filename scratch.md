

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

(.venv) zmarty@zmarty-aorus:/git/vllm-nightly$ uv pip install --pre -U --force-reinstall --no-deps vllm \
  --extra-index-url https://wheels.vllm.ai/nightly
Resolved 1 package in 117ms
Prepared 1 package in 4.91s
Uninstalled 1 package in 53ms
Installed 1 package in 25ms
 - vllm==1.0.0.dev20251107+cu130
 + vllm==0.11.1rc6.dev234+gc4768dcf4.cu129


I think this loaded an older rc. Updated with:

(.venv) zmarty@zmarty-aorus:/git/vllm-nightly$ uv pip install --pre -U --force-reinstall --no-deps vllm \
  --extra-index-url https://wheels.vllm.ai/nightly
Resolved 1 package in 117ms
Prepared 1 package in 4.91s
Uninstalled 1 package in 53ms
Installed 1 package in 25ms
 - vllm==1.0.0.dev20251107+cu130
 + vllm==0.11.1rc6.dev234+gc4768dcf4.cu129


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

export NCCL_P2P_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

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

DOES NOT WORK - vllm currently loads this unquantized: "Your NVFP4 235B MoE checkpoint is effectively being treated as (mostly) unquantized BF16, so each GPU ends up with ~94 GiB of weights, which basically fills a 96 GiB RTX PRO 6000."

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
    --max-num-seqs 1 \
    --max-model-len 1280 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8000
```

```console
WORKS

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

112 tokens/sec?
```

```console
sudo vi /etc/default/grub
At the end of GRUB_CMDLINE_LINUX_DEFAULT add md_iommu=on iommu=pt like so:
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash md_iommu=on iommu=pt"
sudo update-grub


[shm_broadcast.py:501] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
[shm_broadcast.py:501] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
[shm_broadcast.py:501] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
[shm_broadcast.py:501] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).

```

---------------------------

---------------------------

---------------------------

---------------------------

---------------------------

---------------------------

---------------------------

---------------------------

---------------------------

Install vllm:
```console
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

Install OpenAI evals
```
FAIL:

cd /git
mkdir evals
cd evals
uv venv --python 3.12 --seed
source .venv/bin/activate
pip install evals

export OPENAI_API_KEY=none
export OPENAI_BASE_URL=http://localhost:8000/
oaieval --help
oaievalset MiniMax-M2-AWQ test
Could not find CompletionFn/Solver in the registry with ID
```

Install https://github.com/EleutherAI/lm-evaluation-harness
```
mkdir lm-evaluation-harness
cd lm-evaluation-harness/
uv venv --python 3.12 --seed
source .venv/bin/activate
pip install "lm_eval[api]"
```

```console
QuantTrio-MiniMax-M2-AWQ

vllm serve \
    /models/awq/QuantTrio-MiniMax-M2-AWQ \
    --served-model-name MiniMax-M2-AWQ \
    --max-num-seqs 10 \
    --max-model-len 128000 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 1 \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think \
    --host 0.0.0.0 \
    --port 8000

"Tell me a very long story"
tp 2 pp 1 - starts at around 122, ends at around 116 tokens/sec after a long story
tp 1 pp 2 - starts at around 118, ends at around 96 tokens/sec after a VERY long story

lm_eval \
   --model local-completions \
   --tasks gsm8k \
   --model_args model=MiniMax-M2-AWQ,base_url=http://127.0.0.1:8000/v1/completions,tokenizer=QuantTrio/MiniMax-M2-AWQ,trust_remote_code=True,num_concurrent=10 \
   --output_path ./results

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9287|±  |0.0071|
|     |       |strict-match    |     5|exact_match|↑  |0.9272|±  |0.0072|

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9348|±  |0.0068|
|     |       |strict-match    |     5|exact_match|↑  |0.9340|±  |0.0068|

--

vllm serve \
    /models/original/GLM-4.5-Air-FP8 \
    --served-model-name GLM-4.5-Air-FP8 \
    --max-num-seqs 10 \
    --max-model-len 128000 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --host 0.0.0.0 \
    --port 8000

"Tell me a very long story"
tp 2 pp 1 - 86 tokens/sec
tp 1 pp 2 - 58 tokens/sec

lm_eval \
   --model local-completions \
   --tasks gsm8k \
   --model_args model=GLM-4.5-Air-FP8,base_url=http://127.0.0.1:8000/v1/completions,tokenizer=zai-org/GLM-4.5-Air-FP8,trust_remote_code=True,num_concurrent=10 \
   --output_path ./results

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8931|±  |0.0085|
|     |       |strict-match    |     5|exact_match|↑  |0.9105|±  |0.0079|

--log_samples \
--

vllm serve \
    /models/awq/QuantTrio-Qwen3-235B-A22B-Thinking-2507-AWQ \
    --served-model-name Qwen3-235B-A22B-Thinking-2507-AWQ \
    --reasoning-parser deepseek_r1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --swap-space 16 \
    --max-num-seqs 10 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8000

"Tell me a very long story"
tp 2 pp 1 - ~96 tokens/sec
tp 1 pp 2 - 70 tokens/sec ... 68 tokens/sec

lm_eval \
   --model local-completions \
   --tasks gsm8k \
   --model_args model=Qwen3-235B-A22B-Thinking-2507-AWQ,base_url=http://127.0.0.1:8000/v1/completions,tokenizer=QuantTrio/Qwen3-235B-A22B-Thinking-2507-AWQ,trust_remote_code=True,num_concurrent=10 \
   --log_samples \
   --output_path ./results
--

vllm serve \
    /models/awq/QuantTrio-Qwen3-VL-235B-A22B-Thinking-AWQ \
    --served-model-name Qwen3-VL-235B-A22B-Thinking-AWQ \
    --reasoning-parser deepseek_r1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --swap-space 16 \
    --max-num-seqs 1 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8000

"Tell me a very long story"
tp 2 pp 1 - 80 tokens/sec
tp 1 pp 2 -  tokens/sec

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.6331|±  |0.0133|
|     |       |strict-match    |     5|exact_match|↑  |0.6717|±  |0.0129|

--

vllm serve \
  /models/original/openai-gpt-oss-120b \
  --served-model-name gpt-oss-120b \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 2 \
  --max_num_seqs 20 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.85 \
  --tool-call-parser openai \
  --reasoning-parser openai_gptoss \
  --enable-auto-tool-choice \
  --host 0.0.0.0 \
  --port 8000

tp 2 pp 1 hangs after the first turn!! - https://github.com/vllm-project/vllm/issues/22361

lm_eval \
   --model local-completions \
   --tasks gsm8k \
   --model_args model=gpt-oss-120b,base_url=http://127.0.0.1:8000/v1/completions,tokenizer=openai/gpt-oss-120b,trust_remote_code=True,num_concurrent=20 \
   --log_samples \
   --output_path ./results

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.4511|±  |0.0137|
|     |       |strict-match    |     5|exact_match|↑  |0.2904|±  |0.0125|
--

hf download nvidia/Qwen3-235B-A22B-NVFP4 --local-dir /models/nvfp4/nvidia/Qwen3-235B-A22B-NVFP4

vllm serve \
    /models/nvfp4/nvidia/Qwen3-235B-A22B-NVFP4 \
    --served-model-name Qwen3-235B-A22B-NVFP4 \
    --reasoning-parser deepseek_r1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --swap-space 16 \
    --max-num-seqs 10 \
    --max-model-len 40960 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8000

lm_eval \
   --model local-completions \
   --tasks gsm8k \
   --model_args model=Qwen3-235B-A22B-NVFP4,base_url=http://127.0.0.1:8000/v1/completions,tokenizer=nvidia/Qwen3-235B-A22B-NVFP4,trust_remote_code=True,num_concurrent=10 \
   --log_samples \
   --output_path ./results

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8757|±  |0.0091|
|     |       |strict-match    |     5|exact_match|↑  |0.8643|±  |0.0094|

lm_eval \
   --model local-completions \
   --tasks mmlu_pro \
   --model_args model=Qwen3-235B-A22B-NVFP4,base_url=http://127.0.0.1:8000/v1/completions,tokenizer=nvidia/Qwen3-235B-A22B-NVFP4,trust_remote_code=True,num_concurrent=10 \
   --log_samples \
   --output_path ./results

lm_eval \
   --model local-chat-completions \
   --tasks mmlu_pro \
   --model_args model=Qwen3-235B-A22B-NVFP4,base_url=http://127.0.0.1:8000/v1/completions,tokenizer=nvidia/Qwen3-235B-A22B-NVFP4,trust_remote_code=True,num_concurrent=10 \
   --log_samples \
   --output_path ./results

--apply_chat_template --gen_kwargs=max_gen_toks=32000

lm_eval \
  --model local-completions \
  --tasks mmlu_pro \
  --model_args model=Qwen3-235B-A22B-NVFP4,\
base_url=http://127.0.0.1:8000/v1/completions,\
tokenizer=nvidia/Qwen3-235B-A22B-NVFP4,\
trust_remote_code=True,\
num_concurrent=10,\
max_length=32768 \
  --gen_kwargs '{"max_gen_toks":1024}' \
  --log_samples \
  --apply_chat_template \
  --output_path ./results

lm_eval \
  --model local-completions \
  --tasks gsm8k \
  --model_args model=Qwen3-235B-A22B-NVFP4,\
base_url=http://127.0.0.1:8000/v1/completions,\
tokenizer=nvidia/Qwen3-235B-A22B-NVFP4,\
trust_remote_code=True,\
num_concurrent=64,\
max_length=32768 \
  --gen_kwargs '{"max_gen_toks":8192}' \
  --log_samples \
  --apply_chat_template \
  --batch_size 64 \
  --num_fewshot 5 \
  --output_path ./results

---

https://huggingface.co/RedHatAI/Qwen3-VL-235B-A22B-Instruct-NVFP4

lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Qwen3-VL-235B-A22B-Instruct-NVFP4",dtype=auto,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True\
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks openllm \
  --batch_size auto

More similar commands at the bottom of this: https://huggingface.co/nm-testing/DeepSeek-R1-Distill-Qwen-32B-NVFP4

---

vllm serve \
    /models/original/GLM-4.6V-FP8/ \
    --served-model-name GLM-4.6V-FP8 \
    --tensor-parallel-size 2 \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --max-num-seqs 10 \
    --max-model-len 131072 \
    --mm-encoder-tp-mode data \
    --mm_processor_cache_type shm \
    --allowed-local-media-path / \
    --host 0.0.0.0 \
    --port 8000

lm_eval \
   --model local-completions \
   --tasks gsm8k \
   --model_args model=GLM-4.6V-FP8,base_url=http://127.0.0.1:8000/v1/completions,tokenizer=zai-org/GLM-4.6V-FP8,trust_remote_code=True,num_concurrent=10 \
   --log_samples \
   --output_path ./results

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.5269|±  |0.0138|
|     |       |strict-match    |     5|exact_match|↑  |0.8711|±  |0.0092|

---

ImportError: cannot import name 'MistralCommonTokenizer' from 'transformers.tokenization_mistral_common' (/git/vllm/.venv/lib/python3.12/site-packages/transformers/tokenization_mistral_common.py). Did you mean: 'MistralTokenizer'?

vllm serve \
    /models/original/Devstral-Small-2-24B-Instruct-2512 \
    --served-model-name Devstral-Small-2-24B-Instruct-2512 \
    --tensor-parallel-size 2 \
    --tool-call-parser mistral \
    --enable-auto-tool-choice \
    --max-num-seqs 10 \
    --max-model-len 262144 \
    --host 0.0.0.0 \
    --port 8000

---

vllm serve \
    /models/gptq/Qwen-Qwen3-235B-A22B-GPTQ-Int4 \
    --served-model-name Qwen3-235B-A22B-GPTQ-Int4 \
    --reasoning-parser deepseek_r1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --swap-space 16 \
    --max-num-seqs 10 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8000

---

# https://docs.unsloth.ai/new/how-to-fine-tune-llms-with-unsloth-and-docker

sudo apt install docker.io
sudo apt-get install -y nvidia-container-toolkit

# Create the docker group (it may already exist, but this command ensures it does):
sudo groupadd docker

# Add your user to the group:
sudo usermod -aG docker $USER

# REBOOT!!!

sudo mkdir -p /unsloth_work/.cache
sudo chown -R $USER:$USER /unsloth_work

docker run -d \
  -e JUPYTER_PASSWORD="mypassword" \
  -e HF_HOME=/workspace/work/.cache/huggingface \
  -e HUGGINGFACE_HUB_CACHE=/workspace/work/.cache/huggingface/hub \
  -e TRANSFORMERS_CACHE=/workspace/work/.cache/huggingface/transformers \
  -e HF_DATASETS_CACHE=/workspace/work/.cache/huggingface/datasets \
  -e XDG_CACHE_HOME=/workspace/work/.cache \
  -p 8888:8888 -p 2222:22 \
  -v /unsloth_work:/workspace/work \
  --gpus all \
  unsloth/unsloth:latest

Access http://localhost:8888/

```

<img width="3809" height="1912" alt="image" src="https://github.com/user-attachments/assets/2e5b0a51-98d5-4b03-9583-c3eeb7de5345" />

<img width="2330" height="1686" alt="image" src="https://github.com/user-attachments/assets/37decead-a119-4889-91d1-982ce5d164e7" />

<img width="2254" height="1318" alt="image" src="https://github.com/user-attachments/assets/10b4b1a2-6585-4099-aecb-cf16753dd095" />

---

```console
vllm nightly!

vllm serve \
    /models/awq/cyankiwi-Devstral-2-123B-Instruct-2512-AWQ-4bit \
    --served-model-name Devstral-2-123B-Instruct-2512-AWQ-4bit \
    --enable-auto-tool-choice \
    --tool-call-parser mistral \
    --max-num-seqs 4 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8000

```

----

```console
cd /git/vllm-nightly
source .venv/bin/activate

export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_MOE_FP16=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=4

vllm serve \
    /models/awq/QuantTrio-MiniMax-M2.1-AWQ \
    --served-model-name MiniMax-M2.1-AWQ \
    --swap-space 16 \
    --max-num-seqs 10 \
    --max-model-len 196608 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000
```

---

```console
# Model configuration (Mandatory)
MODEL="/models/awq/mratsim-MiniMax-M2.1-FP8-INT4-AWQ"
MODELNAME="MiniMax-M2.1-FP8-INT4-AWQ"
GPU_UTIL=0.97
SAMPLER_OVERRIDE='{"temperature": 1, "top_p": 0.95, "top_k": 40, "repetition_penalty": 1.1, "frequency_penalty": 0.40}'

# Prevent memory fragmentation
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

# Prevent vLLM from using 100% CPU when idle (Very Recommended)
export VLLM_SLEEP_WHEN_IDLE=1

vllm serve "${MODEL}" \
  --served-model-name "${MODELNAME}" \
  --trust-remote-code \
  --gpu-memory-utilization ${GPU_UTIL} \
  --tensor-parallel-size 2 \
  --override-generation-config "${SAMPLER_OVERRIDE}" \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think
```

---

```console
vllm serve \
     "/models/original/GLM-4.7-Flash" \
     --served-model-name GLM-4.7-Flash \
     --tensor-parallel-size 2 \
     --speculative-config.method mtp \
     --speculative-config.num_speculative_tokens 1 \
     --tool-call-parser glm47 \
     --reasoning-parser glm45 \
     --enable-auto-tool-choice
```
