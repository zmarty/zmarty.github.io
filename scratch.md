

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
Tried installing nightly

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

```
