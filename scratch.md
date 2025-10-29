
```code
https://huggingface.co/QuantTrio/GLM-4.5-Air-AWQ-FP16Mix

CONTEXT_LENGTH=32768 vllm serve \
    "/models/original/QuantTrio-GLM-4.5-Air-AWQ-FP16Mix/" \
    --served-model-name GLM-4.5-Air-AWQ-FP16Mix \
    --enable-expert-parallel \
    --swap-space 16 \
    --max-num-seqs 512 \
    --max-model-len $CONTEXT_LENGTH \
    --max-seq-len-to-capture $CONTEXT_LENGTH \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000
```
