# M5 Max model-speed animations

These are **replays of recorded SSE chunks**, not desktop screen recordings or
text animations calculated from an assumed token rate. The text and arrival
times come from real, separate requests to the local inference servers.

## Captures

Data: [`../../assets/data/m5-max-speed/`](../../assets/data/m5-max-speed/)

Hardware: MacBook Pro, M5 Max, 18-core CPU, 40-core GPU, 128 GB unified memory.
MTPLX 2.12.2 for both Qwens; ds4 revision
`c97964b8aa77034a8e17afb10ca7a686c40ab511` for DeepSeek Q4 with SSD streaming.
Captured October 31, 2026 Pacific time (November 1 UTC).

One model loaded at a time, one measured request per model. Each received a
brief, different warmup prompt before the measured request. Model loading is
excluded; prompt processing and first-text latency are included. These are warm
server runs, not controlled cold-SSD-cache experiments. No cached prompt tokens
were reported. All three received this exact prompt:

> Write a Python function that merges overlapping intervals. Include type hints,
> comments, a short explanation, and three example calls. Return the code first.

Temperature 0, thinking disabled, 512-token output cap. Flash-Next generated 506
tokens and stopped, 27B hit the 512-token cap, and DeepSeek generated 489 tokens
and stopped. Different answers and different tokenizers mean this is a speed
demonstration, not a comparison of solution quality or equal amounts of code.

| Model | First text | Total request time | Streamed rate |
| --- | ---: | ---: | ---: |
| Qwen3.8-Flash-Next, mixed 4-bit | 0.167 s | 4.418 s | 119.86 tok/s |
| Qwen3.8-27B, 8-bit | 0.190 s | 9.052 s | 58.17 tok/s |
| DeepSeek-V4.1-Flash, mixed Q4, SSD streaming | 7.087 s | 75.411 s | 7.14 tok/s |

The displayed rate is `(completion_tokens - 1) / (last_text_time - first_text_time)`
for the full response. It is an approximate client-observed rate: SSE can batch
multiple tokens in a chunk. It is not an instantaneous rate or GPU-kernel timing.
JSON files retain every text chunk, timestamp, API usage count and finish reason.

## Reproduce

Start the desired model before each capture. Stop it before starting another.
The capture tool never starts or stops servers itself; inspect `/v1/models` to
confirm the intended server before running it. Changing runtime/kernel versions,
cache state, sampling settings or prompts can change the results.

```bash
python3 tools/model-speed/capture.py \
  --base-url http://127.0.0.1:8000/v1 \
  --label Qwen3.8-Flash-Next --quant '4-bit mixed' \
  --output assets/data/m5-max-speed/flash-next.json

# Switch to Qwen 27B first:
python3 tools/model-speed/capture.py \
  --base-url http://127.0.0.1:8000/v1 \
  --label Qwen3.8-27B --quant '8-bit' \
  --output assets/data/m5-max-speed/qwen-27b.json

# Switch to DeepSeek first:
python3 tools/model-speed/capture.py \
  --base-url http://127.0.0.1:8001/v1 \
  --label DeepSeek-V4.1-Flash --quant 'Q4 mixed / SSD streaming' \
  --output assets/data/m5-max-speed/deepseek.json
```

Render without loading any models:

```bash
python3 -m venv /tmp/model-speed-render
/tmp/model-speed-render/bin/pip install -r tools/model-speed/requirements.txt
/tmp/model-speed-render/bin/python tools/model-speed/render.py
```

Default font: macOS Courier New. On another platform, pass `--font /path/to/monospace.ttf`.
The imageio-ffmpeg package supplies the ffmpeg executable; no global installation
is needed.

## Outputs

- `assets/img/m5-max-speed/request-aligned.gif`: first 20 seconds, requests
  aligned, including initial wait. 1080x750, 5 fps, normal speed, looping.
- `assets/img/m5-max-speed/decode-aligned.gif`: first 15 seconds after each
  model's first text. Same playback speed, but each initial wait is removed.
- Matching PNGs show the 12-second frame for static previews or article covers.
- `assets/video/m5-max-speed/*.mp4`: 1440x1000, H.264, 15 fps, silent,
  normal-speed versions for LinkedIn uploads and downloadable blog alternatives.

The GIFs end before DeepSeek completes. The renderer samples actual chunk arrival
times at the export frame rate; it does not interpolate or invent text arrivals.
The models were run separately, then aligned for display, not run concurrently.

For LinkedIn placement and paste-ready captions, see [LINKEDIN.md](LINKEDIN.md).
