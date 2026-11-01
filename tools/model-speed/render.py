#!/usr/bin/env python3
"""Render captured SSE timing data; requires Pillow and imageio-ffmpeg."""
import argparse
import json
from pathlib import Path
import subprocess
import textwrap

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "assets/data/m5-max-speed"
OUT = ROOT / "assets/img/m5-max-speed"
VIDEO = ROOT / "assets/video/m5-max-speed"
W, H = 1440, 1000
BG = "#0b1020"
PANEL = "#131c2f"
TEXT = "#e6edf7"
MUTED = "#9caec8"
COLORS = ["#57dfc3", "#91acff", "#ffbf78"]


def load_font(size, bold=False):
    # Override with --font for rendering on Linux or another platform.
    name = ARGS.font or "/System/Library/Fonts/Supplemental/Courier New.ttf"
    if bold and not ARGS.font:
        name = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf"
    return ImageFont.truetype(name, size)


def render(records, t, aligned):
    im = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(im)
    draw.text((42, 28), "Three local models. One MacBook Pro.", font=load_font(39, True), fill=TEXT)
    draw.text((44, 83), "M5 Max / 40-core GPU / 128 GB unified memory", font=load_font(22), fill=MUTED)
    draw.rounded_rectangle((38, 126, W-38, 216), radius=14, fill=PANEL)
    draw.text((58, 138), "PROMPT", font=load_font(17, True), fill=COLORS[0])
    draw.text((58, 167), "Merge overlapping intervals in Python; add types, comments and examples.", font=load_font(23), fill=TEXT)
    subtitle = "FIRST TEXT ALIGNED / wait removed" if aligned else "REQUEST START ALIGNED / includes the wait"
    draw.text((44, 235), subtitle, font=load_font(19, True), fill=MUTED)
    draw.text((1112, 231), f"{t:04.1f} seconds", font=load_font(26, True), fill=TEXT)
    margin, gap = 38, 18
    pw = (W-2*margin-2*gap)//3
    for i, r in enumerate(records):
        x = margin+i*(pw+gap)
        color = COLORS[i]
        draw.rounded_rectangle((x, 280, x+pw, 864), radius=16, fill=PANEL, outline="#29364d", width=2)
        draw.rounded_rectangle((x+17, 301, x+22, 351), radius=2, fill=color)
        draw.text((x+36, 301), r["label"], font=load_font(23, True), fill=TEXT)
        draw.text((x+36, 337), r["quant"], font=load_font(18), fill=MUTED)
        draw.text((x+20, 378), f'{r["streamed_tokens_per_s"]:.1f} tok/s', font=load_font(32, True), fill=color)
        draw.text((x+20, 421), f'First text: {r["time_to_first_text_s"]:.2f}s', font=load_font(19), fill=MUTED)
        draw.line((x+19, 457, x+pw-19, 457), fill="#29364d", width=2)
        origin = r["events"][0]["t"] if aligned else 0
        elapsed = t+origin
        text = "".join(e["text"] for e in r["events"] if e["t"] <= elapsed)
        lines = []
        for line in text.replace("\t", "    ").splitlines():
            lines.extend(textwrap.wrap(line, width=34, replace_whitespace=False, drop_whitespace=False) or [""])
        if not text:
            draw.text((x+22, 485), "Waiting for first text...", font=load_font(18), fill=MUTED)
        else:
            for n, line in enumerate(lines[-12:]):
                draw.text((x+22, 480+n*25), line, font=load_font(19), fill=TEXT)
        done = elapsed >= r["total_s"]
        if done:
            status = "Output cap reached" if r.get("finish_reason") == "length" else "Finished"
            status += f' / {r["usage"]["completion_tokens"]} tokens'
        else:
            status = "Streaming / latest lines" if text else "Processing prompt"
        draw.text((x+20, 810), status, font=load_font(17, True), fill=color if done else MUTED)
        fraction = min(1, elapsed/r["total_s"])
        draw.rounded_rectangle((x+20, 846, x+pw-20, 850), radius=2, fill="#29364d")
        if fraction > 0:
            draw.rounded_rectangle((x+20, 846, x+20+(pw-40)*fraction, 850), radius=2, fill=color)
    draw.text((42, 886), "REAL STREAM TIMINGS / separate runs / normal speed / thinking off", font=load_font(21, True), fill=TEXT)
    draw.text((42, 925), "Same prompt / temperature 0 / 512-token cap / brief warmup before each run", font=load_font(19), fill=MUTED)
    draw.text((42, 958), "Rates: full-run client-observed average after first text. One run each; answers differ.", font=load_font(17), fill=MUTED)
    return im


def export(records, aligned):
    duration = 15 if aligned else 20
    stem = "decode-aligned" if aligned else "request-aligned"
    OUT.mkdir(parents=True, exist_ok=True)
    VIDEO.mkdir(parents=True, exist_ok=True)
    target = VIDEO / (stem+".mp4")
    fps = 15
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
           "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    gif_frames = []
    # All frames use one palette to avoid color flicker.
    palette = render(records, duration, aligned).resize((1080, 750), Image.Resampling.LANCZOS).quantize(colors=128)
    try:
        for n in range(duration*fps):
            frame = render(records, n/fps, aligned)
            proc.stdin.write(frame.tobytes())
            if n % 3 == 0:  # 5 fps GIF, 15 fps MP4; both play at real speed.
                gif_frames.append(frame.resize((1080, 750), Image.Resampling.LANCZOS).quantize(palette=palette, dither=Image.Dither.NONE))
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg encoding failed")
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    gif = OUT / (stem+".gif")
    gif_frames[0].save(gif, save_all=True, append_images=gif_frames[1:], duration=200, loop=0, optimize=True, disposal=1)
    still = OUT / (stem+".png")
    render(records, 12, aligned).save(still, optimize=True)
    print(f"{target}: {target.stat().st_size:,} bytes")
    print(f"{gif}: {gif.stat().st_size:,} bytes / {len(gif_frames)} frames")
    print(still)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", help="Path to a monospace TTF (default: macOS Courier New)")
    ARGS = parser.parse_args()
    records = [json.loads((DATA / name).read_text()) for name in ["flash-next.json", "qwen-27b.json", "deepseek.json"]]
    for record in records:
        assert all(a["t"] <= b["t"] for a,b in zip(record["events"], record["events"][1:]))
        assert record["usage"].get("completion_tokens_details", {}).get("reasoning_tokens", 0) == 0
    export(records, aligned=False)
    export(records, aligned=True)
