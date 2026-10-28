#!/usr/bin/env python3
"""Make gameplay and top-left stats GIFs from captured WebM clips.

Requires system Python with GI/GStreamer and Pillow. Crop coordinates refer to
the 1920x1080 recordings, not the original monitor resolution.
"""
import argparse
from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from PIL import Image


def frames(path, start=0, end=None):
    pipeline = Gst.parse_launch('uridecodebin name=source ! videoconvert ! videorate '
        '! video/x-raw,format=RGB,framerate=12/1 ! appsink name=sink sync=false max-buffers=2 drop=false')
    pipeline.get_by_name('source').set_property('uri', path.resolve().as_uri())
    sink = pipeline.get_by_name('sink')
    pipeline.set_state(Gst.State.PLAYING)
    images = []
    try:
        while True:
            sample = sink.emit('try-pull-sample', 15 * Gst.SECOND)
            if sample is None:
                message = pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR)
                if message:
                    raise RuntimeError(str(message.parse_error()))
                if not sink.get_property('eos'):
                    raise RuntimeError('Timed out decoding video')
                break
            caps = sample.get_caps().get_structure(0)
            width, height = caps.get_value('width'), caps.get_value('height')
            buffer = sample.get_buffer()
            timestamp = buffer.pts / Gst.SECOND
            if timestamp < start:
                continue
            if end is not None and timestamp >= end:
                break
            data = buffer.extract_dup(0, buffer.get_size())
            stride = ((width * 3 + 3) // 4) * 4
            images.append(Image.frombytes('RGB', (width, height), data, 'raw', 'RGB', stride))
            if len(images) > 600:
                raise ValueError('Input too long: use short clips (50 seconds maximum)')
    finally:
        pipeline.set_state(Gst.State.NULL)
    if not images:
        raise ValueError(f'No frames in {path}')
    return images


def save(images, target, max_width):
    # Keep under 5 MB, under 400 frames, and conservatively under 36M total pixels.
    for width in [max_width, min(640, max_width), min(480, max_width), min(320, max_width)]:
        for step in [1, 2, 3]:
            selected = images[::step]
            if len(selected) >= 400:
                continue
            size = (min(width, images[0].width),
                    max(1, round(images[0].height * min(width, images[0].width) / images[0].width)))
            if size[0] * size[1] * len(selected) > 36_000_000:
                continue
            output = [im.resize(size, Image.Resampling.LANCZOS).quantize(colors=128)
                      for im in selected]
            # GIF times are in 10ms units; alternate delays to preserve duration.
            duration = [round((i + 1) * step * 100 / 12) * 10 -
                        round(i * step * 100 / 12) * 10 for i in range(len(output))]
            output[0].save(target, save_all=True, append_images=output[1:],
                           duration=duration, loop=0, optimize=True, disposal=2)
            if target.stat().st_size < 5_000_000:
                with Image.open(target) as check:
                    assert check.n_frames < 400
                print(f'{target}: {size[0]}x{size[1]}, {len(output)} frames, {target.stat().st_size:,} bytes')
                return
    target.unlink(missing_ok=True)
    raise ValueError('Cannot fit GIF under limits; shorten clip')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('clips', type=Path, nargs='+')
    parser.add_argument('--stats-crop', default='0,0,640,360', help='x,y,width,height in recording pixels')
    parser.add_argument('--start', type=float, default=0, help='Start time in seconds')
    parser.add_argument('--end', type=float, help='End time in seconds (exclusive)')
    args = parser.parse_args()
    if args.start < 0 or (args.end is not None and args.end <= args.start):
        parser.error('Require 0 <= start < end')
    try:
        x, y, w, h = map(int, args.stats_crop.split(','))
        assert min(x, y) >= 0 and min(w, h) > 0
    except (ValueError, AssertionError):
        parser.error('--stats-crop must be x,y,width,height with nonnegative origin and positive size')
    Gst.init(None)
    for clip in args.clips:
        images = frames(clip, args.start, args.end)
        if x + w > images[0].width or y + h > images[0].height:
            parser.error('Stats crop extends outside the recording')
        save(images, clip.with_name(clip.stem + '-gameplay.gif'), 960)
        save([im.crop((x, y, x + w, y + h)) for im in images],
             clip.with_name(clip.stem + '-stats.gif'), w)


if __name__ == '__main__':
    main()
