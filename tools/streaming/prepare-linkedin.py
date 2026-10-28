#!/usr/bin/env python3
"""Export selected 30 fps WebM clips for LinkedIn's three-second minimum.

The intro and gameplay selections are copied unchanged. A combined
reel contains each selection once. Requires GI, GStreamer and VP8 plugins.
"""
import argparse
from pathlib import Path
import shutil

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst


def encode(paths, target):
    pipeline = Gst.parse_launch(
        'appsrc name=input format=time block=true max-bytes=16777216 '
        '! queue ! vp8enc deadline=1 cpu-used=4 target-bitrate=12000000 threads=4 '
        '! webmmux ! filesink name=output')
    pipeline.get_by_name('output').set_property('location', str(target))
    source = pipeline.get_by_name('input')
    offset = 0
    frames = 0
    try:
        for path in paths:
            decoder = Gst.parse_launch(
                'uridecodebin name=input ! videoconvert '
                '! video/x-raw,format=I420 ! appsink name=sink sync=false max-buffers=2 drop=false')
            decoder.get_by_name('input').set_property('uri', path.as_uri())
            sink = decoder.get_by_name('sink')
            count = 0
            try:
                decoder.set_state(Gst.State.PLAYING)
                while True:
                    sample = sink.emit('try-pull-sample', 15 * Gst.SECOND)
                    if sample is None:
                        error = decoder.get_bus().pop_filtered(Gst.MessageType.ERROR)
                        if error:
                            raise RuntimeError(str(error.parse_error()))
                        if not sink.get_property('eos'):
                            raise RuntimeError(f'Decoder timed out: {path}')
                        break
                    if frames == 0:
                        source.set_property('caps', sample.get_caps())
                        pipeline.set_state(Gst.State.PLAYING)
                    buffer = sample.get_buffer().copy_deep()
                    # These source recordings are 30 fps. Normalize timestamps
                    # to make joins and repeated selections continuous.
                    buffer.pts = (offset + count) * Gst.SECOND // 30
                    buffer.dts = Gst.CLOCK_TIME_NONE
                    buffer.duration = ((offset + count + 1) * Gst.SECOND // 30) - buffer.pts
                    if source.emit('push-buffer', buffer) != Gst.FlowReturn.OK:
                        raise RuntimeError('Encoder rejected a frame')
                    frames += 1
                    count += 1
            finally:
                decoder.set_state(Gst.State.NULL)
            if not count:
                raise RuntimeError(f'No frames decoded: {path}')
            offset += count
        source.emit('end-of-stream')
        message = pipeline.get_bus().timed_pop_filtered(
            60 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
        if message is None:
            raise RuntimeError('Encoder timed out')
        if message.type == Gst.MessageType.ERROR:
            raise RuntimeError(str(message.parse_error()))
    finally:
        pipeline.set_state(Gst.State.NULL)
    print(f'{target.name}: {frames / 30:.2f}s, 30 fps, {target.stat().st_size / 1e6:.2f} MB', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path.home() / 'Videos/streaming-captures/linkedin')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    Gst.init(None)
    root = Path(__file__).resolve().parents[2] / 'assets/video/game-streaming'
    names = ['uncharted-intro', 'uncharted-gameplay']
    paths = [root / (name + '.webm') for name in names]
    for path in paths:
        target = output / path.name
        shutil.copy2(path, target)
        print(f'{target.name}: copied full 30 fps selection', flush=True)
    encode(paths, output / 'puget-sound-streaming-reel.webm')
    print(f'Exports ready: {output}')


if __name__ == '__main__':
    main()
