#!/usr/bin/env python3
"""KDE/Wayland capture: select monitor, wait 60s, six 5s clips with 10s gaps.

Uses Ubuntu's python3-dbus, python3-gi, gir1.2-gstreamer-1.0,
gstreamer1.0-pipewire and GStreamer good/base plugins. No ffmpeg needed.
"""
import argparse
import os
from pathlib import Path
import subprocess
import time
import uuid

import dbus
import dbus.mainloop.glib
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst


def request(bus, method, *args, **options):
    token = 'capture_' + uuid.uuid4().hex
    sender = bus.get_unique_name()[1:].replace('.', '_')
    path = f'/org/freedesktop/portal/desktop/request/{sender}/{token}'
    loop = GLib.MainLoop()
    result = []
    def response(code, data):
        result.extend((code, data))
        loop.quit()
    match = bus.add_signal_receiver(response, signal_name='Response',
        dbus_interface='org.freedesktop.portal.Request', path=path)
    try:
        method(*args, dbus.Dictionary({'handle_token': token, **options}, signature='sv'))
        if not result:
            loop.run()
        if result[0] != 0:
            raise RuntimeError('Screen sharing cancelled or denied')
        return result[1]
    finally:
        match.remove()


def beep():
    sound = '/usr/share/sounds/freedesktop/stereo/complete.oga'
    try:
        subprocess.Popen(['paplay', sound], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print('\a', end='', flush=True)


def record(fd, node, path, seconds, test=False):
    source = 'videotestsrc is-live=true' if test else f'pipewiresrc fd={fd} path={node} do-timestamp=true'
    pipeline = Gst.parse_launch(source + ' ! queue ! videoconvert ! videoscale ! videorate '
        '! video/x-raw,width=1920,height=1080,framerate=30/1 '
        '! vp8enc deadline=1 cpu-used=8 target-bitrate=20000000 threads=4 '
        '! webmmux ! filesink name=output')
    pipeline.get_by_name('output').set_property('location', str(path))
    bus = pipeline.get_bus()
    try:
        pipeline.set_state(Gst.State.PLAYING)
        # Start duration when frames actually reach the encoder, not during setup.
        _, state, _ = pipeline.get_state(10 * Gst.SECOND)
        if state != Gst.State.PLAYING:
            raise RuntimeError('Recorder did not start')
        beep()
        message = bus.timed_pop_filtered(int(seconds * Gst.SECOND), Gst.MessageType.ERROR)
        if message:
            raise RuntimeError(str(message.parse_error()))
    finally:
        pipeline.send_event(Gst.Event.new_eos())
        message = bus.timed_pop_filtered(10 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
        pipeline.set_state(Gst.State.NULL)
    if message and message.type == Gst.MessageType.ERROR:
        raise RuntimeError(str(message.parse_error()))
    if not message:
        raise RuntimeError('Recorder did not finish writing the clip')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path.home() / 'Videos' / 'streaming-captures')
    parser.add_argument('--delay', type=float, default=60)
    parser.add_argument('--gap', type=float, default=10)
    parser.add_argument('--duration', type=float, default=5)
    parser.add_argument('--count', type=int, default=6)
    parser.add_argument('--test', action='store_true', help='Synthetic video; no screen sharing')
    args = parser.parse_args()
    if min(args.delay, args.gap) < 0 or args.duration <= 0 or args.count < 1:
        parser.error('Durations/count must be positive; delay/gap may be zero')
    Gst.init(None)
    for element in ['vp8enc', 'webmmux', 'videoconvert', 'videoscale', 'videorate',
                    'videotestsrc' if args.test else 'pipewiresrc']:
        if not Gst.ElementFactory.find(element):
            parser.error(f'Missing GStreamer element: {element}')
    output = args.output / time.strftime('%Y%m%d-%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    session = None
    fd = None
    try:
        node = None
        if not args.test:
            dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
            bus = dbus.SessionBus()
            portal = dbus.Interface(bus.get_object('org.freedesktop.portal.Desktop',
                '/org/freedesktop/portal/desktop'), 'org.freedesktop.portal.ScreenCast')
            session = request(bus, portal.CreateSession,
                session_handle_token='session_' + uuid.uuid4().hex)['session_handle']
            request(bus, portal.SelectSources, session, types=dbus.UInt32(1),
                    multiple=dbus.Boolean(False), cursor_mode=dbus.UInt32(2))
            print('Select the monitor with the game in the screen-sharing dialog.', flush=True)
            stream = request(bus, portal.Start, session, '')['streams'][0]
            node = stream[0]
            fd = portal.OpenPipeWireRemote(session, dbus.Dictionary({}, signature='sv')).take()
        print(f'Get into the game. First clip in {args.delay:g} seconds. Saving to {output}', flush=True)
        time.sleep(args.delay)
        for index in range(args.count):
            if index:
                time.sleep(args.gap)
            path = output / f'clip-{index + 1:02d}.webm'
            print(f'Recording {path.name} ({args.duration:g}s)', flush=True)
            record(fd, node, path, args.duration, args.test)
        print(f'Done: {output}')
    finally:
        if fd is not None:
            os.close(fd)
        if session is not None:
            dbus.Interface(bus.get_object('org.freedesktop.portal.Desktop', session),
                           'org.freedesktop.portal.Session').Close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped.')
