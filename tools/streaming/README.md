# Streaming captures

Run from the repository root on the gaming client:

```bash
python3 tools/streaming/capture.py
```

Select the monitor containing the game in KDE's screen-sharing dialog. The
60-second countdown begins after selection. Enable the streaming statistics
overlay and enter the game. The script records six five-second clips, sounding
a cue when each recording starts, with ten seconds between clips. The first
clip starts after the initial countdown. Ctrl+C stops recording and finalizes
the current clip. Nothing starts on the remote server.

Recordings are silent 1920x1080, 30 fps WebM files, saved under
`~/Videos/streaming-captures/<timestamp>/`. The entire selected monitor is
recorded, including overlays. Use an SDR client display for these GIF captures;
this pipeline does not implement HDR tone mapping. Check the resulting video
and stats readability before choosing a clip. Software video encoding can add
load, so the stats during capture may differ from normal streaming.

Convert a chosen session (replace the timestamp):

```bash
python3 tools/streaming/make-gifs.py ~/Videos/streaming-captures/20261027-220000/*.webm
```

This generates separate `-gameplay.gif` and `-stats.gif` files beside each
recording. The default top-left crop is 640x360 **in recording pixels**. Adjust
it to the overlay's actual bounds:

```bash
python3 tools/streaming/make-gifs.py /path/to/clip-01.webm --stats-crop 0,0,720,420
```

The converter starts at 12 fps, uses a GIF palette, and reduces size/frame rate
as needed to stay below 5,000,000 bytes, 400 frames, and 36 million aggregate
frame pixels. It preserves the source aspect ratio. Review the first frame as
a potential static preview. These are conservative export targets, not a
guarantee of animation in every LinkedIn placement. Current LinkedIn general
media help lists 500 frames or 36,152,320 pixels; it does not establish the
article cover behavior described in the original request:
https://www.linkedin.com/help/linkedin/answer/a792945

Dependencies already present on the inspected KDE Wayland client: Python 3,
`dbus`, PyGObject, GStreamer with PipeWire/VP8/WebM plugins, and Pillow. `paplay`
plays the cue; if absent the script uses a terminal bell. Screen selection
uses the standard XDG ScreenCast portal:
https://flatpak.github.io/xdg-desktop-portal/docs/doc-org.freedesktop.portal.ScreenCast.html

Synthetic smoke test, without recording the desktop:

```bash
python3 tools/streaming/capture.py --test --delay 0 --gap 0 --count 1 --duration 1 --output /tmp/streaming-test
python3 tools/streaming/make-gifs.py /tmp/streaming-test/*/clip-01.webm
```

## Blog and LinkedIn video

The blog plays the selected 1080p WebM recordings at their original 30 fps.
The underlying remote game stream ran at 4K60. GIF previews are hidden by the
blog stylesheet and remain visible in GitHub's Markdown viewer, which strips
the playable video sources. The enlarged stats GIF remains visible on both.

Prepare separate LinkedIn uploads:

```bash
python3 tools/streaming/prepare-linkedin.py
```

Exports go to `~/Videos/streaming-captures/linkedin/`. The intro and gameplay
clips are copied unchanged. `puget-sound-streaming-reel.webm` combines these
two selections once, at 30 fps, for a roughly 10-second cover or body video.
Exports are silent VP8 WebM. The menu and short driving examples were removed
from both the blog assets and the LinkedIn export folder.

LinkedIn documents WebM support and desktop requirements of at least three
seconds, 10–60 fps, and 192 kbps–30 Mbps. Its article editor supports direct
cover and body video uploads; GIF cover images are no longer supported.
Sources checked October 27, 2026:

- https://www.linkedin.com/help/linkedin/answer/a548372
- https://www.linkedin.com/help/linkedin/answer/a521719/
