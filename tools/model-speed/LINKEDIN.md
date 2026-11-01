# LinkedIn article: where to put the visuals

## Main visual

In **Local speed tests**, keep the prose version of the original speed table.
Immediately after the paragraph explaining those tests, add a heading:

> Watch them generate

Then paste:

> I also recorded all three writing a Python function to merge overlapping
> intervals. Each ran on its own, with thinking off, temperature 0 and a 512-token
> output cap. Here are the responses side by side, replayed at normal speed.

Upload `assets/video/m5-max-speed/request-aligned.mp4` as a body video at this
point. If you want an image instead, use
`assets/img/m5-max-speed/request-aligned.png`. Use the MP4 for motion; the PNG
is static. The matching GIF is available for placements that support animation.

Caption:

> M5 Max, 128 GB. Same prompt, separate runs, requests aligned. Flash-Next
> finished in 4.4 seconds; 27B hit its output cap in 9.1 seconds. DeepSeek took
> 7.1 seconds to start writing and 75.4 seconds to finish. This clip shows the
> first 20 seconds.

Follow with:

> On this short prompt with thinking off, the generation rates were about 120
> tokens/second for Flash-Next, 58 for the 8-bit 27B, and 7 for DeepSeek Q4
> streaming from SSD. These were fresh measurements from the streamed responses;
> the earlier tests used longer outputs and included Qwen's reasoning tokens.

Then continue with **DeepSeek: possible, but bring patience**.

## Optional second visual

If you want both clips, insert `assets/video/m5-max-speed/decode-aligned.mp4`
immediately after the main clip's explanation, before the DeepSeek section.
Otherwise omit it to keep the article short.

Caption:

> Same recordings, with each model's initial wait removed. The first text is
> lined up at zero so you can compare generation speed. Fifteen seconds,
> normal speed.

Static alternative: `assets/img/m5-max-speed/decode-aligned.png`.

## Cover and promotion

The request-aligned PNG can also be the article cover. It repeats the body
visual, so it is optional. For a separate feed post promoting the article,
upload the request-aligned MP4 and put the article link in the accompanying
text. Keep the MP4s rather than relying on GIF animation in LinkedIn.

All paths are relative to the repository root. Public files after deployment:

- https://www.ovidiudan.com/assets/video/m5-max-speed/request-aligned.mp4
- https://www.ovidiudan.com/assets/video/m5-max-speed/decode-aligned.mp4
- https://www.ovidiudan.com/assets/img/m5-max-speed/request-aligned.png
- https://www.ovidiudan.com/assets/img/m5-max-speed/decode-aligned.png
