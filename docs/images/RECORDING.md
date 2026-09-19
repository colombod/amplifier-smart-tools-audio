# chain-animation — how it was made

Rendered with `unfold` (robotdad/amplifier-smart-tool-unfold) in a throwaway Linux container,
destroyed afterwards. Linux support for unfold was proven in the same effort and contributed
upstream as robotdad/amplifier-smart-tool-unfold#2.

Source: `chain-animation.mp4` — 1280x720, 30 fps, 24.000 s, 720 frames confirmed by
`ffprobe -count_frames`.

GIF conversion, two-pass palette at NATIVE resolution. Do not downscale: at 800 px the 1-pixel
grey ceiling and floor rules alias away, and those rules are the only thing the limiter beat is
about.

```bash
ffmpeg -i chain-animation.mp4 -vf "fps=15,palettegen=stats_mode=diff" -y palette.png
ffmpeg -i chain-animation.mp4 -i palette.png \
  -lavfi "fps=15[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3" -y chain-animation.gif
```

## The one rule that made this take work

`unfold` cannot morph geometry — an element's points are fixed at construction, and a tween
moves only opacity, position, scale, rotation and draw. So every shape change is two elements
swapped, and the swap schedule is the whole problem. **Shapes and text need opposite
schedules:**

- **Shapes overlap.** The incoming waveform fades in at `t`, the outgoing one fades out at
  `t + d`. Both sit at full opacity for an instant, so the waveform band is never empty. Two
  amber strokes briefly on top of each other read as one stroke thickening.
- **Text substitutes with a gap.** The outgoing string fades out at `t`, the incoming one starts
  only at `t + d`. Two amber *strings* on top of each other do not read as a thicker string —
  they read as a third string nobody wrote. An earlier take applied the shape rule to text and
  printed the literal word `bounlpressss` in the caption slot, and the readout pair
  `LUFS -21.0  dBTP -0.0`, a number that was neither the old value nor the new one. An empty
  caption slot for a third of a second asserts nothing; a garbled measurement asserts something
  false about the tool the animation exists to explain.

## What was verified, and how

Five checks, all measured on the shipped file, none by eye alone.

1. **`ffprobe` independently**: `h264`, 1280x720, 24.000 s, 30/1 fps, 720 frames counted.
2. **Blank-frame scan, every frame, waveform band (rows 317–442): 0 blank frames**, and 0 frames
   holding 1–39 lit pixels. Lit pixels are counted at full resolution with `bytes.count`; an
   earlier checker reduced each row with an averaging `scale` filter, where 2 lit pixels out of
   1280 average to 0.4, quantize to 0, and invent hundreds of blanks that do not exist.
3. **Ceiling and floor scan** from the rules' first frame (656) to the end: **0 violations each.**
   Negative control: a fake ceiling placed inside the waveform's range fires on 64 frames, so the
   scan is known to be capable of failing. The rules occupy rows 334–335 and 466–467; the
   waveform body sits at row 336 and below, with only the antialiased top edge of the stroke
   (~2 px per frame) reaching row 333 where the flat-topped peaks press against the rule.
4. **Doubled-text scan.** Mechanically, every text slot's amber pixel count falls to exactly 0
   between strings at every one of its transitions — a superimposed pair would count strictly
   more pixels than either string alone, never zero. Then 30 frames spanning every caption
   change and every readout change were transcribed by a vision model cold, with no hint of what
   they should say: 30 single, clean, correctly-spelled strings, and readouts that always move as
   a pair (`-21.4/-3.2` → `-14.0/-0.6` → `-14.0/-1.0`).
5. **Multiband bars and the cut.** The four fills measure 198 / 166 / 106 / 50 px inside 200 px
   tracks; a cold vision read named the least- and most-reduced band unprompted and called the
   levels easy to tell apart. The `cut` beat shortens the waveform 980 px → 730 px (25.5%) and it
   stays short to the final frame; shown the before/after frames aligned, a cold reader said
   "something was removed" without being told an edit had happened.

Both grey rules survive the GIF conversion: 730 grey pixels on each of rows 334–335 and 466–467
in the GIF's own final frames.

## Known limitations

- The caption slot is deliberately empty for about a quarter of a second at each stage change.
  That is the cost of the text rule above and it is the cheaper of the two failures.
- The shortening at `cut` is 25.5%. It is unmistakable when the before/after frames are aligned,
  and unmistakable in motion, but a cold side-by-side of the first and last frames — which differ
  in several other ways at once — did not single it out.
