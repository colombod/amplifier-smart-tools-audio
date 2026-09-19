# chain-animation — how it was made

Rendered with `unfold` (robotdad/amplifier-smart-tool-unfold) in a throwaway Linux container,
destroyed afterwards. Linux support for unfold was proven in the same effort and contributed
upstream as robotdad/amplifier-smart-tool-unfold#2.

Source: `chain-animation.mp4` — 1280x720, 30 fps, 26.000 s, 780 frames confirmed by
`ffprobe -count_frames`.

GIF conversion, two-pass palette at NATIVE resolution. Do not downscale: at 800 px the 1-pixel
grey ceiling and floor rules alias away, and those rules are the only thing the limiter beat is
about.

```bash
ffmpeg -i chain-animation.mp4 -vf "fps=15,palettegen=stats_mode=diff" -y palette.png
ffmpeg -i chain-animation.mp4 -i palette.png \
  -lavfi "fps=15[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3" -y chain-animation.gif
```

## What was verified, and how

- **Every frame from the ceiling line's first appearance to the end was scanned at the pixel
  level: 0 frames have a waveform pixel above the line.** The checker was calibrated against a
  rejected earlier take that *does* cross, and correctly reported 21 violating frames at −10 px
  for it. A checker never seen to fail is not a checker.
- The multiband bars were shown to a vision model cold, with no hint of what to look for; it
  read four distinct fill levels and the right direction unprompted (`high -0.8 dB`,
  `high-mid -3.0 dB`, `low-mid -6.2 dB`, `low -9.5 dB`).
- Readouts were transcribed across 14 frames spanning the loudness and limit beats: every
  sampled frame is a coherent pair, never half-updated.

## Known defect

The waveform blanks entirely for five short stretches (17, 9, 8, 10 and 10 frames) because each
geometry change is built as remove-then-add. Ten of those frames fall inside the `limit` beat,
so 74 of the 84 audited frames positively demonstrate the ceiling invariant and the other 10
show nothing at all. A targeted fix rendered clean on continuity but drifted the geometry so the
envelope punched through the floor rule — the same lie on the bottom half — so it was not
shipped.
