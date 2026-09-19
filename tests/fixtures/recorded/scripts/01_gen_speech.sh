#!/usr/bin/env bash
# Generate the source audio for the recordings.
#   - real speech containing filler words, via piper-tts (MIT, no credential)
#   - the same content resampled to 16000 / 22050 / 44100 / 48000 Hz
#   - a long (~1 min) version made by tiling the same speech 8 times
#   - a no-speech file (silence) and a no-speech file (1 kHz tone)
set -euo pipefail

WAV=/rec/wav
mkdir -p "$WAV" /opt/voices

TEXT="So, um, the first thing we need to do is, uh, check the levels. And then, erm, we can start recording."

# Voice: en_US-lessac-medium (MIT-licensed piper voice, 22050 Hz native).
if [ ! -f /opt/voices/en_US-lessac-medium.onnx ]; then
  cd /opt/voices
  /opt/piper/bin/python -m piper.download_voices en_US-lessac-medium
fi

# Native-rate synthesis. --sentence-silence 1.0 puts a >700 ms gap between
# sentences so the hesitation branch of detect fillers has something real to find.
printf '%s\n' "$TEXT" | /opt/piper/bin/piper \
  -m /opt/voices/en_US-lessac-medium.onnx \
  --data-dir /opt/voices \
  --sentence-silence 1.0 \
  -f "$WAV/speech_native.wav"

ffprobe -v error -show_entries stream=sample_rate,channels,duration -of default=nw=1 "$WAV/speech_native.wav"

# Same CONTENT, four rates. Mono 16-bit PCM throughout; only the rate differs.
for SR in 16000 22050 44100 48000; do
  ffmpeg -y -v error -i "$WAV/speech_native.wav" -ac 1 -ar "$SR" -c:a pcm_s16le "$WAV/speech_short_${SR}.wav"
done

# Long form: the same speech tiled 8 times (~1 minute). The degenerate
# (start == end) word case was previously seen only on a file built this way.
ffmpeg -y -v error -stream_loop 7 -i "$WAV/speech_native.wav" -ac 1 -ar 16000 -c:a pcm_s16le "$WAV/speech_long_16000.wav"
for SR in 22050 44100 48000; do
  ffmpeg -y -v error -stream_loop 7 -i "$WAV/speech_native.wav" -ac 1 -ar "$SR" -c:a pcm_s16le "$WAV/speech_long_${SR}.wav"
done

# No speech at all: digital silence, and a steady 1 kHz tone.
ffmpeg -y -v error -f lavfi -i anullsrc=r=16000:cl=mono -t 5 -c:a pcm_s16le "$WAV/nospeech_silence_16000.wav"
ffmpeg -y -v error -f lavfi -i "sine=frequency=1000:sample_rate=16000:duration=5" -ac 1 -c:a pcm_s16le "$WAV/nospeech_tone_16000.wav"

echo "--- generated ---"
ls -l "$WAV"
sha256sum "$WAV"/*.wav
