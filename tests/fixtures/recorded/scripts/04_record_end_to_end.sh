#!/usr/bin/env bash
# RECORD the end-to-end truth: what `aud detect fillers` actually produces
# for each source file, so there is a captured expected-output to regress
# a replay-backed test against.
set -uo pipefail

OUT=/rec/out/end_to_end
mkdir -p "$OUT"

AUD_MANIFEST=$(aud manifest)
STAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

for WAV in \
  /rec/wav/speech_short_16000.wav \
  /rec/wav/speech_short_22050.wav \
  /rec/wav/speech_short_44100.wav \
  /rec/wav/speech_short_48000.wav \
  /rec/wav/speech_long_16000.wav \
  /rec/wav/nospeech_silence_16000.wav \
  /rec/wav/nospeech_tone_16000.wav
do
  BASE=$(basename "$WAV" .wav)
  echo "detect fillers: $BASE" >&2
  STDOUT=$(aud detect fillers "$WAV" 2>/tmp/e2e.err)
  RC=$?
  SHA=$(sha256sum "$WAV" | cut -d' ' -f1)
  jq -n \
    --arg stamp "$STAMP" \
    --arg wav "$(basename "$WAV")" \
    --arg sha "$SHA" \
    --arg cmd "aud detect fillers $WAV" \
    --argjson manifest "$AUD_MANIFEST" \
    --argjson rc "$RC" \
    --arg stderr "$(cat /tmp/e2e.err)" \
    --argjson out "${STDOUT:-null}" \
    '{provenance:{recorded_at_utc:$stamp, source_wav:$wav, source_wav_sha256:$sha, command:$cmd, aud_manifest:$manifest}, exit_code:$rc, stderr:$stderr, stdout:$out}' \
    > "$OUT/fillers_${BASE}.json"
  echo "  -> $OUT/fillers_${BASE}.json (rc=$RC)" >&2
done

ls -l "$OUT"
