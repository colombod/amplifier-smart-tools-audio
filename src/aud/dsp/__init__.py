"""DSP engine: the deterministic mastering chain.

Everything in this package is pure numpy/scipy signal processing -- no AI
provider, no network calls, no credentials. It runs the same way with or
without a model configured, which is the whole point of a "deterministic
path" in this tool.

Sub-modules:
    io          -- read/write audio files (soundfile)
    filters     -- RBJ biquads (peaking/shelf/hpf/lpf) as scipy sos sections
    crossover   -- Linkwitz-Riley band splitting/recombination
    dynamics    -- feed-forward (multiband) compressor
    limiter     -- true-peak aware lookahead brickwall limiter
    saturation  -- oversampled waveshaping (soft/tape/tube)
    loudness    -- BS.1770 / EBU R128 wrapper (pyloudnorm)
    stft        -- STFT analysis / WOLA resynthesis (transform spine only --
                   no masking, ducking, band mapping or gain law)
    bands       -- perceptual band mapping (Hz<->Bark/ERB, band edges,
                   bin->band energy summation -- no spreading functions,
                   masking thresholds, gain laws or ducking)
    analysis    -- honest, numeric-only measurement report
    engine      -- applies an ordered mastering plan (list of stages) in one pass
"""

from __future__ import annotations

__all__: list[str] = []
