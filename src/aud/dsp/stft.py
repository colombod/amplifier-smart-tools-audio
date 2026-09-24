"""STFT analysis / WOLA resynthesis: the transform spine, nothing else.

This module is deliberately narrow: it turns a signal into a complex
time-frequency array and back again, exactly. It does NOT mask, duck, map
bands or apply a gain law -- those are a separate concern (multi-track /
sidechain work lives in a separate tool; see AGENTS.md). Every caller that
wants to *do* something with the spectrum (denoise, de-ess, de-reverb, ...)
sits on top of `analyze`/`resynthesize`; this module only has to guarantee
that with an untouched (unity-gain) spectrum, what comes back out is what
went in, to float64 precision.

ShortTimeFFT vs. the legacy `scipy.signal.stft`/`istft` trio
--------------------------------------------------------------
This module uses `scipy.signal.ShortTimeFFT` exclusively. `stft`/`istft`
were marked legacy in scipy 1.12 ("new code should use `ShortTimeFFT`"), and
`ShortTimeFFT.istft` itself had two correctness bugs fixed in scipy 1.15 --
so this module requires `scipy>=1.15` (bumped in `pyproject.toml`; the
previous `>=1.14` floor predates the fix). Building new code on a
documented-legacy, previously-buggy API would be choosing the worse of two
already-installed options for no reason.

Weighted overlap-add (WOLA), not bare overlap-add
--------------------------------------------------
`analyze`/`resynthesize` implement WOLA: the SAME window is applied both at
analysis (before the forward FFT) and at synthesis (after the inverse FFT,
before the overlap-add), matching the "sqrt-Hann x sqrt-Hann" / "Hann x
Hann" style pairings most audio engines actually ship, rather than bare
overlap-add (an analysis window with no taper on the resynthesis side).
Why WOLA rather than bare OLA, stated with its evidence and that evidence's
limits. The standard argument is Smith's: a synthesis window is the correct
form for instantaneous spectral modification, because it tapers away the
discontinuities a modified frame introduces at its own edges, which bare
overlap-add carries straight into the output.

An earlier design note for this module also cited a specific figure -- that
against a brick-wall mask targeting -80 dB, bare OLA reached only -61 dB
while WOLA reached the full -80 dB. **That figure did not reproduce when
this module was built**: a second attempt measured roughly -80 dB for both,
on the reasoning that either is an exact filter-bank inverse once correctly
normalised against its own LTI gain. The disagreement is unresolved and the
number is therefore NOT relied upon here. It is recorded rather than deleted
so that nobody re-derives it and assumes it was never checked.

What does hold, and is verified by this module's own tests, is narrower and
sufficient: the literal same-window-twice WOLA scheme has reconstruction
failure modes that depend on the window/hop pair, and this module surfaces
them instead of hiding them (see below). Masking behaviour is out of scope
here -- any comparison involving a frequency mask belongs to whichever
module applies one.

Correctness here is NOT "any dual window scipy can solve for". `ShortTimeFFT`
will, if left to compute its own canonical (minimum-energy) dual window,
find a mathematically valid inverse for almost any NOLA-compliant analysis
window -- even ones for which the literal "same window twice" WOLA scheme
would reconstruct badly. That would silently hide exactly the failure mode
this module needs to surface (a window/hop choice that is fine on its own
but a bad WOLA pair with itself). So `resynthesize` always passes its own
explicit `dual_win` -- the analysis window itself, scaled by the measured
constant-overlap-add sum of `window**2` at the given hop -- rather than
leaving scipy to compute a different, non-literal dual. Verified empirically
(see tests/test_dsp_stft.py and `check_cola_nola`):

    periodic Hann,           hop N/4, N/8  -> exact reconstruction
    Hann x Hann (same win),  hop N/2       -> reconstruction FAILS
    sqrt-Hann x sqrt-Hann,   hop N/2, N/4  -> exact reconstruction

`scipy.signal.get_window` returns a periodic (DFT-even) window by default
(`fftbins=True`) when given a name -- the right choice for FFT analysis, but
this module asserts it rather than assuming it, by checking the standard
identity that an N-point periodic window equals an (N+1)-point symmetric
window with its last sample dropped.

Phase
-----
A real, non-negative per-bin gain (any mask/EQ curve built by a caller on
top of this module) leaves phase untouched -- multiplying a complex number
by a non-negative real scales its magnitude and leaves its angle alone. What
actually goes wrong when a caller gets this wrong is CONSISTENCY (the same
gain reused frame-to-frame, the same window used for analysis and
resynthesis, the correct COLA normalization) -- not phase. No phase-aware
reconstruction (phase vocoder-style phase locking, instantaneous frequency
correction, etc.) is implemented or needed here; a later contributor should
not add it without a concrete, measured reason.

Latency: offline, not streaming
--------------------------------
A real-time WOLA processor cannot emit its p-th output frame until it has
received the p-th frame's worth of new input, so a STREAMING implementation
has an unavoidable latency of one window length (`N` samples, converted to
ms via `stft_properties`' `streaming_latency_ms`). `aud` is an offline file
processor: `analyze`/`resynthesize` here process the whole array at once,
internally zero-padding a signal shorter than the minimum `ShortTimeFFT`
needs and trimming the resynthesized result back to the caller's requested
`length`. There is no streaming boundary to wait on, so the NET latency of
the offline path is zero. `stft_properties`' `streaming_latency_ms` is
reported for informational/interoperability purposes only (e.g. comparing
against a streaming implementation elsewhere) -- it is not a delay this
module's own functions introduce.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import ShortTimeFFT, check_COLA, check_NOLA, get_window

__all__ = [
    "analyze",
    "check_cola_nola",
    "resynthesize",
    "stft_properties",
]

_SQRT_HANN_ALIASES = ("sqrt_hann", "hann_sqrt")


def _min_input_len(n_fft: int) -> int:
    """`ShortTimeFFT` requires at least `ceil(n_fft / 2)` input samples."""
    return -(-n_fft // 2)


def _analysis_window(window: str | tuple, n_fft: int) -> np.ndarray:
    """Build the (periodic) analysis window used for both analysis and synthesis.

    Args:
        window: Any window name/tuple accepted by `scipy.signal.get_window`
            (e.g. `"hann"`, `"blackman"`, `("kaiser", 8.6)`), plus the extra
            alias `"sqrt_hann"` (equivalently `"hann_sqrt"`) for the square
            root of the periodic Hann window -- the other named WOLA pairing
            this module's docstring and tests call out (50% overlap).
        n_fft: Window length in samples.

    Returns:
        A length-`n_fft` float64 array.

    Raises:
        ValueError: `get_window` did not return a periodic (DFT-even)
            window -- see the module docstring's "assert it, do not assume"
            note. This would indicate a scipy behaviour change, not a bad
            caller input, since periodic is `get_window`'s documented
            default for a name/tuple spec.
    """
    base_name = "hann" if window in _SQRT_HANN_ALIASES else window
    periodic = np.asarray(get_window(base_name, n_fft, fftbins=True), dtype=np.float64)
    symmetric_np1 = np.asarray(get_window(base_name, n_fft + 1, fftbins=False), dtype=np.float64)[:-1]
    if not np.allclose(periodic, symmetric_np1, atol=1e-9):
        raise ValueError(
            f"get_window({base_name!r}, {n_fft}, fftbins=True) did not return a periodic "
            "(DFT-even) window (checked via the periodic == symmetric[:-1] identity); "
            "this module's COLA/NOLA reasoning assumes a periodic window throughout."
        )
    return np.sqrt(periodic) if window in _SQRT_HANN_ALIASES else periodic


def _resolve_hop(n_fft: int, hop: int | None) -> int:
    if hop is None:
        hop = n_fft // 4
    hop = int(hop)
    if not (0 < hop <= n_fft):
        raise ValueError(f"hop must satisfy 0 < hop <= n_fft; got hop={hop}, n_fft={n_fft}")
    return hop


def _cola_sum(window_power: np.ndarray, hop: int) -> float:
    """Numerically overlap-add `window_power` at `hop` and return the
    steady-state (middle-of-buffer) plateau value.

    This is exact and window-agnostic -- no closed-form COLA constant is
    assumed for any particular window family. It equals the constant every
    resynthesized sample is divided by, and it is only genuinely constant
    (independent of sample position) when `window_power` is COLA-compliant
    at `hop`; callers must check `check_cola_nola` to know whether the
    single value returned here actually holds everywhere, or only at this
    one sampled position.
    """
    n = len(window_power)
    reps = max(8, 2 * (n // hop) + 4)
    buf = np.zeros(n + reps * hop, dtype=np.float64)
    for i in range(reps):
        start = i * hop
        buf[start : start + n] += window_power
    return float(buf[len(buf) // 2])


def check_cola_nola(window: np.ndarray, hop: int, tol: float = 1e-10) -> dict[str, bool]:
    """Report constant/nonzero-overlap-add compliance for an explicit window array.

    A thin, explicit wrapper over `scipy.signal.check_COLA`/`check_NOLA`,
    exposed directly (not only buried inside `stft_properties`) so a caller
    -- or a test -- can check ANY window array against ANY hop, including
    windows this module never selects internally (a symmetric window, an
    arbitrary product window). That is what proves the check is actually
    sensitive rather than vacuous: see tests/test_dsp_stft.py's symmetric-Hann
    and Hann-squared-at-50%-hop cases, which must come back non-compliant.

    Args:
        window: 1-D real window array (already built -- not a name).
        hop: Hop size in samples.
        tol: Tolerance passed through to both scipy checks.

    Returns:
        `{"cola": bool, "nola": bool}`. COLA (constant overlap-add) implies
        NOLA (nonzero overlap-add); COLA is the bar this module's own
        literal-WOLA-with-scaled-same-window scheme needs to reconstruct
        exactly (see `resynthesize`). NOLA alone is the weaker condition
        under which *some* dual window exists (relevant only if a caller
        reaches past this module to `ShortTimeFFT`'s own automatic dual --
        this module deliberately does not use that path; see the module
        docstring).
    """
    window = np.asarray(window, dtype=np.float64)
    n = len(window)
    noverlap = n - int(hop)
    return {
        "cola": bool(check_COLA(window, n, noverlap, tol=tol)),
        "nola": bool(check_NOLA(window, n, noverlap, tol=tol)),
    }


def stft_properties(
    window: str | tuple = "hann",
    n_fft: int = 2048,
    hop: int | None = None,
    sr: int = 48000,
) -> dict[str, float | bool]:
    """Report the COLA/NOLA compliance and time/frequency tradeoff of a choice.

    Compliance is reported for `window**2` -- the product window this
    module's WOLA scheme actually overlap-adds, since `window` is reused
    unmodified as both the analysis and the synthesis window (see the
    module docstring). A window can be perfectly well-behaved on its own and
    still make a non-compliant WOLA pair with itself at a given hop (plain
    Hann at 50% hop is COLA on its own, but Hann-squared at 50% hop is not --
    see tests/test_dsp_stft.py).

    Args:
        window: Window spec, as accepted by `analyze`/`resynthesize`.
        n_fft: Window/FFT length in samples.
        hop: Hop size in samples. Defaults to `n_fft // 4` (75% overlap --
            this module's defensible default; see the module docstring).
        sr: Sample rate in Hz, for the ms/Hz conversions.

    Returns:
        {
            "window_ms": float,       # window length
            "hop_ms": float,
            "overlap_pct": float,
            "bin_hz": float,          # frequency bin spacing
            "streaming_latency_ms": float,  # informational; see module docstring
            "cola": bool,             # window**2 constant-overlap-add at this hop
            "nola": bool,             # window**2 nonzero-overlap-add at this hop
        }
    """
    hop = _resolve_hop(n_fft, hop)
    win = _analysis_window(window, n_fft)
    compliance = check_cola_nola(win**2, hop)
    return {
        "window_ms": n_fft / sr * 1000.0,
        "hop_ms": hop / sr * 1000.0,
        "overlap_pct": 100.0 * (1.0 - hop / n_fft),
        "bin_hz": sr / n_fft,
        "streaming_latency_ms": n_fft / sr * 1000.0,
        "cola": compliance["cola"],
        "nola": compliance["nola"],
    }


def analyze(
    x: np.ndarray,
    sr: int,
    window: str | tuple = "hann",
    n_fft: int = 2048,
    hop: int | None = None,
) -> np.ndarray:
    """Analyse a signal to a complex short-time spectrum.

    Args:
        x: Array of shape `(n_samples,)` (mono) or `(n_samples, n_channels)`.
        sr: Sample rate in Hz.
        window: Window spec, as accepted by `scipy.signal.get_window`, plus
            the extra alias `"sqrt_hann"`. Reused unmodified as the
            synthesis window by `resynthesize` (WOLA; see module docstring)
            -- pass the SAME `window`/`n_fft`/`hop` to both calls.
        n_fft: Window/FFT length in samples.
        hop: Hop size in samples. Defaults to `n_fft // 4` (75% overlap).

    Returns:
        Complex array: `(n_bins, n_frames)` for mono input, or
        `(n_bins, n_channels, n_frames)` for multi-channel input (`n_bins`
        = `n_fft // 2 + 1`).

        A signal shorter than `ShortTimeFFT`'s minimum (`ceil(n_fft / 2)`
        samples) is zero-padded at the end, internally, up to that minimum
        before analysis, so a short input is always well-defined rather than
        raising -- this is the "pad" half of the offline pad/process/trim
        path described in the module docstring. `resynthesize`'s `length`
        argument trims the padding back off.
    """
    hop = _resolve_hop(n_fft, hop)
    win = _analysis_window(window, n_fft)

    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    x2 = x[:, None] if squeeze else x

    min_len = _min_input_len(n_fft)
    if x2.shape[0] < min_len:
        pad = np.zeros((min_len - x2.shape[0], x2.shape[1]), dtype=np.float64)
        x2 = np.concatenate([x2, pad], axis=0)

    sft = ShortTimeFFT(win, hop, fs=sr)
    spectrum = sft.stft(x2, axis=0)  # (n_bins, n_channels, n_frames)
    return spectrum[:, 0, :] if squeeze else spectrum


def resynthesize(
    spectrum: np.ndarray,
    sr: int,
    length: int,
    window: str | tuple = "hann",
    n_fft: int = 2048,
    hop: int | None = None,
) -> np.ndarray:
    """Resynthesise a complex short-time spectrum back to a signal.

    Args:
        spectrum: As returned by `analyze` -- `(n_bins, n_frames)` for mono,
            `(n_bins, n_channels, n_frames)` for multi-channel.
        sr: Sample rate in Hz (must match the `analyze` call).
        length: Number of samples to return -- the exact length of the
            original signal passed to `analyze`. The reconstructed signal is
            trimmed to this length (this is the "trim" half of the offline
            pad/process/trim path; see the module docstring).
        window: Must match the `window` passed to `analyze`.
        n_fft: Must match the `n_fft` passed to `analyze`.
        hop: Must match the `hop` passed to `analyze`.

    Returns:
        Real array of shape `(length,)` (mono) or `(length, n_channels)`
        (multi-channel).

    Raises:
        ValueError: `window`/`n_fft`/`hop` is not COLA-compliant as its own
            WOLA pair (`window**2` has a non-positive overlap-add sum at
            `hop`) -- the degenerate case where no meaningful reconstruction
            is possible at all. A non-COLA-but-still-positive-sum choice is
            NOT rejected here (it will simply reconstruct with audible
            ripple, exactly as measured in the module docstring); use
            `stft_properties`/`check_cola_nola` beforehand to know which
            regime a given choice is in.
    """
    hop = _resolve_hop(n_fft, hop)
    win = _analysis_window(window, n_fft)

    spectrum = np.asarray(spectrum)
    squeeze = spectrum.ndim == 2
    spectrum2 = spectrum[:, None, :] if squeeze else spectrum

    cola_sum = _cola_sum(win**2, hop)
    if cola_sum <= 0:
        raise ValueError(
            f"window={window!r}, n_fft={n_fft}, hop={hop} has a non-positive "
            f"constant-overlap-add sum ({cola_sum}) for its own WOLA pair; no "
            "meaningful reconstruction is possible with this choice"
        )
    dual_win = win / cola_sum

    sft = ShortTimeFFT(win, hop, fs=sr, dual_win=dual_win)
    k1 = max(int(length), _min_input_len(n_fft))
    y = sft.istft(spectrum2, k1=k1, f_axis=0, t_axis=-1)
    y = y[: int(length)]
    return y[:, 0] if squeeze else y
