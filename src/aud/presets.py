"""Named, pre-built mastering chains for common destinations.

`aud preset` is a third mechanism serving the same idea as the pipe chain
and `master`: one shell command, not a conversation. `aud preset show NAME`
prints a plan document raw on stdout, exactly like every other stage verb,
so it pipes straight into `render`:

    aud preset show podcast | aud render in.wav out.wav

Every preset is built through the exact same `aud.lib` stage builders every
other verb uses (`eq`, `compress`, `deess`, `saturate`, `loudness`,
`limit`), so a preset's plan is validated identically to a hand-built one
and cannot drift from the stage contracts documented in
contracts/plan.v1.md. There is deliberately no second, preset-only source
of truth for what a valid stage looks like -- a preset that hardcoded a
params dict would be exactly that.

Numbers are not invented. Each preset targets a real, named, checkable
loudness convention (cited in its one-line `description`, and expanded on
in the comment above its builder function below):

    podcast          -16 LUFS / -1.0 dBTP  -- Apple Podcasts' documented
                                              stereo delivery recommendation
    music-streaming  -14 LUFS / -1.0 dBTP  -- Spotify / Apple Music /
                                              YouTube Music normalization
    broadcast        -23 LUFS / -1.0 dBTP  -- EBU R128's specified average
                                              programme loudness and
                                              maximum true peak level
    voiceover        -16 LUFS / -1.0 dBTP  -- no single universal VO
                                              delivery spec exists, so this
                                              reuses the spoken-word target
                                              (see the comment on _voiceover)

`gate`/`expand` (the quiet-end repair stages -- see `aud.dsp.gate`'s module
docstring) are used, or deliberately not used, per destination below, and
each choice is argued in the comment above that preset's builder function,
the same way its loudness target is. The two spoken-word presets
(`podcast`, `voiceover`) get `expand`: an audible noise floor between
phrases is unambiguous, and every later stage in the chain (`compress`,
`loudness`) LIFTS that floor rather than lowering it, so leaving it alone
here means it ships lifted. `broadcast` and `music-streaming` do not --
see their own builder comments for why each is a genuine "no", not an
oversight.

Both stages take their threshold RELATIVE TO the file's own measured noise
floor (`threshold_above_floor_db`), never a fixed dBFS number, for the same
reason `strip_silence` and `detect silence` do: a preset is meant to be
reused across many recordings, and only a floor-relative threshold means
the same thing on all of them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from aud.lib import compress, deess, eq, expand, limit, loudness, saturate
from aud.plan import Plan, new_plan
from aud.schemas import AudError

__all__ = ["PRESET_NAMES", "build_preset", "list_presets"]


@dataclass(frozen=True)
class _Preset:
    """One named preset: a one-line description, and how to build its plan."""

    description: str
    build: Callable[[], Plan]


# -- podcast ------------------------------------------------------------
#
# Target: -16 LUFS integrated / -1.0 dBTP true peak. Apple's "Audio
# requirements" for Apple Podcasts specify -16 LUFS (+/-1 dB) for stereo
# episodes with a true peak no higher than -1 dBTP; this also lands within
# about 2 dB of Spotify's and YouTube's own loudness normalization, so one
# file serves every major platform without a second master.
#
# Processing: an 80 Hz high-pass clears mic rumble and handling noise --
# safely below the fundamental of adult speech (~85-255 Hz) so it never
# touches the voice itself. A gentle downward expander cleans up room tone
# between phrases before anything else touches level: podcasts routinely
# mix a host and one or more guests recorded in different rooms on
# different mics within the same episode, so each speaker's own
# floor-relative threshold (not a fixed dBFS number) is what keeps this
# preset meaningful across all of them, and `expand` (not `gate`) is the
# gentler device this module's docstring says is usually right for voice.
# 8 dB above the measured floor -- a little more headroom than `expand`'s
# own 6 dB default -- avoids ducking quieter consonants and breaths that a
# single, closer-mic'd speaker could tolerate at the tighter default; a
# mixed multi-guest recording cannot assume that. Multiband compression at
# 300/3000 Hz (a 3-band split typical for voice) with a gentle 2.0 ratio
# evens out level without audibly pumping. De-essing at 4 dB / 6.5 kHz (the
# contract's own default sibilance centre) tames harshness without dulling
# the recording.
def _podcast() -> Plan:
    plan = new_plan()
    plan = eq(plan, hpf=80.0)
    plan = expand(plan, threshold_above_floor_db=8.0, ratio=2.0)
    plan = compress(plan, bands=[300.0, 3000.0], ratio=2.0)
    plan = deess(plan, amount_db=4.0, freq_hz=6500.0)
    plan = loudness(plan, target_lufs=-16.0)
    plan = limit(plan, ceiling_dbtp=-1.0)
    return plan


# -- music-streaming ------------------------------------------------------
#
# Target: -14 LUFS integrated / -1.0 dBTP true peak -- the Spotify / Apple
# Music / YouTube Music loudness-normalization convention. This is also the
# exact target contracts/plan.v1.md itself uses as its own worked example.
#
# Processing: a 30 Hz high-pass clears subsonic content without touching
# bass. Multiband compression at 120/900/5500 Hz, ratio 2.5, is the
# crossover/ratio combination this tool's own end-to-end testing measured
# landing at -14.01 LUFS / -2.04 dBTP against a -1.0 dBTP ceiling on a real
# file (see the shared findings this project keeps). A light saturation
# touch (drive 1.2, mix 0.15) adds subtle analog-style glue without
# becoming the recording's defining character.
#
# Deliberately no `gate`/`expand`: a music "noise floor" is frequently
# intentional -- room ambience, tape hiss that is part of the sound, a
# quiet sustained passage or a decaying reverb tail -- not noise to be
# removed, and a gate/expander cannot tell the difference from silence
# between phrases the way it can on speech. Ducking or expanding against a
# decaying note or reverb tail pumps audibly on exactly the material a
# mastering engineer would leave alone. Voice has an unambiguous "between
# phrases" to target; music does not, so this preset does not guess.
def _music_streaming() -> Plan:
    plan = new_plan()
    plan = eq(plan, hpf=30.0)
    plan = compress(plan, bands=[120.0, 900.0, 5500.0], ratio=2.5)
    plan = saturate(plan, drive=1.2, mix=0.15)
    plan = loudness(plan, target_lufs=-14.0)
    plan = limit(plan, ceiling_dbtp=-1.0)
    return plan


# -- broadcast ------------------------------------------------------------
#
# Target: -23 LUFS integrated / -1.0 dBTP true peak -- EBU R128's specified
# average programme loudness (-23.0 LUFS) and maximum true peak level
# (-1 dBTP).
#
# Processing: a 30 Hz high-pass removes subsonic content ahead of the
# broadcast chain's own processing. Compression is deliberately gentle
# (200/2000 Hz, ratio 1.8) -- broadcast content usually arrives already
# produced, so this preset should not re-squash it, only bring it to spec.
#
# Deliberately no `gate`/`expand`, for the same reason compression here is
# gentle rather than aggressive: broadcast content usually arrives already
# produced, with any noise-floor cleanup already done upstream in the
# original production chain. Expanding an already-controlled programme a
# second time risks audibly pumping material this preset was never asked
# to touch. A caller mastering an unproduced, single-source recording FOR
# broadcast delivery should reach for `podcast` or `voiceover` (or run
# `aud expand` explicitly) before applying broadcast's own gentle spec-only
# processing.
def _broadcast() -> Plan:
    plan = new_plan()
    plan = eq(plan, hpf=30.0)
    plan = compress(plan, bands=[200.0, 2000.0], ratio=1.8)
    plan = loudness(plan, target_lufs=-23.0)
    plan = limit(plan, ceiling_dbtp=-1.0)
    return plan


# -- voiceover ------------------------------------------------------------
#
# Target: -16 LUFS integrated / -1.0 dBTP true peak. There is no single
# universal voiceover delivery spec the way there is for podcasts (Apple)
# or broadcast (EBU); this reuses the same documented spoken-word target as
# 'podcast' rather than inventing a number nobody can check.
#
# Processing is what differentiates it: a 100 Hz high-pass (voiceover is
# typically closer-mic'd than a podcast interview, so more low end can
# safely go), a +2.5 dB presence peak at 3.5 kHz (a standard VO clarity
# lift), a gentle downward expander, tighter multiband compression
# (250/2500 Hz, ratio 3.5) for session-to-session consistency, and stronger
# de-essing (8 dB / 7 kHz) for the sibilance a close mic picks up.
#
# The expander uses `expand`'s own 6 dB-above-floor default rather than
# podcast's widened 8 dB: VO is a single speaker in one controlled booth
# per session (unlike a podcast's mixed multi-guest, multi-room reality),
# which is exactly the closer, more consistent recording condition that
# default already assumes. `ratio=2.5` -- steeper than podcast's 2.0 --
# matches the same "tighter, more consistent" theme as the tighter
# compression above: a controlled booth affords cleaning its floor a
# little more assertively without the risk multi-source podcast audio runs.
def _voiceover() -> Plan:
    plan = new_plan()
    plan = eq(plan, hpf=100.0, peaks=[(3500.0, 2.5, 1.0)])
    plan = expand(plan, ratio=2.5)
    plan = compress(plan, bands=[250.0, 2500.0], ratio=3.5)
    plan = deess(plan, amount_db=8.0, freq_hz=7000.0)
    plan = loudness(plan, target_lufs=-16.0)
    plan = limit(plan, ceiling_dbtp=-1.0)
    return plan


_PRESETS: dict[str, _Preset] = {
    "podcast": _Preset(
        description=(
            "Spoken-word podcast hosting -- -16 LUFS / -1.0 dBTP (Apple Podcasts' stereo "
            "delivery recommendation); 80 Hz high-pass, gentle downward expansion of room tone "
            "between phrases, gentle multiband compression, light de-essing."
        ),
        build=_podcast,
    ),
    "music-streaming": _Preset(
        description=(
            "Music for streaming platforms -- -14 LUFS / -1.0 dBTP (Spotify / Apple Music / "
            "YouTube Music convention, also this tool's own contract example); multiband "
            "compression, light saturation glue."
        ),
        build=_music_streaming,
    ),
    "broadcast": _Preset(
        description=(
            "Broadcast delivery -- -23 LUFS / -1.0 dBTP (EBU R128's programme loudness and "
            "true-peak spec); transparent high-pass, gentle two-band compression."
        ),
        build=_broadcast,
    ),
    "voiceover": _Preset(
        description=(
            "Voiceover (commercials, corporate, promos) -- -16 LUFS / -1.0 dBTP (spoken-word "
            "convention; no single universal VO spec exists); presence lift, gentle downward "
            "expansion of booth noise floor, tighter compression, stronger de-essing."
        ),
        build=_voiceover,
    ),
}

PRESET_NAMES: tuple[str, ...] = tuple(sorted(_PRESETS))


def list_presets() -> list[dict[str, str]]:
    """Every preset name with its one-line description, sorted by name."""
    return [{"name": name, "description": _PRESETS[name].description} for name in PRESET_NAMES]


def build_preset(name: str) -> Plan:
    """Build the named preset's plan document.

    Raises:
        AudError: code "unknown_preset" if `name` is not one of PRESET_NAMES.
    """
    preset = _PRESETS.get(name)
    if preset is None:
        raise AudError(
            code="unknown_preset",
            message=f"'{name}' is not a known preset.",
            remedy=f"Use one of: {', '.join(PRESET_NAMES)}. See 'aud preset --list'.",
        )
    return preset.build()
