"""
chimera.types
~~~~~~~~~~~~~
Shared dataclasses, type aliases and enumerations used across the library.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum, auto

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing import Any as TypeAlias

import numpy as np

# ── Type aliases ──────────────────────────────────────────────────────────────
AudioArray: TypeAlias = np.ndarray  # float64, mono, range [-1, 1]
SegmentList: TypeAlias = list[tuple[float, float]]  # list of (start_sec, end_sec)


# ── Enumerations ──────────────────────────────────────────────────────────────


class MaskMode(Enum):
    """How to apply masking when multiple speakers are detected."""

    ALL_SAME = auto()  # same key-derived params for every speaker
    ALL_UNIQUE = auto()  # independent key per speaker (key + "_spk{N}")
    SELECTED = auto()  # only mask speakers listed in speaker_ids


class OutputFormat(Enum):
    """Supported output file formats."""

    WAV = "wav"
    FLAC = "flac"
    OGG = "ogg"


# ── Masking parameters ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MaskParams:
    """
    Complete, immutable parameter set for one voice transformation.

    Frozen so that the same object can safely be passed across threads
    and cached for repeated use.

    Fields
    ------
    pitch_shift_semitones : float
        F0 shift in semitones  [−10, +10].
    formant_warp : float
        Spectral-envelope frequency warp factor  [0.78, 1.22].
        < 1 → longer virtual vocal tract (deeper);  > 1 → shorter (brighter).
    spectral_tilt : float
        Linear gain slope in dB / kHz  [−4, +4].
    breathiness : float
        Aperiodicity blend toward 1.0  [0, 0.45].
    temporal_jitter : float
        Std-dev of F0 multiplicative noise  [0, 0.018].
    vibrato_rate : float
        Sinusoidal F0 modulation rate in Hz  [0, 7].
    vibrato_depth : float
        Sinusoidal F0 modulation depth in semitones  [0, 0.4].
    subharmonic_mix : float
        Fraction of sub-octave energy blended into SP  [0, 0.15].
    intensity : float
        Master scalar applied to all perceptual offsets  [0, 1].
    seed : int
        Deterministic 64-bit seed for all stochastic operations.
    speaker_label : str | None
        Optional tag identifying which speaker this params set belongs to.
    """

    pitch_shift_semitones: float
    formant_warp: float
    spectral_tilt: float
    breathiness: float
    temporal_jitter: float
    vibrato_rate: float
    vibrato_depth: float
    subharmonic_mix: float
    intensity: float
    seed: int
    speaker_label: str | None = None

    def scaled(self, factor: float) -> MaskParams:
        """Return a copy with all perceptual offsets scaled by *factor*."""
        return MaskParams(
            pitch_shift_semitones=self.pitch_shift_semitones * factor,
            formant_warp=1.0 + (self.formant_warp - 1.0) * factor,
            spectral_tilt=self.spectral_tilt * factor,
            breathiness=self.breathiness * factor,
            temporal_jitter=self.temporal_jitter * factor,
            vibrato_rate=self.vibrato_rate * factor,
            vibrato_depth=self.vibrato_depth * factor,
            subharmonic_mix=self.subharmonic_mix * factor,
            intensity=self.intensity * factor,
            seed=self.seed,
            speaker_label=self.speaker_label,
        )

    def summary(self) -> str:
        lines = [
            f"{'Chimera MaskParams':^44}",
            "─" * 44,
            f"  Speaker label      : {self.speaker_label or '(none)'}",
            f"  Pitch shift        : {self.pitch_shift_semitones:+.3f} st",
            f"  Formant warp       : {self.formant_warp:.5f}×",
            f"  Spectral tilt      : {self.spectral_tilt:+.3f} dB/kHz",
            f"  Breathiness        : {self.breathiness:.4f}",
            f"  Temporal jitter    : {self.temporal_jitter:.5f} σ",
            f"  Vibrato rate       : {self.vibrato_rate:.3f} Hz",
            f"  Vibrato depth      : {self.vibrato_depth:.4f} st",
            f"  Subharmonic mix    : {self.subharmonic_mix:.4f}",
            f"  Master intensity   : {self.intensity:.3f}",
            "─" * 44,
        ]
        return "\n".join(lines)


# ── Speaker segment ───────────────────────────────────────────────────────────


@dataclass
class SpeakerSegment:
    """
    A contiguous audio interval attributed to a single speaker.

    Attributes
    ----------
    speaker_id : str
        Canonical label (e.g. "SPEAKER_0", "SPEAKER_1", …).
    start_sec : float
        Segment start time in seconds.
    end_sec : float
        Segment end time in seconds.
    is_voiced : bool
        True if VAD detected speech in this interval.
    params : MaskParams | None
        The masking parameters assigned to this segment (set by pipeline).
    """

    speaker_id: str
    start_sec: float
    end_sec: float
    is_voiced: bool
    params: MaskParams | None = None

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


# ── Pipeline result ───────────────────────────────────────────────────────────


@dataclass
class ChimeraResult:
    """
    Encapsulates the full output of a Chimera processing job.

    Attributes
    ----------
    audio : np.ndarray
        Processed audio as float64 mono in [-1, 1].
    sample_rate : int
        Audio sample rate.
    segments : list[SpeakerSegment]
        All diarized segments with their masking assignments.
    speakers_masked : list[str]
        Speaker IDs that were actually masked.
    speakers_skipped : list[str]
        Speaker IDs that were left untouched (noise / excluded).
    processing_time_s : float
        Wall-clock seconds taken for the full pipeline.
    """

    audio: AudioArray
    sample_rate: int
    segments: list[SpeakerSegment]
    speakers_masked: list[str]
    speakers_skipped: list[str]
    processing_time_s: float

    @property
    def num_speakers(self) -> int:
        return len({s.speaker_id for s in self.segments})

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.sample_rate
