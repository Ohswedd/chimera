"""
chimera.pipeline
~~~~~~~~~~~~~~~~
The central processing pipeline that ties together:

  VAD → Diarization → Parameter Assignment → Vocoder Transforms → COWL

It operates on a fully segmented representation of the audio, processing
each voiced segment independently and leaving silence and noise untouched.

Processing flow
---------------

  ┌─────────────────────────────────────────────────────────────────┐
  │  INPUT audio (mono float64, any supported sample rate)          │
  └────────────────────────┬────────────────────────────────────────┘
                           │
                    [1] VAD  (chimera.vad)
                           │
                    [2] Speaker Diarization  (chimera.diarize)
                           │
                    [3] Speaker → MaskParams assignment
                           │
                    [4] Per-segment WORLD vocoder transform
                           │
                    [5] COWL irreversibility layer  (chimera.irreversible)
                           │
                    [6] Segment stitching with cross-fade
                           │
  ┌─────────────────────────────────────────────────────────────────┐
  │  OUTPUT audio (mono float64, same sample rate)                  │
  └─────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import time

import numpy as np

from .diarize import SpeakerDiarizer
from .exceptions import PipelineError
from .irreversible import apply_cowl
from .keygen import derive_params
from .transform import apply_all_layers
from .types import (
    AudioArray,
    ChimeraResult,
    MaskMode,
    MaskParams,
    SpeakerSegment,
)
from .vad import VoiceActivityDetector

# Cross-fade duration (seconds) applied at segment boundaries to avoid clicks
_CROSSFADE_S: float = 0.005


def _crossfade(a: np.ndarray, b: np.ndarray, fade_len: int) -> np.ndarray:
    """Blend the tail of *a* into the head of *b* over *fade_len* samples."""
    if fade_len <= 0 or len(a) < fade_len or len(b) < fade_len:
        return np.concatenate([a, b])
    win = np.linspace(0.0, 1.0, fade_len)
    a[-fade_len:] *= 1.0 - win
    b[:fade_len] *= win
    return np.concatenate([a[:-fade_len], a[-fade_len:] + b[:fade_len], b[fade_len:]])


def _seg_audio(audio: AudioArray, seg: SpeakerSegment, sr: int) -> AudioArray:
    """Extract the audio slice for *seg*."""
    s = int(seg.start_sec * sr)
    e = int(seg.end_sec * sr)
    e = min(e, len(audio))
    return audio[s:e].copy()


class ChimeraPipeline:
    """
    Main processing pipeline for Chimera voice anonymisation.

    Parameters
    ----------
    key : str
        Master masking key / passphrase.
    salt : str
        Domain-separation salt (default "chimera-v1").
    intensity : float
        Master intensity in [0, 1].
    preset : str | None
        Named preset — overrides *intensity* if set.
    mode : MaskMode
        How to handle multiple speakers.
    speaker_ids : list[str] | None
        In MaskMode.SELECTED, only these speaker IDs are masked.
    n_speakers : int | None
        Expected number of speakers for the diarizer (None = auto).
    apply_cowl : bool
        Whether to apply the Cryptographic One-Way Layer (default True).
    vad_energy_db : float
        VAD energy threshold.
    crossfade : bool
        Apply cross-fade at segment boundaries (default True).
    """

    def __init__(
        self,
        key: str,
        salt: str = "chimera-v1",
        intensity: float = 1.0,
        preset: str | None = None,
        mode: MaskMode = MaskMode.ALL_UNIQUE,
        speaker_ids: list[str] | None = None,
        n_speakers: int | None = None,
        apply_cowl_layer: bool = True,
        vad_energy_db: float = -45.0,
        crossfade: bool = True,
    ) -> None:
        if not key:
            raise PipelineError("key must be a non-empty string.")

        # Resolve preset → intensity
        if preset is not None:
            from .presets import resolve_intensity

            intensity = resolve_intensity(preset)

        self.key = key
        self.salt = salt
        self.intensity = intensity
        self.mode = mode
        self.speaker_ids_mask: set[str] | None = set(speaker_ids) if speaker_ids else None
        self.n_speakers = n_speakers
        self.use_cowl = apply_cowl_layer
        self.crossfade = crossfade

        self._vad = VoiceActivityDetector(energy_threshold_db=vad_energy_db)
        self._diarizer = SpeakerDiarizer(n_speakers=n_speakers)

    # ── Public entry points ───────────────────────────────────────────────────

    def process(
        self,
        audio: AudioArray,
        sr: int,
    ) -> ChimeraResult:
        """
        Run the full pipeline on *audio*.

        Parameters
        ----------
        audio : np.ndarray
            Mono or stereo float64 PCM in [-1, 1].
        sr : int
            Sample rate.

        Returns
        -------
        ChimeraResult
        """
        t0 = time.perf_counter()

        audio = self._prepare(audio)

        # [1] VAD
        vad_intervals = self._vad.detect(audio, sr)

        # [2] Speaker Diarization
        try:
            segments = self._diarizer.diarize(audio, sr, vad_intervals)
        except Exception:
            # Fallback: treat entire audio as single speaker
            segments = [SpeakerSegment("SPEAKER_0", 0.0, len(audio) / sr, True)]

        # [3] Parameter assignment
        speaker_ids = list({s.speaker_id for s in segments if s.is_voiced})
        params_map = self._assign_params(speaker_ids)
        for seg in segments:
            seg.params = params_map.get(seg.speaker_id)

        # [4+5] Transform each segment
        out_chunks: list[np.ndarray] = []
        speakers_masked: set[str] = set()
        speakers_skipped: set[str] = set()

        for seg in segments:
            chunk = _seg_audio(audio, seg, sr)
            if len(chunk) == 0:
                continue

            if not seg.is_voiced or seg.params is None or seg.speaker_id == "SILENCE":
                speakers_skipped.add(seg.speaker_id)
                out_chunks.append(chunk)
                continue

            # [4] Vocoder transform
            masked = apply_all_layers(chunk, sr, seg.params)

            # [5] COWL
            if self.use_cowl:
                masked = apply_cowl(masked, sr, seg.params)

            speakers_masked.add(seg.speaker_id)
            out_chunks.append(masked)

        # [6] Stitch
        if not out_chunks:
            final = audio.copy()
        else:
            fade_len = int(sr * _CROSSFADE_S) if self.crossfade else 0
            final = out_chunks[0]
            for chunk in out_chunks[1:]:
                final = _crossfade(final, chunk, fade_len)

        # Trim / pad to original length
        final = final[: len(audio)]
        if len(final) < len(audio):
            final = np.pad(final, (0, len(audio) - len(final)))

        peak = np.max(np.abs(final))
        if peak > 1e-6:
            final /= peak

        t1 = time.perf_counter()

        return ChimeraResult(
            audio=final.astype(np.float64),
            sample_rate=sr,
            segments=segments,
            speakers_masked=sorted(speakers_masked),
            speakers_skipped=sorted(speakers_skipped - speakers_masked),
            processing_time_s=t1 - t0,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _prepare(self, audio: np.ndarray) -> AudioArray:
        """Ensure mono, float64, normalised."""
        audio = np.asarray(audio, dtype=np.float64)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        peak = np.max(np.abs(audio))
        if peak > 1e-6:
            audio = audio / peak
        return audio

    def _assign_params(self, speaker_ids: list[str]) -> dict[str, MaskParams | None]:
        """
        Build a speaker_id → MaskParams mapping according to self.mode.
        """
        result: dict[str, MaskParams | None] = {}

        for sid in speaker_ids:
            if sid == "SILENCE":
                result[sid] = None
                continue

            # In SELECTED mode, only mask specified speakers
            if (
                self.mode == MaskMode.SELECTED
                and self.speaker_ids_mask
                and sid not in self.speaker_ids_mask
            ):
                result[sid] = None
                continue

            if self.mode == MaskMode.ALL_SAME:
                # All speakers share the same params (derived from key alone)
                result[sid] = derive_params(self.key, salt=self.salt, intensity=self.intensity)
            else:
                # ALL_UNIQUE or SELECTED: each speaker gets independent params
                result[sid] = derive_params(
                    self.key,
                    salt=self.salt,
                    intensity=self.intensity,
                    speaker_label=sid,
                )

        return result
