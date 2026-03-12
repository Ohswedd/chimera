"""
chimera.core
~~~~~~~~~~~~
High-level public API — the only module most users will need to import.

All functions are stateless wrappers around ChimeraPipeline that cover
the most common use cases with sensible defaults.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .exceptions import AudioLoadError
from .keygen import derive_params, derive_speaker_params
from .pipeline import ChimeraPipeline
from .presets import resolve_intensity
from .types import ChimeraResult, MaskMode, MaskParams

_SUPPORTED_SR = {8_000, 16_000, 22_050, 24_000, 44_100, 48_000}


def _load(path: str | Path) -> tuple[np.ndarray, int]:
    try:
        audio, sr = sf.read(str(path), dtype="float64", always_2d=False)
    except Exception as exc:
        raise AudioLoadError(f"Cannot load '{path}': {exc}") from exc
    return audio, sr


def _save(path: str | Path, audio: np.ndarray, sr: int, subtype: str | None) -> None:
    sf.write(str(path), audio, sr, subtype=subtype)


# ── File-level API ────────────────────────────────────────────────────────────


def mask_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    key: str,
    salt: str = "chimera-v1",
    intensity: float = 1.0,
    preset: str | None = None,
    mode: MaskMode = MaskMode.ALL_UNIQUE,
    n_speakers: int | None = None,
    speaker_ids: list[str] | None = None,
    apply_cowl: bool = True,
    subtype: str | None = None,
) -> ChimeraResult:
    """
    Apply Chimera voice anonymisation to an audio file.

    Parameters
    ----------
    input_path : str | Path
        Source audio file (WAV, FLAC, OGG, …).
    output_path : str | Path
        Destination audio file.
    key : str
        Masking passphrase.
    salt : str
        Domain-separation salt (default "chimera-v1").
    intensity : float
        Masking strength in [0, 1].
    preset : str | None
        Named preset — overrides *intensity*.
        Choices: "whisper", "subtle", "moderate", "strong", "extreme".
    mode : MaskMode
        How to treat multiple speakers.
        ALL_UNIQUE (default) — each speaker receives independent params.
        ALL_SAME              — all speakers masked identically.
        SELECTED              — only speakers in *speaker_ids* are masked.
    n_speakers : int | None
        Expected number of speakers.  None = auto-detect (2–8).
    speaker_ids : list[str] | None
        Used with MaskMode.SELECTED.
    apply_cowl : bool
        Apply the Cryptographic One-Way Layer (recommended True).
    subtype : str | None
        Output bit-depth (e.g. "PCM_16", "PCM_24", "FLOAT").

    Returns
    -------
    ChimeraResult
        Contains the output audio array and full diarization metadata.
    """
    audio, sr = _load(input_path)
    result = mask_array(
        audio,
        sr,
        key=key,
        salt=salt,
        intensity=intensity,
        preset=preset,
        mode=mode,
        n_speakers=n_speakers,
        speaker_ids=speaker_ids,
        apply_cowl=apply_cowl,
    )
    _save(output_path, result.audio, sr, subtype)
    return result


def mask_array(
    audio: np.ndarray,
    sr: int,
    *,
    key: str,
    salt: str = "chimera-v1",
    intensity: float = 1.0,
    preset: str | None = None,
    mode: MaskMode = MaskMode.ALL_UNIQUE,
    n_speakers: int | None = None,
    speaker_ids: list[str] | None = None,
    apply_cowl: bool = True,
) -> ChimeraResult:
    """
    Apply Chimera voice anonymisation to a NumPy array.

    Parameters
    ----------
    audio : np.ndarray
        Mono or stereo PCM (float32 or float64).
    sr : int
        Sample rate in Hz.
    (all other parameters — see mask_file)

    Returns
    -------
    ChimeraResult
    """
    pipeline = ChimeraPipeline(
        key=key,
        salt=salt,
        intensity=intensity,
        preset=preset,
        mode=mode,
        n_speakers=n_speakers,
        speaker_ids=speaker_ids,
        apply_cowl_layer=apply_cowl,
    )
    return pipeline.process(audio, sr)


# ── Convenience helpers ───────────────────────────────────────────────────────


def get_params(
    key: str,
    *,
    salt: str = "chimera-v1",
    intensity: float = 1.0,
    preset: str | None = None,
    speaker_label: str | None = None,
) -> MaskParams:
    """
    Return the MaskParams that would be derived from *key* without
    processing any audio.  Useful for inspection, logging, and testing.
    """
    if preset is not None:
        intensity = resolve_intensity(preset)
    return derive_params(key, salt=salt, intensity=intensity, speaker_label=speaker_label)


def get_speaker_params(
    key: str,
    speaker_ids: list[str],
    *,
    salt: str = "chimera-v1",
    intensity: float = 1.0,
    preset: str | None = None,
) -> dict[str, MaskParams]:
    """Return per-speaker MaskParams for inspection or external use."""
    if preset is not None:
        intensity = resolve_intensity(preset)
    return derive_speaker_params(key, speaker_ids, salt=salt, intensity=intensity)
