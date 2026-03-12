"""
chimera.transform
~~~~~~~~~~~~~~~~~
Voice transformation using Praat's speech-specific algorithms.

Pitch + formant modification uses Praat's "Change gender" command,
which internally combines:

  1. LPC (Linear Predictive Coding) analysis to extract the vocal tract
     filter — the spectral shape that determines formant positions.
  2. Inverse filtering to isolate the glottal source excitation.
  3. Frequency-scaling of the LPC filter coefficients to shift formants
     (equivalent to changing the vocal tract length).
  4. PSOLA resynthesis with the new pitch and modified LPC filter.

This is the gold standard for natural-sounding voice identity change
because it operates on actual speech models (source-filter decomposition),
not generic spectral processing.

The transformation is near-deterministic (LPC floating-point jitter is
below -100 dB SNR and inaudible).
"""

from __future__ import annotations

import numpy as np
import parselmouth
from parselmouth.praat import call

from .types import MaskParams

# ── Voice transform (Praat LPC + PSOLA) ─────────────────────────────────────


def change_voice(
    audio: np.ndarray,
    sr: int,
    semitones: float,
    formant_ratio: float,
) -> np.ndarray:
    """
    Change voice identity using Praat's LPC source-filter model.

    Parameters
    ----------
    audio : np.ndarray
        Mono float64 PCM.
    sr : int
        Sample rate.
    semitones : float
        Pitch shift in semitones.
    formant_ratio : float
        Formant frequency ratio (>1 = shorter vocal tract / brighter,
        <1 = longer vocal tract / deeper).

    Returns
    -------
    np.ndarray
        Transformed audio, same length as input.
    """
    if len(audio) == 0:
        return audio

    no_pitch = abs(semitones) < 0.05
    no_formant = abs(formant_ratio - 1.0) < 0.005
    if no_pitch and no_formant:
        return audio

    sound = parselmouth.Sound(audio, sampling_frequency=sr)

    # Compute pitch factor
    pitch_factor = 2.0 ** (semitones / 12.0)

    # Get the original median pitch so we can calculate the target median
    pitch_obj = call(sound, "To Pitch", 0.0, 75.0, 600.0)
    original_median = call(pitch_obj, "Get quantile", 0.0, 0.0, 0.5, "Hertz")

    if original_median == 0 or np.isnan(original_median):
        # No voiced content detected — return as-is
        return audio

    new_median = original_median * pitch_factor

    # Praat "Change gender" parameters:
    #   min_pitch, max_pitch   — pitch analysis range
    #   formant_shift_ratio    — scales LPC filter coefficients (VTL change)
    #   new_pitch_median       — target F0 median in Hz
    #   pitch_range_factor     — scales F0 variation around median (1.0 = keep)
    #   duration_factor        — time-stretch (1.0 = keep)
    result_sound = call(
        sound,
        "Change gender",
        75.0,  # minimum pitch (Hz)
        600.0,  # maximum pitch (Hz)
        formant_ratio,  # formant shift ratio
        new_median,  # new pitch median (Hz)
        1.0,  # pitch range factor
        1.0,  # duration factor
    )

    audio_out = result_sound.values[0]

    # Match length to input
    if len(audio_out) > len(audio):
        audio_out = audio_out[: len(audio)]
    elif len(audio_out) < len(audio):
        audio_out = np.pad(audio_out, (0, len(audio) - len(audio_out)))

    return audio_out


# ── Full layer stack ──────────────────────────────────────────────────────────


def apply_all_layers(
    audio: np.ndarray,
    sr: int,
    params: MaskParams,
) -> np.ndarray:
    """
    Apply the complete voice transformation to *audio*.

    Uses Praat's LPC source-filter model for combined pitch + formant
    shifting, producing natural voice identity changes.

    Returns float64 mono audio level-matched to the input.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if len(audio) == 0:
        return audio

    # Remember original loudness
    input_rms = np.sqrt(np.mean(audio**2))

    # Combined pitch + formant transform via Praat LPC
    audio = change_voice(
        audio,
        sr,
        params.pitch_shift_semitones,
        params.formant_warp,
    )

    # Level-match to preserve original loudness
    out_rms = np.sqrt(np.mean(audio**2))
    if out_rms > 1e-8 and input_rms > 1e-8:
        audio = audio * (input_rms / out_rms)

    # Safety clip
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio /= peak

    return audio.astype(np.float64)
