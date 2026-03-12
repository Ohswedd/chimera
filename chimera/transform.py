"""
chimera.transform
~~~~~~~~~~~~~~~~~
Low-level audio transformation layers built on the WORLD vocoder.

WORLD (Morise 2016) decomposes speech into three orthogonal streams:
  F0  – fundamental frequency contour (perceived pitch)
  SP  – spectral envelope   (vocal-tract shape → formants + timbre)
  AP  – aperiodicity band   (breathiness / noise ratio)

Each stream is modified independently before re-synthesis, preserving
the natural glottal pulse structure that characterises clean speech.

Layer summary
-------------
Layer   Stream   Operation
──────  ───────  ─────────────────────────────────────────────────────────
L1      F0       Global semitone shift  (pitch register change)
L2      F0       Sinusoidal vibrato modulation  (disguises pitch dynamics)
L3      F0       Micro-temporal jitter  (breaks rhythmic voice-print)
L4      SP       Frequency-domain formant warp  (vocal-tract length change)
L5      SP       Linear spectral tilt  (bright ↔ dark timbral balance)
L6      SP       Sub-harmonic injection  (extra low-frequency resonance)
L7      AP       Breathiness blend  (noise-to-periodic ratio shift)
"""

from __future__ import annotations

import sys

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing import Any as TypeAlias

import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d

_F64: TypeAlias = np.ndarray  # float64 alias

_FRAME_PERIOD_MS: float = 5.0
_SUPPORTED_SR = {8_000, 16_000, 22_050, 24_000, 44_100, 48_000}

_FFT_SIZES = {
    8_000: 512,
    16_000: 1024,
    22_050: 1024,
    24_000: 1024,
    44_100: 2048,
    48_000: 2048,
}


def _fft_size(sr: int) -> int:
    return _FFT_SIZES.get(sr, pw.get_cheaptrick_fft_size(sr))


# ── Analysis ──────────────────────────────────────────────────────────────────


def analyse(audio: _F64, sr: int) -> tuple[_F64, _F64, _F64]:
    """
    Decompose speech into (f0, sp, ap) using WORLD.

    Uses DIO + StoneMask for robust F0 estimation, CheapTrick for the
    spectral envelope, and D4C for the aperiodicity band.
    """
    audio = np.asarray(audio, dtype=np.float64)
    f0, t = pw.dio(audio, sr, frame_period=_FRAME_PERIOD_MS)
    f0 = pw.stonemask(audio, f0, t, sr)
    fft = _fft_size(sr)
    sp = pw.cheaptrick(audio, f0, t, sr, fft_size=fft)
    ap = pw.d4c(audio, f0, t, sr, fft_size=fft)
    return f0, sp, ap


# ── F0 layers ─────────────────────────────────────────────────────────────────


def l1_pitch_shift(f0: _F64, semitones: float) -> _F64:
    """L1 – Global pitch shift by *semitones*."""
    if abs(semitones) < 1e-4:
        return f0
    ratio = 2.0 ** (semitones / 12.0)
    out = f0.copy()
    out[out > 0.0] *= ratio
    return out


def l2_vibrato(
    f0: _F64,
    rate_hz: float,
    depth_st: float,
    sr: int,
) -> _F64:
    """
    L2 – Sinusoidal vibrato modulation.

    Adds a periodic F0 oscillation at *rate_hz* Hz with *depth_st*
    semitone amplitude.  This disguises the speaker's natural pitch
    dynamics without sounding artificial when depth_st < 0.3 st.
    """
    if rate_hz < 0.1 or depth_st < 1e-4:
        return f0
    n_frames = len(f0)
    t = np.arange(n_frames) * _FRAME_PERIOD_MS / 1000.0
    modulation = 2.0 ** (depth_st * np.sin(2.0 * np.pi * rate_hz * t) / 12.0)
    out = f0.copy()
    voiced = out > 0.0
    out[voiced] *= modulation[voiced]
    return out


def l3_temporal_jitter(
    f0: _F64,
    coefficient: float,
    rng: np.random.Generator,
) -> _F64:
    """
    L3 – Micro-temporal F0 jitter.

    Adds multiplicative Gaussian noise to the voiced F0 trajectory.
    *coefficient* is the fractional std-dev (~0.005–0.018).
    Breaks speaker-specific pitch micro-rhythm patterns without
    introducing perceptible roughness (threshold ≈ 0.5% jitter).
    """
    if coefficient < 1e-5:
        return f0
    out = f0.copy()
    voiced = out > 0.0
    noise = rng.normal(1.0, coefficient, size=out.shape)
    out[voiced] *= noise[voiced]
    # Clip to physiologically plausible range [50, 600] Hz
    out[voiced] = np.clip(out[voiced], 50.0, 600.0)
    return out


# ── SP layers ─────────────────────────────────────────────────────────────────


def l4_formant_warp(sp: _F64, warp_factor: float) -> _F64:
    """
    L4 – Frequency-domain formant warping.

    Resamples the spectral envelope in the frequency domain by
    *warp_factor*, effectively simulating a different vocal-tract length.
    Operates in log-power domain for smoother interpolation.

    warp_factor < 1 → longer tract (darker, deeper)
    warp_factor > 1 → shorter tract (brighter, lighter)
    """
    if abs(warp_factor - 1.0) < 1e-4:
        return sp

    T, N = sp.shape
    src_idx = np.arange(N, dtype=np.float64)
    dst_idx = src_idx * warp_factor

    out = np.empty_like(sp)
    log_sp = np.log(sp + 1e-30)

    for t in range(T):
        interp_fn = interp1d(
            src_idx,
            log_sp[t],
            kind="linear",
            bounds_error=False,
            fill_value=(log_sp[t, 0], log_sp[t, -1]),
        )
        out[t] = np.exp(interp_fn(dst_idx))

    return np.clip(out, 1e-30, None)


def l5_spectral_tilt(sp: _F64, sr: int, tilt_db_per_khz: float) -> _F64:
    """
    L5 – Linear spectral tilt.

    Applies a frequency-linear gain ramp (in dB/kHz) to the spectral
    envelope, altering the perceived brightness/darkness balance.
    """
    if abs(tilt_db_per_khz) < 1e-4:
        return sp
    N = sp.shape[1]
    freq_khz = np.linspace(0.0, sr / 2000.0, N)
    gain_db = tilt_db_per_khz * freq_khz
    # sp is a power spectrum → squared linear gain
    gain_power = 10.0 ** (gain_db / 10.0)
    return sp * gain_power[np.newaxis, :]


def l6_subharmonic_injection(
    sp: _F64,
    f0: _F64,
    mix: float,
    sr: int,
) -> _F64:
    """
    L6 – Sub-harmonic resonance injection.

    Slightly boosts the spectral envelope in the sub-fundamental region,
    creating a subtle added resonance that alters the perceived vocal
    weight without sounding artificial.  *mix* ∈ [0, 0.15].
    """
    if mix < 1e-4:
        return sp

    N = sp.shape[1]
    freq_hz = np.linspace(0.0, sr / 2.0, N)
    out = sp.copy()

    # For each voiced frame, boost the band below F0
    for t in range(len(sp)):
        if f0[t] < 1.0:
            continue
        sub_hz = f0[t] / 2.0  # sub-octave
        # Gaussian boost centred at sub_hz
        sigma = sub_hz * 0.4
        boost = mix * np.exp(-0.5 * ((freq_hz - sub_hz) / sigma) ** 2)
        out[t] *= 1.0 + boost

    return out


# ── AP layers ─────────────────────────────────────────────────────────────────


def l7_breathiness(ap: _F64, amount: float) -> _F64:
    """
    L7 – Aperiodicity (breathiness) blend.

    Blends AP toward 1.0 (fully aperiodic / noisy) by *amount* ∈ [0, 0.45].
    Adds a speaker-independent breathy quality that masks fine spectral
    identity cues present in the aperiodic band.
    """
    if amount < 1e-4:
        return ap
    return np.clip(ap + amount * (1.0 - ap), 0.0, 1.0)


# ── Synthesis ─────────────────────────────────────────────────────────────────


def synthesise(f0: _F64, sp: _F64, ap: _F64, sr: int, target_len: int = 0) -> _F64:
    """Resynthesize from modified WORLD streams; normalise to [-1, 1].

    If *target_len* > 0, the output is trimmed or zero-padded to that length.
    """
    audio = pw.synthesize(f0, sp, ap, sr, frame_period=_FRAME_PERIOD_MS)
    if target_len > 0:
        if len(audio) > target_len:
            audio = audio[:target_len]
        elif len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak
    return audio.astype(np.float64)


# ── Full layer stack ──────────────────────────────────────────────────────────


def apply_all_layers(
    audio: _F64,
    sr: int,
    params,  # MaskParams  (imported lazily to avoid circular)
) -> _F64:
    """
    Apply the complete 7-layer vocoder masking stack to *audio*.

    Returns normalised float64 mono audio.
    """
    from .types import MaskParams as _MP

    assert isinstance(params, _MP)

    audio = np.asarray(audio, dtype=np.float64)
    if len(audio) == 0:
        return audio

    # Normalise input
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak

    rng = np.random.default_rng(params.seed)

    f0, sp, ap = analyse(audio, sr)

    # F0 layers
    f0 = l1_pitch_shift(f0, params.pitch_shift_semitones)
    f0 = l2_vibrato(f0, params.vibrato_rate, params.vibrato_depth, sr)
    f0 = l3_temporal_jitter(f0, params.temporal_jitter, rng)

    # SP layers
    sp = l4_formant_warp(sp, params.formant_warp)
    sp = l5_spectral_tilt(sp, sr, params.spectral_tilt)
    sp = l6_subharmonic_injection(sp, f0, params.subharmonic_mix, sr)

    # AP layer
    ap = l7_breathiness(ap, params.breathiness)

    return synthesise(f0, sp, ap, sr, target_len=len(audio))
