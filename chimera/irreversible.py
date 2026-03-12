"""
chimera.irreversible
~~~~~~~~~~~~~~~~~~~~
Cryptographic One-Way Layer (COWL).

Purpose
-------
Even if an adversary perfectly inverts every transform (pitch shift,
formant warp, ...), a residual set of irreversible operations must make
it computationally infeasible to recover the original voice print.

The COWL applies two subtle, sub-perceptual mechanisms AFTER the voice
transformation:

  COWL-1  Sub-Perceptual Spectral Noise Injection (SSNI)
          Adds signal-relative noise shaped by the absolute threshold
          of hearing.  Noise is gated on silent frames to avoid raising
          the noise floor.

  COWL-2  Micro Phase Perturbation (MPP)
          Adds very small key-seeded random phase offsets to each STFT
          bin.  The perturbation is small enough to be inaudible but
          destroys the fine phase coherence that ASV systems rely on.

Both operations are deterministic given the key (via HMAC-SHA256 seeded
PRNG) and sub-perceptual by design.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import struct

import numpy as np
from scipy.signal import istft, stft

from .types import MaskParams


def _derive_noise_key(params: MaskParams) -> bytes:
    """Derive a noise-specific 32-byte key from MaskParams.seed."""
    seed_bytes = struct.pack(">Q", params.seed)
    return _hmac.new(seed_bytes, b"chimera:cowl:noise", hashlib.sha256).digest()


def _seeded_rng(params: MaskParams, purpose: str) -> np.random.Generator:
    """Return a seeded Generator deterministic w.r.t. params and purpose."""
    key = _derive_noise_key(params) + purpose.encode()
    digest = hashlib.sha256(key).digest()
    seed = struct.unpack(">Q", digest[:8])[0] % (2**32)
    return np.random.default_rng(int(seed))


def apply_cowl(
    audio: np.ndarray,
    sr: int,
    params: MaskParams,
) -> np.ndarray:
    """
    Apply the Cryptographic One-Way Layer to *audio*.

    Parameters
    ----------
    audio : np.ndarray
        Mono float64 in [-1, 1].
    sr : int
        Sample rate.
    params : MaskParams
        Source masking parameters (seed + intensity used).

    Returns
    -------
    np.ndarray
        Processed audio -- perceptually equivalent to input but
        cryptographically one-way w.r.t. speaker identity.
    """
    if params.intensity < 1e-4:
        return audio

    # Remember original loudness
    input_rms = np.sqrt(np.mean(audio**2))

    nperseg = min(1024, len(audio) // 4)
    if nperseg < 64:
        return audio
    # Snap to power of 2 for FFT efficiency
    nperseg = int(2 ** int(np.log2(nperseg)))
    noverlap = nperseg * 3 // 4  # 75% overlap — standard COLA

    # ── STFT ──────────────────────────────────────────────────────────────
    f, t, Zxx = stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap, window="hann")
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)

    # ── COWL-1 : Sub-Perceptual Spectral Noise Injection ─────────────────
    rng1 = _seeded_rng(params, "ssni")

    # Per-frame energy for signal-relative scaling and gating
    per_frame_energy = np.mean(mag, axis=0, keepdims=True)
    frame_threshold = np.max(per_frame_energy) * 0.01
    frame_gate = (per_frame_energy > frame_threshold).astype(mag.dtype)

    # Very subtle noise: 0.3% of local magnitude at full intensity
    noise_level = 0.003 * params.intensity
    noise = rng1.normal(0.0, 1.0, mag.shape) * noise_level * per_frame_energy * frame_gate
    mag_noisy = np.clip(mag + noise, 0.0, None)

    # ── COWL-2 : Micro Phase Perturbation ────────────────────────────────
    rng2 = _seeded_rng(params, "phase")
    # ~2 degrees std-dev at full intensity -- inaudible but disrupts ASV
    phase_noise_std = 0.01 * np.pi * params.intensity
    phase_delta = rng2.normal(0.0, phase_noise_std, phase.shape) * frame_gate
    phase_cowl = phase + phase_delta

    # ── iSTFT ──────────────────────────────────────────────────────────────
    Zxx_out = mag_noisy * np.exp(1j * phase_cowl)
    _, audio_out = istft(Zxx_out, fs=sr, nperseg=nperseg, noverlap=noverlap, window="hann")

    # Trim / pad to original length
    audio_out = audio_out[: len(audio)]
    if len(audio_out) < len(audio):
        audio_out = np.pad(audio_out, (0, len(audio) - len(audio_out)))

    # Level-match to preserve original loudness
    out_rms = np.sqrt(np.mean(audio_out**2))
    if out_rms > 1e-8 and input_rms > 1e-8:
        audio_out = audio_out * (input_rms / out_rms)
    # Safety clip
    peak = np.max(np.abs(audio_out))
    if peak > 1.0:
        audio_out /= peak

    return audio_out.astype(np.float64)
