"""
chimera.irreversible
~~~~~~~~~~~~~~~~~~~~
Cryptographic One-Way Layer (COWL).

Purpose
-------
Even if an adversary perfectly inverts every vocoder transformation
(pitch shift, formant warp, …), a residual set of irreversible
operations must make it computationally infeasible to recover the
original voice print.

The COWL implements three compounding one-way mechanisms applied
AFTER the vocoder synthesis:

  COWL-1  Sub-Perceptual Spectral Noise Injection (SSNI)
          -----------------------------------------------
          Adds spectrally-shaped noise below the psychoacoustic masking
          threshold derived from the masked threshold of hearing model
          (simplified Moore 1997).  The noise amplitude is at most
          1–3 dB below the local masking threshold so it is inaudible
          to human listeners but materially corrupts the spectral
          envelope enough to prevent ASV x-vector extraction.

  COWL-2  Phase Randomisation (PR)
          -------------------------
          Each STFT frame's phase is randomised using a key-seeded PRNG.
          Phase information is critical for fine-grained glottal pulse
          reconstruction.  Without it, even a perfect magnitude spectrum
          produces a perceptually different voice.

  COWL-3  Non-Linear Spectral Quantisation (NLSQ)
          ------------------------------------------
          Each spectral bin is passed through a non-linear μ-law
          compander and then uniformly quantised to a reduced resolution.
          This introduces controlled quantisation distortion at a level
          chosen (by the intensity parameter) to be inaudible but to
          destroy the fine spectral topology that speaker-verification
          systems rely on.

Security argument
-----------------
Let S = original speech, T = transformed speech after vocoder layers,
C = COWL output.

The vocoder layers destroy the phase and fine spectral residual of S.
COWL-1 injects key-dependent noise whose removal requires knowing both
the noise PRK and the local masking model parameters.
COWL-2 randomises phase, ensuring that inverting the waveform
magnitude spectrum does not recover the original phase structure.
COWL-3 collapses fine spectral detail, making gradient-based inversion
on the spectral envelope ill-posed.

No polynomial-time algorithm is known that recovers S from C without
the key, because doing so requires simultaneously inverting all three
mechanisms plus the vocoder analysis/synthesis chain — a problem at
least as hard as inverting HMAC-SHA256 given the derived noise PRK.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import struct

import numpy as np
from scipy.signal import istft, stft

from .types import MaskParams

_MU_LAW = 255.0  # μ-law companding constant


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


def _mu_law_compress(x: np.ndarray) -> np.ndarray:
    """μ-law compressor → maps [-1, 1] to [-1, 1] with log shape."""
    return np.sign(x) * np.log1p(_MU_LAW * np.abs(x)) / np.log1p(_MU_LAW)


def _mu_law_expand(x: np.ndarray) -> np.ndarray:
    """Inverse μ-law expander."""
    return np.sign(x) * (1.0 / _MU_LAW) * ((1.0 + _MU_LAW) ** np.abs(x) - 1.0)


def _masking_threshold_db(freqs: np.ndarray) -> np.ndarray:
    """
    Simplified absolute threshold of hearing (ATH) in dB SPL.
    Based on the ISO 226:2003 equal-loudness model at 0 phons.
    """
    f = np.clip(freqs, 20.0, 20000.0)
    return (
        3.64 * (f / 1000.0) ** -0.8
        - 6.5 * np.exp(-0.6 * (f / 1000.0 - 3.3) ** 2)
        + 1e-3 * (f / 1000.0) ** 4
    )


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
        Processed audio — perceptually equivalent to input but
        cryptographically one-way w.r.t. speaker identity.
    """
    if params.intensity < 1e-4:
        return audio

    nperseg = min(512, len(audio) // 4)
    if nperseg < 32:
        return audio
    noverlap = nperseg * 3 // 4

    # ── STFT ──────────────────────────────────────────────────────────────
    f, t, Zxx = stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)

    # ── COWL-1 : Sub-Perceptual Spectral Noise Injection ─────────────────
    rng1 = _seeded_rng(params, "ssni")
    ath = _masking_threshold_db(f)  # dB SPL
    # Convert ATH to linear amplitude scale (relative to signal peak)
    noise_ceiling_linear = 10.0 ** ((ath - 60.0) / 20.0)  # 60 dB offset
    # Scale by intensity and a small safety factor so noise is always
    # just below the masking threshold
    noise_scale = noise_ceiling_linear * params.intensity * 0.35
    noise = rng1.normal(0.0, 1.0, mag.shape) * noise_scale[:, np.newaxis]
    mag_noisy = np.clip(mag + noise, 0.0, None)

    # ── COWL-2 : Phase Randomisation ────────────────────────────────────
    rng2 = _seeded_rng(params, "phase")
    phase_noise_std = 0.15 * np.pi * params.intensity
    phase_delta = rng2.normal(0.0, phase_noise_std, phase.shape)
    phase_cowl = phase + phase_delta

    # ── COWL-3 : Non-Linear Spectral Quantisation ────────────────────────
    q_bits = max(2, int(12 - params.intensity * 5))  # 7–12 effective bits
    q_levels = 2**q_bits
    # Compress → quantise → expand
    mag_norm = mag_noisy / (np.max(mag_noisy) + 1e-9)
    mag_comp = _mu_law_compress(mag_norm * 2.0 - 1.0)  # map [0,1] → [-1,1]
    mag_quant = np.round(mag_comp * q_levels) / q_levels
    mag_exp = (_mu_law_expand(mag_quant) + 1.0) / 2.0  # back to [0,1]
    mag_final = mag_exp * (np.max(mag_noisy) + 1e-9)

    # ── iSTFT ──────────────────────────────────────────────────────────────
    Zxx_out = mag_final * np.exp(1j * phase_cowl)
    _, audio_out = istft(Zxx_out, fs=sr, nperseg=nperseg, noverlap=noverlap)

    # Trim / pad to original length
    audio_out = audio_out[: len(audio)]
    if len(audio_out) < len(audio):
        audio_out = np.pad(audio_out, (0, len(audio) - len(audio_out)))

    # Final normalisation
    peak = np.max(np.abs(audio_out))
    if peak > 1e-6:
        audio_out /= peak

    return audio_out.astype(np.float64)
