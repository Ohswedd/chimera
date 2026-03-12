"""
chimera.keygen
~~~~~~~~~~~~~~
Deterministic, cryptographically sound parameter derivation.

Design
------
We implement a two-step HKDF-like scheme (RFC 5869) entirely with the
Python standard library (hmac + hashlib), so there are zero additional
cryptographic dependencies.

Step 1 -- Extract:
    PRK = HMAC-SHA256(salt, key)

Step 2 -- Expand (one call per parameter, with unique label):
    T(label) = HMAC-SHA256(PRK, label || 0x01)

Each T(label) is 32 bytes. We read the first 4 bytes as a big-endian
uint32 and linearly map to the parameter's natural range.

The derivation is:
  - Deterministic (same key + salt -> same params, always).
  - Domain-separated (each parameter uses a unique label).
  - Collision-resistant up to HMAC-SHA256 security (128-bit).
  - Opaque (parameters cannot be reversed to the key without
    inverting HMAC-SHA256).

Speaker-specific keys
---------------------
When per-speaker masking is requested, the key for speaker N is:

    key_N = key + ":chimera:spk:" + speaker_id

ensuring that speaker keys are independent and non-overlapping even if
the master key is short.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import struct

from .exceptions import KeyDerivationError
from .types import MaskParams

# Domain-separation salt version tag (bump on breaking changes)
_DEFAULT_SALT = "chimera-v1"


def _prk(key: str, salt: str) -> bytes:
    """HKDF Extract step -> 32-byte pseudo-random key."""
    try:
        return _hmac.new(
            salt.encode("utf-8"),
            key.encode("utf-8"),
            hashlib.sha256,
        ).digest()
    except Exception as exc:  # pragma: no cover
        raise KeyDerivationError(f"Key derivation failed: {exc}") from exc


def _expand_float(prk: bytes, label: str, lo: float, hi: float) -> float:
    """Expand PRK under *label* and map result to [lo, hi]."""
    info = label.encode("utf-8") + b"\x01"
    digest = _hmac.new(prk, info, hashlib.sha256).digest()
    raw = struct.unpack(">I", digest[:4])[0]  # uint32, [0, 2^32-1]
    t = raw / 0xFFFF_FFFF  # -> [0.0, 1.0]
    return lo + t * (hi - lo)


def _expand_int(prk: bytes, label: str) -> int:
    """Expand PRK under *label* to a 64-bit unsigned integer."""
    info = label.encode("utf-8") + b"\x01"
    digest = _hmac.new(prk, info, hashlib.sha256).digest()
    return struct.unpack(">Q", digest[:8])[0]


# ── Public API ────────────────────────────────────────────────────────────────


def derive_params(
    key: str,
    *,
    salt: str = _DEFAULT_SALT,
    intensity: float = 1.0,
    speaker_label: str | None = None,
) -> MaskParams:
    """
    Derive a complete MaskParams from a passphrase.

    Parameters
    ----------
    key : str
        Master masking key / passphrase.  Any Unicode string is valid.
    salt : str
        Domain-separation salt.  Must match across encode/decode pairs.
        Default "chimera-v1".
    intensity : float
        Master intensity scalar in [0.0, 1.0].  0 -> identity transform;
        1 -> maximum masking.
    speaker_label : str | None
        If set, appended to the key before derivation so that different
        speakers receive independent parameters even with the same master key.

    Returns
    -------
    MaskParams
        Fully specified, immutable parameter set.

    Raises
    ------
    KeyDerivationError
        If *key* is empty or *intensity* is out of range.
    """
    if not key:
        raise KeyDerivationError("key must be a non-empty string.")
    if not (0.0 <= intensity <= 1.0):
        raise KeyDerivationError(f"intensity must be in [0, 1], got {intensity!r}.")

    # Inject speaker label into the key path so params are fully independent
    effective_key = f"{key}:chimera:spk:{speaker_label}" if speaker_label else key

    prk = _prk(effective_key, salt)

    # ── Parameter derivation ─────────────────────────────────────────────────
    # Identity parameters (pitch, formant) have wide ranges for effective
    # voice masking.  Quality parameters (tilt, breathiness) are kept gentle
    # to preserve natural speech.
    pitch = _expand_float(prk, "pitch", -8.0, 8.0)
    formant = _expand_float(prk, "formant", 0.80, 1.20)
    tilt = _expand_float(prk, "tilt", -1.5, 1.5)
    breathiness = _expand_float(prk, "breathiness", 0.01, 0.10)
    seed = _expand_int(prk, "seed")

    raw = MaskParams(
        pitch_shift_semitones=pitch,
        formant_warp=formant,
        spectral_tilt=tilt,
        breathiness=breathiness,
        intensity=intensity,
        seed=seed,
        speaker_label=speaker_label,
    )

    # Apply intensity scaling, then enforce minimum identity shift so the
    # output never sounds like the original speaker.
    scaled = raw.scaled(intensity)

    if intensity < 1e-4:
        return scaled

    # Minimum 4 semitones pitch shift (clearly different register)
    p = scaled.pitch_shift_semitones
    if abs(p) < 4.0:
        p = 4.0 if p >= 0 else -4.0
    # Minimum 12% formant warp (clearly different vocal tract length)
    fw = scaled.formant_warp
    if abs(fw - 1.0) < 0.12:
        fw = 1.12 if fw >= 1.0 else 0.88

    return MaskParams(
        pitch_shift_semitones=p,
        formant_warp=fw,
        spectral_tilt=scaled.spectral_tilt,
        breathiness=scaled.breathiness,
        intensity=scaled.intensity,
        seed=scaled.seed,
        speaker_label=scaled.speaker_label,
    )


def derive_speaker_params(
    master_key: str,
    speaker_ids: list[str],
    *,
    salt: str = _DEFAULT_SALT,
    intensity: float = 1.0,
) -> dict[str, MaskParams]:
    """
    Derive independent MaskParams for each speaker in *speaker_ids*.

    Returns a dict mapping speaker_id -> MaskParams.
    """
    return {
        sid: derive_params(master_key, salt=salt, intensity=intensity, speaker_label=sid)
        for sid in speaker_ids
    }
