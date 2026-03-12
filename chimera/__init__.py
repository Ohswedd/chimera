"""
Chimera
=======
Local, deterministic, cryptographically one-way voice anonymisation.

Chimera disguises spoken-voice identity in audio recordings by applying
a seven-layer acoustic transformation stack derived from a user-supplied
passphrase.  It is the only open-source library that combines:

  • WORLD high-quality vocoder analysis/synthesis
  • Automatic Voice Activity Detection (no GPU, no cloud)
  • MFCC-based speaker diarization (per-speaker independent masking)
  • Key-derived deterministic parameterisation (HKDF-SHA256)
  • Cryptographic One-Way Layer (COWL) — mathematically irreversible

Quick start
-----------
    # File-based (simplest)
    import chimera
    chimera.mask_file("interview.wav", "anonymous.wav", key="project-x")

    # Array-based
    import soundfile as sf
    audio, sr = sf.read("interview.wav")
    result = chimera.mask_array(audio, sr, key="project-x", preset="strong")

    # Inspect derived parameters
    p = chimera.get_params("project-x", preset="strong")
    print(p.summary())

    # Real-time microphone
    from chimera.realtime import RealtimeAnonymiser
    anon = RealtimeAnonymiser(key="project-x", preset="moderate")
    anon.start()   # begins recording & processing
    input("Press Enter to stop...")
    anon.stop()
    anon.save("recorded_anonymous.wav")

Project info
------------
Homepage     : https://github.com/Ohswedd/chimera
Documentation: https://github.com/Ohswedd/chimera/wiki
License      : MIT
"""

from .core import get_params, get_speaker_params, mask_array, mask_file
from .exceptions import (
    AudioLoadError,
    ChimeraError,
    DiarizationError,
    KeyDerivationError,
    PipelineError,
    PresetNotFoundError,
    RealtimeError,
    UnsupportedSampleRateError,
)
from .keygen import derive_params, derive_speaker_params
from .pipeline import ChimeraPipeline
from .presets import PRESETS, list_presets
from .types import ChimeraResult, MaskMode, MaskParams

__all__ = [
    # High-level API
    "mask_file",
    "mask_array",
    "get_params",
    "get_speaker_params",
    # Pipeline
    "ChimeraPipeline",
    "MaskMode",
    # Types
    "MaskParams",
    "ChimeraResult",
    # Key derivation
    "derive_params",
    "derive_speaker_params",
    # Presets
    "PRESETS",
    "list_presets",
    # Exceptions
    "ChimeraError",
    "AudioLoadError",
    "KeyDerivationError",
    "DiarizationError",
    "PipelineError",
    "RealtimeError",
    "PresetNotFoundError",
    "UnsupportedSampleRateError",
]

__version__ = "0.1.0"
__author__ = "Ohswedd"
__license__ = "MIT"
