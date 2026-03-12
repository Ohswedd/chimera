"""
chimera.exceptions
~~~~~~~~~~~~~~~~~~
Custom exception hierarchy for the Chimera library.

All public exceptions inherit from ChimeraError, allowing callers to
catch the entire family with a single except clause.
"""

from __future__ import annotations


class ChimeraError(Exception):
    """Base class for all Chimera exceptions."""


class KeyDerivationError(ChimeraError):
    """Raised when key derivation fails due to invalid input."""


class AudioLoadError(ChimeraError):
    """Raised when an audio file cannot be loaded or decoded."""


class UnsupportedSampleRateError(ChimeraError):
    """Raised when the sample rate is not supported by the vocoder."""

    def __init__(self, sr: int) -> None:
        super().__init__(
            f"Sample rate {sr} Hz is not supported. "
            "Supported rates: 8000, 16000, 22050, 24000, 44100, 48000."
        )


class DiarizationError(ChimeraError):
    """Raised when speaker diarization cannot segment the audio."""


class RealtimeError(ChimeraError):
    """Raised when the real-time audio stream encounters a fatal error."""


class PipelineError(ChimeraError):
    """Raised when the processing pipeline is misconfigured."""


class IrreversibilityError(ChimeraError):
    """Raised when the irreversibility layer cannot be applied correctly."""


class PresetNotFoundError(ChimeraError):
    """Raised when a named preset does not exist."""

    def __init__(self, name: str) -> None:
        super().__init__(
            f"Preset '{name}' not found. " "Use chimera.list_presets() to see available options."
        )
