"""
Shared pytest fixtures for the Chimera test suite.
"""

from __future__ import annotations

import numpy as np
import pytest

SR = 22_050  # default sample rate for all tests
SHORT_DUR = 0.8  # seconds — fast tests
LONG_DUR = 2.5  # seconds — integration tests


def _tone(freq: float = 200.0, duration: float = 2.0, sr: int = SR) -> np.ndarray:
    """Synthetic voiced tone with harmonics — mimics a sustained vowel."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (
        0.50 * np.sin(2 * np.pi * freq * t)
        + 0.25 * np.sin(2 * np.pi * freq * 2 * t)
        + 0.12 * np.sin(2 * np.pi * freq * 3 * t)
        + 0.06 * np.sin(2 * np.pi * freq * 4 * t)
    )
    fade = int(sr * 0.02)
    audio[:fade] *= np.linspace(0, 1, fade)
    audio[-fade:] *= np.linspace(1, 0, fade)
    return (audio * 0.8).astype(np.float64)


def _silence(duration: float = 1.0, sr: int = SR) -> np.ndarray:
    return np.zeros(int(sr * duration), dtype=np.float64)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sr():
    return SR


@pytest.fixture
def short_audio():
    """Short voiced tone (0.8 s)."""
    return _tone(freq=200.0, duration=SHORT_DUR)


@pytest.fixture
def long_audio():
    """Longer voiced tone (2.5 s)."""
    return _tone(freq=180.0, duration=LONG_DUR)


@pytest.fixture
def silence():
    return _silence(duration=1.0)


@pytest.fixture
def stereo_audio():
    mono = _tone(freq=220.0, duration=1.0)
    return np.stack([mono, mono * 0.9], axis=1)


@pytest.fixture
def two_speaker_audio():
    """Two concatenated tones at different pitches — simulates two speakers."""
    spk0 = _tone(freq=160.0, duration=1.2)
    spk1 = _tone(freq=280.0, duration=1.2)
    return np.concatenate([spk0, spk1])


@pytest.fixture
def default_params():
    from chimera.keygen import derive_params

    return derive_params("test-fixture-key", intensity=0.7)


@pytest.fixture
def strong_params():
    from chimera.keygen import derive_params

    return derive_params("test-fixture-key-strong", intensity=0.9)
