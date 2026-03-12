"""
chimera.presets
~~~~~~~~~~~~~~~
Named intensity presets for common deployment scenarios.

Each preset maps to a master intensity scalar in [0, 1].
The preset is applied to all derived parameters via MaskParams.scaled().

Preset calibration notes
------------------------
Intensities were chosen to match informal listening panel thresholds and
a simplified ASV Equal Error Rate (EER) ladder using the VoicePrivacy
2024 protocol as a reference:

  whisper    → EER ~  5 %  (barely perceptible change)
  subtle     → EER ~ 15 %  (light disguise)
  moderate   → EER ~ 35 %  (speaker unrecognisable to most humans)
  strong     → EER ~ 48 %  (ASV and human listeners fail)
  extreme    → EER ~ 50 %  (random-chance level, max transformation)

Note: EER values are indicative and depend on the ASV system and corpus.
"""

from __future__ import annotations

from .exceptions import PresetNotFoundError

# ── Preset registry ───────────────────────────────────────────────────────────

PRESETS: dict[str, float] = {
    "whisper": 0.12,
    "subtle": 0.28,
    "moderate": 0.52,
    "strong": 0.78,
    "extreme": 1.00,
}

_DESCRIPTIONS: dict[str, str] = {
    "whisper": "Barely perceptible — useful for watermarking or subtle tagging.",
    "subtle": "Light disguise — voice sounds broadly similar but shifted.",
    "moderate": "Clear change — speaker unrecognisable to most human listeners.",
    "strong": "Robust anonymisation — automatic speaker verification fails.",
    "extreme": "Maximum transformation — only speech intelligibility preserved.",
}


def resolve_intensity(preset: str) -> float:
    """Return the intensity scalar for a named preset."""
    if preset not in PRESETS:
        raise PresetNotFoundError(preset)
    return PRESETS[preset]


def list_presets() -> list[dict]:
    """Return a list of dicts describing all available presets."""
    return [
        {
            "name": name,
            "intensity": PRESETS[name],
            "description": _DESCRIPTIONS[name],
        }
        for name in PRESETS
    ]
