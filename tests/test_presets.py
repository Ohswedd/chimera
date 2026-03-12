"""Tests for chimera.presets."""

from __future__ import annotations

import pytest

from chimera.exceptions import PresetNotFoundError
from chimera.presets import PRESETS, list_presets, resolve_intensity


class TestPresets:
    def test_all_five_presets_exist(self):
        for name in ("whisper", "subtle", "moderate", "strong", "extreme"):
            assert name in PRESETS

    def test_intensities_in_range(self):
        for name, intensity in PRESETS.items():
            assert 0.0 <= intensity <= 1.0, f"{name} intensity out of [0,1]"

    def test_intensities_ordered(self):
        order = ["whisper", "subtle", "moderate", "strong", "extreme"]
        values = [PRESETS[n] for n in order]
        assert values == sorted(values), "Presets must be in ascending intensity order"

    def test_resolve_intensity_returns_float(self):
        for name in PRESETS:
            assert isinstance(resolve_intensity(name), float)

    def test_resolve_invalid_raises(self):
        with pytest.raises(PresetNotFoundError):
            resolve_intensity("nonexistent-preset")

    def test_list_presets_count(self):
        presets = list_presets()
        assert len(presets) == len(PRESETS)

    def test_list_presets_has_required_fields(self):
        for p in list_presets():
            assert "name" in p
            assert "intensity" in p
            assert "description" in p

    def test_extreme_is_max_intensity(self):
        assert PRESETS["extreme"] == 1.0

    def test_whisper_is_lowest(self):
        assert PRESETS["whisper"] == min(PRESETS.values())
