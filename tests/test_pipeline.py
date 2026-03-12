"""Integration tests for chimera.pipeline and chimera.core."""

from __future__ import annotations

import numpy as np
import pytest

import chimera
from chimera import ChimeraResult, MaskMode


class TestMaskArray:
    def test_returns_chimera_result(self, long_audio, sr):
        result = chimera.mask_array(long_audio, sr, key="pipeline-test", preset="moderate")
        assert isinstance(result, ChimeraResult)

    def test_output_length_equals_input(self, long_audio, sr):
        result = chimera.mask_array(long_audio, sr, key="length-test")
        assert len(result.audio) == len(long_audio)

    def test_output_float64(self, long_audio, sr):
        result = chimera.mask_array(long_audio, sr, key="dtype-test")
        assert result.audio.dtype == np.float64

    def test_output_normalised(self, long_audio, sr):
        result = chimera.mask_array(long_audio, sr, key="norm-test")
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6

    def test_determinism(self, long_audio, sr):
        r1 = chimera.mask_array(long_audio, sr, key="det-key", intensity=0.6)
        r2 = chimera.mask_array(long_audio, sr, key="det-key", intensity=0.6)
        np.testing.assert_array_equal(r1.audio, r2.audio)

    def test_different_keys_different_output(self, long_audio, sr):
        r1 = chimera.mask_array(long_audio, sr, key="key-A", intensity=0.8)
        r2 = chimera.mask_array(long_audio, sr, key="key-B", intensity=0.8)
        assert not np.allclose(r1.audio, r2.audio)

    def test_output_differs_from_input(self, long_audio, sr):
        result = chimera.mask_array(long_audio, sr, key="diff-test", preset="strong")
        assert not np.allclose(long_audio, result.audio)

    def test_stereo_input_produces_mono_output(self, stereo_audio, sr):
        result = chimera.mask_array(stereo_audio, sr, key="stereo-test")
        assert result.audio.ndim == 1

    def test_processing_time_recorded(self, long_audio, sr):
        result = chimera.mask_array(long_audio, sr, key="timing-test")
        assert result.processing_time_s > 0.0

    def test_duration_property(self, long_audio, sr):
        result = chimera.mask_array(long_audio, sr, key="duration-test")
        expected = len(long_audio) / sr
        assert abs(result.duration_s - expected) < 0.2


class TestPresets:
    @pytest.mark.parametrize("preset", ["whisper", "subtle", "moderate", "strong", "extreme"])
    def test_all_presets_produce_output(self, short_audio, sr, preset):
        result = chimera.mask_array(short_audio, sr, key="preset-test", preset=preset)
        assert len(result.audio) > 0

    def test_higher_intensity_has_larger_parameter_offsets(self):
        """Intensity scaling must strictly increase all parameter offsets."""
        import chimera

        p_low = chimera.get_params("monotone-key", intensity=0.1)
        p_high = chimera.get_params("monotone-key", intensity=0.9)
        # All offsets from identity must be larger at higher intensity
        assert abs(p_high.pitch_shift_semitones) >= abs(p_low.pitch_shift_semitones)
        assert abs(p_high.formant_warp - 1.0) >= abs(p_low.formant_warp - 1.0)
        assert p_high.breathiness >= p_low.breathiness
        assert p_high.temporal_jitter >= p_low.temporal_jitter


class TestMaskMode:
    def test_all_same_mode(self, long_audio, sr):
        result = chimera.mask_array(
            long_audio, sr, key="same-mode", mode=MaskMode.ALL_SAME, preset="moderate"
        )
        assert isinstance(result, ChimeraResult)

    def test_all_unique_mode(self, long_audio, sr):
        result = chimera.mask_array(
            long_audio, sr, key="unique-mode", mode=MaskMode.ALL_UNIQUE, preset="moderate"
        )
        assert isinstance(result, ChimeraResult)

    def test_selected_mode_with_no_ids(self, long_audio, sr):
        result = chimera.mask_array(
            long_audio, sr, key="sel-mode", mode=MaskMode.SELECTED, speaker_ids=None
        )
        assert isinstance(result, ChimeraResult)


class TestGetParams:
    def test_returns_mask_params(self):
        from chimera.types import MaskParams

        p = chimera.get_params("inspect-key", preset="strong")
        assert isinstance(p, MaskParams)

    def test_deterministic(self):
        p1 = chimera.get_params("same-key", preset="moderate")
        p2 = chimera.get_params("same-key", preset="moderate")
        assert p1.pitch_shift_semitones == p2.pitch_shift_semitones

    def test_summary_returns_string(self):
        p = chimera.get_params("summary-key", intensity=0.5)
        s = p.summary()
        assert isinstance(s, str)
        assert "Pitch shift" in s


class TestNoCOWL:
    def test_cowl_false_still_produces_output(self, long_audio, sr):
        result = chimera.mask_array(
            long_audio, sr, key="no-cowl", preset="moderate", apply_cowl=False
        )
        assert len(result.audio) > 0
        assert not np.allclose(long_audio, result.audio)
