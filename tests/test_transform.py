"""Tests for chimera.transform — Praat LPC voice transformation."""

from __future__ import annotations

import numpy as np

from chimera.transform import apply_all_layers, change_voice


class TestChangeVoice:
    def test_zero_shift_noop(self, long_audio, sr):
        result = change_voice(long_audio, sr, semitones=0.0, formant_ratio=1.0)
        np.testing.assert_array_equal(result, long_audio)

    def test_pitch_changes_audio(self, long_audio, sr):
        result = change_voice(long_audio, sr, semitones=4.0, formant_ratio=1.0)
        assert not np.allclose(long_audio, result)

    def test_formant_changes_audio(self, long_audio, sr):
        result = change_voice(long_audio, sr, semitones=0.0, formant_ratio=1.15)
        assert not np.allclose(long_audio, result)

    def test_combined_changes_audio(self, long_audio, sr):
        result = change_voice(long_audio, sr, semitones=4.0, formant_ratio=1.15)
        assert not np.allclose(long_audio, result)

    def test_preserves_length(self, long_audio, sr):
        result = change_voice(long_audio, sr, semitones=-4.0, formant_ratio=0.90)
        assert len(result) == len(long_audio)

    def test_near_deterministic(self, long_audio, sr):
        r1 = change_voice(long_audio, sr, semitones=3.0, formant_ratio=1.10)
        r2 = change_voice(long_audio, sr, semitones=3.0, formant_ratio=1.10)
        # LPC resynthesis has small floating-point jitter;
        # correlation > 0.99 confirms near-identical output
        corr = np.corrcoef(r1, r2)[0, 1]
        assert corr > 0.99, f"Expected corr > 0.99, got {corr:.4f}"

    def test_empty_audio(self):
        empty = np.zeros(0, dtype=np.float64)
        result = change_voice(empty, 22050, semitones=3.0, formant_ratio=1.10)
        assert len(result) == 0


class TestApplyAllLayers:
    def test_output_differs_from_input(self, long_audio, sr, default_params):
        result = apply_all_layers(long_audio, sr, default_params)
        assert not np.allclose(long_audio, result)

    def test_output_same_length(self, long_audio, sr, default_params):
        result = apply_all_layers(long_audio, sr, default_params)
        assert len(result) == len(long_audio)

    def test_output_bounded(self, long_audio, sr, default_params):
        result = apply_all_layers(long_audio, sr, default_params)
        assert np.max(np.abs(result)) <= 1.0 + 1e-6

    def test_output_float64(self, long_audio, sr, default_params):
        result = apply_all_layers(long_audio, sr, default_params)
        assert result.dtype == np.float64

    def test_near_deterministic(self, long_audio, sr, default_params):
        r1 = apply_all_layers(long_audio, sr, default_params)
        r2 = apply_all_layers(long_audio, sr, default_params)
        corr = np.corrcoef(r1, r2)[0, 1]
        assert corr > 0.99, f"Expected corr > 0.99, got {corr:.4f}"

    def test_different_params_different_output(self, long_audio, sr):
        from chimera.keygen import derive_params

        p1 = derive_params("key-one", intensity=0.8)
        p2 = derive_params("key-two", intensity=0.8)
        r1 = apply_all_layers(long_audio, sr, p1)
        r2 = apply_all_layers(long_audio, sr, p2)
        assert not np.allclose(r1, r2)

    def test_empty_audio_returns_empty(self, sr, default_params):
        empty = np.zeros(0, dtype=np.float64)
        result = apply_all_layers(empty, sr, default_params)
        assert len(result) == 0
