"""Tests for chimera.transform — 7-layer WORLD vocoder stack."""

from __future__ import annotations

import numpy as np

from chimera.transform import (
    analyse,
    apply_all_layers,
    l1_pitch_shift,
    l2_vibrato,
    l3_temporal_jitter,
    l4_formant_warp,
    l5_spectral_tilt,
    l6_subharmonic_injection,
    l7_breathiness,
)


class TestAnalysis:
    def test_returns_three_streams(self, long_audio, sr):
        f0, sp, ap = analyse(long_audio, sr)
        assert f0.ndim == 1
        assert sp.ndim == 2
        assert ap.ndim == 2

    def test_f0_has_voiced_frames(self, long_audio, sr):
        f0, _, _ = analyse(long_audio, sr)
        assert np.any(f0 > 0.0), "Voiced tone must have voiced F0 frames"

    def test_sp_positive(self, long_audio, sr):
        _, sp, _ = analyse(long_audio, sr)
        assert np.all(sp > 0.0)

    def test_ap_in_range(self, long_audio, sr):
        _, _, ap = analyse(long_audio, sr)
        assert np.all(ap >= 0.0)
        assert np.all(ap <= 1.0 + 1e-6)


class TestF0Layers:
    def test_l1_pitch_shift_up(self, long_audio, sr):
        f0, _, _ = analyse(long_audio, sr)
        shifted = l1_pitch_shift(f0, semitones=4.0)
        voiced = f0 > 0
        if voiced.any():
            ratio = shifted[voiced].mean() / f0[voiced].mean()
            expected = 2.0 ** (4.0 / 12.0)
            assert abs(ratio - expected) < 0.05

    def test_l1_pitch_shift_zero_noop(self, long_audio, sr):
        f0, _, _ = analyse(long_audio, sr)
        result = l1_pitch_shift(f0, semitones=0.0)
        np.testing.assert_array_equal(f0, result)

    def test_l1_unvoiced_frames_unchanged(self, long_audio, sr):
        f0, _, _ = analyse(long_audio, sr)
        shifted = l1_pitch_shift(f0, semitones=5.0)
        unvoiced = f0 == 0.0
        np.testing.assert_array_equal(shifted[unvoiced], f0[unvoiced])

    def test_l2_vibrato_zero_depth_noop(self, long_audio, sr):
        f0, _, _ = analyse(long_audio, sr)
        result = l2_vibrato(f0, rate_hz=5.0, depth_st=0.0, sr=sr)
        np.testing.assert_array_equal(f0, result)

    def test_l3_jitter_changes_f0(self, long_audio, sr):
        f0, _, _ = analyse(long_audio, sr)
        rng = np.random.default_rng(42)
        jittered = l3_temporal_jitter(f0, coefficient=0.01, rng=rng)
        voiced = f0 > 0
        if voiced.sum() > 10:
            assert not np.allclose(f0[voiced], jittered[voiced])

    def test_l3_jitter_zero_noop(self, long_audio, sr):
        f0, _, _ = analyse(long_audio, sr)
        rng = np.random.default_rng(0)
        result = l3_temporal_jitter(f0, coefficient=0.0, rng=rng)
        np.testing.assert_array_equal(f0, result)

    def test_l3_jitter_clips_to_physiological_range(self, long_audio, sr):
        f0, _, _ = analyse(long_audio, sr)
        rng = np.random.default_rng(1)
        result = l3_temporal_jitter(f0, coefficient=0.018, rng=rng)
        voiced = result > 0
        assert np.all(result[voiced] >= 50.0)
        assert np.all(result[voiced] <= 600.0)


class TestSPLayers:
    def test_l4_warp_one_noop(self, long_audio, sr):
        _, sp, _ = analyse(long_audio, sr)
        result = l4_formant_warp(sp, warp_factor=1.0)
        np.testing.assert_allclose(result, sp, rtol=1e-5)

    def test_l4_warp_changes_sp(self, long_audio, sr):
        _, sp, _ = analyse(long_audio, sr)
        warped = l4_formant_warp(sp, warp_factor=0.90)
        assert not np.allclose(sp, warped)

    def test_l4_warp_output_positive(self, long_audio, sr):
        _, sp, _ = analyse(long_audio, sr)
        warped = l4_formant_warp(sp, warp_factor=1.15)
        assert np.all(warped > 0.0)

    def test_l5_tilt_zero_noop(self, long_audio, sr):
        _, sp, _ = analyse(long_audio, sr)
        result = l5_spectral_tilt(sp, sr, tilt_db_per_khz=0.0)
        np.testing.assert_allclose(result, sp, rtol=1e-5)

    def test_l5_tilt_changes_sp(self, long_audio, sr):
        _, sp, _ = analyse(long_audio, sr)
        tilted = l5_spectral_tilt(sp, sr, tilt_db_per_khz=2.0)
        assert not np.allclose(sp, tilted)

    def test_l6_zero_mix_noop(self, long_audio, sr):
        f0, sp, _ = analyse(long_audio, sr)
        result = l6_subharmonic_injection(sp, f0, mix=0.0, sr=sr)
        np.testing.assert_allclose(result, sp, rtol=1e-5)


class TestAPLayer:
    def test_l7_zero_noop(self, long_audio, sr):
        _, _, ap = analyse(long_audio, sr)
        result = l7_breathiness(ap, amount=0.0)
        np.testing.assert_array_equal(result, ap)

    def test_l7_increases_aperiodicity(self, long_audio, sr):
        _, _, ap = analyse(long_audio, sr)
        result = l7_breathiness(ap, amount=0.3)
        assert result.mean() > ap.mean()

    def test_l7_stays_in_range(self, long_audio, sr):
        _, _, ap = analyse(long_audio, sr)
        result = l7_breathiness(ap, amount=0.45)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0 + 1e-6)


class TestApplyAllLayers:
    def test_output_differs_from_input(self, long_audio, sr, default_params):
        result = apply_all_layers(long_audio, sr, default_params)
        assert not np.allclose(long_audio, result)

    def test_output_same_length(self, long_audio, sr, default_params):
        result = apply_all_layers(long_audio, sr, default_params)
        assert len(result) == len(long_audio)

    def test_output_normalised(self, long_audio, sr, default_params):
        result = apply_all_layers(long_audio, sr, default_params)
        assert np.max(np.abs(result)) <= 1.0 + 1e-6

    def test_output_float64(self, long_audio, sr, default_params):
        result = apply_all_layers(long_audio, sr, default_params)
        assert result.dtype == np.float64

    def test_deterministic(self, long_audio, sr, default_params):
        r1 = apply_all_layers(long_audio, sr, default_params)
        r2 = apply_all_layers(long_audio, sr, default_params)
        np.testing.assert_array_equal(r1, r2)

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
