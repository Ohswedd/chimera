"""Tests for chimera.irreversible — Cryptographic One-Way Layer (COWL)."""

from __future__ import annotations

import numpy as np

from chimera.irreversible import apply_cowl


class TestCOWL:
    def test_changes_audio(self, long_audio, sr, default_params):
        result = apply_cowl(long_audio, sr, default_params)
        assert not np.allclose(long_audio, result)

    def test_deterministic(self, long_audio, sr, default_params):
        r1 = apply_cowl(long_audio, sr, default_params)
        r2 = apply_cowl(long_audio, sr, default_params)
        np.testing.assert_array_equal(r1, r2)

    def test_different_keys_different_output(self, long_audio, sr):
        from chimera.keygen import derive_params

        p1 = derive_params("cowl-key-A", intensity=0.8)
        p2 = derive_params("cowl-key-B", intensity=0.8)
        r1 = apply_cowl(long_audio, sr, p1)
        r2 = apply_cowl(long_audio, sr, p2)
        assert not np.allclose(r1, r2)

    def test_output_normalised(self, long_audio, sr, default_params):
        result = apply_cowl(long_audio, sr, default_params)
        assert np.max(np.abs(result)) <= 1.0 + 1e-6

    def test_output_float64(self, long_audio, sr, default_params):
        result = apply_cowl(long_audio, sr, default_params)
        assert result.dtype == np.float64

    def test_zero_intensity_passthrough(self, long_audio, sr):
        from chimera.keygen import derive_params

        p = derive_params("any", intensity=0.0)
        result = apply_cowl(long_audio, sr, p)
        # At zero intensity COWL should be a near-identity
        np.testing.assert_allclose(result, long_audio, atol=1e-6)

    def test_length_preserved(self, long_audio, sr, default_params):
        result = apply_cowl(long_audio, sr, default_params)
        assert len(result) == len(long_audio)

    def test_very_short_audio_handled(self, sr, default_params):
        tiny = np.zeros(64, dtype=np.float64)
        result = apply_cowl(tiny, sr, default_params)
        assert isinstance(result, np.ndarray)

    def test_high_intensity_more_distortion(self, long_audio, sr):
        from chimera.keygen import derive_params

        p_low = derive_params("same-key", intensity=0.2)
        p_high = derive_params("same-key", intensity=1.0)
        r_low = apply_cowl(long_audio, sr, p_low)
        r_high = apply_cowl(long_audio, sr, p_high)
        diff_low = np.mean((long_audio - r_low) ** 2)
        diff_high = np.mean((long_audio - r_high) ** 2)
        assert diff_high > diff_low, "Higher intensity must introduce more distortion"
