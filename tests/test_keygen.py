"""Tests for chimera.keygen — HKDF-SHA256 parameter derivation."""

from __future__ import annotations

import pytest

from chimera.exceptions import KeyDerivationError
from chimera.keygen import derive_params, derive_speaker_params


class TestDeriveParams:
    def test_determinism_same_key(self):
        p1 = derive_params("hello", intensity=0.7)
        p2 = derive_params("hello", intensity=0.7)
        assert p1.pitch_shift_semitones == p2.pitch_shift_semitones
        assert p1.formant_warp == p2.formant_warp
        assert p1.seed == p2.seed

    def test_different_keys_produce_different_params(self):
        p1 = derive_params("key-alpha")
        p2 = derive_params("key-beta")
        assert p1.pitch_shift_semitones != p2.pitch_shift_semitones
        assert p1.formant_warp != p2.formant_warp

    def test_intensity_zero_is_identity(self):
        p = derive_params("any-key", intensity=0.0)
        assert abs(p.pitch_shift_semitones) < 1e-9
        assert abs(p.formant_warp - 1.0) < 1e-9
        assert abs(p.breathiness) < 1e-9
        assert abs(p.temporal_jitter) < 1e-9
        assert abs(p.spectral_tilt) < 1e-9
        assert abs(p.vibrato_rate) < 1e-9
        assert abs(p.vibrato_depth) < 1e-9
        assert abs(p.subharmonic_mix) < 1e-9

    def test_different_salts_produce_different_params(self):
        p1 = derive_params("key", salt="salt-A")
        p2 = derive_params("key", salt="salt-B")
        assert p1.pitch_shift_semitones != p2.pitch_shift_semitones

    def test_speaker_label_independence(self):
        p0 = derive_params("key", intensity=0.8, speaker_label="SPEAKER_0")
        p1 = derive_params("key", intensity=0.8, speaker_label="SPEAKER_1")
        assert p0.pitch_shift_semitones != p1.pitch_shift_semitones

    def test_speaker_label_vs_no_label(self):
        p_bare = derive_params("key", intensity=0.8)
        p_spk0 = derive_params("key", intensity=0.8, speaker_label="SPEAKER_0")
        assert p_bare.pitch_shift_semitones != p_spk0.pitch_shift_semitones

    def test_param_ranges_at_full_intensity(self):
        for trial in range(5):
            p = derive_params(f"range-test-{trial}", intensity=1.0)
            assert -10.5 <= p.pitch_shift_semitones <= 10.5
            assert 0.75 <= p.formant_warp <= 1.25
            assert 0.0 <= p.breathiness <= 0.50
            assert 0.0 <= p.temporal_jitter <= 0.02
            assert -4.5 <= p.spectral_tilt <= 4.5
            assert 0.0 <= p.vibrato_rate <= 7.5
            assert 0.0 <= p.vibrato_depth <= 0.45
            assert 0.0 <= p.subharmonic_mix <= 0.16

    def test_empty_key_raises(self):
        with pytest.raises(KeyDerivationError):
            derive_params("")

    def test_intensity_too_high_raises(self):
        with pytest.raises(KeyDerivationError):
            derive_params("key", intensity=1.1)

    def test_intensity_negative_raises(self):
        with pytest.raises(KeyDerivationError):
            derive_params("key", intensity=-0.1)

    def test_unicode_key_accepted(self):
        p = derive_params("chiave-セキュリティ-🔐", intensity=0.5)
        assert p is not None

    def test_speaker_label_stored(self):
        p = derive_params("key", speaker_label="SPEAKER_2")
        assert p.speaker_label == "SPEAKER_2"

    def test_scaled_preserves_identity_at_zero(self):
        p = derive_params("key", intensity=1.0)
        scaled = p.scaled(0.0)
        assert abs(scaled.pitch_shift_semitones) < 1e-9
        assert abs(scaled.formant_warp - 1.0) < 1e-9


class TestDeriveSpeakerParams:
    def test_returns_one_entry_per_speaker(self):
        result = derive_speaker_params("key", ["SPEAKER_0", "SPEAKER_1", "SPEAKER_2"])
        assert set(result.keys()) == {"SPEAKER_0", "SPEAKER_1", "SPEAKER_2"}

    def test_speakers_independent(self):
        result = derive_speaker_params("key", ["SPEAKER_0", "SPEAKER_1"])
        p0 = result["SPEAKER_0"]
        p1 = result["SPEAKER_1"]
        assert p0.pitch_shift_semitones != p1.pitch_shift_semitones

    def test_speaker_label_set(self):
        result = derive_speaker_params("key", ["SPEAKER_0"])
        assert result["SPEAKER_0"].speaker_label == "SPEAKER_0"
