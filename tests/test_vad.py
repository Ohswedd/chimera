"""Tests for chimera.vad — Voice Activity Detector."""

from __future__ import annotations

import numpy as np

from chimera.vad import VoiceActivityDetector


class TestVoiceActivityDetector:
    def test_voiced_tone_produces_voiced_intervals(self, long_audio, sr):
        vad = VoiceActivityDetector()
        intervals = vad.detect(long_audio, sr)
        voiced = [v for _, _, v in intervals if v]
        assert any(voiced), "A clear voiced tone must contain voiced intervals"

    def test_silence_contains_no_voiced_intervals(self, silence, sr):
        vad = VoiceActivityDetector()
        intervals = vad.detect(silence, sr)
        assert not any(v for _, _, v in intervals)

    def test_intervals_are_non_overlapping(self, long_audio, sr):
        vad = VoiceActivityDetector()
        intervals = vad.detect(long_audio, sr)
        for i in range(len(intervals) - 1):
            _, end_i, _ = intervals[i]
            start_next, _, _ = intervals[i + 1]
            assert end_i <= start_next + 1e-6, "Intervals must not overlap"

    def test_intervals_cover_full_duration(self, long_audio, sr):
        vad = VoiceActivityDetector()
        intervals = vad.detect(long_audio, sr)
        total = sum(e - s for s, e, _ in intervals)
        expected = len(long_audio) / sr
        assert abs(total - expected) < 0.15, "Intervals must cover the full audio"

    def test_all_starts_non_negative(self, long_audio, sr):
        vad = VoiceActivityDetector()
        intervals = vad.detect(long_audio, sr)
        assert all(s >= 0.0 for s, _, _ in intervals)

    def test_very_short_audio_handled(self, sr):
        """Audio shorter than one frame must not crash."""
        tiny = np.zeros(100, dtype=np.float64)
        vad = VoiceActivityDetector()
        intervals = vad.detect(tiny, sr)
        assert isinstance(intervals, list)

    def test_custom_thresholds_respected(self, long_audio, sr):
        """A very aggressive threshold should detect everything as voiced."""
        vad = VoiceActivityDetector(
            energy_threshold_db=-100.0,
            zcr_threshold=1.0,
            periodicity_threshold=0.0,
            min_speech_ms=1.0,
            min_silence_ms=1.0,
        )
        intervals = vad.detect(long_audio, sr)
        voiced_duration = sum(e - s for s, e, v in intervals if v)
        assert voiced_duration > 0.5

    def test_noise_not_classified_as_speech(self, sr):
        """White noise should score low on periodicity and high on ZCR."""
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.1, int(sr * 1.0))
        vad = VoiceActivityDetector()
        intervals = vad.detect(noise, sr)
        voiced_duration = sum(e - s for s, e, v in intervals if v)
        total_duration = len(noise) / sr
        # VAD may classify some noise as voiced; it must not classify all of it
        assert voiced_duration < total_duration * 0.9
