"""Tests for chimera.diarize — speaker diarization."""

from __future__ import annotations

from chimera.diarize import SpeakerDiarizer
from chimera.vad import VoiceActivityDetector


def run_diarize(audio, sr, n_speakers=None):
    vad = VoiceActivityDetector()
    intervals = vad.detect(audio, sr)
    diarizer = SpeakerDiarizer(n_speakers=n_speakers)
    return diarizer.diarize(audio, sr, intervals)


class TestSpeakerDiarizer:
    def test_returns_list_of_segments(self, long_audio, sr):
        from chimera.types import SpeakerSegment

        segs = run_diarize(long_audio, sr)
        assert isinstance(segs, list)
        assert all(isinstance(s, SpeakerSegment) for s in segs)

    def test_silence_labelled_correctly(self, silence, sr):
        segs = run_diarize(silence, sr)
        assert all(s.speaker_id == "SILENCE" for s in segs)

    def test_segments_cover_full_audio(self, long_audio, sr):
        segs = run_diarize(long_audio, sr)
        total = sum(s.duration for s in segs)
        expected = len(long_audio) / sr
        assert abs(total - expected) < 0.2

    def test_no_negative_durations(self, long_audio, sr):
        segs = run_diarize(long_audio, sr)
        assert all(s.duration >= 0.0 for s in segs)

    def test_voiced_segments_have_speaker_label(self, long_audio, sr):
        segs = run_diarize(long_audio, sr)
        for s in segs:
            if s.is_voiced:
                assert s.speaker_id.startswith("SPEAKER_")

    def test_two_speaker_audio_detects_speakers(self, two_speaker_audio, sr):
        segs = run_diarize(two_speaker_audio, sr, n_speakers=2)
        speaker_ids = {s.speaker_id for s in segs if s.is_voiced}
        assert len(speaker_ids) >= 1  # at least one speaker detected
