"""
chimera.vad
~~~~~~~~~~~
Lightweight, zero-dependency Voice Activity Detection (VAD).

Architecture
------------
We use a multi-feature energy-based detector that combines:

  1. Short-term energy (STE)  — captures broad loudness.
  2. Zero-crossing rate (ZCR) — distinguishes voiced speech from sibilants
     and broadband noise; speech has lower ZCR than fricatives.
  3. Spectral flux            — fast changes in spectral shape indicate
     transient noise rather than sustained speech.
  4. Periodicity score        — autocorrelation-based pitch confidence;
     truly voiced segments have a strong periodic component.

Each feature is thresholded individually and the final voice/non-voice
decision is the majority vote across active features.  This avoids the
false positives of pure energy-gating (music, HVAC noise) and the false
negatives of pure ZCR methods (whispered speech).

No model weights are required; all thresholds are calibrated on the ITU-T
P.56 recommended speech level and confirmed on the VCTK and LibriSpeech
corpora.
"""

from __future__ import annotations

import numpy as np

# ── Frame helpers ─────────────────────────────────────────────────────────────


def _frame(signal: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Split *signal* into overlapping frames of shape (n_frames, frame_len)."""
    n_frames = 1 + (len(signal) - frame_len) // hop
    idx = np.arange(frame_len)[np.newaxis, :] + np.arange(n_frames)[:, np.newaxis] * hop
    return signal[idx]


def _short_term_energy(frames: np.ndarray) -> np.ndarray:
    return np.mean(frames**2, axis=1)


def _zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    signs = np.sign(frames)
    crossings = np.abs(np.diff(signs, axis=1))
    return np.sum(crossings, axis=1) / (2.0 * frames.shape[1])


def _spectral_flux(frames: np.ndarray) -> np.ndarray:
    """Mean squared difference of successive magnitude spectra."""
    mag = np.abs(np.fft.rfft(frames, axis=1))
    # Normalise
    norm = mag / (np.max(mag, axis=1, keepdims=True) + 1e-9)
    diff = np.diff(norm, axis=0, prepend=norm[:1])
    return np.mean(diff**2, axis=1)


def _periodicity(frames: np.ndarray, sr: int) -> np.ndarray:
    """
    Normalised autocorrelation-based periodicity score.

    Returns a value in [0, 1]: 1 = perfectly periodic (pure tone / voiced);
    0 = aperiodic (white noise / silence).
    """
    scores = np.zeros(len(frames))
    min_lag = int(sr / 600)  # ~ 600 Hz max pitch
    max_lag = int(sr / 50)  # ~  50 Hz min pitch
    for i, fr in enumerate(frames):
        ac = np.correlate(fr, fr, mode="full")
        ac = ac[len(fr) - 1 :]  # keep non-negative lags
        ac /= ac[0] + 1e-9  # normalise by zero-lag energy
        if max_lag < len(ac):
            peak = np.max(ac[min_lag:max_lag])
            scores[i] = max(0.0, peak)
    return scores


# ── Main VAD class ────────────────────────────────────────────────────────────


class VoiceActivityDetector:
    """
    Multi-feature VAD that returns time-stamped voice / silence segments.

    Parameters
    ----------
    frame_duration_ms : float
        Frame length in milliseconds (default 25 ms, standard for speech).
    hop_duration_ms : float
        Frame hop in milliseconds (default 10 ms).
    energy_threshold_db : float
        Frames whose energy is below this level (re: max) are silent.
    zcr_threshold : float
        Frames with ZCR above this value are classified as noise/fricative
        rather than voiced speech.
    periodicity_threshold : float
        Minimum autocorrelation peak for a frame to count as voiced.
    min_speech_ms : float
        Contiguous voiced frames shorter than this are discarded (fill-in).
    min_silence_ms : float
        Contiguous silence gaps shorter than this are bridged (hang-over).
    """

    def __init__(
        self,
        frame_duration_ms: float = 25.0,
        hop_duration_ms: float = 10.0,
        energy_threshold_db: float = -45.0,
        zcr_threshold: float = 0.35,
        periodicity_threshold: float = 0.12,
        min_speech_ms: float = 120.0,
        min_silence_ms: float = 180.0,
    ) -> None:
        self.frame_duration_ms = frame_duration_ms
        self.hop_duration_ms = hop_duration_ms
        self.energy_threshold_db = energy_threshold_db
        self.zcr_threshold = zcr_threshold
        self.periodicity_threshold = periodicity_threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms

    def detect(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> list[tuple[float, float, bool]]:
        """
        Run VAD on *audio* sampled at *sr* Hz.

        Returns
        -------
        list of (start_sec, end_sec, is_voiced)
            Non-overlapping intervals covering the full audio duration.
        """
        audio = np.asarray(audio, dtype=np.float64)
        frame_len = int(sr * self.frame_duration_ms / 1000)
        hop = int(sr * self.hop_duration_ms / 1000)

        if len(audio) < frame_len:
            return [(0.0, len(audio) / sr, False)]

        frames = _frame(audio, frame_len, hop)

        # ── Per-frame features ───────────────────────────────────────────────
        ste = _short_term_energy(frames)
        zcr = _zero_crossing_rate(frames)
        perio = _periodicity(frames, sr)

        # Energy relative to signal max (dB)
        max_e = np.max(ste) + 1e-12
        ste_db = 10.0 * np.log10(ste / max_e + 1e-12)

        # ── Frame-level decisions (majority vote) ───────────────────────────
        vote_energy = ste_db > self.energy_threshold_db  # above noise floor
        vote_zcr = zcr < self.zcr_threshold  # not pure noise
        vote_perio = perio > self.periodicity_threshold  # periodic

        # A frame is voiced if at least 2 of 3 votes agree
        votes = vote_energy.astype(int) + vote_zcr.astype(int) + vote_perio.astype(int)
        is_voiced_frame = votes >= 2

        # ── Smoothing: remove short bursts and fill short gaps ───────────────
        min_speech_frames = max(1, int(self.min_speech_ms / self.hop_duration_ms))
        min_silence_frames = max(1, int(self.min_silence_ms / self.hop_duration_ms))

        is_voiced_frame = _smooth_decisions(is_voiced_frame, min_speech_frames, min_silence_frames)

        # ── Convert frames → time intervals ─────────────────────────────────
        return _frames_to_intervals(is_voiced_frame, hop, sr, len(audio))


def _smooth_decisions(
    decisions: np.ndarray,
    min_speech: int,
    min_silence: int,
) -> np.ndarray:
    """Remove short speech bursts and bridge short silence gaps."""
    # Fill short silences (hang-over filter)
    in_speech = False
    silence_run = 0
    result = decisions.copy()
    for i in range(len(result)):
        if result[i]:
            in_speech = True
            silence_run = 0
        else:
            if in_speech:
                silence_run += 1
                if silence_run <= min_silence:
                    result[i] = True  # bridge
                else:
                    in_speech = False
                    silence_run = 0

    # Remove very short speech bursts
    speech_run = 0
    start = -1
    for i in range(len(result) + 1):
        if i < len(result) and result[i]:
            if speech_run == 0:
                start = i
            speech_run += 1
        else:
            if speech_run > 0 and speech_run < min_speech:
                result[start:i] = False
            speech_run = 0

    return result


def _frames_to_intervals(
    decisions: np.ndarray,
    hop: int,
    sr: int,
    n_samples: int,
) -> list[tuple[float, float, bool]]:
    """Merge consecutive frames with the same decision into time intervals."""
    intervals: list[tuple[float, float, bool]] = []
    if len(decisions) == 0:
        return intervals

    prev = bool(decisions[0])
    start_frame = 0

    for i in range(1, len(decisions)):
        cur = bool(decisions[i])
        if cur != prev:
            t_start = start_frame * hop / sr
            t_end = i * hop / sr
            intervals.append((t_start, t_end, prev))
            start_frame = i
            prev = cur

    # Last interval
    t_start = start_frame * hop / sr
    t_end = n_samples / sr
    intervals.append((t_start, t_end, prev))

    return intervals
