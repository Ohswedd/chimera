"""
chimera.diarize
~~~~~~~~~~~~~~~
Lightweight, model-free speaker diarization via MFCC-based clustering.

Algorithm
---------
1. Extract 13-dimensional MFCC vectors (+ delta + delta-delta → 39 dims)
   for every voiced segment found by the VAD.
2. Pool and standardise the vectors (zero-mean, unit-variance per dim).
3. Cluster with k-means (scikit-learn).  If *n_speakers* is unknown, we
   use the silhouette score over k ∈ [2, 8] to select automatically.
4. Assign each voiced frame to its cluster label.
5. Apply a majority-vote smoothing within each VAD segment to avoid
   rapid speaker switches inside a single contiguous speech region.

Limitations
-----------
- Works best with 2–6 speakers and segments longer than ~2 s.
- Not a replacement for DNN-based diarization (pyannote.audio) when
  high accuracy is required.  It is intentionally dependency-light.
- Silence segments are always labelled SILENCE, never as a speaker.

Integration
-----------
chimera.diarize is called automatically by the pipeline when
*n_speakers* ≥ 1 or *mode* requires per-speaker masking.  It can also
be used standalone for inspection purposes.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .exceptions import DiarizationError
from .types import SpeakerSegment

# ── MFCC extraction (pure numpy, no librosa dependency) ──────────────────────


def _preemphasis(audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    return np.append(audio[0], audio[1:] - coeff * audio[:-1])


def _mel_filterbank(
    n_filters: int, n_fft: int, sr: int, fmin: float = 0.0, fmax: float | None = None
) -> np.ndarray:
    fmax = fmax or sr / 2.0
    mel_lo = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_hi = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel_pts = np.linspace(mel_lo, mel_hi, n_filters + 2)
    hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fb = np.zeros((n_filters, n_fft // 2 + 1))
    for m in range(1, n_filters + 1):
        lo, center, hi = bins[m - 1], bins[m], bins[m + 1]
        fb[m - 1, lo:center] = (np.arange(lo, center) - lo) / (center - lo + 1e-9)
        fb[m - 1, center:hi] = (hi - np.arange(center, hi)) / (hi - center + 1e-9)
    return fb


def _extract_mfcc(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    n_filters: int = 40,
) -> np.ndarray:
    """Return MFCC matrix of shape (n_frames, n_mfcc * 3) including deltas."""
    audio = _preemphasis(audio)
    frame_len = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    n_fft = 2 ** int(np.ceil(np.log2(frame_len)))

    n_frames = 1 + (len(audio) - frame_len) // hop
    if n_frames < 1:
        return np.zeros((0, n_mfcc * 3))

    # Framing + Hamming window
    idx = np.arange(frame_len)[np.newaxis, :] + np.arange(n_frames)[:, np.newaxis] * hop
    frames = audio[idx] * np.hamming(frame_len)

    # Power spectrum
    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2

    # Mel filterbank
    fb = _mel_filterbank(n_filters, n_fft, sr)
    mel_spec = np.dot(spec, fb.T)
    mel_spec = np.where(mel_spec > 0, mel_spec, np.finfo(float).eps)
    log_mel = np.log(mel_spec)

    # DCT
    log_mel.shape[0]
    dct_matrix = np.cos(
        np.pi
        / n_filters
        * (np.arange(n_mfcc)[:, np.newaxis] + 0.5)
        * np.arange(n_filters)[np.newaxis, :]
    )
    mfcc = np.dot(log_mel, dct_matrix.T)  # (n_frames, n_mfcc)

    # Deltas
    def _delta(m: np.ndarray, w: int = 2) -> np.ndarray:
        """Numerical delta (first-order derivative) of MFCC matrix."""
        dm = np.zeros_like(m)
        denom = 2.0 * np.sum(np.arange(1, w + 1) ** 2) + 1e-9
        for t in range(len(m)):
            for k in range(1, w + 1):
                t_fwd = min(len(m) - 1, t + k)
                t_bwd = max(0, t - k)
                dm[t] += k * (m[t_fwd] - m[t_bwd])
            dm[t] /= denom
        return dm

    delta1 = _delta(mfcc)
    delta2 = _delta(delta1)
    return np.concatenate([mfcc, delta1, delta2], axis=1)


# ── Speaker Diarizer ──────────────────────────────────────────────────────────


class SpeakerDiarizer:
    """
    MFCC + k-means speaker diarizer.

    Parameters
    ----------
    n_speakers : int | None
        Expected number of speakers.  If None, automatically selected
        from 2..max_speakers using the silhouette criterion.
    max_speakers : int
        Upper bound when auto-selecting (ignored if n_speakers is given).
    n_mfcc : int
        Number of MFCC coefficients (default 13; set to 20 for noisy audio).
    random_state : int
        Seed for k-means reproducibility.
    """

    def __init__(
        self,
        n_speakers: int | None = None,
        max_speakers: int = 8,
        n_mfcc: int = 13,
        random_state: int = 42,
    ) -> None:
        self.n_speakers = n_speakers
        self.max_speakers = max_speakers
        self.n_mfcc = n_mfcc
        self.random_state = random_state

    def diarize(
        self,
        audio: np.ndarray,
        sr: int,
        voiced_intervals: list[tuple[float, float, bool]],
    ) -> list[SpeakerSegment]:
        """
        Assign speaker labels to voiced intervals.

        Parameters
        ----------
        audio : np.ndarray
            Full audio signal (mono, float64).
        sr : int
            Sample rate.
        voiced_intervals : list of (start_sec, end_sec, is_voiced)
            Output of VoiceActivityDetector.detect().

        Returns
        -------
        list[SpeakerSegment]
            All intervals with speaker_id set.  Silence intervals receive
            speaker_id = "SILENCE".

        Raises
        ------
        DiarizationError
            If insufficient voiced audio is available.
        """
        segments: list[SpeakerSegment] = []

        # Separate voiced from silence early
        voiced = [(s, e) for s, e, v in voiced_intervals if v]
        if not voiced:
            for s, e, _v in voiced_intervals:
                segments.append(SpeakerSegment("SILENCE", s, e, False))
            return segments

        # ── Extract MFCCs for each voiced chunk ──────────────────────────────
        chunk_features: list[np.ndarray] = []
        chunk_frames: list[int] = []
        hop_ms = 10.0
        int(sr * hop_ms / 1000)

        for s, e in voiced:
            start = int(s * sr)
            end = int(e * sr)
            chunk = audio[start:end]
            if len(chunk) < int(sr * 0.1):  # skip < 100 ms
                chunk_features.append(np.zeros((0, self.n_mfcc * 3)))
                chunk_frames.append(0)
                continue
            mfcc = _extract_mfcc(chunk, sr, n_mfcc=self.n_mfcc, hop_ms=hop_ms)
            chunk_features.append(mfcc)
            chunk_frames.append(len(mfcc))

        all_feat = np.concatenate([f for f in chunk_features if len(f) > 0], axis=0)
        if len(all_feat) < 10:
            raise DiarizationError(
                "Not enough voiced frames for diarization. "
                "Ensure the audio contains at least 1 second of speech."
            )

        # ── Standardise features ─────────────────────────────────────────────
        scaler = StandardScaler()
        all_feat_scaled = scaler.fit_transform(all_feat)

        # ── Select n_speakers (if unknown) ───────────────────────────────────
        k = self._select_k(all_feat_scaled)

        # ── K-means clustering ───────────────────────────────────────────────
        km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        frame_labels = km.fit_predict(all_feat_scaled)

        # ── Map frame labels back to voiced intervals ─────────────────────────
        # frame_labels is a flat array; split it by chunk_frames
        voiced_labels: list[np.ndarray] = []
        ptr = 0
        for n_fr in chunk_frames:
            if n_fr == 0:
                voiced_labels.append(np.array([], dtype=int))
            else:
                voiced_labels.append(frame_labels[ptr : ptr + n_fr])
                ptr += n_fr

        # ── Assign majority-vote label to each voiced interval ───────────────
        vi = 0  # index into voiced list
        for s, e, is_v in voiced_intervals:
            if not is_v:
                segments.append(SpeakerSegment("SILENCE", s, e, False))
                continue

            labels_for_chunk = voiced_labels[vi]
            vi += 1
            if len(labels_for_chunk) == 0:
                # chunk too short – assign to most common speaker so far
                most_common = "SPEAKER_0"
                if segments:
                    spoken = [sg.speaker_id for sg in segments if sg.is_voiced]
                    most_common = max(set(spoken), key=spoken.count) if spoken else "SPEAKER_0"
                segments.append(SpeakerSegment(most_common, s, e, True))
                continue

            # Majority vote
            majority = int(np.bincount(labels_for_chunk).argmax())
            speaker_id = f"SPEAKER_{majority}"
            segments.append(SpeakerSegment(speaker_id, s, e, True))

        return segments

    def _select_k(self, features: np.ndarray) -> int:
        """Select optimal k via silhouette score."""
        if self.n_speakers is not None:
            return max(1, self.n_speakers)
        if len(features) < 4:
            return 1

        best_k = 2
        best_score = -1.0
        max_k = min(self.max_speakers, len(features) // 2)
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=5)
            labels = km.fit_predict(features)
            if len(set(labels)) < 2:
                continue
            try:
                score = silhouette_score(features[::4], labels[::4])  # subsample
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_k = k
        return best_k
