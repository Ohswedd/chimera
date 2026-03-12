# Changelog

All notable changes to Chimera are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] -- 2026-03-12

### Changed

- **chimera.transform** -- Complete rewrite: replaced 7-layer WORLD vocoder with
  Praat's LPC "Change gender" command.  Uses Linear Predictive Coding to
  decompose voice into glottal source + vocal tract filter, scales LPC filter
  coefficients to shift formants (vocal tract length), and uses PSOLA for pitch
  resynthesis.  This is the gold standard for natural-sounding voice identity
  change because it operates on actual speech models (source-filter
  decomposition), not generic spectral processing.

- **chimera.irreversible** -- Simplified COWL to two sub-perceptual mechanisms:
  COWL-1 (spectral noise injection at 0.3% of signal) and COWL-2 (micro phase
  perturbation at ~2 degrees).  Removed COWL-3 (non-linear spectral quantisation).
  Added silent frame gating to avoid raising noise floor in quiet passages.

- **chimera.keygen** -- Updated parameter ranges: pitch [-8, +8] semitones,
  formant warp [0.80, 1.20], tilt [-1.5, +1.5] dB/kHz, breathiness [0.01, 0.10].
  Added minimum identity shift enforcement: |pitch| >= 4 semitones, |formant - 1| >= 12%
  after intensity scaling.

- **chimera.types** -- Simplified MaskParams: removed `vibrato_rate`, `vibrato_depth`,
  `temporal_jitter`, and `subharmonic_mix` fields.  Kept `pitch_shift_semitones`,
  `formant_warp`, `spectral_tilt`, `breathiness`, `intensity`, `seed`, `speaker_label`.

- **chimera.pipeline** -- Changed input preparation from peak normalisation to
  safety-clip-only (preserves original loudness).  Changed final output from
  peak normalisation to RMS level-matching.

- **Audio quality** -- LPC source-filter model produces natural-sounding output
  even at large parameter shifts.  RMS level-matching preserves original loudness.

### Removed

- Removed `pyworld` dependency entirely.
- Removed 5 vocoder layers: sinusoidal vibrato, micro-temporal jitter,
  spectral tilt application, sub-harmonic injection, breathiness blend.
- Removed COWL-3 (non-linear spectral quantisation / mu-law).

### Added

- `praat-parselmouth>=0.4` dependency for Praat LPC voice transformation.

---

## [0.1.0] -- 2026-03-12

### Added

- **chimera.vad** -- Multi-feature Voice Activity Detector using short-term energy,
  zero-crossing rate, and autocorrelation-based periodicity with majority-vote decision
  and hang-over / burst-removal smoothing.

- **chimera.diarize** -- MFCC (13 + delta + delta-delta = 39 dimensions) k-means speaker
  diarizer with automatic k selection via silhouette score (k in 2-8).

- **chimera.keygen** -- HKDF-SHA256 key derivation producing domain-separated
  transformation parameters from any passphrase.

- **chimera.transform** -- Voice transformation engine (initially WORLD vocoder based).

- **chimera.irreversible** -- Cryptographic One-Way Layer (COWL) for irreversibility.

- **chimera.pipeline** -- `ChimeraPipeline` orchestrating all stages with cross-fade
  stitching, per-speaker parameter assignment, and three `MaskMode` options
  (`ALL_UNIQUE`, `ALL_SAME`, `SELECTED`).

- **chimera.realtime** -- `RealtimeAnonymiser` for microphone capture with ring-buffer
  threading, and `mask_stream()` generator for push-based chunk streaming.

- **chimera.core** -- High-level API: `mask_file()`, `mask_array()`, `get_params()`,
  `get_speaker_params()`.

- **chimera.presets** -- Five named intensity presets: `whisper`, `subtle`, `moderate`,
  `strong`, `extreme`.

- **chimera.types** -- Fully typed `MaskParams` (frozen dataclass), `SpeakerSegment`,
  `ChimeraResult`, `MaskMode` enum.

- **chimera.exceptions** -- Custom exception hierarchy with eight typed exceptions.

- **tests/** -- Test suite covering key derivation, VAD, transform layers, COWL,
  pipeline integration, preset resolution, and multi-speaker modes.

- **examples/** -- Six runnable examples: basic file masking, multi-speaker, array API,
  real-time microphone, streaming, and parameter inspection.

- **docs/** -- Architecture overview, quick-start guide, full API reference, security
  model, and changelog.

- **paper/** -- Full academic paper PDF (11 sections, 15 references, 11 tables).

- **GitHub Actions** -- CI workflow (lint + test on Python 3.9-3.12),
  release workflow (PyPI publish on tag), weekly security audit.

- PEP 561 `py.typed` marker -- Chimera ships with inline type annotations.

---

## [Unreleased]

### Planned

- Neural diarizer plug-in interface (pyannote.audio compatibility shim).
- Prosody-preserving F0 contour warp.
- Formal EER / WER benchmark suite.
- Reversible mode with symmetric key.
- WebAssembly build target.
- Graphical user interface (Qt / web).
