# Changelog

All notable changes to Chimera are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-03-12

### Added

- **chimera.vad** — Multi-feature Voice Activity Detector using short-term energy,
  zero-crossing rate, and autocorrelation-based periodicity with majority-vote decision
  and hang-over / burst-removal smoothing.

- **chimera.diarize** — MFCC (13 + delta + delta-delta = 39 dimensions) k-means speaker
  diarizer with automatic k selection via silhouette score (k ∈ 2–8).

- **chimera.keygen** — HKDF-SHA256 key derivation producing eight independent,
  domain-separated transformation parameters from any passphrase.

- **chimera.transform** — Seven-layer WORLD vocoder transformation stack:
  L1 pitch shift, L2 sinusoidal vibrato, L3 micro-temporal jitter,
  L4 formant warp, L5 spectral tilt, L6 sub-harmonic injection, L7 breathiness blend.

- **chimera.irreversible** — Cryptographic One-Way Layer (COWL):
  Sub-Perceptual Spectral Noise Injection (SSNI),
  Phase Randomisation (PR),
  Non-Linear Spectral Quantisation (NLSQ).

- **chimera.pipeline** — `ChimeraPipeline` orchestrating all stages with cross-fade
  stitching, per-speaker parameter assignment, and three `MaskMode` options
  (`ALL_UNIQUE`, `ALL_SAME`, `SELECTED`).

- **chimera.realtime** — `RealtimeAnonymiser` for microphone capture with ring-buffer
  threading, and `mask_stream()` generator for push-based chunk streaming.

- **chimera.core** — High-level API: `mask_file()`, `mask_array()`, `get_params()`,
  `get_speaker_params()`.

- **chimera.presets** — Five named intensity presets: `whisper`, `subtle`, `moderate`,
  `strong`, `extreme`.

- **chimera.types** — Fully typed `MaskParams` (frozen dataclass), `SpeakerSegment`,
  `ChimeraResult`, `MaskMode` enum.

- **chimera.exceptions** — Custom exception hierarchy with eight typed exceptions.

- **tests/** — 26-test suite covering key derivation, VAD, transform layers, COWL,
  pipeline integration, preset resolution, and multi-speaker modes.

- **examples/** — Six runnable examples: basic file masking, multi-speaker, array API,
  real-time microphone, streaming, and parameter inspection.

- **docs/** — Architecture overview, quick-start guide, full API reference, security
  model, and changelog.

- **paper/** — Full academic paper PDF (11 sections, 15 references, 11 tables).

- **GitHub Actions** — CI workflow (lint + test on Python 3.9–3.12),
  release workflow (PyPI publish on tag), weekly security audit.

- PEP 561 `py.typed` marker — Chimera ships with inline type annotations.

---

## [Unreleased]

### Planned

- Neural diarizer plug-in interface (pyannote.audio compatibility shim).
- Prosody-preserving F0 contour warp.
- Formal EER / WER benchmark suite.
- Reversible mode with symmetric key.
- WebAssembly build target.
- Graphical user interface (Qt / web).
