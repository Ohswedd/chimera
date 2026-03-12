<div align="center">

# Chimera

### Cryptographically Irreversible, Speaker-Aware Voice Anonymisation

[![CI](https://github.com/Ohswedd/chimera/actions/workflows/ci.yml/badge.svg)](https://github.com/Ohswedd/chimera/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen)](https://github.com/Ohswedd/chimera)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue)](https://pypi.org/project/chimera-voice/)
[![PyPI](https://img.shields.io/pypi/v/chimera-voice)](https://pypi.org/project/chimera-voice/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Chimera** disguises the identity of one or more speakers in any audio recording.
It runs entirely on CPU, requires no cloud services, no GPU, and no pre-trained model weights.
Given only the output audio and any key, no known algorithm can recover the original speaker's
acoustic identity.

[**Wiki**](https://github.com/Ohswedd/chimera/wiki) · [**Quick Start**](#quick-start) · [**Paper**](paper/chimera_paper.pdf)

</div>

---

## Why Chimera?

| Feature | Chimera | Pitch shift | Neural VC |
|---|:---:|:---:|:---:|
| Fully local / offline | :white_check_mark: | :white_check_mark: | :warning: |
| No model weights | :white_check_mark: | :white_check_mark: | :x: |
| Multi-speaker, per-speaker keys | :white_check_mark: | :x: | :x: |
| Automatic VAD (ignores noise) | :white_check_mark: | :x: | :x: |
| Natural-sounding output | :white_check_mark: | :x: | :white_check_mark: |
| Deterministic / auditable | :white_check_mark: | :x: | :x: |
| Cryptographically one-way | :white_check_mark: | :x: | :x: |
| Real-time microphone support | :white_check_mark: | :white_check_mark: | GPU |

---

## How It Works

Chimera applies six processing stages to every audio input:

```
Input audio
    |
    v  [1] VAD -- isolates speech from noise, music, silence
    |
    v  [2] Diarization -- MFCC k-means, assigns segments to speakers
    |
    v  [3] Key Derivation -- HKDF-SHA256 per speaker -> 5 parameters
    |
    v  [4] Voice Transform (Praat LPC source-filter model)
    |       Pitch shift     via PSOLA resynthesis
    |       Formant shift   via LPC vocal-tract length scaling
    |
    v  [5] COWL -- Cryptographic One-Way Layer
    |       Sub-perceptual spectral noise (SSNI)
    |       Micro phase perturbation (MPP)
    |
    v  [6] Cross-fade stitching + level matching
    |
Output audio (mono float64, same sample rate)
```

All randomness is derived from the key via HKDF, making the transformation **fully deterministic** and **fully one-way**: same key -> same output; output -> original is computationally infeasible.

---

## Installation

```bash
pip install chimera-voice
```

With real-time microphone support:

```bash
pip install "chimera-voice[realtime]"
```

From source:

```bash
git clone https://github.com/Ohswedd/chimera
cd chimera
pip install -e ".[dev]"
```

---

## Quick Start

### Anonymise a file

```python
import chimera

chimera.mask_file("interview.wav", "anonymous.wav", key="my-secret", preset="strong")
```

### Work with NumPy arrays

```python
import chimera
import soundfile as sf

audio, sr = sf.read("interview.wav")
result = chimera.mask_array(audio, sr, key="my-secret", preset="strong")
sf.write("anonymous.wav", result.audio, sr)
```

### Multi-speaker: independent key per speaker

```python
result = chimera.mask_array(
    audio, sr,
    key        = "my-secret",
    preset     = "moderate",
    mode       = chimera.MaskMode.ALL_UNIQUE,   # default
    n_speakers = 3,
)
print(result.speakers_masked)    # ['SPEAKER_0', 'SPEAKER_1', 'SPEAKER_2']
print(f"Processed in {result.processing_time_s:.2f}s")
```

### Mask only selected speakers

```python
result = chimera.mask_array(
    audio, sr,
    key          = "my-secret",
    preset       = "strong",
    mode         = chimera.MaskMode.SELECTED,
    speaker_ids  = ["SPEAKER_0"],   # only mask speaker 0
)
```

### Real-time microphone

```python
from chimera.realtime import RealtimeAnonymiser

anon = RealtimeAnonymiser(key="my-secret", preset="moderate")
anon.start()
input("Recording -- press Enter to stop...")
anon.stop()
anon.save("recorded_anonymous.wav")
```

### Streaming (generator-based)

```python
from chimera.realtime import mask_stream

for masked_chunk in mask_stream(my_chunk_generator, sr=22050,
                                key="my-secret", preset="strong"):
    send_to_output(masked_chunk)
```

### Inspect parameters

```python
p = chimera.get_params("my-secret", preset="strong")
print(p.summary())

#        Chimera MaskParams
# --------------------------------------------
#   Speaker label      : (none)
#   Pitch shift        : +5.743 st
#   Formant warp       : 0.91234x
#   Spectral tilt      : -1.214 dB/kHz
#   Breathiness        : 0.0812
#   Master intensity   : 0.780
# --------------------------------------------
```

---

## Presets

| Preset | Intensity | ASV EER | Use case |
|---|---|---|---|
| `whisper` | 0.12 | ~5 % | Soft watermarking |
| `subtle` | 0.28 | ~15 % | Light disguise |
| `moderate` | 0.52 | ~35 % | Speaker unrecognisable to humans |
| `strong` | 0.78 | ~48 % | ASV systems fail |
| `extreme` | 1.00 | ~50 % | Maximum -- content only |

*Indicative Equal Error Rate against ECAPA-TDNN (VoicePrivacy 2024 protocol).*

---

## MaskMode

| Mode | Behaviour |
|---|---|
| `MaskMode.ALL_UNIQUE` | Each speaker gets independent parameters (default) |
| `MaskMode.ALL_SAME` | All speakers share the same transformation |
| `MaskMode.SELECTED` | Only speakers listed in `speaker_ids` are masked |

---

## Security

Chimera is designed with three security properties:

**Determinism** -- `F(audio, key)` always returns the same output.
Every bit of randomness is derived from the key via HKDF-SHA256.

**One-way** -- Given `F(audio, key)`, recovering the original speaker's acoustic identity requires simultaneously inverting:
1. Key-seeded micro phase perturbation (requires HKDF seed)
2. Key-derived sub-perceptual noise injection (requires HMAC sub-key)
3. Praat LPC pitch + formant transform (non-invertible without all params)

**Key independence** -- HKDF-SHA256 provides 128-bit second-preimage resistance.
Per-speaker keys are domain-separated: `key + ":chimera:spk:" + speaker_id`.

> Chimera is a privacy-enhancing tool, not an encryption scheme. For high-stakes deployments, combine it with access control, key rotation, and additional anonymisation measures. See the [Security wiki page](https://github.com/Ohswedd/chimera/wiki/Security) for the full threat model.

---

## Supported Audio Formats

Any format supported by [libsndfile](http://www.mega-nerd.com/libsndfile/): WAV, FLAC, OGG, AIFF, and more.

Supported sample rates: `8 000`, `16 000`, `22 050`, `24 000`, `44 100`, `48 000` Hz.

---

## Project Structure

```
chimera/
|-- chimera/              # Library source
|   |-- __init__.py       # Public API surface
|   |-- core.py           # High-level functions: mask_file, mask_array, get_params
|   |-- pipeline.py       # ChimeraPipeline -- full orchestration
|   |-- keygen.py         # HKDF-SHA256 parameter derivation
|   |-- vad.py            # Voice Activity Detector
|   |-- diarize.py        # MFCC k-means speaker diarizer
|   |-- transform.py      # Praat LPC source-filter voice transform
|   |-- irreversible.py   # Cryptographic One-Way Layer (COWL)
|   |-- realtime.py       # Real-time microphone / streaming engine
|   |-- presets.py        # Named intensity presets
|   |-- types.py          # MaskParams, ChimeraResult, MaskMode, ...
|   |-- exceptions.py     # Exception hierarchy
|   `-- py.typed          # PEP 561 marker
|-- tests/                # Full test suite
|-- examples/             # Runnable usage examples
|-- docs/                 # Documentation
|-- benchmarks/           # Performance benchmarks
|-- paper/                # Academic paper (PDF)
|-- .github/workflows/    # CI and release workflows
|-- pyproject.toml
|-- CHANGELOG.md
|-- CONTRIBUTING.md
`-- LICENSE
```

---

## Development

### Setup

```bash
git clone https://github.com/Ohswedd/chimera
cd chimera
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Run tests

```bash
pytest                         # all tests
pytest -m "not slow"           # skip slow tests
pytest --cov=chimera           # with coverage
```

### Lint and format

```bash
black chimera tests examples
ruff check chimera tests examples
mypy chimera
```

### Build distribution

```bash
pip install build
python -m build
```

---

## Academic Paper

The full technical paper is available at [`paper/chimera_paper.pdf`](paper/chimera_paper.pdf).

It covers the full threat model (A1-A4), HKDF key derivation, voice transformation layers,
COWL security argument, diarization architecture, real-time latency profile,
comparison with state-of-the-art, and 15 references.

**Citation:**

```bibtex
@software{chimera2026,
  title   = {Chimera: Cryptographically Irreversible Speaker-Aware Voice Anonymisation},
  author  = {Ohswedd},
  year    = {2026},
  url     = {https://github.com/Ohswedd/chimera},
  version = {0.2.0},
  license = {MIT}
}
```

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE) 2026 Ohswedd
