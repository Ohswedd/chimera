# Architecture

## Pipeline overview

```
Input audio (mono or stereo, any supported sample rate)
    │
    ▼ ── Stage 1: VAD ──────────────────────────────────────────────────
    │   Module: chimera.vad.VoiceActivityDetector
    │   Features: Short-term energy · Zero-crossing rate · Periodicity
    │   Output:  List of (start_s, end_s, is_voiced) intervals
    │
    ▼ ── Stage 2: Speaker Diarization ──────────────────────────────────
    │   Module: chimera.diarize.SpeakerDiarizer
    │   Method: 39-dim MFCC (13 + Δ + ΔΔ) → StandardScaler → k-means
    │   k selected by silhouette score when n_speakers is unknown
    │   Output:  List of SpeakerSegment (speaker_id, start, end, is_voiced)
    │
    ▼ ── Stage 3: Key Derivation ────────────────────────────────────────
    │   Module: chimera.keygen
    │   Method: HKDF-SHA256 (Extract + Expand)
    │   PRK = HMAC-SHA256(salt, key)
    │   param_i = map(HMAC-SHA256(PRK, label_i)[0:4], range_i)
    │   Output:  Dict[speaker_id → MaskParams]
    │
    ▼ ── Stage 4: 7-Layer WORLD Vocoder Transform ───────────────────────
    │   Module: chimera.transform
    │   WORLD: DIO + StoneMask (F0) · CheapTrick (SP) · D4C (AP)
    │
    │   F0 layers:
    │     L1  Pitch shift       f0 × 2^(Δst/12)
    │     L2  Sinusoidal vibrato f0 × 2^(d·sin(2πrt)/12)
    │     L3  Micro-jitter      f0 × N(1, σ) per frame
    │
    │   SP layers:
    │     L4  Formant warp      SP resampled at α·freq (log interp)
    │     L5  Spectral tilt     SP × 10^(β·f_kHz/10)
    │     L6  Sub-harmonic inj  SP boosted at F0/2 with Gaussian
    │
    │   AP layer:
    │     L7  Breathiness blend AP + γ(1 − AP)
    │
    │   Output: Resynthesised audio chunks per segment
    │
    ▼ ── Stage 5: COWL (Cryptographic One-Way Layer) ────────────────────
    │   Module: chimera.irreversible
    │   Applied in STFT domain (nperseg adaptive, 75 % overlap)
    │
    │   COWL-1  SSNI   mag += noise shaped by ISO 226 masking threshold
    │   COWL-2  PR     phase += N(0, 0.15π × intensity) per bin
    │   COWL-3  NLSQ   μ-law compress → quantise q bits → expand
    │
    │   Output: COWL-hardened audio chunks
    │
    ▼ ── Stage 6: Stitch ────────────────────────────────────────────────
        Module: chimera.pipeline
        Method: 5 ms cross-fade at segment boundaries
        Output: Final mono float64 array, normalised to [-1, 1]
```

---

## Module responsibilities

| Module | Single responsibility |
|---|---|
| `exceptions` | Typed exception hierarchy; no logic |
| `types` | Frozen dataclasses and enums; no side-effects |
| `keygen` | Pure function: key → params |
| `vad` | Pure function: audio → intervals |
| `diarize` | Pure function: audio + intervals → segments |
| `transform` | Pure function: audio + params → audio |
| `irreversible` | Pure function: audio + params → audio |
| `pipeline` | Orchestrates stages 1–6; manages threading model |
| `realtime` | Thread-safe ring-buffer wrapper around pipeline |
| `presets` | Simple registry dict; no logic |
| `core` | Thin I/O wrappers; delegates to pipeline |

All modules are independently unit-testable because they communicate through
typed interfaces, not global state.

---

## Data flow types

```
AudioArray = np.ndarray[float64]   # mono, [-1, 1]

MaskParams (frozen dataclass)
    pitch_shift_semitones: float
    formant_warp:          float
    spectral_tilt:         float
    breathiness:           float
    temporal_jitter:       float
    vibrato_rate:          float
    vibrato_depth:         float
    subharmonic_mix:       float
    intensity:             float
    seed:                  int
    speaker_label:         str | None

SpeakerSegment (dataclass)
    speaker_id:  str
    start_sec:   float
    end_sec:     float
    is_voiced:   bool
    params:      MaskParams | None   (assigned in Stage 3)

ChimeraResult (dataclass)
    audio:              AudioArray
    sample_rate:        int
    segments:           list[SpeakerSegment]
    speakers_masked:    list[str]
    speakers_skipped:   list[str]
    processing_time_s:  float
```

---

## Real-time architecture

```
sounddevice InputStream
    │   raw float32 chunks
    ▼
Input thread (daemon)
    │   accumulates samples → fixed-size chunks
    │   puts to queue.Queue(maxsize=32)
    ▼
Processing thread (daemon)
    │   dequeues chunks
    │   → apply_all_layers(chunk, sr, params)
    │   → apply_cowl(chunk, sr, params)
    │   appends to output list / calls on_chunk callback
    ▼
Output
    save()  →  soundfile.write
    get_audio()  →  np.concatenate(output_chunks)
```

Back-pressure: if the processor falls behind, the input thread drops incoming chunks
rather than growing the queue unboundedly — a hard real-time requirement.

---

## Supported sample rates

`8 000` · `16 000` · `22 050` · `24 000` · `44 100` · `48 000` Hz

All other rates raise `UnsupportedSampleRateError`. Use `librosa.resample` or
`scipy.signal.resample_poly` to convert before passing to Chimera.
