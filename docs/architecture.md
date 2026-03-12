# Architecture

## Pipeline overview

```
Input audio (mono or stereo, any supported sample rate)
    |
    v -- Stage 1: VAD --
    |   Module: chimera.vad.VoiceActivityDetector
    |   Features: Short-term energy . Zero-crossing rate . Periodicity
    |   Output:  List of (start_s, end_s, is_voiced) intervals
    |
    v -- Stage 2: Speaker Diarization --
    |   Module: chimera.diarize.SpeakerDiarizer
    |   Method: 39-dim MFCC (13 + delta + delta-delta) -> StandardScaler -> k-means
    |   k selected by silhouette score when n_speakers is unknown
    |   Output:  List of SpeakerSegment (speaker_id, start, end, is_voiced)
    |
    v -- Stage 3: Key Derivation --
    |   Module: chimera.keygen
    |   Method: HKDF-SHA256 (Extract + Expand)
    |   PRK = HMAC-SHA256(salt, key)
    |   param_i = map(HMAC-SHA256(PRK, label_i)[0:4], range_i)
    |   Output:  Dict[speaker_id -> MaskParams]
    |
    v -- Stage 4: Voice Transform --
    |   Module: chimera.transform
    |
    |   Praat LPC "Change gender" command:
    |       1. LPC analysis extracts vocal tract filter (formant positions)
    |       2. Inverse filtering isolates glottal source excitation
    |       3. Frequency-scaling of LPC coefficients shifts formants
    |          (equivalent to changing vocal tract length)
    |       4. PSOLA resynthesis with new pitch and modified LPC filter
    |
    |   Output: Transformed audio chunks per segment
    |
    v -- Stage 5: COWL (Cryptographic One-Way Layer) --
    |   Module: chimera.irreversible
    |   Applied in STFT domain (adaptive window, 75% overlap)
    |
    |   COWL-1  SSNI   mag += 0.3% signal-relative noise, gated on silence
    |   COWL-2  MPP    phase += N(0, 0.01*pi * intensity) per bin
    |
    |   Output: COWL-hardened audio chunks
    |
    v -- Stage 6: Stitch --
        Module: chimera.pipeline
        Method: 5 ms cross-fade at segment boundaries
        Output: Final mono float64 array, level-matched to input
```

---

## Module responsibilities

| Module | Single responsibility |
|---|---|
| `exceptions` | Typed exception hierarchy; no logic |
| `types` | Frozen dataclasses and enums; no side-effects |
| `keygen` | Pure function: key -> params |
| `vad` | Pure function: audio -> intervals |
| `diarize` | Pure function: audio + intervals -> segments |
| `transform` | Pure function: audio + params -> audio (Praat LPC) |
| `irreversible` | Pure function: audio + params -> audio |
| `pipeline` | Orchestrates stages 1-6; manages threading model |
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
    pitch_shift_semitones: float   # [-8, +8], min |4| enforced
    formant_warp:          float   # [0.80, 1.20], min 12% enforced
    spectral_tilt:         float   # [-1.5, +1.5] dB/kHz
    breathiness:           float   # [0.01, 0.10]
    intensity:             float   # [0, 1]
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
    |   raw float32 chunks
    v
Input thread (daemon)
    |   accumulates samples -> fixed-size chunks
    |   puts to queue.Queue(maxsize=32)
    v
Processing thread (daemon)
    |   dequeues chunks
    |   -> apply_all_layers(chunk, sr, params)
    |   -> apply_cowl(chunk, sr, params)
    |   appends to output list / calls on_chunk callback
    v
Output
    save()  ->  soundfile.write
    get_audio()  ->  np.concatenate(output_chunks)
```

Back-pressure: if the processor falls behind, the input thread drops incoming chunks
rather than growing the queue unboundedly -- a hard real-time requirement.

---

## Supported sample rates

`8 000` . `16 000` . `22 050` . `24 000` . `44 100` . `48 000` Hz

All other rates raise `UnsupportedSampleRateError`. Use `librosa.resample` or
`scipy.signal.resample_poly` to convert before passing to Chimera.
