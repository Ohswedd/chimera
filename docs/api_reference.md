# API Reference

## Top-level functions

All functions below are importable directly from `chimera`.

---

### `mask_file`

```python
chimera.mask_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    key: str,
    salt: str = "chimera-v1",
    intensity: float = 1.0,
    preset: str | None = None,
    mode: MaskMode = MaskMode.ALL_UNIQUE,
    n_speakers: int | None = None,
    speaker_ids: list[str] | None = None,
    apply_cowl: bool = True,
    subtype: str | None = None,
) -> ChimeraResult
```

Anonymise an audio file and write the result.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_path` | `str\|Path` | — | Source audio file |
| `output_path` | `str\|Path` | — | Destination audio file |
| `key` | `str` | — | Masking passphrase |
| `salt` | `str` | `"chimera-v1"` | Domain-separation salt |
| `intensity` | `float` | `1.0` | Masking strength in [0, 1] |
| `preset` | `str\|None` | `None` | Named preset; overrides `intensity` |
| `mode` | `MaskMode` | `ALL_UNIQUE` | Multi-speaker policy |
| `n_speakers` | `int\|None` | `None` | Expected speakers (None = auto) |
| `speaker_ids` | `list[str]\|None` | `None` | Used with `SELECTED` mode |
| `apply_cowl` | `bool` | `True` | Apply Cryptographic One-Way Layer |
| `subtype` | `str\|None` | `None` | Output bit-depth, e.g. `"PCM_24"` |

---

### `mask_array`

```python
chimera.mask_array(
    audio: np.ndarray,
    sr: int,
    *,
    key: str,
    salt: str = "chimera-v1",
    intensity: float = 1.0,
    preset: str | None = None,
    mode: MaskMode = MaskMode.ALL_UNIQUE,
    n_speakers: int | None = None,
    speaker_ids: list[str] | None = None,
    apply_cowl: bool = True,
) -> ChimeraResult
```

Anonymise a NumPy audio array. Accepts mono or stereo float32/float64.

---

### `get_params`

```python
chimera.get_params(
    key: str,
    *,
    salt: str = "chimera-v1",
    intensity: float = 1.0,
    preset: str | None = None,
    speaker_label: str | None = None,
) -> MaskParams
```

Derive and return the `MaskParams` that would be applied for a given key
without processing any audio. Useful for inspection and logging.

---

### `get_speaker_params`

```python
chimera.get_speaker_params(
    key: str,
    speaker_ids: list[str],
    *,
    salt: str = "chimera-v1",
    intensity: float = 1.0,
    preset: str | None = None,
) -> dict[str, MaskParams]
```

Return per-speaker `MaskParams` for all IDs in `speaker_ids`.

---

### `list_presets`

```python
chimera.list_presets() -> list[dict]
```

Return a list of preset descriptions:
```python
[{"name": "whisper", "intensity": 0.12, "description": "..."}, ...]
```

---

## Classes

### `ChimeraPipeline`

The main processing engine. All top-level functions are thin wrappers around it.

```python
from chimera import ChimeraPipeline, MaskMode

pipeline = ChimeraPipeline(
    key              = "my-secret",
    preset           = "strong",
    mode             = MaskMode.ALL_UNIQUE,
    n_speakers       = None,           # auto
    apply_cowl_layer = True,
    vad_energy_db    = -45.0,
    crossfade        = True,
)

result = pipeline.process(audio, sr)
```

---

### `MaskParams`

Frozen dataclass holding all eight transformation parameters plus metadata.

```python
@dataclass(frozen=True)
class MaskParams:
    pitch_shift_semitones: float   # [-10, +10]
    formant_warp:          float   # [0.78, 1.22]
    spectral_tilt:         float   # [-4, +4] dB/kHz
    breathiness:           float   # [0, 0.45]
    temporal_jitter:       float   # [0, 0.018] σ
    vibrato_rate:          float   # [0, 7] Hz
    vibrato_depth:         float   # [0, 0.40] st
    subharmonic_mix:       float   # [0, 0.15]
    intensity:             float   # [0, 1]
    seed:                  int
    speaker_label:         str | None

    def scaled(self, factor: float) -> MaskParams: ...
    def summary(self) -> str: ...
```

---

### `ChimeraResult`

```python
@dataclass
class ChimeraResult:
    audio:             np.ndarray       # masked mono float64
    sample_rate:       int
    segments:          list[SpeakerSegment]
    speakers_masked:   list[str]
    speakers_skipped:  list[str]
    processing_time_s: float

    @property
    def num_speakers(self) -> int: ...
    @property
    def duration_s(self) -> float: ...
```

---

### `SpeakerSegment`

```python
@dataclass
class SpeakerSegment:
    speaker_id: str          # "SPEAKER_0", "SPEAKER_1", … or "SILENCE"
    start_sec:  float
    end_sec:    float
    is_voiced:  bool
    params:     MaskParams | None   # None for SILENCE or skipped speakers

    @property
    def duration(self) -> float: ...
```

---

### `MaskMode`

```python
class MaskMode(Enum):
    ALL_UNIQUE  # Each speaker gets independent key-derived params (default)
    ALL_SAME    # All speakers share the same params
    SELECTED    # Only speakers in speaker_ids are masked
```

---

## Real-time API

### `RealtimeAnonymiser`

```python
from chimera.realtime import RealtimeAnonymiser

anon = RealtimeAnonymiser(
    key                = "my-secret",
    preset             = "moderate",
    sample_rate        = 22_050,
    chunk_duration_ms  = 200.0,
    device             = None,        # system default mic
    on_chunk           = None,        # optional callback(np.ndarray)
    apply_cowl_layer   = True,
)

anon.start()       # begins recording in background threads
anon.stop()        # stops recording; waits for queue to drain
anon.save("out.wav")
audio = anon.get_audio()   # np.ndarray of all processed chunks
```

### `mask_stream`

```python
from chimera.realtime import mask_stream

for masked in mask_stream(
    chunk_iterator,
    sr             = 22050,
    key            = "my-secret",
    preset         = "strong",
    apply_cowl_layer = True,
):
    # masked is np.ndarray[float64], same length as input chunk
    ...
```

---

## Exceptions

All exceptions are importable from `chimera`.

| Exception | When raised |
|---|---|
| `ChimeraError` | Base class for all Chimera exceptions |
| `KeyDerivationError` | Empty key or intensity out of [0, 1] |
| `AudioLoadError` | File cannot be loaded by libsndfile |
| `UnsupportedSampleRateError` | Sample rate not in supported set |
| `DiarizationError` | Insufficient voiced audio for diarization |
| `RealtimeError` | sounddevice not installed, or stream error |
| `PipelineError` | Pipeline misconfiguration |
| `PresetNotFoundError` | Unknown preset name |

---

## Preset reference

| Name | Intensity | Use case |
|---|---|---|
| `"whisper"` | 0.12 | Watermarking |
| `"subtle"` | 0.28 | Light disguise |
| `"moderate"` | 0.52 | Unrecognisable to humans |
| `"strong"` | 0.78 | ASV systems fail |
| `"extreme"` | 1.00 | Maximum masking |
