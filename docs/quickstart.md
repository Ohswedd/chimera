# Quick Start

## Installation

```bash
pip install chimera-voice
```

For real-time microphone support:

```bash
pip install "chimera-voice[realtime]"
```

---

## 1. Anonymise a single file

```python
import chimera

chimera.mask_file("interview.wav", "anonymous.wav", key="my-secret", preset="strong")
```

That's it. The output file sounds like a different speaker, the transformation is deterministic, and the original identity cannot be recovered from the output.

---

## 2. Choose a preset

```python
# Barely perceptible — watermarking
chimera.mask_file("input.wav", "out.wav", key="k", preset="whisper")

# Light disguise
chimera.mask_file("input.wav", "out.wav", key="k", preset="subtle")

# Clear change — unrecognisable to human listeners
chimera.mask_file("input.wav", "out.wav", key="k", preset="moderate")

# Robust — ASV systems fail  (recommended for most use cases)
chimera.mask_file("input.wav", "out.wav", key="k", preset="strong")

# Maximum — only speech content preserved
chimera.mask_file("input.wav", "out.wav", key="k", preset="extreme")
```

---

## 3. Array API

```python
import chimera
import soundfile as sf

audio, sr = sf.read("interview.wav")

result = chimera.mask_array(audio, sr, key="my-secret", preset="strong")

# result.audio        — masked mono float64 array
# result.speakers_masked  — list of speaker IDs that were processed
# result.processing_time_s — wall-clock seconds

sf.write("anonymous.wav", result.audio, sr)
print(f"Masked {result.num_speakers} speaker(s) in {result.processing_time_s:.2f}s")
```

---

## 4. Multi-speaker: independent key per speaker

```python
result = chimera.mask_array(
    audio, sr,
    key        = "my-secret",
    preset     = "strong",
    mode       = chimera.MaskMode.ALL_UNIQUE,   # default
    n_speakers = 3,                              # auto-detected if omitted
)

for seg in result.segments:
    if seg.is_voiced:
        print(f"{seg.speaker_id}  {seg.start_sec:.1f}s → {seg.end_sec:.1f}s")
```

---

## 5. Mask only selected speakers

```python
result = chimera.mask_array(
    audio, sr,
    key          = "my-secret",
    preset       = "strong",
    mode         = chimera.MaskMode.SELECTED,
    speaker_ids  = ["SPEAKER_0"],
)
# SPEAKER_1, SPEAKER_2 are passed through untouched
```

---

## 6. Same transformation for all speakers

```python
result = chimera.mask_array(
    audio, sr,
    key    = "my-secret",
    preset = "moderate",
    mode   = chimera.MaskMode.ALL_SAME,
)
```

---

## 7. Real-time microphone

```python
from chimera.realtime import RealtimeAnonymiser

anon = RealtimeAnonymiser(key="my-secret", preset="moderate")
anon.start()
input("🎙  Recording — press Enter to stop...")
anon.stop()
anon.save("recorded_anonymous.wav")
```

---

## 8. Streaming (generator-based)

```python
from chimera.realtime import mask_stream

def my_chunk_generator():
    # yield np.ndarray chunks from any source
    ...

for masked_chunk in mask_stream(my_chunk_generator(), sr=22050,
                                key="my-secret", preset="strong"):
    send_to_output(masked_chunk)
```

---

## 9. Inspect derived parameters

```python
p = chimera.get_params("my-secret", preset="strong")
print(p.summary())
```

---

## 10. Per-speaker parameters

```python
params = chimera.get_speaker_params(
    "my-secret",
    speaker_ids=["SPEAKER_0", "SPEAKER_1"],
    preset="strong",
)
for sid, p in params.items():
    print(f"{sid}  pitch={p.pitch_shift_semitones:+.2f}st  warp={p.formant_warp:.3f}x")
```

---

## Next steps

- [Architecture](architecture.md) — understand the full pipeline
- [API Reference](api_reference.md) — all classes and parameters
- [Security](security.md) — threat model and COWL design
