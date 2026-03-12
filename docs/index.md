# Chimera

**Cryptographically irreversible, speaker-aware voice anonymisation.**

Chimera disguises the identity of one or more speakers in any audio recording.
It operates entirely on CPU — no cloud, no GPU, no pre-trained model weights.

## Core guarantees

- **One-way** — given only the output audio and the key, no known algorithm can recover the original speaker's identity.
- **Deterministic** — the same key always produces the same output. The transformation is fully auditable.
- **Natural** — WORLD high-quality vocoder synthesis preserves glottal pulse structure. The output sounds like a real human, not a voice-changer.
- **Speaker-aware** — built-in VAD and diarization automatically identify and independently mask each speaker.

## Navigation

| Page | What you'll find |
|---|---|
| [Quick Start](quickstart.md) | Install, first file mask, multi-speaker, real-time mic |
| [Architecture](architecture.md) | Pipeline stages, module responsibilities, data flow |
| [API Reference](api_reference.md) | Every public class, function, and parameter |
| [Security](security.md) | Threat model, COWL security argument, recommendations |
| [Changelog](changelog.md) | Version history |
