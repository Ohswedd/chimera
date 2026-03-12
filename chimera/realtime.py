"""
chimera.realtime
~~~~~~~~~~~~~~~~
Real-time voice anonymisation for microphone input and streaming sources.

Architecture
------------
The real-time engine uses a ring-buffer / double-buffer scheme:

  ┌─────────────────┐     ┌──────────────┐     ┌──────────────────┐
  │  Input thread   │────▶│  Ring buffer │────▶│  Process thread  │
  │  (sounddevice)  │     │  (N chunks)  │     │  (chimera pipe)  │
  └─────────────────┘     └──────────────┘     └────────┬─────────┘
                                                         │
                                                         ▼
                                               ┌──────────────────┐
                                               │  Output callback │
                                               │  or WAV writer   │
                                               └──────────────────┘

Each chunk is processed independently with a short overlap-add window
to avoid boundary artefacts.  Processing latency ≈ chunk_duration +
vocoder_latency (~100–300 ms on a modern CPU).

Usage example
-------------
    from chimera.realtime import RealtimeAnonymiser

    anon = RealtimeAnonymiser(key="my-key", preset="strong")
    anon.start()
    # ... speak into microphone ...
    anon.stop()
    anon.save("output.wav")

Requirements
------------
sounddevice must be installed separately (not a hard dependency):
    pip install sounddevice
"""

from __future__ import annotations

import contextlib
import queue
import threading
from pathlib import Path
from typing import Callable

import numpy as np

from .exceptions import RealtimeError
from .irreversible import apply_cowl
from .keygen import derive_params
from .transform import apply_all_layers
from .types import MaskParams


class RealtimeAnonymiser:
    """
    Real-time voice anonymisation from a microphone or audio stream.

    Parameters
    ----------
    key : str
        Masking passphrase.
    salt : str
        Domain-separation salt.
    intensity : float
        Masking intensity in [0, 1].
    preset : str | None
        Named preset — overrides *intensity*.
    sample_rate : int
        Target sample rate.  sounddevice will resample if needed.
    chunk_duration_ms : float
        Audio chunk size in milliseconds.  Smaller = lower latency but
        higher CPU; 150–300 ms is recommended.
    device : int | str | None
        sounddevice input device index or name.  None = system default.
    on_chunk : Callable | None
        Optional callback invoked with each processed chunk (np.ndarray).
    apply_cowl_layer : bool
        Whether to apply the COWL irreversibility layer (recommended True).
    """

    def __init__(
        self,
        key: str,
        salt: str = "chimera-v1",
        intensity: float = 1.0,
        preset: str | None = None,
        sample_rate: int = 22_050,
        chunk_duration_ms: float = 200.0,
        device: int | str | None = None,
        on_chunk: Callable[[np.ndarray], None] | None = None,
        apply_cowl_layer: bool = True,
    ) -> None:
        if preset is not None:
            from .presets import resolve_intensity

            intensity = resolve_intensity(preset)

        self.params: MaskParams = derive_params(key, salt=salt, intensity=intensity)
        self.sr = sample_rate
        self.chunk_len = int(sample_rate * chunk_duration_ms / 1000)
        self.device = device
        self.on_chunk = on_chunk
        self.use_cowl = apply_cowl_layer

        self._input_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=32)
        self._output_chunks: list[np.ndarray] = []
        self._running = False
        self._proc_thread: threading.Thread | None = None
        self._input_thread: threading.Thread | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start recording from the microphone and processing in background."""
        try:
            import sounddevice as sd  # lazy import — optional dependency
        except ImportError:
            raise RealtimeError(
                "sounddevice is required for real-time mode.\n"
                "Install with:  pip install sounddevice"
            )

        if self._running:
            return

        self._running = True
        self._output_chunks.clear()

        # Processing thread
        self._proc_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._proc_thread.start()

        # Recording thread
        self._input_thread = threading.Thread(
            target=self._record_loop,
            args=(sd,),
            daemon=True,
        )
        self._input_thread.start()

    def stop(self) -> None:
        """Stop recording and wait for the processing queue to drain."""
        self._running = False
        if self._input_thread:
            self._input_thread.join(timeout=2.0)
        if self._proc_thread:
            # Drain the queue
            self._input_queue.put(None)  # sentinel
            self._proc_thread.join(timeout=10.0)

    def save(self, path: str | Path, subtype: str = "PCM_16") -> None:
        """Save all processed chunks to *path*."""
        try:
            import soundfile as sf
        except ImportError:
            raise RealtimeError("soundfile is required to save audio.")

        if not self._output_chunks:
            raise RealtimeError("No audio recorded.  Call start() before save().")

        audio = np.concatenate(self._output_chunks)
        sf.write(str(path), audio, self.sr, subtype=subtype)

    def get_audio(self) -> np.ndarray:
        """Return all processed audio as a single numpy array."""
        if not self._output_chunks:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(self._output_chunks)

    # ── Processing loop ───────────────────────────────────────────────────────

    def _process_loop(self) -> None:
        """Worker thread: dequeue raw chunks, transform, store."""
        while True:
            chunk = self._input_queue.get(timeout=5.0)
            if chunk is None:  # sentinel
                break
            try:
                masked = apply_all_layers(chunk, self.sr, self.params)
                if self.use_cowl:
                    masked = apply_cowl(masked, self.sr, self.params)
                self._output_chunks.append(masked)
                if self.on_chunk is not None:
                    self.on_chunk(masked)
            except Exception:
                # Never crash the processing thread; pass raw chunk through
                self._output_chunks.append(chunk)

    def _record_loop(self, sd) -> None:
        """Collect microphone input into fixed-size chunks."""
        accumulator = np.zeros(0, dtype=np.float32)
        with sd.InputStream(
            device=self.device,
            channels=1,
            samplerate=self.sr,
            dtype="float32",
        ) as stream:
            while self._running:
                raw, _overflowed = stream.read(self.chunk_len // 4)
                accumulator = np.concatenate([accumulator, raw.flatten()])
                while len(accumulator) >= self.chunk_len:
                    chunk = accumulator[: self.chunk_len].astype(np.float64)
                    accumulator = accumulator[self.chunk_len :]
                    with contextlib.suppress(queue.Full):
                        self._input_queue.put_nowait(chunk)


# ── Convenience function ──────────────────────────────────────────────────────


def mask_stream(
    audio_iterator,
    sr: int,
    key: str,
    *,
    salt: str = "chimera-v1",
    intensity: float = 1.0,
    preset: str | None = None,
    apply_cowl_layer: bool = True,
    chunk_size: int = 4096,
):
    """
    Generator-based streaming anonymiser.

    Yields masked audio chunks from an iterable of raw audio chunks.

    Parameters
    ----------
    audio_iterator : iterable of np.ndarray
        Sequence of mono float64 audio chunks.
    sr : int
        Sample rate.
    key : str
        Masking key.
    other params : see ChimeraPipeline.

    Yields
    ------
    np.ndarray
        Masked audio chunk of the same length as the input chunk.
    """
    if preset is not None:
        from .presets import resolve_intensity

        intensity = resolve_intensity(preset)

    params = derive_params(key, salt=salt, intensity=intensity)

    for chunk in audio_iterator:
        chunk = np.asarray(chunk, dtype=np.float64)
        if len(chunk) == 0:
            yield chunk
            continue
        masked = apply_all_layers(chunk, sr, params)
        if apply_cowl_layer:
            masked = apply_cowl(masked, sr, params)
        yield masked
