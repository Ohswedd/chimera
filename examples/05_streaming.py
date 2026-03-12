"""
Example 05 — Generator-based streaming
=======================================
Demonstrates the mask_stream() generator for processing audio
from any source (file, network socket, pipe) chunk by chunk.

This pattern is useful for:
  - Voice servers that receive audio in packets
  - WebRTC pipelines
  - Custom input devices

Usage:
    python examples/05_streaming.py input.wav output.wav
"""

import sys

import numpy as np
import soundfile as sf

from chimera.realtime import mask_stream

CHUNK_SAMPLES = 4096  # samples per chunk (~185 ms at 22 050 Hz)
KEY = "streaming-pipeline-key"
PRESET = "strong"


def file_chunk_generator(audio: np.ndarray):
    """Simulate a streaming source by yielding fixed-size chunks."""
    for start in range(0, len(audio), CHUNK_SAMPLES):
        yield audio[start : start + CHUNK_SAMPLES].astype(np.float64)


def main(input_path: str, output_path: str) -> None:
    audio, sr = sf.read(input_path, dtype="float64", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    print(f"Streaming {len(audio)/sr:.2f}s of audio in {CHUNK_SAMPLES}-sample chunks …")

    masked_chunks = []
    chunk_count = 0

    for masked in mask_stream(
        file_chunk_generator(audio),
        sr=sr,
        key=KEY,
        preset=PRESET,
        apply_cowl_layer=True,
    ):
        masked_chunks.append(masked)
        chunk_count += 1

    output = np.concatenate(masked_chunks)[: len(audio)]
    sf.write(output_path, output, sr, subtype="PCM_16")

    print(f"  Chunks processed : {chunk_count}")
    print(f"  Output saved     : {output_path!r}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python 05_streaming.py <input.wav> <output.wav>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
