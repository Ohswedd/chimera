"""
Example 02 — Multi-speaker masking
====================================
Demonstrates three MaskMode options for recordings with multiple speakers.

  ALL_UNIQUE  (default) — each speaker receives an independent transformation.
  ALL_SAME              — all speakers receive the same transformation.
  SELECTED              — only speaker 0 is masked; others pass through.

Usage:
    python examples/02_multi_speaker.py input.wav
"""

import sys

import soundfile as sf

import chimera
from chimera import MaskMode


def demo(audio, sr, out_prefix: str) -> None:
    # ── Mode 1: ALL_UNIQUE (recommended) ────────────────────────────────────
    print("Mode: ALL_UNIQUE")
    result = chimera.mask_array(
        audio,
        sr,
        key="shared-project-key",
        mode=MaskMode.ALL_UNIQUE,
        preset="strong",
    )
    sf.write(f"{out_prefix}_unique.wav", result.audio, sr)
    print(f"  Masked : {result.speakers_masked}")
    print(f"  Saved  : {out_prefix}_unique.wav\n")

    # ── Mode 2: ALL_SAME ─────────────────────────────────────────────────────
    print("Mode: ALL_SAME")
    result = chimera.mask_array(
        audio,
        sr,
        key="shared-project-key",
        mode=MaskMode.ALL_SAME,
        preset="moderate",
    )
    sf.write(f"{out_prefix}_same.wav", result.audio, sr)
    print(f"  Masked : {result.speakers_masked}")
    print(f"  Saved  : {out_prefix}_same.wav\n")

    # ── Mode 3: SELECTED ─────────────────────────────────────────────────────
    print("Mode: SELECTED (only SPEAKER_0)")
    result = chimera.mask_array(
        audio,
        sr,
        key="shared-project-key",
        mode=MaskMode.SELECTED,
        speaker_ids=["SPEAKER_0"],
        preset="strong",
    )
    sf.write(f"{out_prefix}_selected.wav", result.audio, sr)
    print(f"  Masked   : {result.speakers_masked}")
    print(f"  Skipped  : {result.speakers_skipped}")
    print(f"  Saved    : {out_prefix}_selected.wav\n")

    # ── Segment details ──────────────────────────────────────────────────────
    print("Segment breakdown:")
    for seg in result.segments:
        tag = "MASKED" if seg.speaker_id in result.speakers_masked else "passed"
        print(
            f"  [{tag:7s}] {seg.speaker_id:12s}  "
            f"{seg.start_sec:5.2f}s → {seg.end_sec:5.2f}s  "
            f"({'voiced' if seg.is_voiced else 'silence'})"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 02_multi_speaker.py <input.wav>")
        sys.exit(1)
    audio, sr = sf.read(sys.argv[1], dtype="float64", always_2d=False)
    demo(audio, sr, out_prefix="multi_speaker_out")
