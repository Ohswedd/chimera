"""
benchmarks/benchmark_pipeline.py
=================================
Wall-clock performance benchmark for the Chimera pipeline.

Measures processing time per second of audio across:
  - Different audio durations (5, 30, 120 seconds)
  - Different presets (moderate, strong, extreme)
  - With and without COWL

Run:
    python benchmarks/benchmark_pipeline.py

Requirements: standard chimera dependencies only (no extra packages).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

import chimera

# -- Synthetic audio generator --

SR = 22_050


def synthetic_speech(duration_s: float, sr: int = SR) -> np.ndarray:
    """Multi-frequency tone that exercises the Praat LPC voice transform."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    audio = (
        0.45 * np.sin(2 * np.pi * 180 * t)
        + 0.22 * np.sin(2 * np.pi * 360 * t)
        + 0.11 * np.sin(2 * np.pi * 540 * t)
        + 0.06 * np.sin(2 * np.pi * 720 * t)
    )
    fade = int(sr * 0.03)
    audio[:fade] *= np.linspace(0, 1, fade)
    audio[-fade:] *= np.linspace(1, 0, fade)
    return (audio * 0.85).astype(np.float64)


# -- Benchmark runner --


@dataclass
class BenchResult:
    duration_s: float
    preset: str
    apply_cowl: bool
    elapsed_s: float
    rtf: float = field(init=False)  # real-time factor (< 1 = faster than RT)

    def __post_init__(self) -> None:
        self.rtf = self.elapsed_s / self.duration_s


def run_benchmark(
    durations: list[float] = (5, 30, 120),
    presets: list[str] = ("moderate", "strong", "extreme"),
    n_runs: int = 3,
) -> list[BenchResult]:
    results: list[BenchResult] = []
    key = "benchmark-key-2026"

    for dur in durations:
        audio = synthetic_speech(dur)
        for preset in presets:
            for apply_cowl in (True, False):
                times = []
                for _ in range(n_runs):
                    t0 = time.perf_counter()
                    chimera.mask_array(audio, SR, key=key, preset=preset, apply_cowl=apply_cowl)
                    times.append(time.perf_counter() - t0)
                elapsed = min(times)  # best of N
                results.append(BenchResult(dur, preset, apply_cowl, elapsed))

    return results


def print_table(results: list[BenchResult]) -> None:
    print()
    print("=" * 76)
    print(" Chimera Pipeline Benchmark (Praat LPC Voice Transform)")
    print(f" Platform: CPU-only  |  SR: {SR} Hz")
    print("=" * 76)
    header = f"{'Duration':>10}  {'Preset':>10}  {'COWL':>6}  {'Elapsed':>10}  {'RTF':>8}  {'Assessment'}"
    print(header)
    print("-" * 76)
    for r in results:
        assessment = "real-time" if r.rtf < 1.0 else ("~ near-RT" if r.rtf < 2.0 else "slow")
        print(
            f"{r.duration_s:>9.0f}s  "
            f"{r.preset:>10}  "
            f"{'yes' if r.apply_cowl else 'no':>6}  "
            f"{r.elapsed_s:>9.2f}s  "
            f"{r.rtf:>8.3f}x  "
            f"{assessment}"
        )
    print("-" * 76)
    print("RTF = processing time / audio duration.  RTF < 1.0 = faster than real-time.")
    print()


if __name__ == "__main__":
    print("Running Chimera benchmarks ...")
    results = run_benchmark()
    print_table(results)
