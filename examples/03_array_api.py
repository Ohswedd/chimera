"""
Example 03 — Array API and batch processing
============================================
Processes multiple audio files from a directory using the array API,
then writes the results in parallel using a ThreadPoolExecutor.

Usage:
    python examples/03_array_api.py ./recordings/ ./anonymised/
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf

import chimera

KEY = "batch-processing-key-2026"
PRESET = "strong"


def process_one(src: Path, dst: Path) -> tuple[str, float]:
    audio, sr = sf.read(str(src), dtype="float64", always_2d=False)
    result = chimera.mask_array(audio, sr, key=KEY, preset=PRESET)
    sf.write(str(dst), result.audio, sr, subtype="PCM_16")
    return src.name, result.processing_time_s


def batch(input_dir: str, output_dir: str, max_workers: int = 4) -> None:
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files = list(in_path.glob("*.wav")) + list(in_path.glob("*.flac"))
    if not files:
        print(f"No .wav/.flac files found in {input_dir!r}")
        return

    print(f"Processing {len(files)} file(s) with {max_workers} workers …")
    total_time = 0.0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(process_one, f, out_path / f.name): f for f in files}
        for fut in as_completed(futures):
            try:
                name, elapsed = fut.result()
                total_time += elapsed
                print(f"  ✓ {name}  ({elapsed:.2f}s)")
            except Exception as exc:
                print(f"  ✗ {futures[fut].name}: {exc}")

    print(f"\nDone. Total processing time: {total_time:.2f}s")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python 03_array_api.py <input_dir> <output_dir>")
        sys.exit(1)
    batch(sys.argv[1], sys.argv[2])
