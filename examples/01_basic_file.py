"""
Example 01 — Basic file masking
================================
Anonymise a single audio file with a passphrase and preset.
No other configuration needed.

Usage:
    python examples/01_basic_file.py input.wav output.wav
"""

import sys

import chimera


def main(input_path: str, output_path: str) -> None:
    print(f"Masking: {input_path!r} → {output_path!r}")

    result = chimera.mask_file(
        input_path,
        output_path,
        key="my-secret-passphrase",
        preset="strong",
    )

    print(f"  Speakers detected : {result.num_speakers}")
    print(f"  Speakers masked   : {result.speakers_masked}")
    print(f"  Duration          : {result.duration_s:.2f}s")
    print(f"  Processed in      : {result.processing_time_s:.2f}s")
    print(f"  Saved to          : {output_path!r}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python 01_basic_file.py <input.wav> <output.wav>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
