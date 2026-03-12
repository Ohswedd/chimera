"""
Example 06 -- Parameter inspection
===================================
Shows how to inspect and compare the transformation parameters that
Chimera derives from a key without processing any audio.

Useful for:
  - Logging and auditing which transformation was applied to a recording.
  - Verifying that two recordings used the same (or different) key.
  - Understanding parameter sensitivity to key changes.

Usage:
    python examples/06_inspect_params.py
"""

import chimera
from chimera import list_presets


def main() -> None:
    print("=" * 60)
    print("Chimera -- Parameter Inspector")
    print("=" * 60)

    # -- 1. Show all presets --
    print("\nAvailable presets:\n")
    for p in list_presets():
        bar = "#" * int(p["intensity"] * 20)
        print(f"  {p['name']:10s}  [{bar:<20s}]  {p['intensity']:.2f}")
        print(f"             {p['description']}")
        print()

    # -- 2. Inspect a specific key at different presets --
    key = "project-documentation"
    print(f"Parameters for key: {key!r}\n")
    for preset in ("subtle", "moderate", "strong"):
        p = chimera.get_params(key, preset=preset)
        print(f"  preset={preset!r}")
        print(f"    pitch shift  = {p.pitch_shift_semitones:+.3f} st")
        print(f"    formant warp = {p.formant_warp:.5f}x")
        print(f"    spectral tilt= {p.spectral_tilt:+.3f} dB/kHz")
        print(f"    breathiness  = {p.breathiness:.4f}")
        print()

    # -- 3. Full summary --
    p = chimera.get_params("my-secret", preset="strong")
    print(p.summary())

    # -- 4. Per-speaker inspection --
    print("\nPer-speaker parameters (same master key, ALL_UNIQUE):\n")
    params = chimera.get_speaker_params(
        "my-secret",
        ["SPEAKER_0", "SPEAKER_1", "SPEAKER_2"],
        preset="strong",
    )
    for sid, p in params.items():
        print(
            f"  {sid}  pitch={p.pitch_shift_semitones:+.2f}st  "
            f"warp={p.formant_warp:.3f}x  "
            f"breathiness={p.breathiness:.3f}"
        )


if __name__ == "__main__":
    main()
