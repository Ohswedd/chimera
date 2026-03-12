"""
Example 04 — Real-time microphone anonymisation
================================================
Records from the default system microphone, anonymises in real-time,
and saves the result to a WAV file.

Requirements:
    pip install "chimera-voice[realtime]"

Usage:
    python examples/04_realtime_mic.py [output.wav] [preset]
"""

import sys


def main(output_path: str = "recorded_anonymous.wav", preset: str = "moderate") -> None:
    try:
        from chimera.realtime import RealtimeAnonymiser
    except ImportError:
        print("sounddevice is required for real-time mode.")
        print("Install with:  pip install 'chimera-voice[realtime]'")
        sys.exit(1)

    print("Chimera Real-Time Anonymiser")
    print(f"  Preset  : {preset}")
    print(f"  Output  : {output_path!r}")
    print()

    anon = RealtimeAnonymiser(
        key="realtime-session-key",
        preset=preset,
        sample_rate=22_050,
        chunk_duration_ms=200,
        apply_cowl_layer=True,
    )

    anon.start()
    print("🎙  Recording… Press Enter to stop.")
    try:
        input()
    except KeyboardInterrupt:
        pass
    finally:
        anon.stop()

    audio = anon.get_audio()
    if len(audio) == 0:
        print("No audio recorded.")
        return

    anon.save(output_path)
    duration = len(audio) / 22_050
    print(f"✓ Saved {duration:.1f}s of anonymised audio to {output_path!r}")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "recorded_anonymous.wav"
    preset_ = sys.argv[2] if len(sys.argv) > 2 else "moderate"
    main(output, preset_)
