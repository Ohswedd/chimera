# Security Model

## Threat model

Chimera is designed to resist four classes of adversary, all of whom have
full access to the output audio `C = F(S, K)`.

| ID | Adversary | Capability | Chimera defence |
|---|---|---|---|
| A1 | Human listener | Trained forensic phonetician | Praat LPC source-filter pitch + formant transform alters all perceptual vocal cues |
| A2 | ASV system | ECAPA-TDNN or x-vector embedder | Formant warp destroys embedding geometry; COWL removes residual cues |
| A3 | Parameter inverter | Knows all transformation parameters; attempts to undo them | COWL is applied after voice transform -- irreversible even with full parameter knowledge |
| A4 | Key guesser | Brute-force or dictionary attack on passphrase | HKDF-SHA256 provides 2^128 preimage resistance |

---

## Key derivation security

```
PRK = HMAC-SHA256(salt, key)

param_i = map(HMAC-SHA256(PRK, label_i || 0x01)[0:4], range_i)
```

Properties:
- **Preimage resistance:** recovering `key` from `PRK` requires inverting HMAC-SHA256.
- **Second-preimage resistance:** finding a different `key` that produces the same parameters requires colliding HMAC-SHA256 (2^128 security).
- **Domain separation:** each parameter uses a unique label, ensuring statistical independence across all parameters even for short keys.
- **Speaker isolation:** per-speaker keys use `key + ":chimera:spk:" + speaker_id`, ensuring that keys for different speakers are independent sub-keys.

---

## COWL security argument

Let `T(S, params)` be the voice transform output.
Let `C = COWL(T(S, params), params_cowl)` be the final output.

Recovering `S` from `C` requires both of the following simultaneously:

**Step 1 -- Undo Micro Phase Perturbation:** Recover the phase perturbation
applied to each STFT bin. The perturbation seed is derived from
`HMAC-SHA256(seed_bytes, "chimera:cowl:phase")`.
Without the key, the seed is unknown, reducing the problem to blind phase
retrieval -- NP-hard in general.

**Step 2 -- Remove SSNI noise:** The injected noise is signal-relative and
scaled by the key-derived noise PRK. Removing it requires knowing the noise
PRK, which is derived from the key. Without it, single-channel noise
separation is an ill-posed problem.

No polynomial-time algorithm accomplishing both steps simultaneously is
known. Chimera therefore provides **computational one-wayness** -- not
information-theoretic secrecy.

---

## Recommendations for high-stakes deployments

1. **Use intensity >= 0.78 (`strong`)** for any deployment where speaker
   re-identification would cause harm.

2. **Rotate keys regularly.** If the same key is reused across many recordings,
   a statistical attack could attempt to correlate COWL noise across outputs.

3. **Use passphrases with >= 80 bits of entropy** (Diceware or a password
   manager). A 12-character random alphanumeric key provides ~71 bits.

4. **Combine with transcript-level pseudonymisation** for recordings that
   contain spoken names, addresses, or other identifying content.

5. **Chimera is not an encryption scheme.** Do not use it as a substitute for
   AES encryption when confidentiality of the audio content (not just speaker
   identity) is required.

6. **Consider pyannote.audio** for diarization if speaker-attribution accuracy
   is critical (e.g., legal depositions). Chimera's built-in MFCC diarizer is
   reliable for 2-4 speakers in clean conditions; complex overlapping speech may
   need a more powerful model.

---

## Known limitations

| Limitation | Mitigation |
|---|---|
| Diarization accuracy drops with >6 overlapping speakers | Set `n_speakers` if known; use a neural diarizer shim |
| `extreme` preset may reduce intelligibility for non-native speakers | Use `strong` instead |
| COWL adds ~20-40 ms latency | Acceptable for batch and recording; disable with `apply_cowl=False` if latency is critical |
| No information-theoretic security | Supplement with key management and access control |
