# Reproduction and release contract

This branch is the standalone supplement snapshot for the two Zenodo preprints.
The files in this branch correspond to **language preprint version
`paper-language-v1.0`** and **speech preprint version `paper-speech-v1.0`**.
It intentionally excludes the historical PersonaVoice runtime, model weights,
private/raw audio, API keys, and deployment code.

## Quick verification

From the repository root, Python 3.10+ is sufficient for the offline checks:

```bash
python reproduce.py
```

This verifies every release file against [`reproducibility/SHA256SUMS.txt`](reproducibility/SHA256SUMS.txt) and runs the four CPU-only tests. To regenerate the derived JSON summaries or publication figures:

```bash
python reproduce.py --derive
python reproduce.py --figures
```

The figure command emits both PDF and editable SVG for the planner mechanism;
all checked-in paper figures are already rendered release artifacts.

## Frozen item-level predictions

- Language: three seed snapshots in [`supplement/language_provenance/artifacts`](supplement/language_provenance/artifacts), with immutable episode IDs and predictions.
- Speech: the 200-item per-item metric snapshot [`eval_200_samples.json`](supplement/speech_shortref/artifacts/eval_200_samples.json), using seed 42 and one-second references.

The package-level `SHA256SUMS.txt` files and the top-level manifest provide two
independent integrity checks. Configuration, dataset fingerprints, evaluator
definitions, and random seeds are recorded in [`reproducibility/config.json`](reproducibility/config.json).

## Paper links

Code, frozen predictions, and SHA-256 manifests are available at:
<https://github.com/FinalSunFlower/PersonaVoice/tree/paper-supplement-v1.0>.
The exact artifact snapshot corresponding to the language preprint is tagged
`paper-language-v1.0`; the speech snapshot is tagged `paper-speech-v1.0`.

Large model weights remain upstream under their original licenses. The frozen
JSON snapshots are sufficient to audit the numerical claims without distributing
a voice-cloning service.
