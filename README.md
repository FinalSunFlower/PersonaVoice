# PersonaVoice: preprint supplement snapshot

This branch is the reproducibility companion for two independent Zenodo
preprints by Jiang Luchang (ORCID
[`0009-0004-4579-4243`](https://orcid.org/0009-0004-4579-4243)). It is deliberately
separate from the historical runtime repository: switching to
`paper-supplement-v1.0` exposes only public derived artifacts and reproduction
code. The paper PDFs and LaTeX sources are intentionally deposited separately
on Zenodo and are not duplicated in this GitHub support branch.

## Canonical papers

1. **Provenance-Constrained Selective Language Style Planning from Sparse Evidence** - typed evidence, citation attribution, conflict preservation, selective abstention, and a measured calibration-transfer failure. The matching support snapshot is tagged `paper-language-v1.0`.
2. **PersonaVoice: One-Second Voice Cloning with Length-Adaptive Flow Matching** - a 200-item one-second reference evaluation, protocol-matched baselines, ablations, and short-text catastrophic-failure analysis. The matching support snapshot is tagged `paper-speech-v1.0`.

## Reproducibility package

- [`supplement/language_provenance`](supplement/language_provenance) contains three frozen seed-level prediction files, ID-only evidence, failure audits, and deterministic figures.
- [`supplement/speech_shortref`](supplement/speech_shortref) contains the frozen 200-item metric rows, length-stratified analyses, trade-off/ablation ledgers, and a clearly marked pending-condition manifest.
- [`reproducibility/config.json`](reproducibility/config.json) records versions, seeds, metrics, data fingerprints, and the policy for excluded model weights.
- [`reproducibility/SHA256SUMS.txt`](reproducibility/SHA256SUMS.txt) is the release-wide SHA-256 manifest; each supplement also carries a package-level manifest.
- [`REPRODUCE.md`](REPRODUCE.md) and [`reproduce.py`](reproduce.py) provide the minimal CPU-only verification and derivation commands.

```bash
python reproduce.py
```

The package does not distribute checkpoints, private/raw audio, API keys, or a
voice-cloning service. Objective SECS/WER and the controlled language proxy are
measurement endpoints, not evidence of psychological recovery, authorization to
impersonate a person, or a deployment safety guarantee.

See [`ZENODO.md`](ZENODO.md) and [`CITATION.cff`](CITATION.cff) for deposition
metadata. The paper package is uploaded to Zenodo separately from this branch.
