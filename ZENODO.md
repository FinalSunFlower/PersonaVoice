# Zenodo release package

This repository is released as version `1.0.0` (25 August 2026) by Jiang
Luchang. The release contains two canonical research objects. The primary
language preprint:

`paper/personavoice_preprint.pdf`

It reports a provenance-constrained selective language planner and measured
results on a public, speaker-disjoint controlled composition. It also includes
a separately labelled DailyDialog lexical-evidence audit; that auxiliary audit
is not pooled with the planner endpoint and does not use entailment or
authorization labels. The paper does not claim natural memorial speech,
psychological reconstruction, safe deployment, or an end-to-end acoustic
result.

The second research object is the independent speech preprint:
`paper/personavoice_speech_preprint.pdf`. It reports measured one-second
voice-cloning results from the repository's v10.4.x release, including
SECS/WER trade-offs, external baselines, ablations, and catastrophic-failure
analysis. It does not claim validated perceptual persona or emotion control.

The protocol-only draft is not part of the canonical release.

## Upload set

Upload the two PDFs, `CITATION.cff`, `LICENSE`, the `paper/` source bundle, and
the two `supplement/` packages. Do not upload private speaker material,
raw consent records, model weights, API keys, or local environment files.

## Suggested Zenodo metadata

- **Titles:** Provenance-Constrained Selective Language Style Planning from Sparse Evidence; PersonaVoice: One-Second Voice Cloning with Length-Adaptive Flow Matching
- **Creator:** Jiang Luchang
- **ORCID:** 0009-0004-4579-4243
- **Resource type:** Publication / Preprint
- **Publication date:** 2026-08-25
- **License:** MIT for code; cite dataset licenses separately
- **Keywords:** provenance-constrained generation; selective prediction; citation attribution; language style; abstention; voice safety
- **Related identifiers:** project repository URL and the ORCID URL
- **Version:** 1.0.0

Zenodo assigns the DOI after deposition. Once assigned, add it to the final
record and the preferred citation; do not invent a DOI in source files.

## PDF checksums

- `paper/personavoice_preprint.pdf`: `84d4efecc910f10ec700b197ac593a52d7144426df38b8909668e903ae149ce5`
- `paper/personavoice_speech_preprint.pdf`: `6a46025035d9c5e09f1048da0d83b62de08a864590ee52af1f240c879949ee0a`
