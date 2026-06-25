# Contributing to PersonaVoice

Thanks for your interest in contributing to PersonaVoice! This project is an academic open-source effort, and we welcome contributions of all kinds: bug fixes, new experiments, documentation improvements, and architectural extensions.

## Code of Conduct

Be respectful, constructive, and academic. Cite prior work properly. Critique ideas, not people.

## How to Contribute

### 1. Fork & Branch
- Fork the repository.
- Create a feature branch: `git checkout -b feature/my-feature`.

### 2. Develop
- Follow the existing code style (PEP 8, line length ≤ 120).
- Add docstrings to all new functions / classes (Google style).
- Keep modules focused — PersonaVoice follows a strict architecture (LAAG / OES / FiLM / CEAG / Persona). Place new code in the correct subpackage:
  - `personavoice/tts_backbone/` — TTS backbone, adapters, sampler, vocoder
  - `personavoice/microaug/` — Sample enhancement (OES)
  - `personavoice/persona/` — Persona extraction
  - `personavoice/common/` — Shared utilities
  - `personavoice/demo/` — Web demo
  - `personavoice/experiment/` — Evaluation scripts
  - `scripts/` — One-off setup / data-prep scripts

### 3. Test
- Run smoke tests:
  ```bash
  python -c "import personavoice; print(personavoice.__version__)"
  python -c "from personavoice.config import SOTA_CONFIG; print(SOTA_CONFIG.version)"
  ```
- For experiments, run on the bundled `1111.mp3` first.

### 4. Submit a PR
- Fill in the PR template.
- If your change affects an architectural component (LAAG / OES / FiLM / CEAG), provide ablation evidence (200-sample paired t-test, p < 0.05, Cohen's d > 0.1).
- Update `README.md` / `README_zh.md` / `ARCHITECTURE.md` if behavior changes.

## Architectural Principles

1. **Plug-in Adapter Design** — F5-TTS backbone stays frozen. Train only lightweight adapters (~2M params).
2. **Ablation-Driven Decisions** — Every module must prove its worth via 200-sample paired t-test.
3. **Academic Honesty** — Report trade-offs honestly. Do not over-claim SOTA.
4. **Configuration Uniqueness** — All inference parameters live in `personavoice/config.py`. Do not hardcode parameters in scripts.
5. **1-Second Constraint** — All designs must work with as little as 1 second of reference audio.

## Reporting Issues

- **Bug**: use the Bug Report template.
- **Reproduction discrepancy**: use the Experiment Reproduction template.
- **Feature idea**: use the Feature Request template.

## Citation

If PersonaVoice helps your research, please cite it (see `CITATION.cff`).
