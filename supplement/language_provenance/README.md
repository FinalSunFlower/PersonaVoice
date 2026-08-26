# Language provenance supplement

This package is a standalone, CPU-only replay layer for *Provenance-Constrained Selective Language Style Planning from Sparse Evidence*. It deliberately contains no import from the historical `personavoice` implementation. The `artifacts/` directory contains ID-only public test evidence, three frozen prediction snapshots, the measured error audit source, and a manifest.

Run from this directory:

```bash
python audit_failures.py
python failure_by_budget.py
python cross_dataset_audit.py
python budget_curve.py
python make_figures.py
python make_architecture_figure.py
```

The budget curve uses the recorded constrained decisions and a provenance-free score (`support_probability * style_probability`). The requested 1/2/3/5/10 grid is recorded in the artifact; only 5 and 10 are genuine cells in the frozen predictions. The 1/2/3 cells are intentionally marked unavailable, because truncating a full-budget row would not be a new planner inference. If validation snapshots are available, pass `--validation ../path/snli_semantic_mlp_seed{seed}.validation.json` to fit the no-provenance threshold on validation only. No threshold is tuned on test.

The benchmark is a programmatic SNLI--PersonaChat proxy. It is not consent evidence, natural memorial speech, or a deployment safety guarantee. The taxonomy intentionally reports that conflict failures are absent in this construction rather than manufacturing a conflict result.

`cross_dataset_audit.py` downloads the official DailyDialog test archive from
Hugging Face, verifies the archive through the emitted SHA256, and computes a
deterministic lexical-evidence availability audit at budgets 1/2/3/5. Its
labels are token-overlap proxies, not entailment annotations, and its numbers
must not be pooled with the planner endpoint.

`make_architecture_figure.py` emits the mechanism schematic as both PDF and
editable SVG. It is drawn on a wide paper-scale canvas with a fixed minimum
label size, so the figure remains legible when embedded as a two-column figure.
