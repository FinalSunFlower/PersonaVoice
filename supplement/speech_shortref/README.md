# Short-reference speech supplement

This is a standalone analysis layer for *PersonaVoice: One-Second Voice Cloning with Length-Adaptive Flow Matching*. It contains only public aggregate snapshots and scripts for recomputing tables, the SECS/WER trade-off, length stratification, ablation summaries, and the catastrophic-failure ledger. It does not import the historical implementation and does not distribute checkpoints, model weights, or audio.

```bash
python aggregate_metrics.py
python tradeoff_analysis.py
python failure_analysis.py
python fine_length_analysis.py
python experiment_manifest.py
python make_figures.py
```

The per-item `eval_200_samples.json` snapshot contains only public derived
metrics (sample ID, word count, SECS, WER, and status). `fine_length_analysis.py`
computes the preregistered 1--3, 4--8, 9--15, and >15 word strata. The external
baseline artifact is protocol-matched but not a paired rerun of every final
checkpoint. SECS is an objective speaker proxy, WER is an automatic text
proxy, and CFR is WER > 0.5. `extension_manifest.json` records the frozen
reference-duration and LAAG/OES conditions that remain pending execution; no
pending condition is presented as a measured result. No perceptual claim about
recovered personality or emotion is made.
