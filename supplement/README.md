# Standalone supplements

The two directories are independent analysis packages. Each reads only the
public snapshots stored in its own `artifacts/` directory, writes derived JSON
and publication figures, and can be tested without the removed PersonaVoice
runtime:

```bash
python -m pytest -q supplement
```

`language_provenance` covers provenance failure auditing, observed evidence
budgets, and a fixed no-provenance replay. `speech_shortref` covers aggregate
metrics, the SECS/WER trade-off, length-stratified CFR, and ablation summaries.
