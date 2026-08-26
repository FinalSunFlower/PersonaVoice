"""Recompute headline tables from the public aggregate snapshots."""
from __future__ import annotations

import argparse
from pathlib import Path

from jsonio import dump_json, read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--output", default="artifacts/aggregate_metrics.json")
    args = parser.parse_args()
    root = Path(args.artifacts)
    main_stats = read_json(root / "eval_200_statistics.json")
    ablation = read_json(root / "ablation_200_statistics.json")
    external = read_json(root / "baseline_external_200.json")
    honest = read_json(root / "honest_metrics_v10.4.7.json")
    length = {
        "short_text": {"n_samples": honest["short_text"]["n_samples"], "secs_mean": honest["short_text"]["secs_mean"], "wer_mean": honest["short_text"]["wer_mean"], "cfr": honest["short_text"]["cfr"]},
        "long_text": {"n_samples": honest["long_text"]["n_samples"], "secs_mean": honest["long_text"]["secs_mean"], "wer_mean": honest["long_text"]["wer_mean"], "cfr": honest["long_text"]["cfr"]},
        "ultra_short_text": {"n_samples": honest["ultra_short_text"]["n_samples"], "secs_mean": honest["ultra_short_text"]["secs_mean"], "wer_mean": honest["ultra_short_text"]["wer_mean"], "cfr": honest["ultra_short_text"]["cfr"]},
    }
    payload = {
        "schema_version": "personavoice-speech-shortref-aggregate-v1",
        "protocol": {"n": 200, "seed": 42, "reference_seconds": 1.0, "speaker_metric": "ECAPA cosine SECS", "text_metric": "Whisper WER", "cfr": "WER > 0.5"},
        "personavoice": main_stats["personavoice"],
        "length": length,
        "baselines": {k: {m: v for m, v in value.items() if m in {"secs_mean", "secs_std", "wer_mean", "wer_std", "n_samples"}} for k, value in external.get("results", {}).items() if value.get("available", False)},
        "ablations": {k: {m: v for m, v in value.items() if m in {"secs_mean", "secs_std", "wer_mean", "wer_std", "n_samples"}} for k, value in ablation.items()},
        "source_artifacts": ["eval_200_statistics.json", "ablation_200_statistics.json", "baseline_external_200.json", "cfr_analysis.json", "honest_metrics_v10.4.7.json"],
    }
    dump_json(payload, args.output)


if __name__ == "__main__":
    main()
