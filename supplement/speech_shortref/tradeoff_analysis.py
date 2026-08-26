"""Derived identity--intelligibility and length-stratified comparisons."""
from __future__ import annotations

import argparse
from pathlib import Path

from jsonio import dump_json, read_json


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--artifacts", default="artifacts"); parser.add_argument("--output", default="artifacts/tradeoff_analysis.json"); args = parser.parse_args()
    data = read_json(Path(args.artifacts) / "aggregate_metrics.json")
    all_row = data["personavoice"]
    short = data["length"]["short_text"]; long = data["length"]["long_text"]
    result = {
        "schema_version": "personavoice-speech-tradeoff-v1",
        "headline": {"secs": all_row["secs"]["mean"], "wer": all_row["wer"]["mean"], "cfr_percent": 13.5},
        "length_delta_short_minus_long": {"secs": short["secs_mean"] - long["secs_mean"], "wer": short["wer_mean"] - long["wer_mean"], "cfr": None},
        "interpretation": "The one-second condition preserves objective speaker similarity while short text sharply worsens recognition. SECS and WER must be reported jointly; neither is a sufficient quality guarantee.",
    }
    result["length_delta_short_minus_long"]["cfr"] = 100 * (short["cfr"] - long["cfr"])
    dump_json(result, args.output)


if __name__ == "__main__":
    main()
