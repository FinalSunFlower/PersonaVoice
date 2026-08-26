"""Emit the frozen extension manifest for conditions without released runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="artifacts/extension_manifest.json")
    args = p.parse_args()
    payload = {
        "schema_version": "personavoice-speech-shortref-extension-v1",
        "status": "protocol_registered_pending_execution",
        "public_data": "LibriTTS dev-clean official split; optional VCTK test-clean validation",
        "seed": 42,
        "reference_duration_seconds": [0.5, 1.0, 2.0, 3.0],
        "length_buckets_words": ["1-3", "4-8", "9-15", ">15"],
        "ablation_factors": {
            "LAAG": ["on", "off"],
            "OES": ["on", "off"],
            "FiLM": ["off (reported release default)", "on (control path only)"],
        },
        "metrics": ["SECS", "WER", "CFR (WER > 0.5)"],
        "controls": [
            "same 200-item ID list where supported",
            "same evaluator revisions and transcript normalization",
            "frozen checkpoint and decoder configuration per condition",
            "no test-set model selection",
        ],
        "claim_boundary": "No duration curve or LAAG/OES causal effect is reported until condition-level audio and metric snapshots are released.",
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
