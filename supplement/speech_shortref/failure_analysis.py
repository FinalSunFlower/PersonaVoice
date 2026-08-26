"""Create an explicit failure ledger from the published CFR statistics."""
from __future__ import annotations

import argparse
from pathlib import Path

from jsonio import dump_json, read_json


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--artifacts", default="artifacts"); parser.add_argument("--output", default="artifacts/failure_ledger.json"); args = parser.parse_args()
    cfr = read_json(Path(args.artifacts) / "cfr_analysis.json")
    payload = {"schema_version": "personavoice-speech-failure-ledger-v1", "source": "cfr_analysis.json", "groups": cfr, "operational_rule": "catastrophic text failure is WER > 0.5", "finding": "Ultra-short text is the dominant measured failure stratum; stronger guidance and Best-of-N historically increased WER/CFR and are excluded from the release configuration."}
    dump_json(payload, args.output)


if __name__ == "__main__":
    main()
