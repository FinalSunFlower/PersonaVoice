"""Compute preregistered fine-grained text-length diagnostics."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


BUCKETS = (("1-3", 1, 3), ("4-8", 4, 8), ("9-15", 9, 15), (">15", 16, None))


def wilson(k: int, n: int, z: float = 1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [center - half, center + half]


def summarize(rows):
    n = len(rows)
    secs = [float(r["secs"]) for r in rows]
    wer = [float(r["wer"]) for r in rows]
    catastrophic = sum(x > 0.5 for x in wer)
    return {
        "n": n,
        "secs_mean": sum(secs) / n if n else None,
        "wer_mean": sum(wer) / n if n else None,
        "cfr_wer_gt_0.5": sum(x > 0.5 for x in wer) / n if n else None,
        "catastrophic_count": catastrophic,
        "cfr_wilson_95": wilson(catastrophic, n),
        "missing_length_count": sum(r.get("text_len_words") is None for r in rows),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="artifacts/eval_200_samples.json")
    p.add_argument("--output", default="artifacts/fine_length_analysis.json")
    args = p.parse_args()
    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = source["rows"]
    result = {
        "schema_version": "personavoice-speech-shortref-fine-length-v1",
        "status": "measured",
        "source_artifact": Path(args.input).name,
        "bucket_definition": "word count; inclusive 1-3, 4-8, 9-15, and >=16",
        "protocol": source["protocol"],
        "buckets": {},
    }
    for name, low, high in BUCKETS:
        selected = [
            r for r in rows
            if r.get("text_len_words") is not None
            and r["text_len_words"] >= low
            and (high is None or r["text_len_words"] <= high)
        ]
        result["buckets"][name] = summarize(selected)
    result["unbucketed"] = summarize([r for r in rows if r.get("text_len_words") is None])
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
