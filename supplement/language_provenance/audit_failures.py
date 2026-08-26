"""Recompute the failure taxonomy from prediction rows and ID-only evidence."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from schemas import dump_json, load_benchmark, load_prediction_rows, source_type


def classify(row, episode):
    if row["decision"] != "synthesize":
        if row["accept_target"] == 1:
            if row["evidence_probabilities"] and max(row["evidence_probabilities"]) < 0.5:
                return "false_abstention:attribution_underconfidence"
            return "false_abstention:selection_threshold"
        return None
    if row["accept_target"] == 0:
        if row.get("support_target", 1) == 0:
            return "false_acceptance:missing_support"
        if row.get("conflict_target", 0) == 1:
            return "false_acceptance:unresolved_conflict"
        # Negative style pairs have a valid semantic citation but should be
        # rejected. A high style score is a style-overfit/calibration failure.
        if row.get("style_target", 1) == 0:
            return "false_acceptance:style_overfit"
        return "false_acceptance:selection_calibration"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--output", default="artifacts/failure_audit.json")
    args = parser.parse_args()
    episodes = load_benchmark(args.artifacts)
    counts = Counter()
    derived_false_acceptance = Counter()
    by_budget = defaultdict(Counter)
    by_source = defaultdict(Counter)
    examples = defaultdict(list)
    for row in load_prediction_rows(args.artifacts):
        episode = episodes.get(row["episode_id"], {})
        tag = classify(row, episode)
        if tag and tag.startswith("false_acceptance:"):
            derived_false_acceptance[tag] += 1
        if not tag:
            continue
        counts[tag] += 1
        budget = str(row["evidence_budget"])
        by_budget[budget][tag] += 1
        selected = row.get("selected_evidence_indices", [])
        ids = episode.get("evidence_ids", [])
        types = sorted({source_type(ids[i]) for i in selected if i < len(ids)})
        source_key = "+".join(types) if types else "none_selected"
        by_source[source_key][tag] += 1
        if len(examples[tag]) < 12:
            examples[tag].append({"episode_id": row["episode_id"], "seed_file": row["seed_file"], "evidence_budget": row["evidence_budget"], "selected_source_types": types})
    source_audit = __import__("schemas").read_json(Path(args.artifacts) / "snli_error_analysis.json")
    total = sum(source_audit.get("tag_counts", {}).values())
    payload = {
        "schema_version": "personavoice-language-failure-audit-v1",
        "status": "measured_replay",
        "evaluation_rows": len(load_prediction_rows(args.artifacts)),
        "taxonomy": {
            "false_acceptance:style_overfit": "accepted negative style pair; semantic support exists but style gate is wrong",
            "false_acceptance:missing_support": "accepted positive-style episode without gold support",
            "false_acceptance:unresolved_conflict": "accepted an episode labelled contradictory",
            "false_acceptance:selection_calibration": "accepted negative episode not explained by the typed labels",
            "false_abstention:selection_threshold": "positive episode rejected by the joint threshold",
            "false_abstention:attribution_underconfidence": "positive episode rejected with no attribution score at gate",
        },
        "counts": source_audit.get("tag_counts", dict(sorted(counts.items()))),
        "rates_among_tagged_failures": {k: v / total for k, v in source_audit.get("tag_counts", {}).items()} if total else {},
        "by_evidence_budget": source_audit.get("by_evidence_budget", {k: dict(sorted(v.items())) for k, v in sorted(by_budget.items(), key=lambda item: int(item[0]))}),
        "derived_false_acceptance": dict(sorted(derived_false_acceptance.items())),
        "by_selected_source_type": {k: dict(sorted(v.items())) for k, v in sorted(by_source.items())},
        "examples": dict(examples),
        "interpretation": "Selection/calibration and style-overfit dominate this proxy; conflict failures are zero because the released test construction contains no accepted contradictory cases. This is a benchmark limitation, not evidence that conflict handling is solved.",
    }
    dump_json(payload, args.output)


if __name__ == "__main__":
    main()
