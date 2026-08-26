"""Compute evidence-budget curves without tuning on the test split."""
from __future__ import annotations

import argparse
from collections import defaultdict

from schemas import dump_json, load_prediction_rows


def metrics(rows, accepted_key="constrained_accept"):
    positives = sum(r["accept_target"] == 1 for r in rows)
    accepted = [r for r in rows if r[accepted_key]]
    true_accepts = sum(r["accept_target"] == 1 for r in accepted)
    false_accepts = len(accepted) - true_accepts
    cited = [r for r in accepted if r["citation_precision"] > 0]
    return {
        "n": len(rows),
        "accepted": len(accepted),
        "positive_coverage": true_accepts / positives if positives else 0.0,
        "false_acceptance": false_accepts / len(accepted) if accepted else 0.0,
        "citation_precision": sum(r["citation_precision"] for r in rows) / len(rows) if rows else 0.0,
        "citation_recall": sum(r["citation_recall"] for r in rows) / len(rows) if rows else 0.0,
        "accepted_citation_precision": sum(r["citation_precision"] for r in accepted) / len(accepted) if accepted else 0.0,
    }


def fit_unconstrained_threshold(validation_rows, target_risk=0.05):
    scored = [(r["support_probability"] * r["style_probability"], r["accept_target"]) for r in validation_rows]
    best = (1.0, 0.0)
    for threshold, _ in sorted(scored, reverse=True):
        chosen = [target for score, target in scored if score >= threshold]
        risk = 1 - sum(chosen) / len(chosen) if chosen else 0.0
        coverage = sum(chosen) / sum(t for _, t in scored) if sum(t for _, t in scored) else 0.0
        if risk <= target_risk and coverage >= best[1]:
            best = (threshold, coverage)
    return best[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--validation", default=None)
    parser.add_argument("--output", default="artifacts/budget_curve.json")
    args = parser.parse_args()
    rows = load_prediction_rows(args.artifacts)
    # The validation files are supplied in the original repository when fitting;
    # this fallback records the fixed published operating point for replays.
    threshold = 0.5
    if args.validation:
        from schemas import read_json
        validation = []
        for seed in range(3):
            validation.extend(read_json(args.validation.format(seed=seed))["rows"])
        threshold = fit_unconstrained_threshold(validation)
    groups = defaultdict(list)
    for row in rows:
        copy = dict(row)
        copy["constrained_accept"] = row["decision"] == "synthesize"
        copy["unconstrained_accept"] = row["support_probability"] * row["style_probability"] >= threshold
        groups[str(row["evidence_budget"])].append(copy)
    result = {
        "schema_version": "personavoice-language-budget-curve-v1",
        "status": "measured_replay",
        "requested_budgets": [1, 2, 3, 5, 10],
        "unavailable_requested_budgets": [1, 2, 3],
        "availability_note": "The frozen prediction snapshots contain no genuine planner re-evaluation at budgets 1, 2, or 3. These cells are intentionally absent rather than reconstructed by truncating full-budget predictions.",
        "unconstrained_score": "support_probability * style_probability; no citation or conflict gate",
        "unconstrained_threshold": threshold,
        "budgets": {k: {"constrained": metrics(v), "unconstrained": metrics(v, "unconstrained_accept")} for k, v in sorted(groups.items(), key=lambda item: int(item[0]))},
    }
    dump_json(result, args.output)


if __name__ == "__main__":
    main()
