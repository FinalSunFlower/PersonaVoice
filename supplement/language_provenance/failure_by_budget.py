"""Normalize the measured failure taxonomy by observed evidence budget."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audit", default="artifacts/failure_audit.json")
    p.add_argument("--curve", default="artifacts/budget_curve.json")
    p.add_argument("--output", default="artifacts/failure_by_budget.json")
    args = p.parse_args()
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    curve = json.loads(Path(args.curve).read_text(encoding="utf-8"))
    result = {}
    for budget, info in curve["budgets"].items():
        counts = audit["by_evidence_budget"].get(budget, {})
        n = info["constrained"]["n"]
        result[budget] = {
            "n": n,
            "false_abstention_selection_rate": counts.get("false_abstention:selection_threshold", 0) / n,
            "false_abstention_attribution_rate": counts.get("false_abstention:attribution_underconfidence", 0) / n,
            "false_acceptance_missing_support_rate": counts.get("false_acceptance:missing_support", 0) / n,
            "false_acceptance_selection_calibration_rate": counts.get("false_acceptance:selection_calibration", 0) / n,
            "counts": counts,
        }
    payload = {
        "schema_version": "personavoice-language-failure-by-budget-v1",
        "status": "measured_replay",
        "interpretation": "Rates use pooled frozen rows from three registered seeds; cells are the observed budgets only.",
        "budgets": result,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
