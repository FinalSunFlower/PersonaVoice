import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from audit_failures import classify
from budget_curve import metrics


def test_failure_tags_are_explicit():
    base = {"decision": "synthesize", "accept_target": 0, "style_target": 0}
    assert classify(base, {}) == "false_acceptance:style_overfit"
    base = {"decision": "synthesize", "accept_target": 0, "support_target": 0, "conflict_target": 0}
    assert classify(base, {}) == "false_acceptance:missing_support"


def test_metrics_empty():
    result = metrics([])
    assert result["n"] == 0 and result["false_acceptance"] == 0
