import json
from pathlib import Path


def test_source_snapshots_are_present():
    root = Path(__file__).parents[1] / "artifacts"
    for name in ["eval_200_statistics.json", "ablation_200_statistics.json", "baseline_external_200.json", "cfr_analysis.json"]:
        assert (root / name).exists()


def test_headline_is_in_expected_range():
    root = Path(__file__).parents[1] / "artifacts"
    payload = json.loads((root / "eval_200_statistics.json").read_text(encoding="utf-8-sig"))
    assert 0.45 < payload["personavoice"]["secs"]["mean"] < 0.55
    assert 0.1 < payload["personavoice"]["wer"]["mean"] < 0.3
