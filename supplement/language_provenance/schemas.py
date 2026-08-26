"""Small, dependency-free readers for the frozen language artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_prediction_rows(artifact_dir: str | Path) -> List[Dict[str, Any]]:
    root = Path(artifact_dir)
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.glob("snli_semantic_mlp_seed[0-9].json")):
        payload = read_json(path)
        if payload.get("status") not in {"measured_evaluation", "measured", "measured_evaluation_run"}:
            raise ValueError(f"unexpected status in {path}: {payload.get('status')}")
        for row in payload["rows"]:
            copy = dict(row)
            copy["seed_file"] = path.name
            rows.append(copy)
    if not rows:
        raise FileNotFoundError(f"no frozen prediction files under {root}")
    return rows


def load_benchmark(artifact_dir: str | Path) -> Dict[str, Dict[str, Any]]:
    path = Path(artifact_dir) / "snli_compositional_test.jsonl"
    records = {}
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                records[item["episode_id"]] = item
    return records


def source_type(evidence_id: str) -> str:
    if evidence_id.startswith("style:"):
        return "style_observation"
    if evidence_id.startswith("snli:"):
        return "factual_relation"
    return "other"


def dump_json(payload: Any, path: str | Path) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
