"""Cross-domain lexical evidence audit on the public DailyDialog test split.

This is an auxiliary structural audit, not a rerun of the trained planner. It
uses only the official test archive and deterministic token-overlap labels:
previous turns that share a non-stopword with the target turn are candidate
lexical evidence. The artifact reports this limitation explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://huggingface.co/datasets/roskoN/dailydialog/resolve/main/test.zip?download=true"
STOP = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "to", "of",
    "in", "on", "at", "for", "from", "with", "is", "are", "was", "were", "be",
    "been", "being", "i", "you", "he", "she", "it", "we", "they", "this", "that",
    "these", "those", "my", "your", "our", "their", "me", "him", "her", "us", "them",
    "do", "does", "did", "have", "has", "had", "will", "would", "can", "could", "not",
}


def toks(text: str) -> set[str]:
    return {x.lower() for x in re.findall(r"[A-Za-z]{3,}", text) if x.lower() not in STOP}


def parse_dialogues(path: Path):
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        turns = [x.strip() for x in line.split("__eou__") if x.strip()]
        if len(turns) >= 3:
            yield turns


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="artifacts/dailydialog_cross_dataset.json")
    p.add_argument("--max-episodes", type=int, default=1000)
    args = p.parse_args()
    with tempfile.TemporaryDirectory(prefix="personavoice-dd-") as tmp:
        archive = Path(tmp) / "test.zip"
        urllib.request.urlretrieve(URL, archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        with zipfile.ZipFile(archive) as zf:
            member = next(x for x in zf.namelist() if x.endswith("dialogues_test.txt"))
            text_path = Path(tmp) / "dialogues_test.txt"
            text_path.write_bytes(zf.read(member))
        dialogues = list(parse_dialogues(text_path))[: args.max_episodes]

    budgets = (1, 2, 3, 5)
    results = {}
    for budget in budgets:
        rows = []
        for dialogue in dialogues:
            target_i = min(len(dialogue) - 1, 5)
            candidate = toks(dialogue[target_i])
            evidence = dialogue[max(0, target_i - budget) : target_i]
            evidence_tokens = [toks(x) for x in evidence]
            overlaps = [len(candidate & x) / max(1, len(candidate)) for x in evidence_tokens]
            gold = {i for i, score in enumerate(overlaps) if score > 0}
            if not candidate or not gold:
                continue
            top = max(range(len(overlaps)), key=lambda i: (overlaps[i], -i))
            rows.append({"gold_count": len(gold), "top_correct": top in gold, "coverage": 1})
        n = len(rows)
        results[str(budget)] = {
            "n_supported_episodes": n,
            "support_coverage_over_eligible": n / len(dialogues) if dialogues else 0.0,
            "top1_citation_precision": sum(1 / r["gold_count"] for r in rows) / n if n else 0.0,
            "top1_citation_recall": sum(r["top_correct"] for r in rows) / n if n else 0.0,
        }

    payload = {
        "schema_version": "personavoice-language-dailydialog-audit-v1",
        "status": "measured_auxiliary_audit",
        "dataset": "DailyDialog official test archive",
        "source_url": URL.split("?", 1)[0],
        "source_sha256": digest,
        "split": "test",
        "episodes_considered": len(dialogues),
        "target_turn": "sixth turn when available; otherwise final turn",
        "budgets": results,
        "label_definition": "lexical evidence = prior turn sharing at least one non-stopword with the target",
        "claim_boundary": "This is not a planner rerun, entailment annotation, or natural-person authorization result; it is a deterministic cross-domain input-structure audit.",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
