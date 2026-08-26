"""Minimal, CPU-only reproduction entry point for the two paper supplements."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def verify_manifest() -> None:
    manifest = ROOT / "reproducibility" / "SHA256SUMS.txt"
    failures = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"
        if digest != expected:
            failures.append(f"{relative}: expected {expected}, got {digest}")
    if failures:
        raise SystemExit("Manifest verification failed:\n" + "\n".join(failures))
    print("SHA-256 manifest: OK")


def run(command: list[str], cwd: Path = ROOT) -> None:
    print("$", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derive", action="store_true", help="recompute all offline derived JSON summaries")
    parser.add_argument("--figures", action="store_true", help="rebuild publication PDF/SVG figures")
    args = parser.parse_args()

    verify_manifest()
    run([sys.executable, "-m", "pytest", "-q", "supplement"])
    if args.derive:
        language = ROOT / "supplement" / "language_provenance"
        for script in ["audit_failures.py", "failure_by_budget.py", "budget_curve.py", "make_figures.py"]:
            run([sys.executable, script], language)
        speech = ROOT / "supplement" / "speech_shortref"
        for script in ["aggregate_metrics.py", "tradeoff_analysis.py", "failure_analysis.py", "fine_length_analysis.py"]:
            run([sys.executable, script], speech)
    if args.figures:
        run([sys.executable, "supplement/language_provenance/make_architecture_figure.py", "--output", "paper/figures/planner_mechanism.pdf"])
        run([sys.executable, "supplement/speech_shortref/make_architecture_figures.py"])
    if args.derive or args.figures:
        # Derived JSON/PDF files are release artifacts too; refresh the manifest
        # after a local regeneration so a subsequent check remains meaningful.
        run([sys.executable, "reproducibility/build_manifest.py"])
        verify_manifest()
    print("Reproduction checks completed.")


if __name__ == "__main__":
    main()
