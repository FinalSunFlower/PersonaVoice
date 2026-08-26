"""Build the deterministic SHA-256 manifest for the paper supplement release."""
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reproducibility" / "SHA256SUMS.txt"
SKIP_DIRS = {".git", ".tmp_render2", "__pycache__", ".pytest_cache", "paper"}
SKIP_NAMES = {OUTPUT.name}


def iter_release_files():
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(ROOT).parts
        if any(part in SKIP_DIRS for part in rel_parts) or path.name in SKIP_NAMES:
            continue
        if path.suffix.lower() in {
            ".pyc", ".aux", ".bbl", ".blg", ".log", ".out", ".fdb_latexmk",
            ".fls", ".synctex.gz",
        }:
            continue
        yield path


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    lines = [
        "# SHA-256 manifest for paper-supplement-v1.0",
        "# Format: <digest>  <repository-relative-path>",
    ]
    for path in iter_release_files():
        lines.append(f"{digest(path)}  {path.relative_to(ROOT).as_posix()}")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines) - 2} file hashes to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
