"""Draw title-free, publication-scale speech architecture schematics."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


INK = "#243447"
MUTED = "#667085"
BLUE = "#DCEBFA"
BLUE_EDGE = "#245C9C"
GREEN = "#E0F1E2"
GREEN_EDGE = "#3E8B4B"
RED = "#FBE0DE"
RED_EDGE = "#B53D3A"
GRAY = "#F3F5F7"


def box(ax, x, y, w, h, title, body, face, edge, title_size=15, body_size=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor=face, edgecolor=edge, linewidth=1.5))
    multiline = "\n" in title
    title_y = 0.73 if multiline else 0.68
    body_y = 0.24 if multiline else 0.30
    if body.count("\n") >= 3:
        body_y = 0.19
    ax.text(x + w / 2, y + h * title_y, title, ha="center", va="center", color=edge, fontsize=title_size, fontweight="bold", linespacing=1.0)
    ax.text(x + w / 2, y + h * body_y, body, ha="center", va="center", color=INK, fontsize=body_size, linespacing=1.10)


def arrow(ax, start, end, color=MUTED, lw=1.4):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=lw, color=color, shrinkA=4, shrinkB=5))


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def speech_overview(out: Path):
    fig, ax = plt.subplots(figsize=(8.8, 3.85), dpi=150)
    ax.set_xlim(0, 10.8); ax.set_ylim(0, 4); ax.axis("off")
    box(ax, 0.12, 2.30, 2.35, 1.25, "Reference\nevidence", "nested segments\n0.5 / 1 / 2 / 3 / full", GRAY, MUTED, 15.0, 10.8)
    box(ax, 0.12, 0.40, 2.35, 1.25, "Approved\ntext", "fixed target text\nverified transcript", GRAY, MUTED, 15.0, 10.8)
    box(ax, 2.90, 1.45, 2.15, 1.30, "Factorized\ncontrols", "persona effects\nemotion effects", GREEN, GREEN_EDGE, 15.0, 11.0)
    box(ax, 5.50, 0.85, 2.65, 2.45, "Frozen\nF5-TTS", "official flow path\nfinal-block FiLM\nhooks; adapter-only", BLUE, BLUE_EDGE, 17.5, 10.0)
    box(ax, 8.65, 2.30, 1.95, 1.25, "Generated\nspeech", "same decoding\nsame seeds", GRAY, MUTED, 15.0, 10.8)
    box(ax, 8.65, 0.25, 1.95, 1.50, "Separate\nevidence\naxes", "CER / WER\nidentity / failures\ncontrol / leakage", RED, RED_EDGE, 11.5, 9.5)
    arrow(ax, (2.52, 2.92), (5.45, 2.75))
    arrow(ax, (2.52, 1.02), (5.45, 1.35))
    arrow(ax, (5.10, 2.05), (5.45, 2.35), GREEN_EDGE)
    arrow(ax, (8.20, 2.35), (8.60, 2.82))
    arrow(ax, (9.62, 2.25), (9.62, 1.82))
    save(fig, out)


def film_mechanism(out: Path):
    fig, ax = plt.subplots(figsize=(8.8, 3.25), dpi=150)
    ax.set_xlim(0, 10.7); ax.set_ylim(0, 3.2); ax.axis("off")
    box(ax, 0.12, 1.78, 2.45, 1.20, "Persona\ncontrols", "5 centered\ncoordinates\nmain-effect bases", GREEN, GREEN_EDGE, 15.0, 9.5)
    box(ax, 0.12, 0.22, 2.45, 1.20, "Emotion\ncontrols", "4 transient\ncoordinates\nmain-effect bases", RED, RED_EDGE, 15.0, 9.5)
    box(ax, 2.95, 0.88, 2.25, 1.45, "Interaction\nbank", "trait x emotion\nseparately ablatable", GRAY, MUTED, 15.0, 11.0)
    box(ax, 5.75, 0.88, 2.10, 1.45, "FiLM\nmodulation", "$h'=(1+\\Delta\\gamma)h+\\beta$\nzero initialized", BLUE, BLUE_EDGE, 14.0, 10.0)
    box(ax, 8.35, 0.88, 2.10, 1.45, "Final DiT\nblocks", "frozen weights\nverified hooks", BLUE, BLUE_EDGE, 15.0, 11.0)
    arrow(ax, (2.62, 2.35), (2.89, 1.85), GREEN_EDGE)
    arrow(ax, (2.62, 0.85), (2.89, 1.35), RED_EDGE)
    arrow(ax, (5.25, 1.60), (5.69, 1.60))
    arrow(ax, (7.90, 1.60), (8.29, 1.60), BLUE_EDGE)
    save(fig, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir or root / "supplement/speech_shortref/figures"
    speech_overview(output_dir / "speech_overview.pdf")
    film_mechanism(output_dir / "film_mechanism.pdf")
