"""Draw title-free, publication-scale speech architecture schematics."""
from __future__ import annotations

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


def box(ax, x, y, w, h, title, body, face, edge, title_size=11, body_size=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor=face, edgecolor=edge, linewidth=1.5))
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", color=edge, fontsize=title_size, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.31, body, ha="center", va="center", color=INK, fontsize=body_size, linespacing=1.15)


def arrow(ax, start, end, color=MUTED, lw=1.4):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=lw, color=color, shrinkA=4, shrinkB=5))


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def speech_overview(out: Path):
    fig, ax = plt.subplots(figsize=(8.8, 3.55), dpi=150)
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    box(ax, 0.25, 2.35, 1.85, 1.05, "Reference\nevidence", "nested segments\n0.5 / 1 / 2 / 3 / full", GRAY, MUTED, 9.5, 7.6)
    box(ax, 0.25, 0.55, 1.85, 1.05, "Approved text", "fixed target text\nverified transcript", GRAY, MUTED, 9.5, 7.6)
    box(ax, 2.55, 1.55, 1.85, 0.95, "Factorized\ncontrols", "persona main effects\nemotion main effects", GREEN, GREEN_EDGE, 9.2, 7.3)
    box(ax, 4.95, 1.05, 2.25, 2.0, "Frozen F5-TTS", "official flow path\nfinal-block FiLM hooks\nadapter-only update", BLUE, BLUE_EDGE, 12, 8.6)
    box(ax, 7.8, 2.35, 1.7, 1.05, "Generated\nspeech", "same decoding\nsame seeds", GRAY, MUTED, 9.3, 7.5)
    box(ax, 7.8, 0.55, 1.7, 1.05, "Separate\nevidence axes", "CER / WER\nidentity / failures\ncontrol / leakage", RED, RED_EDGE, 8.8, 7.3)
    arrow(ax, (1.97, 2.88), (4.9, 2.78))
    arrow(ax, (1.97, 1.08), (4.9, 1.28))
    arrow(ax, (4.35, 2.0), (4.9, 2.32), GREEN_EDGE)
    arrow(ax, (7.25, 2.45), (7.74, 2.82))
    arrow(ax, (8.65, 2.33), (8.65, 1.68))
    save(fig, out)


def film_mechanism(out: Path):
    fig, ax = plt.subplots(figsize=(8.8, 2.8), dpi=150)
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.2); ax.axis("off")
    box(ax, 0.25, 1.75, 1.85, 0.95, "Persona controls", "5 centered coordinates\nmain-effect bases", GREEN, GREEN_EDGE, 10, 7.5)
    box(ax, 0.25, 0.45, 1.85, 0.95, "Emotion controls", "4 transient coordinates\nmain-effect bases", RED, RED_EDGE, 10, 7.5)
    box(ax, 2.75, 1.0, 2.0, 1.1, "Interaction bank", "trait x emotion\nseparately ablatable", GRAY, MUTED, 10, 8)
    box(ax, 5.35, 1.0, 1.8, 1.1, "FiLM modulation", "$h'=(1+\\Delta\\gamma)h+\\beta$\nzero initialized", BLUE, BLUE_EDGE, 9.0, 6.4)
    box(ax, 7.9, 1.0, 1.9, 1.1, "Final DiT blocks", "frozen weights\nverified hooks", BLUE, BLUE_EDGE, 9.3, 7.2)
    arrow(ax, (2.15, 2.22), (2.69, 1.78), GREEN_EDGE)
    arrow(ax, (2.15, 0.92), (2.69, 1.32), RED_EDGE)
    arrow(ax, (4.8, 1.55), (5.39, 1.55))
    arrow(ax, (7.2, 1.55), (7.84, 1.55), BLUE_EDGE)
    save(fig, out)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    speech_overview(root / "paper/figures/tmlr_speech_overview.pdf")
    film_mechanism(root / "paper/figures/tmlr_film_mechanism.pdf")
