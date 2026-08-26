"""Draw the title-free language planner overview used in the paper."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


INK = "#243447"
MUTED = "#667085"
BLUE = "#DCEBFA"
BLUE_EDGE = "#245C9C"
GREEN = "#E0F1E2"
GREEN_EDGE = "#3E8B4B"
RED = "#FBE0DE"
RED_EDGE = "#B53D3A"
GRAY = "#F3F5F7"


def box(ax, x, y, w, h, title, body, face, edge, title_size=14.5, body_size=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor=face, edgecolor=edge, linewidth=1.5))
    ax.text(x + w / 2, y + h * 0.7, title, ha="center", va="center", color=edge, fontsize=title_size, fontweight="bold", linespacing=1.0)
    ax.text(x + w / 2, y + h * 0.3, body, ha="center", va="center", color=INK, fontsize=body_size, linespacing=1.08)


def arrow(ax, start, end, color=MUTED, lw=1.4):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=lw, color=color, shrinkA=4, shrinkB=5))


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / "paper/figures/method_overview.pdf"
    fig, ax = plt.subplots(figsize=(12.0, 3.6), dpi=150)
    ax.set_xlim(0, 12.2); ax.set_ylim(0, 4); ax.axis("off")
    box(ax, 0.2, 1.75, 2.3, 1.25, "Typed evidence", "claim + source\nconsent / time", GRAY, MUTED, 13.8, 10.2)
    box(ax, 0.2, 0.35, 2.3, 0.85, "Candidate\nutterance", "candidate $y$", GRAY, MUTED, 12.2, 10.2)
    ax.add_patch(FancyBboxPatch((3.05, 0.45), 4.65, 2.8, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor=BLUE, edgecolor=BLUE_EDGE, linewidth=1.8))
    ax.text(5.38, 2.95, "Provenance-constrained planner", ha="center", va="center", color=BLUE_EDGE, fontsize=15.5, fontweight="bold")
    box(ax, 3.35, 1.78, 1.85, 0.75, "Attribution", "citation $a_i$", "white", BLUE_EDGE, 11.5, 9.3)
    box(ax, 5.55, 1.78, 1.85, 0.75, "Support /\nconflict", "$p_s$, $p_c$", "white", BLUE_EDGE, 10.7, 9.3)
    box(ax, 3.35, 0.78, 1.85, 0.75, "Observable\nstyle", "match $p_y$", "white", BLUE_EDGE, 10.7, 9.3)
    box(ax, 5.55, 0.78, 1.85, 0.75, "Calibrated\nselection", "risk $p_a$", "white", BLUE_EDGE, 10.4, 9.3)
    diamond = Polygon([[8.15, 1.85], [8.85, 2.55], [9.55, 1.85], [8.85, 1.15]], closed=True, facecolor=RED, edgecolor=RED_EDGE, linewidth=1.8)
    ax.add_patch(diamond)
    ax.text(8.85, 1.98, "Fail-closed\ndecision", ha="center", va="center", color=RED_EDGE, fontsize=12.8, fontweight="bold", linespacing=1.0)
    box(ax, 10.0, 1.95, 1.75, 0.75, "Accept", "text + citations", GREEN, GREEN_EDGE, 12.5, 9.8)
    box(ax, 10.0, 0.75, 1.75, 0.75, "Abstain", "typed reasons", RED, RED_EDGE, 12.5, 9.8)
    arrow(ax, (2.52, 2.4), (3.0, 2.25)); arrow(ax, (2.52, 0.78), (3.0, 1.2))
    arrow(ax, (7.75, 2.15), (8.1, 2.15), BLUE_EDGE)
    arrow(ax, (9.58, 2.15), (9.95, 2.35), GREEN_EDGE)
    arrow(ax, (9.58, 1.55), (9.95, 1.15), RED_EDGE)
    ax.plot([3.05, 9.55], [0.28, 0.28], color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(6.3, 0.03, "Approved text only -> separately evaluated realizer", ha="center", va="bottom", color=MUTED, fontsize=10.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


if __name__ == "__main__":
    main()
