"""Draw the language planner mechanism as a clean, paper-scale vector schematic."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


COLORS = {
    "ink": "#243447",
    "muted": "#667085",
    "input": "#F3F5F7",
    "blue": "#DCEBFA",
    "blue_edge": "#245C9C",
    "teal": "#DDF4F0",
    "teal_edge": "#147D78",
    "gold": "#FFF0C9",
    "gold_edge": "#B7791F",
    "red": "#FBE0DE",
    "red_edge": "#B53D3A",
    "green": "#E0F1E2",
    "green_edge": "#3E8B4B",
    "white": "#FFFFFF",
}


def box(ax, x, y, w, h, title, body, face, edge, title_size=14, body_size=11.5, lw=1.8):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.06", facecolor=face, edgecolor=edge, linewidth=lw, zorder=2)
    ax.add_patch(patch)
    multiline = "\n" in title
    title_y = 0.75 if multiline else 0.68
    body_y = 0.25 if multiline else 0.32
    ax.text(x + w / 2, y + h * title_y, title, ha="center", va="center", color=edge, fontsize=title_size, fontweight="bold", linespacing=1.0, zorder=3)
    ax.text(x + w / 2, y + h * body_y, body, ha="center", va="center", color=COLORS["ink"], fontsize=body_size, linespacing=1.15, zorder=3)


def arrow(ax, start, end, color=COLORS["muted"], lw=1.7, rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}", shrinkA=4, shrinkB=5, zorder=1))


def draw(output: Path):
    plt.rcParams.update({"font.family": ["DejaVu Sans", "sans-serif"], "pdf.fonttype": 42, "ps.fonttype": 42})
    # A compact 12-inch canvas keeps labels large when the figure is placed at
    # text width in the preprint. Long labels are deliberately wrapped inside
    # their modules rather than relying on clipping or tiny type.
    fig = plt.figure(figsize=(12.0, 4.35), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 12.0), ylim=(0, 4.35))
    ax.axis("off")

    # Inputs and shared representation.
    box(ax, 0.20, 2.95, 1.55, 0.95, "Typed\nevidence", "source / time / stance", COLORS["input"], COLORS["muted"], 13.5, 9.5)
    box(ax, 0.20, 0.85, 1.55, 0.95, "Candidate", "utterance $y$", COLORS["input"], COLORS["muted"], 13.5, 10.0)
    box(ax, 2.05, 0.90, 1.90, 2.95, "Shared\nencoder", "frozen MiniLM or\nhashed CPU baseline\n\nseparate $h_E$ and $h_y$", COLORS["blue"], COLORS["blue_edge"], 15.0, 9.5, 2.0)
    arrow(ax, (1.80, 3.42), (2.00, 3.25))
    arrow(ax, (1.80, 1.32), (2.00, 1.65))

    # Three auditable branches.
    box(ax, 4.35, 2.95, 1.85, 0.9, "Context\naggregation", "candidate-conditioned\nnot a citation", COLORS["teal"], COLORS["teal_edge"], 12.5, 8.8)
    box(ax, 4.35, 1.75, 1.85, 0.9, "Evidence\nattribution", "per-item $a_i$\nevidence IDs", COLORS["green"], COLORS["green_edge"], 12.5, 8.8)
    box(ax, 4.35, 0.55, 1.85, 0.9, "Observable\nstyle", "evidence/candidate\ndifference", COLORS["gold"], COLORS["gold_edge"], 12.5, 8.8)
    arrow(ax, (4.00, 3.25), (4.30, 3.35), COLORS["blue_edge"])
    arrow(ax, (4.00, 2.35), (4.30, 2.20), COLORS["blue_edge"])
    arrow(ax, (4.00, 1.55), (4.30, 1.00), COLORS["blue_edge"])

    # Inspectable gates before the final decision.
    box(ax, 6.65, 2.95, 1.45, 0.9, "Task heads", "$p_s, p_c, p_y$\nsupport / conflict / style", COLORS["blue"], COLORS["blue_edge"], 11.0, 7.8)
    box(ax, 6.65, 1.75, 1.45, 0.9, "Citation gate", "$\\max_i a_i \\geq \\theta_e$\nselected IDs", COLORS["green"], COLORS["green_edge"], 10.5, 7.8)
    box(ax, 6.65, 0.55, 1.45, 0.9, "Style gate", "$p_y$ observable\nnot a trait claim", COLORS["gold"], COLORS["gold_edge"], 10.5, 7.8)
    arrow(ax, (6.25, 3.40), (6.60, 3.40), COLORS["teal_edge"])
    arrow(ax, (6.25, 2.20), (6.60, 2.20), COLORS["green_edge"])
    arrow(ax, (6.25, 1.00), (6.60, 1.00), COLORS["gold_edge"])

    # Decision diamond and approved-text boundary.
    diamond = Polygon([[8.45, 2.2], [9.10, 2.85], [9.75, 2.2], [9.10, 1.55]], closed=True, facecolor=COLORS["red"], edgecolor=COLORS["red_edge"], linewidth=2.0, zorder=2)
    ax.add_patch(diamond)
    ax.text(9.10, 2.36, "Calibrated", ha="center", va="center", fontsize=11.5, fontweight="bold", color=COLORS["red_edge"], zorder=3)
    ax.text(9.10, 1.98, "accept or abstain", ha="center", va="center", fontsize=9.5, color=COLORS["ink"], zorder=3)
    arrow(ax, (8.10, 3.40), (8.40, 2.78), COLORS["blue_edge"])
    arrow(ax, (8.10, 2.20), (8.40, 2.20), COLORS["green_edge"])
    arrow(ax, (8.10, 1.00), (8.40, 1.62), COLORS["gold_edge"])
    box(ax, 10.15, 1.65, 1.55, 1.1, "Output", "approved text\n+ evidence IDs", COLORS["input"], COLORS["muted"], 12.5, 9.0, 1.6)
    arrow(ax, (9.78, 2.2), (10.10, 2.2), COLORS["red_edge"], 2.0)

    # The claim boundary is explicit and visually separate from the model path.
    ax.plot([2.05, 8.10], [0.22, 0.22], color=COLORS["muted"], linewidth=1.0, linestyle=(0, (4, 3)), zorder=0)
    ax.text(5.08, 0.02, "Approved text and evidence IDs cross the realization boundary", ha="center", va="bottom", fontsize=8.8, color=COLORS["muted"])

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output.with_suffix(".svg"), format="svg", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="../../paper/figures/planner_mechanism.pdf")
    draw(Path(parser.parse_args().output).resolve())
