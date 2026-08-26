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
    # Export near the physical width of a two-column figure*.  The 5-inch
    # height lets every label remain legible after LaTex scales it to textwidth.
    fig = plt.figure(figsize=(9.3, 5.0), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 15.5), ylim=(0, 5.6))
    ax.axis("off")

    # Inputs and shared representation.
    box(ax, 0.30, 3.85, 2.00, 1.20, "Typed\nevidence", "source / time\nstance", COLORS["input"], COLORS["muted"], 16.0, 13.0)
    box(ax, 0.30, 1.05, 2.00, 1.20, "Candidate", "utterance $y$", COLORS["input"], COLORS["muted"], 15.0, 13.0)
    box(ax, 2.75, 1.05, 2.35, 4.00, "Shared\nencoder", "frozen MiniLM\nor hashed CPU\nbaseline\n\nseparate $h_E$ and $h_y$", COLORS["blue"], COLORS["blue_edge"], 18.0, 13.0, 2.2)
    arrow(ax, (2.35, 4.45), (2.68, 4.20))
    arrow(ax, (2.35, 1.60), (2.68, 2.10))

    # Three auditable branches.
    box(ax, 5.75, 3.85, 2.45, 1.20, "Context\naggregation", "candidate-\nconditioned\nnot provenance", COLORS["teal"], COLORS["teal_edge"], 15.0, 12.0)
    box(ax, 5.75, 2.25, 2.45, 1.20, "Evidence\nattribution", "per-item $a_i$\nevidence IDs", COLORS["green"], COLORS["green_edge"], 15.0, 12.5)
    box(ax, 5.75, 0.65, 2.45, 1.20, "Observable\nstyle", "evidence /\ncandidate\ndifference", COLORS["gold"], COLORS["gold_edge"], 15.0, 12.0)
    arrow(ax, (5.15, 4.15), (5.68, 4.40), COLORS["blue_edge"])
    arrow(ax, (5.15, 3.05), (5.68, 2.85), COLORS["blue_edge"])
    arrow(ax, (5.15, 1.95), (5.68, 1.25), COLORS["blue_edge"])

    # Inspectable gates before the final decision.
    box(ax, 8.90, 3.85, 2.05, 1.20, "Task\nheads", "$p_s, p_c, p_y$\nsupport /\nconflict / style", COLORS["blue"], COLORS["blue_edge"], 14.0, 11.5)
    box(ax, 8.90, 2.25, 2.05, 1.20, "Citation gate", "$\\max_i a_i \\geq \\theta_e$\nselected IDs", COLORS["green"], COLORS["green_edge"], 14.0, 11.5)
    box(ax, 8.90, 0.65, 2.05, 1.20, "Style gate", "$p_y$ observable\nnot a trait claim", COLORS["gold"], COLORS["gold_edge"], 14.0, 11.5)
    arrow(ax, (8.25, 4.45), (8.83, 4.45), COLORS["teal_edge"])
    arrow(ax, (8.25, 2.85), (8.83, 2.85), COLORS["green_edge"])
    arrow(ax, (8.25, 1.25), (8.83, 1.25), COLORS["gold_edge"])

    # Decision diamond and approved-text boundary.
    diamond = Polygon([[11.15, 2.85], [12.50, 4.05], [13.85, 2.85], [12.50, 1.65]], closed=True, facecolor=COLORS["red"], edgecolor=COLORS["red_edge"], linewidth=2.4, zorder=2)
    ax.add_patch(diamond)
    ax.text(12.50, 3.20, "Calibrated\nselection", ha="center", va="center", fontsize=15.0, fontweight="bold", color=COLORS["red_edge"], linespacing=1.0, zorder=3)
    ax.text(12.50, 2.35, "accept / abstain", ha="center", va="center", fontsize=12.5, color=COLORS["ink"], zorder=3)
    arrow(ax, (10.95, 4.45), (11.08, 3.99), COLORS["blue_edge"])
    arrow(ax, (10.95, 2.85), (11.08, 2.85), COLORS["green_edge"])
    arrow(ax, (10.95, 1.25), (11.08, 1.71), COLORS["gold_edge"])
    box(ax, 14.10, 1.92, 1.20, 1.86, "Output", "approved\ntext\n+ IDs", COLORS["input"], COLORS["muted"], 13.0, 12.0, 1.8)
    arrow(ax, (13.88, 2.85), (14.03, 2.85), COLORS["red_edge"], 2.0)

    # The claim boundary is explicit and visually separate from the model path.
    ax.plot([2.75, 10.95], [0.27, 0.27], color=COLORS["muted"], linewidth=1.2, linestyle=(0, (4, 3)), zorder=0)
    ax.text(6.85, 0.03, "Approved text and evidence IDs cross the realization boundary", ha="center", va="bottom", fontsize=12.0, color=COLORS["muted"])

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output.with_suffix(".svg"), format="svg", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = Path(args.output) if args.output else Path(__file__).resolve().parent / "figures/planner_mechanism.pdf"
    draw(output.resolve())
