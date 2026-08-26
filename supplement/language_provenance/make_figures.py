"""Publication figures for the language extension analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {"blue": "#0F4D92", "green": "#8BCF8B", "red": "#B64342", "gray": "#767676"}


def style():
    plt.rcParams.update({"font.family": ["DejaVu Sans", "sans-serif"], "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8, "legend.frameon": False, "svg.fonttype": "none"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--output", default="figures")
    args = parser.parse_args()
    style()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    curve = json.loads((Path(args.artifacts) / "budget_curve.json").read_text(encoding="utf-8"))
    audit = json.loads((Path(args.artifacts) / "failure_audit.json").read_text(encoding="utf-8"))
    budgets = list(curve["budgets"])
    x = list(range(len(budgets)))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55), constrained_layout=True)
    for name, color, key in [("Constrained", PALETTE["blue"], "constrained"), ("No provenance", PALETTE["red"], "unconstrained")]:
        axes[0].plot(x, [curve["budgets"][b][key]["positive_coverage"] * 100 for b in budgets], marker="o", lw=2, color=color, label=name)
        axes[1].plot(x, [curve["budgets"][b][key]["false_acceptance"] * 100 for b in budgets], marker="o", lw=2, color=color, label=name)
    for ax, ylabel in zip(axes, ["Positive coverage (%)", "False acceptance among accepted (%)"]):
        ax.set_xticks(x, budgets); ax.set_xlabel("Evidence items"); ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=.18); ax.legend()
    fig.savefig(out / "evidence_budget_tradeoff.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out / "evidence_budget_tradeoff.png", dpi=300, bbox_inches="tight")
    tags = audit["counts"]
    labels = [k.split(":", 1)[1].replace("_", " ") for k in tags]
    fig, ax = plt.subplots(figsize=(5.6, 2.6), constrained_layout=True)
    bars = ax.barh(labels, list(tags.values()), color=[PALETTE["red"] if "acceptance" in k else PALETTE["gray"] for k in tags])
    ax.bar_label(bars, padding=3, fontsize=8); ax.set_xlabel("Tagged rows (pooled three seeds)")
    ax.grid(axis="x", alpha=.18); fig.savefig(out / "failure_taxonomy.pdf", dpi=300, bbox_inches="tight"); fig.savefig(out / "failure_taxonomy.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
