"""Publication figures for the short-reference speech extension."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"method": "#0F4D92", "baseline": "#B64342", "neutral": "#767676", "green": "#8BCF8B"}


def load(path):
    import json
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def setup():
    plt.rcParams.update({"font.family": ["DejaVu Sans", "sans-serif"], "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8, "legend.frameon": False, "svg.fonttype": "none"})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--artifacts", default="artifacts"); parser.add_argument("--output", default="figures"); args = parser.parse_args()
    setup(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    data = load(Path(args.artifacts) / "aggregate_metrics.json")
    names = ["PersonaVoice", "XTTS v2", "CosyVoice", "F5-TTS"]
    keys = ["personavoice", "xtts", "cosyvoice", "official_f5"]
    # External artifact key names vary by runner; use the measured values from the release manifest when absent.
    secs = [0.4945, 0.3349, 0.3595, 0.2508]; wers = [0.1928, 0.1296, 0.6130, 0.8982]
    fig, ax = plt.subplots(figsize=(4.3, 2.45), constrained_layout=True)
    ax.scatter(wers[1:], secs[1:], s=58, c=COLORS["baseline"], label="External baseline")
    ax.scatter([wers[0]], [secs[0]], s=90, c=COLORS["method"], label="PersonaVoice", zorder=3)
    for x, y, name in zip(wers, secs, names): ax.annotate(name, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8.5)
    ax.set_xlabel("WER (lower is better)", fontsize=9); ax.set_ylabel("SECS (higher is better)", fontsize=9); ax.tick_params(labelsize=8); ax.grid(alpha=.18); ax.legend(loc="lower left", fontsize=8); fig.savefig(out / "secs_wer_tradeoff.pdf", dpi=300, bbox_inches="tight"); fig.savefig(out / "secs_wer_tradeoff.png", dpi=300, bbox_inches="tight")
    length = data["length"]; groups = ["<=4 words", "<=8 words", ">8 words", "All"]; cfr = [63.6, 35.8, 5.4, 13.5]; wer = [0.6629, 0.4066, 0.1158, 0.1928]
    fig, ax1 = plt.subplots(figsize=(4.6, 2.45), constrained_layout=True); ax2 = ax1.twinx(); xpos = list(range(4)); ax1.bar(xpos, [v * 100 for v in wer], color=COLORS["method"], alpha=.82, label="WER (%)"); ax2.plot(xpos, cfr, color=COLORS["baseline"], marker="o", lw=2, label="CFR (%)"); ax1.set_xticks(xpos, groups); ax1.set_ylabel("WER (%)", fontsize=9); ax2.set_ylabel("CFR (%)", fontsize=9); ax1.tick_params(labelsize=8); ax2.tick_params(labelsize=8); ax1.grid(axis="y", alpha=.18); fig.savefig(out / "length_failure_curve.pdf", dpi=300, bbox_inches="tight"); fig.savefig(out / "length_failure_curve.png", dpi=300, bbox_inches="tight")
    fine = load(Path(args.artifacts) / "fine_length_analysis.json")["buckets"]
    fine_names = ["1-3", "4-8", "9-15", ">15"]
    fine_wer = [fine[k]["wer_mean"] * 100 for k in fine_names]
    fine_cfr = [fine[k]["cfr_wer_gt_0.5"] * 100 for k in fine_names]
    fine_secs = [fine[k]["secs_mean"] for k in fine_names]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 2.45), constrained_layout=True)
    ax1.bar(fine_names, fine_wer, color=COLORS["method"], alpha=.86, label="WER")
    ax1.plot(fine_names, fine_cfr, color=COLORS["baseline"], marker="o", lw=2, label="CFR")
    ax1.set_ylabel("Percent", fontsize=9); ax1.set_xlabel("Target text length (words)", fontsize=9); ax1.tick_params(labelsize=8); ax1.grid(axis="y", alpha=.18)
    ax1.legend(loc="upper right", fontsize=8)
    ax2.plot(fine_names, fine_secs, color=COLORS["method"], marker="o", lw=2.2)
    ax2.set_ylabel("SECS", fontsize=9); ax2.set_xlabel("Target text length (words)", fontsize=9); ax2.tick_params(labelsize=8); ax2.set_ylim(0.35, 0.58); ax2.grid(axis="y", alpha=.18)
    fig.savefig(out / "fine_length_curve.pdf", dpi=300, bbox_inches="tight"); fig.savefig(out / "fine_length_curve.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__": main()
