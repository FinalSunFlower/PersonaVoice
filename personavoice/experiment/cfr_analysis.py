"""CFR (Catastrophic Failure Rate) 计算.

创新点: 在1s极限信息缺失下,创新模块的主要贡献不是均值优化,
而是截断长尾灾难分布,显著提升系统鲁棒性.

灾难边界定义:
    - 语义灾难边界 (Semantic Disaster): WER > 50%
    - 音色崩塌边界 (Timbre Collapse): SECS < 0.25

Usage:
    python -m personavoice.experiment.cfr_analysis
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

# 灾难边界
WER_DISASTER_THRESHOLD = 0.50  # WER > 50% = 语义灾难
SECS_DISASTER_THRESHOLD = 0.25  # SECS < 0.25 = 音色崩塌


def load_ablation_data() -> Dict[str, List[Dict]]:
    """加载消融实验数据."""
    path = RESULTS_DIR / "ablation_200_samples.json"
    with open(str(path), "r", encoding="utf-8") as f:
        data = json.load(f)

    # 配置名映射 (中文友好)
    config_map = {
        "full_v8": "Full v8.0 (完整)",
        "wo_td_cfg": "-TD-CFG (无时间衰减引导)",
        "wo_ibop": "-IBOP (无正交投影)",
        "wo_sbm": "-SBM (无薛定谔桥)",
    }

    result = {}
    for key, label in config_map.items():
        if key in data:
            result[label] = data[key]
    return result


def compute_cfr(samples: List[Dict]) -> Dict[str, float]:
    """计算单个配置的CFR指标.

    Args:
        samples: 样本列表,每个含wer和secs字段

    Returns:
        dict: CFR指标
    """
    n = len(samples)
    if n == 0:
        return {}

    wer_values = [s["wer"] for s in samples]
    secs_values = [s["secs"] for s in samples]

    # 灾难率
    wer_disaster = sum(1 for w in wer_values if w > WER_DISASTER_THRESHOLD)
    secs_disaster = sum(1 for s in secs_values if s < SECS_DISASTER_THRESHOLD)
    combined_disaster = sum(
        1 for w, s in zip(wer_values, secs_values)
        if w > WER_DISASTER_THRESHOLD or s < SECS_DISASTER_THRESHOLD
    )

    # 均值
    wer_mean = np.mean(wer_values)
    secs_mean = np.mean(secs_values)

    # 标准差 (鲁棒性指标)
    wer_std = np.std(wer_values)
    secs_std = np.std(secs_values)

    # 分位数 (长尾分析)
    wer_p90 = np.percentile(wer_values, 90)
    wer_p95 = np.percentile(wer_values, 95)
    secs_p10 = np.percentile(secs_values, 10)
    secs_p05 = np.percentile(secs_values, 5)

    return {
        "n_samples": n,
        "wer_mean": float(wer_mean),
        "wer_std": float(wer_std),
        "wer_p90": float(wer_p90),
        "wer_p95": float(wer_p95),
        "wer_disaster_rate": float(wer_disaster / n * 100),
        "wer_disaster_count": wer_disaster,
        "secs_mean": float(secs_mean),
        "secs_std": float(secs_std),
        "secs_p10": float(secs_p10),
        "secs_p05": float(secs_p05),
        "secs_disaster_rate": float(secs_disaster / n * 100),
        "secs_disaster_count": secs_disaster,
        "combined_disaster_rate": float(combined_disaster / n * 100),
        "combined_disaster_count": combined_disaster,
    }


def run_cfr_analysis():
    """运行CFR分析."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("=" * 70)
    logger.info("CFR (Catastrophic Failure Rate) Analysis")
    logger.info("灾难边界: WER>50%% (语义灾难), SECS<0.25 (音色崩塌)")
    logger.info("=" * 70)

    data = load_ablation_data()
    if not data:
        logger.error("No ablation data found!")
        return

    # 计算每个配置的CFR
    results = {}
    for config_name, samples in data.items():
        results[config_name] = compute_cfr(samples)

    # 打印对比表
    logger.info("\n" + "=" * 90)
    logger.info("CFR对比表 (200 samples)")
    logger.info("=" * 90)
    header = f"{'Config':<30} {'WER Mean':<10} {'WER CFR%':<10} {'SECS Mean':<10} {'SECS CFR%':<10} {'Combined CFR%':<14}"
    logger.info(header)
    logger.info("-" * 90)

    for config, metrics in results.items():
        line = (
            f"{config:<30} "
            f"{metrics['wer_mean']:<10.4f} "
            f"{metrics['wer_disaster_rate']:<10.1f} "
            f"{metrics['secs_mean']:<10.4f} "
            f"{metrics['secs_disaster_rate']:<10.1f} "
            f"{metrics['combined_disaster_rate']:<14.1f}"
        )
        logger.info(line)

    logger.info("-" * 90)

    # 长尾分析
    logger.info("\n" + "=" * 90)
    logger.info("长尾分布分析 (分位数)")
    logger.info("=" * 90)
    header = f"{'Config':<30} {'WER P90':<10} {'WER P95':<10} {'SECS P10':<10} {'SECS P05':<10}"
    logger.info(header)
    logger.info("-" * 90)

    for config, metrics in results.items():
        line = (
            f"{config:<30} "
            f"{metrics['wer_p90']:<10.4f} "
            f"{metrics['wer_p95']:<10.4f} "
            f"{metrics['secs_p10']:<10.4f} "
            f"{metrics['secs_p05']:<10.4f}"
        )
        logger.info(line)

    logger.info("-" * 90)

    # 灾难率改善 (相对Full v8.0)
    if "Full v8.0 (完整)" in results:
        full_cfr = results["Full v8.0 (完整)"]["combined_disaster_rate"]
        logger.info("\n" + "=" * 90)
        logger.info("灾难率改善 (相对Full v8.0)")
        logger.info("=" * 90)
        for config, metrics in results.items():
            if config == "Full v8.0 (完整)":
                continue
            delta = metrics["combined_disaster_rate"] - full_cfr
            logger.info(
                f"  {config:<30} CFR={metrics['combined_disaster_rate']:.1f}% "
                f"(vs Full={full_cfr:.1f}%, delta=+{delta:.1f}%)"
            )

    # 保存结果
    output_path = RESULTS_DIR / "cfr_analysis.json"
    with open(str(output_path), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nCFR results saved to: {output_path}")

    # 生成可视化
    try:
        _plot_cfr_comparison(results)
    except Exception as e:
        logger.warning(f"Plot failed: {e}")

    return results


def _plot_cfr_comparison(results: Dict):
    """生成CFR对比图."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = list(results.keys())
    wer_cfrs = [results[c]["wer_disaster_rate"] for c in configs]
    secs_cfrs = [results[c]["secs_disaster_rate"] for c in configs]
    combined_cfrs = [results[c]["combined_disaster_rate"] for c in configs]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 简化配置名
    short_names = [c.split("(")[0].strip() for c in configs]

    # WER CFR
    bars1 = axes[0].bar(short_names, wer_cfrs, color=["#2ecc71", "#e74c3c", "#e67e22", "#9b59b6"])
    axes[0].set_title("Semantic Disaster Rate (WER > 50%)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Disaster Rate (%)")
    axes[0].set_ylim(0, max(wer_cfrs) * 1.2 + 5)
    for bar, val in zip(bars1, wer_cfrs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=30)

    # SECS CFR
    bars2 = axes[1].bar(short_names, secs_cfrs, color=["#2ecc71", "#e74c3c", "#e67e22", "#9b59b6"])
    axes[1].set_title("Timbre Collapse Rate (SECS < 0.25)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Disaster Rate (%)")
    axes[1].set_ylim(0, max(secs_cfrs) * 1.2 + 5)
    for bar, val in zip(bars2, secs_cfrs):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=30)

    # Combined CFR
    bars3 = axes[2].bar(short_names, combined_cfrs, color=["#2ecc71", "#e74c3c", "#e67e22", "#9b59b6"])
    axes[2].set_title("Combined Disaster Rate", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Disaster Rate (%)")
    axes[2].set_ylim(0, max(combined_cfrs) * 1.2 + 5)
    for bar, val in zip(bars3, combined_cfrs):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
    axes[2].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    output_path = RESULTS_DIR / "cfr_comparison.png"
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"CFR plot saved to: {output_path}")


if __name__ == "__main__":
    run_cfr_analysis()
