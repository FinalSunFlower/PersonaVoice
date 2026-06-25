"""实验模块: SOTA 评估脚本集合.

架构位置: 与核心库代码分离的实验脚本集合, 引用 personavoice 主架构,
不包含任何架构本体的实现, 仅负责评估.

核心评估工具 (复用):
- ecapa_evaluator: 真实 ECAPA SECS 评估 (Vocos/Griffin-Lim 声码器)
- wer_evaluator: Whisper-based WER 评估
- utils: 共享工具函数 (数据加载, tokenizer, logger)

SOTA 评估 (顶会 Table 1):
- eval_200_samples: 200 样本统计显著性评估 (PV vs F5-TTS Baseline)
- baseline_external: 外部 SOTA 对比 (CosyVoice / XTTS v2, 1s 极限克隆)
- cfr_analysis: 灾难失败率分析 (CFR 长尾截断)

顶会可视化 (Figure 4/5/6):
- visualize: Pareto 前沿 + 波形/频谱对比 + 注意力热力图
"""
