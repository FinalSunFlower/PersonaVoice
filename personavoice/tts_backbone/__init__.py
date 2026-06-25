"""TTS 骨干: F5-TTS 预训练骨干 + 适配器 + 声码器 (v10.4.8 满血版).

架构位置: 语音生成核心, 接收条件编码器输出的嵌入, 通过 Flow Matching
+ DiT 生成 mel 频谱, 再经 Vocos 声码器合成波形.

v10.4.8 核心组件:
- F5TTSPretrainedBackbone: F5-TTS 预训练骨干 + FiLM + OES
- ceag_sampler: CEAG 采样器 (交叉熵注意力引导, 升级自 IEAG; 当前 disabled)
- laag_generator: LAAG 长度自适应生成 (动态 Chunking + 动态 CFG + 动态 FiLM)
- VocosVocoder: 神经声码器

v10.3+ 时长计算: 使用 F5-TTS 官方公式 (infer_batch_process 内部自动)
"""

from personavoice.tts_backbone.f5_pretrained_backbone import (
    F5TTSPretrainedBackbone,
    PersonaEmotionFiLM,
    load_pretrained_f5tts_backbone,
)
from personavoice.tts_backbone.vocoder import VocosVocoder

__all__ = [
    "F5TTSPretrainedBackbone",
    "PersonaEmotionFiLM",
    "load_pretrained_f5tts_backbone",
    "VocosVocoder",
]
