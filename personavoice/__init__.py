"""PersonaVoice: 1秒极限语音克隆核心架构 (v10.4.8 满血 SOTA).

本模块是 PersonaVoice 项目的入口, 暴露 v10.4.8 满血 SOTA 核心架构组件.

v10.4.8 核心架构 (LAAG + OES + FiLM + F5 官方流程):
1. F5-TTS 预训练骨干 (冻结) + FiLM Adapter (persona/emotion 注入)
2. OES (正交环境子流形): env_scale=0.1 渐进初始化 (解决 1111.mp3 SECS 灾难)
3. CEAG (交叉熵注意力引导): 带 padding mask, v10.4.7 实验验证对当前基线无显著效果, disabled
4. LAAG (长度自适应生成): 动态 Chunking + 动态 CFG + 动态 FiLM 激活
5. Silero VAD + RMS 归一化: 前端精准预处理 (替代老旧 WebRTC VAD)
6. Persona 管线: BERT 编码聊天记录 → Big Five → persona_emb

v10.3+ 时长计算: 使用 F5-TTS 官方 infer_batch_process 内部公式
v10.4.8 诚实定位: SECS SOTA (+47.6% vs XTTS v2), 短文本 WER 为 Future Work

架构位置: 顶层包入口, 对外暴露核心骨干与声码器.
"""

from personavoice.tts_backbone.f5_pretrained_backbone import (
    F5TTSPretrainedBackbone,
    PersonaEmotionFiLM,
    load_pretrained_f5tts_backbone,
)
from personavoice.tts_backbone.vocoder import VocosVocoder
from personavoice.microaug.cross_manifold_refiner import (
    OrthogonalEnvironmentSubmanifold,
)
from personavoice.config import SOTA_CONFIG, get_config, get_inference_kwargs

__version__ = "10.4.8"

__all__ = [
    "F5TTSPretrainedBackbone",
    "PersonaEmotionFiLM",
    "load_pretrained_f5tts_backbone",
    "VocosVocoder",
    "OrthogonalEnvironmentSubmanifold",
    "SOTA_CONFIG",
    "get_config",
    "get_inference_kwargs",
]
