"""MicroAug: 正交环境子流形 (v9.1 精简版).

架构位置: TTS 骨干的条件预处理层, 对参考 speaker_emb 进行环境解缠.

v9.1 核心组件 (消融实验验证有效):
- OrthogonalEnvironmentSubmanifold (OES): 正交环境子流形, 分离录音环境与音色

已移除 (消融实验证实无效):
- IBOP (CrossManifoldAttention/NullSpaceProjector): p=0.85/0.51, Cohen's d<0.05
- AM-ODE (AcousticMomentumODE): 零初始化未贡献
- MINE (MINEEstimator): 已废弃
"""

from personavoice.microaug.cross_manifold_refiner import (
    OrthogonalEnvironmentSubmanifold,
)

__all__ = [
    "OrthogonalEnvironmentSubmanifold",
]
