"""PersonaVoice 人格提取子包.

架构位置: 人格管线入口, 从聊天记录/结构化回忆提取人格表示.
被 demo API 和 TTS 骨干的 FiLM Adapter 消费.
"""
from personavoice.persona.extractor import (
    extract_persona_emb,
    extract_big_five_traits,
    calibrate_traits,
)

__all__ = [
    "extract_persona_emb",
    "extract_big_five_traits",
    "calibrate_traits",
]
