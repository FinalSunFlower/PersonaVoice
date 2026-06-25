"""实验脚本共享工具.

架构位置: 实验模块的基础设施层, 提供 v7.x 实验脚本共用的
数据加载/tokenizer/logger 等工具, 从历史消融模块抽取, 保持实验目录整洁.

提供公共辅助函数 (数据加载, tokenizer, 日志配置), 被 v7.x 实验脚本使用.
从历史消融模块抽取, 保持实验目录整洁, 避免跨版本依赖.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logger(name: str = "experiment") -> logging.Logger:
    """Configure and return a logger with standard formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    return logger


def load_test_samples(n_samples: int = 20, seed: int = 42) -> List[Dict]:
    """Load LibriTTS test samples for voice cloning evaluation.

    Each sample contains:
        - mel_ref: 1s reference mel (94, 100)
        - mel_full: Full reference mel (T, 100)
        - target_text: Target text to synthesize
        - speaker_emb: ECAPA speaker embedding (192,)

    Args:
        n_samples: Number of samples to load
        seed: Random seed for reproducible sample selection

    Returns:
        List of sample dictionaries
    """
    data_path = PROJECT_ROOT / "data" / "libritts_processed" / "libritts_devclean_processed.pt"
    data = torch.load(str(data_path), weights_only=False)

    mel = data["tensors"]["mel"]  # (N, 100, T)
    ecapa_emb = data["tensors"]["ecapa_emb"]  # (N, 192)
    texts = data["metadata"]["texts"]

    samples = []
    n_total = mel.shape[0]
    rng = np.random.RandomState(seed)
    indices = rng.choice(n_total, min(n_samples, n_total), replace=False)

    for idx in indices:
        idx = int(idx)
        mel_full = mel[idx]  # (100, T)
        if mel_full.dim() == 2:
            mel_full = mel_full.transpose(0, 1)  # (T, 100)

        # 1s reference (first 94 frames at 24kHz hop=256)
        mel_ref = mel_full[:94, :]  # (94, 100)

        samples.append({
            "mel_ref": mel_ref,
            "mel_full": mel_full,
            "target_text": texts[idx] if idx < len(texts) else "",
            "speaker_emb": ecapa_emb[idx],
            "idx": idx,
        })

    logging.getLogger(__name__).info(
        f"Loaded {len(samples)} test samples (ref=1.0s, ref_samples=94)"
    )
    return samples


def load_tokenizer():
    """Load F5-TTS tokenizer for text-to-token conversion.

    Returns:
        SimpleTokenizer: Object with .encode(text) -> token tensor
    """
    from f5_tts.api import F5TTS

    class SimpleTokenizer:
        def __init__(self):
            self._f5tts = F5TTS()
            self.vocab_char_map = self._f5tts.ema_model.vocab_char_map

        def encode(self, text: str):
            from f5_tts.model.utils import list_str_to_idx
            if isinstance(text, str):
                text = [text]
            return list_str_to_idx(text, self.vocab_char_map)[0]

    return SimpleTokenizer()


def load_backbone(device: str = "cuda", use_film: bool = True):
    """Load the pretrained F5-TTS backbone with PersonaVoice adapters.

    Args:
        device: Target device
        use_film: Enable FiLM adapter

    Returns:
        backbone: F5TTSPretrainedBackbone instance
        stats: Weight loading statistics
    """
    from personavoice.tts_backbone.f5_pretrained_backbone import load_pretrained_f5tts_backbone
    return load_pretrained_f5tts_backbone(
        device=device,
        use_film=use_film,
    )
