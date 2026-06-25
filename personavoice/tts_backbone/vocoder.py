"""神经声码器集成: Vocos 高质量波形合成.

架构位置: TTS 骨干子系统的出口, 将 mel 频谱转换为 24kHz 波形.

v10.1: 满血架构, 只保留 Vocos (删除 BigVGAN/Griffin-Lim 备用方案)
- Vocos 是基于 Fourier 的高效神经声码器, 单次前向传播生成波形
- 比 HiFi-GAN 快 10 倍, 质量相当
- 支持从 mel spectrogram 直接生成 24kHz 波形
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class VocosVocoder(nn.Module):
    """Vocos neural vocoder wrapper.

    Converts mel spectrogram to high-quality waveform using Vocos.

    Attributes:
        vocos: Vocos model instance
        sample_rate: Output sample rate (default 24000)
        mel_sample_rate: Sample rate used for mel extraction
    """

    def __init__(
        self,
        model_name: str = "charactr/vocos-mel-24khz",
        sample_rate: int = 24000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        input_n_mels: int = 100,  # F5-TTS/Vocos 统一用 100 mel bins
    ):
        super().__init__()
        self.model_name = model_name
        self.target_sample_rate = sample_rate
        self.device = device
        self._vocos = None
        # Vocos charactr/vocos-mel-24khz 期望 100 mel bins,
        # 项目内部 mel 为 80 bins, 需要可学习的线性投影做通道转换
        self._input_n_mels = input_n_mels
        self._vocos_n_mels = 100  # charactr/vocos-mel-24khz 固定 100 bins
        self._mel_proj: Optional[nn.Linear] = None
        self._load_vocos()
        # 若输入 mel bins 与 Vocos 期望不一致, 建立投影层
        if self._vocos is not None and self._input_n_mels != self._vocos_n_mels:
            self._mel_proj = nn.Linear(
                self._input_n_mels, self._vocos_n_mels
            ).to(self.device)
            nn.init.eye_(self._mel_proj.weight[:, :self._input_n_mels])
            nn.init.zeros_(self._mel_proj.bias)
            self._mel_proj.eval()

    def _load_vocos(self):
        """Load Vocos vocoder (满血架构, 无备用方案)."""
        from vocos import Vocos
        # 优先使用本地缓存, 避免联网
        try:
            from personavoice.common.local_models import (
                setup_offline_mode, is_model_cached
            )
            setup_offline_mode()
            if not is_model_cached(self.model_name):
                logger.info(
                    f"Vocos model '{self.model_name}' not in local cache, "
                    f"downloading from network."
                )
        except ImportError:
            pass
        self._vocos = Vocos.from_pretrained(self.model_name)
        self._vocos = self._vocos.to(self.device)
        self._vocos.eval()
        logger.info(
            f"Vocos vocoder loaded: {self.model_name} "
            f"(output: {self.target_sample_rate}Hz)"
        )

    @property
    def is_available(self) -> bool:
        """Check if vocoder is available."""
        return self._vocos is not None

    @property
    def sample_rate(self) -> int:
        """Output sample rate."""
        return self.target_sample_rate

    @torch.no_grad()
    def mel_to_waveform(
        self,
        mel_spectrogram: torch.Tensor,
        input_sample_rate: int = 16000,
    ) -> torch.Tensor:
        """Convert mel spectrogram to waveform.

        Args:
            mel_spectrogram: Mel spectrogram (batch, n_mels, time) or (n_mels, time)
            input_sample_rate: Sample rate of the mel extraction (for resampling)

        Returns:
            Waveform (batch, samples) or (samples,)
        """
        # Ensure batch dimension
        squeeze_batch = False
        if mel_spectrogram.dim() == 2:
            mel_spectrogram = mel_spectrogram.unsqueeze(0)
            squeeze_batch = True

        mel_spectrogram = mel_spectrogram.to(self.device)

        # Vocos expects (batch, n_mels, time)
        mel_in = mel_spectrogram
        # 若输入 mel bins (80) 与 Vocos 期望 (100) 不一致, 用投影层转换
        if self._mel_proj is not None:
            # (B, n_mels, T) -> (B, T, n_mels) -> proj -> (B, T, 100) -> (B, 100, T)
            mel_in = self._mel_proj(
                mel_in.transpose(1, 2)
            ).transpose(1, 2)
        waveform = self._vocos.decode(mel_in)

        # Resample if needed
        if self.target_sample_rate != input_sample_rate:
            waveform = self._resample(waveform, self.target_sample_rate, input_sample_rate)

        if squeeze_batch:
            waveform = waveform.squeeze(0)

        return waveform

    def _resample(
        self,
        waveform: torch.Tensor,
        orig_freq: int,
        new_freq: int,
    ) -> torch.Tensor:
        """Resample waveform."""
        import torchaudio
        resampler = torchaudio.transforms.Resample(orig_freq, new_freq).to(self.device)
        return resampler(waveform)

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """Forward pass: mel -> waveform."""
        return self.mel_to_waveform(mel_spectrogram)


class VocoderManager:
    """Singleton manager for vocoder instances.

    Ensures only one vocoder instance is loaded per process.
    """

    _instances: Dict[str, VocosVocoder] = {}

    @classmethod
    def get_vocoder(
        cls,
        model_name: str = "charactr/vocos-mel-24khz",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> VocosVocoder:
        """Get or create a vocoder instance."""
        key = f"{model_name}_{device}"
        if key not in cls._instances:
            cls._instances[key] = VocosVocoder(
                model_name=model_name,
                device=device,
            )
        return cls._instances[key]

    @classmethod
    def clear_cache(cls):
        """Clear cached vocoder instances."""
        cls._instances.clear()


def mel_to_waveform(
    mel_spectrogram: torch.Tensor,
    vocoder: Optional[VocosVocoder] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    input_sample_rate: int = 16000,
) -> torch.Tensor:
    """Convenience function to convert mel to waveform.

    Args:
        mel_spectrogram: Mel spectrogram (batch, n_mels, time) or (n_mels, time)
        vocoder: Optional pre-loaded vocoder
        device: Device for computation
        input_sample_rate: Sample rate of mel extraction

    Returns:
        Waveform tensor
    """
    if vocoder is None:
        vocoder = VocoderManager.get_vocoder(device=device)
    return vocoder.mel_to_waveform(mel_spectrogram, input_sample_rate)


def save_audio(
    waveform: torch.Tensor,
    path: str,
    sample_rate: int = 24000,
):
    """Save waveform to audio file.

    Args:
        waveform: Waveform tensor (samples,) or (batch, samples)
        path: Output file path
        sample_rate: Sample rate
    """
    import torchaudio

    # Ensure 2D (channels, samples)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Ensure CPU
    waveform = waveform.cpu()

    torchaudio.save(path, waveform, sample_rate)
    logger.info(f"Audio saved to {path} ({waveform.shape[-1]/sample_rate:.2f}s)")
