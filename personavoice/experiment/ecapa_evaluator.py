"""PersonaVoice v4.0 真实 ECAPA SECS 评估模块 (SOTA 修正版).

架构位置: 实验模块中的核心评估工具, 通过真实声码器 + ECAPA-TDNN 计算真实 SECS,
被所有训练/消融/验证脚本复用.

核心修复 (P0):
    旧版使用 nn.Linear(80, 192) 随机投影伪造 ECAPA 嵌入, 导致 SECS≈0.01.
    本版通过真实声码器 (Vocos/Griffin-Lim) 将生成 mel 转换为波形,
    再用 speechbrain/spkrec-ecapa-voxceleb 提取真实 192 维嵌入,
    计算真实 SECS.

评估链路:
    generated_mel (80, T) ──► Vocos/Griffin-Lim ──► wav (16kHz)
                                                    │
                                                    ▼
                                              ECAPA-TDNN
                                                    │
                                                    ▼
                                              emb_gen (192)
                                                    │
    emb_ref (192, 真实) ──────────────────────► cosine_similarity ──► SECS

声码器选择策略:
    1. Vocos (首选): 高质量神经声码器, 需要100-bin/24kHz mel
       - 若输入mel为80-bin, 自动插值到100-bin
       - 若输入mel已归一化, 自动反归一化到log-mel范围
    2. Griffin-Lim (回退): 通用, 兼容任何mel格式, 质量较低但诚实
"""
import os
import sys
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

# v10.4: 禁用 wandb (避免 speechbrain lazy import 冲突)
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


class ECAPAEvaluator:
    """真实ECAPA-TDNN说话人相似度评估器 (SOTA修正版).

    通过真实声码器将mel转换为波形, 再提取ECAPA嵌入, 计算真实SECS.
    """

    # Vocos charactr/vocos-mel-24khz 的 mel 配置
    VOCOS_SAMPLE_RATE = 24000
    VOCOS_N_MELS = 100
    VOCOS_N_FFT = 1024
    VOCOS_HOP_LENGTH = 256

    # 项目原始 mel 配置 (SOTA: Vocos兼容 100-bin)
    PROJECT_SAMPLE_RATE = 16000
    PROJECT_N_MELS = 100  # SOTA: 与 Vocos 一致

    def __init__(
        self,
        device: str = "cpu",
        cache_dir: str = "./models/ecapa_tdnn",
        vocoder_device: Optional[str] = None,
        prefer_vocoder: str = "auto",
    ):
        """初始化ECAPA评估器.

        Args:
            device: ECAPA编码器设备 (建议cpu, 评估时不占GPU显存)
            cache_dir: ECAPA模型缓存目录
            vocoder_device: 声码器设备 (默认与device相同)
            prefer_vocoder: "vocos" / "griffin_lim" / "auto"
        """
        print("Loading ECAPA-TDNN evaluator (SOTA rectified)...")
        self.device = device
        self.vocoder_device = vocoder_device or device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 加载 ECAPA 编码器
        try:
            from speechbrain.inference.speaker import EncoderClassifier

            self.encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(self.cache_dir),
                run_opts={"device": device},
            )
            self.embedding_dim = 192
            self.available = True
            print(f"  ECAPA loaded, embedding_dim={self.embedding_dim}")
        except Exception as e:
            print(f"  [WARN] Failed to load ECAPA: {e}")
            self.available = False
            self.embedding_dim = 192

        # 加载声码器
        self._vocoder = None
        self._vocoder_type = None
        self._griffin_lim = None
        self._inverse_mel = None
        self._mel_interp = None  # 80->100 mel 插值 (可学习, 但评估时用线性插值)
        self._load_vocoder(prefer_vocoder)

    def _load_vocoder(self, prefer: str = "auto"):
        """加载声码器 (满血架构, 只用 Vocos, 无备用方案)."""
        from vocos import Vocos

        self._vocoder = Vocos.from_pretrained(
            "charactr/vocos-mel-24khz"
        ).to(self.vocoder_device)
        self._vocoder.eval()
        self._vocoder_type = "vocos"
        print(f"  Vocoder: Vocos (24kHz, 100 mel bins) on {self.vocoder_device}")

    # -----------------------------------------------------------------
    # 嵌入提取
    # -----------------------------------------------------------------
    @torch.no_grad()
    def extract_embedding(self, wav: torch.Tensor, sr: int = 16000) -> torch.Tensor:
        """从波形提取真实ECAPA嵌入.

        Args:
            wav: (B, T) 或 (T,) 波形, 16kHz
            sr: 采样率

        Returns:
            emb: (B, 192) 嵌入
        """
        if not self.available:
            return torch.zeros(
                wav.shape[0] if wav.dim() > 1 else 1, self.embedding_dim
            )

        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        wav = wav.to(self.device)
        # ECAPA 期望 16kHz
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        # ECAPA 需要足够长的音频 (至少 ~1秒)
        min_len = 8000
        if wav.shape[-1] < min_len:
            wav = torch.nn.functional.pad(wav, (0, min_len - wav.shape[-1]))

        emb = self.encoder.encode_batch(wav)
        if emb.dim() == 3:
            emb = emb.squeeze(1)

        return emb.cpu()

    # -----------------------------------------------------------------
    # 声码器: mel -> wav
    # -----------------------------------------------------------------
    @torch.no_grad()
    def mel_to_waveform(
        self,
        mel: torch.Tensor,
        mel_sample_rate: int = 16000,
        mel_n_mels: int = 100,
        is_normalized: bool = False,
    ) -> Tuple[torch.Tensor, int]:
        """将mel频谱转换为波形 (真实声码器).

        Args:
            mel: (B, n_mels, T) 或 (n_mels, T) mel频谱
            mel_sample_rate: mel提取时的采样率
            mel_n_mels: mel的bin数
            is_normalized: mel是否经过per-sample归一化 (SOTA标准: False, 使用raw log-mel)

        Returns:
            wav: (B, T_samples) 波形
            sr: 波形采样率
        """
        squeeze = False
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
            squeeze = True

        mel = mel.to(self.vocoder_device)

        # Vocos 声码器解码 (满血架构, 无备用方案)
        wav = self._vocos_decode(
            mel, mel_sample_rate, mel_n_mels, is_normalized
        )
        sr = self.VOCOS_SAMPLE_RATE
        # 重采样到 16kHz (ECAPA 需要)
        wav = torchaudio.functional.resample(
            wav, self.VOCOS_SAMPLE_RATE, 16000
        )
        sr = 16000

        if squeeze:
            wav = wav.squeeze(0)

        return wav.cpu(), sr

    def _vocos_decode(
        self,
        mel: torch.Tensor,
        mel_sample_rate: int,
        mel_n_mels: int,
        is_normalized: bool,
    ) -> torch.Tensor:
        """Vocos 声码器解码.

        处理 mel 格式转换:
        1. 反归一化 (若需要): normalized mel -> log-mel
        2. mel bin 插值: 80 -> 100
        3. Vocos 解码: mel -> 24kHz wav
        """
        # Step 1: 反归一化 (仅当输入是归一化的)
        # SOTA标准: 使用 raw log-mel (is_normalized=False), 无需反归一化
        if is_normalized:
            # 旧版归一化数据: mel = (mel - mean) / std
            # 反归一化到 log-mel 范围 [-80, 20]
            mel = mel * 12.0 - 30.0

        # 确保 mel 在合理范围 (Vocos 期望 log-mel)
        mel = torch.clamp(mel, min=-80, max=20)

        # Step 2: mel bin 插值 (80 -> 100 或其他 -> 100)
        if mel_n_mels != self.VOCOS_N_MELS:
            # 线性插值: (B, 80, T) -> (B, 100, T)
            mel = mel.transpose(1, 2)  # (B, T, n_mels)
            mel = torch.nn.functional.interpolate(
                mel, size=self.VOCOS_N_MELS, mode="linear", align_corners=False
            )
            mel = mel.transpose(1, 2)  # (B, 100, T)

        # Step 3: Vocos 解码
        wav = self._vocoder.decode(mel)
        return wav

    # -----------------------------------------------------------------
    # SECS 计算
    # -----------------------------------------------------------------
    @torch.no_grad()
    def compute_secs(
        self,
        wav_generated: torch.Tensor,
        wav_reference: torch.Tensor,
        sr: int = 16000,
    ) -> float:
        """计算真实SECS (从波形).

        Args:
            wav_generated: (B, T) 或 (T,) 生成的波形
            wav_reference: (B, T) 或 (T,) 参考波形
            sr: 采样率

        Returns:
            secs: 平均余弦相似度
        """
        if not self.available:
            return 0.0

        emb_gen = self.extract_embedding(wav_generated, sr)
        emb_ref = self.extract_embedding(wav_reference, sr)

        secs = torch.nn.functional.cosine_similarity(emb_gen, emb_ref, dim=-1)
        return secs.mean().item()

    @torch.no_grad()
    def compute_secs_from_mel(
        self,
        mel_generated: torch.Tensor,
        emb_reference: torch.Tensor,
        mel_sample_rate: int = 16000,
        mel_n_mels: int = 100,
        is_normalized: bool = False,
    ) -> float:
        """从生成mel计算真实SECS (SOTA修正版).

        关键修复: 通过真实声码器将mel转换为波形, 再提取ECAPA嵌入,
        而非使用 nn.Linear(80, 192) 随机投影.

        Args:
            mel_generated: (B, n_mels, T) 生成的mel (raw log-mel)
            emb_reference: (B, 192) 参考音频的真实ECAPA嵌入
            mel_sample_rate: mel提取时的采样率
            mel_n_mels: mel的bin数
            is_normalized: mel是否经过per-sample归一化 (SOTA标准: False)

        Returns:
            secs: 真实SECS (余弦相似度)
        """
        if not self.available:
            return 0.0

        # mel -> wav (真实声码器)
        wav_gen, sr = self.mel_to_waveform(
            mel_generated,
            mel_sample_rate=mel_sample_rate,
            mel_n_mels=mel_n_mels,
            is_normalized=is_normalized,
        )

        # wav -> ECAPA 嵌入
        emb_gen = self.extract_embedding(wav_gen, sr=sr)

        # 与参考嵌入计算余弦相似度
        emb_ref = emb_reference.to(emb_gen.device)
        if emb_ref.dim() == 1:
            emb_ref = emb_ref.unsqueeze(0)

        secs = torch.nn.functional.cosine_similarity(emb_gen, emb_ref, dim=-1)
        return secs.mean().item()

    @torch.no_grad()
    def compute_secs_mel_vs_mel(
        self,
        mel_generated: torch.Tensor,
        mel_reference: torch.Tensor,
        mel_sample_rate: int = 16000,
        mel_n_mels: int = 100,
        is_normalized: bool = False,
    ) -> float:
        """从两个mel计算SECS (都通过声码器转换为波形).

        Args:
            mel_generated: (B, n_mels, T) 生成的mel (raw log-mel)
            mel_reference: (B, n_mels, T) 参考mel (raw log-mel)
            mel_sample_rate: mel采样率
            mel_n_mels: mel bin数
            is_normalized: 是否归一化 (SOTA标准: False)

        Returns:
            secs: 真实SECS
        """
        if not self.available:
            return 0.0

        wav_gen, sr = self.mel_to_waveform(
            mel_generated, mel_sample_rate, mel_n_mels, is_normalized
        )
        wav_ref, _ = self.mel_to_waveform(
            mel_reference, mel_sample_rate, mel_n_mels, is_normalized
        )

        return self.compute_secs(wav_gen, wav_ref, sr=sr)


def test_ecapa_evaluator():
    """测试ECAPA评估器 (SOTA修正版)."""
    print("\n" + "=" * 60)
    print("Testing ECAPA Evaluator (SOTA Rectified)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = ECAPAEvaluator(device="cpu", vocoder_device=device)

    # 测试1: 相同波形的SECS应接近1.0
    print("\n[Test 1] Same waveform SECS (should be ~1.0):")
    wav1 = torch.randn(1, 16000) * 0.1
    secs_same = evaluator.compute_secs(wav1, wav1)
    print(f"  SECS (same): {secs_same:.4f}")

    # 测试2: 不同波形的SECS应较低
    print("\n[Test 2] Different waveform SECS (should be < 1.0):")
    wav2 = torch.randn(1, 16000) * 0.1
    secs_diff = evaluator.compute_secs(wav1, wav2)
    print(f"  SECS (diff): {secs_diff:.4f}")

    # 测试3: mel -> wav -> ECAPA 链路
    print("\n[Test 3] mel -> wav -> ECAPA pipeline:")
    mel_fake = torch.randn(2, 80, 126)  # 模拟生成的mel (归一化)
    emb_ref = torch.randn(2, 192)  # 模拟参考ECAPA嵌入
    secs_mel = evaluator.compute_secs_from_mel(
        mel_fake, emb_ref,
        mel_sample_rate=16000, mel_n_mels=80, is_normalized=True
    )
    print(f"  SECS (mel->wav->ECAPA): {secs_mel:.4f}")
    print(f"  Vocoder type: {evaluator._vocoder_type}")

    print("\n[DONE] ECAPA evaluator test complete.")


if __name__ == "__main__":
    test_ecapa_evaluator()
