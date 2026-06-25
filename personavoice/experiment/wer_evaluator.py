"""WER 评估器: 基于 Whisper (SOTA 真实实现).

架构位置: 实验模块中的核心评估工具, 使用本地缓存的 Whisper 模型计算 WER,
被所有训练/消融/验证脚本复用, 与 ecapa_evaluator 共同构成双指标评估体系.

使用本地缓存的 Whisper 模型计算词错误率 (WER).
在 1s 极限克隆下, WER 比 SECS 更能暴露咬字崩溃问题.

设计:
    - 优先使用本地缓存的 whisper-large-v3-turbo
    - 在 CPU 上运行 Whisper (节省 GPU 显存给训练)
    - 支持从 mel 生成 wav 再识别, 或直接从 wav 识别
    - WER 计算: 基于编辑距离的标准实现

使用方式:
    evaluator = WEREvaluator(device="cpu")
    wer = evaluator.compute_wer(wav_gen, target_text)
"""

import os
import torch
import torchaudio
import numpy as np
from typing import Optional, List, Dict
import logging
import re

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

logger = logging.getLogger(__name__)


def edit_distance(s1: List[str], s2: List[str]) -> int:
    """计算两个词序列的编辑距离."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]


def compute_wer(reference: str, hypothesis: str) -> float:
    """计算词错误率(WER).

    WER = (S + D + I) / N
    其中 S=替换, D=删除, I=插入, N=参考词数
    """
    # 标准化文本: 小写, 去标点, 分词
    def normalize(text):
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    ref_words = normalize(reference)
    hyp_words = normalize(hypothesis)

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    # 计算WER (简化版: 编辑距离/参考词数)
    dist = edit_distance(ref_words, hyp_words)
    wer = dist / len(ref_words)
    return min(wer, 1.0)  # 上限1.0


class WEREvaluator:
    """Whisper-based WER evaluator (本地模型, CPU运行)."""

    def __init__(
        self,
        device: str = "cpu",
        model_name: str = "openai/whisper-large-v3-turbo",
    ):
        """初始化WER评估器.

        Args:
            device: "cpu" 或 "cuda" (建议cpu, 节省GPU显存)
            model_name: Whisper模型名 (使用本地缓存)
        """
        self.device = device
        self.model_name = model_name
        self.available = False

        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            logger.info(f"Loading Whisper model: {model_name} (local cache)")

            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()

            self.available = True
            logger.info(f"Whisper loaded on {device}, params={sum(p.numel() for p in self.model.parameters()):,}")

        except Exception as e:
            logger.warning(f"Whisper not available: {e}")
            self.processor = None
            self.model = None

    @torch.no_grad()
    def transcribe(self, wav: torch.Tensor, sr: int = 16000) -> str:
        """将波形转换为文本.

        对长音频进行分块转录 (Whisper在30s段内最准确),
        然后拼接结果. 这解决了长音频幻觉和遗漏问题.

        Args:
            wav: (B, T) 或 (T,) 波形, 16kHz
            sr: 采样率

        Returns:
            text: 识别的文本
        """
        if not self.available:
            return ""

        # 确保wav是1D
        if wav.dim() == 2:
            wav = wav.squeeze(0)  # (T,)
        wav = wav.float().cpu()

        # 重采样到16kHz (Whisper要求)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        # 分块转录: 每块最大25秒 (Whisper最佳范围)
        chunk_samples = 25 * 16000  # 25秒
        total_samples = wav.shape[0]

        if total_samples <= chunk_samples:
            # 短音频, 直接转录
            return self._transcribe_chunk(wav)

        # 长音频, 分块转录
        chunks = []
        for start in range(0, total_samples, chunk_samples):
            end = min(start + chunk_samples, total_samples)
            chunk_wav = wav[start:end]
            # 跳过过短的块
            if chunk_wav.shape[0] < 1600:  # <0.1s
                continue
            chunk_text = self._transcribe_chunk(chunk_wav)
            if chunk_text:
                chunks.append(chunk_text)

        return " ".join(chunks).strip()

    @torch.no_grad()
    def _transcribe_chunk(self, wav: torch.Tensor) -> str:
        """转录单个音频块 (<=30秒).

        Args:
            wav: (T,) 波形, 16kHz

        Returns:
            text: 识别的文本
        """
        if not self.available:
            return ""

        wav = wav.float().cpu()

        # 处理
        inputs = self.processor(
            wav.numpy(),
            sampling_rate=16000,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device)

        # 生成
        predicted_ids = self.model.generate(
            input_features,
            max_new_tokens=440,
            language="english",
            task="transcribe",
        )

        text = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )[0].strip()

        return text

    @torch.no_grad()
    def compute_wer(
        self,
        wav: torch.Tensor,
        target_text: str,
        sr: int = 16000,
    ) -> float:
        """计算生成波形的WER.

        Args:
            wav: (B, T) 或 (T,) 生成波形
            target_text: 目标文本
            sr: 采样率

        Returns:
            wer: 词错误率 (0.0-1.0)
        """
        if not self.available:
            return 1.0

        # 识别文本
        transcribed = self.transcribe(wav, sr=sr)

        # 计算WER
        wer = compute_wer(target_text, transcribed)
        return wer

    @torch.no_grad()
    def compute_wer_batch(
        self,
        wavs: torch.Tensor,  # (B, T)
        target_texts: List[str],
        sr: int = 16000,
    ) -> Dict:
        """批量计算WER.

        Returns:
            results: {
                "wer_mean": float,
                "wer_std": float,
                "wer_per_sample": List[float],
                "transcriptions": List[str],
            }
        """
        wer_scores = []
        transcriptions = []

        for i in range(wavs.shape[0]):
            wav_i = wavs[i]  # (T,)
            target = target_texts[i]

            transcribed = self.transcribe(wav_i, sr=sr)
            wer = compute_wer(target, transcribed)

            wer_scores.append(wer)
            transcriptions.append(transcribed)

        return {
            "wer_mean": float(np.mean(wer_scores)) if wer_scores else 1.0,
            "wer_std": float(np.std(wer_scores)) if wer_scores else 0.0,
            "wer_per_sample": wer_scores,
            "transcriptions": transcriptions,
        }


def quick_test():
    """快速测试WER评估器."""
    logging.basicConfig(level=logging.INFO)

    evaluator = WEREvaluator(device="cpu")

    if not evaluator.available:
        print("Whisper not available, test skipped")
        return

    # 测试1: 完美匹配
    print("\n[Test 1] Perfect match:")
    # 生成一段简单波形 (这里用噪声模拟)
    wav = torch.randn(1, 16000) * 0.1
    wer = evaluator.compute_wer(wav, "hello world", sr=16000)
    print(f"  WER (noise vs 'hello world'): {wer:.4f} (expected ~1.0)")

    # 测试2: WER计算逻辑
    print("\n[Test 2] WER calculation logic:")
    print(f"  WER('hello world', 'hello world'): {compute_wer('hello world', 'hello world'):.4f}")
    print(f"  WER('hello world', 'hello'): {compute_wer('hello world', 'hello'):.4f}")
    print(f"  WER('hello world', 'hi world'): {compute_wer('hello world', 'hi world'):.4f}")
    print(f"  WER('the cat sat', 'the cat sat'): {compute_wer('the cat sat', 'the cat sat'):.4f}")
    print(f"  WER('the cat sat', 'the dog sat'): {compute_wer('the cat sat', 'the dog sat'):.4f}")


if __name__ == "__main__":
    quick_test()
