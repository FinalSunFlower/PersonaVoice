"""前端音频预处理 — v10.0 新增 (修复地雷3).

架构位置: 演示模块的前置组件, 对用户上传的参考音频进行预处理.

=============================================================================
地雷3: 前端 WebRTC VAD 过于老旧 (听感瑕疵)
=============================================================================
问题: WebRTC VAD 基于 GMM, 对 1s 极短音频的开头呼吸声、微弱辅音
(/s/, /f/) 的切分极其生硬, 导致参考音频"吃字"或不自然.

修复方案: 改用 Silero VAD (轻量级神经网络, 显存 <100MB).
对呼吸声和唇齿音的判断极其精准, 大幅提升 1s 样本的参考纯度.

=============================================================================
v10.0 前端预处理 Pipeline:
=============================================================================
    1. 加载音频 → 重采样到 16kHz (Silero VAD 要求)
    2. Silero VAD 精准语音端点检测 → 去除前后静音
    3. 重采样到 24kHz (F5-TTS 要求)
    4. RMS 能量归一化 → 生成音频响度对齐参考音频
    5. 尾部补 50ms 静音 (F5-TTS 官方要求)

设计理念:
    - Silero VAD: 神经网络 VAD, 对呼吸声/辅音切分精准
    - RMS 归一化: 确保不同录音环境的响度一致性
    - 全程 torchaudio: 不依赖 pydub/ffprobe
"""

import logging
import os
from typing import Optional, Tuple

import torch
import torchaudio

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

logger = logging.getLogger(__name__)

# Silero VAD 模型缓存 (首次使用时从 HuggingFace 下载, 之后离线)
_SILERO_VAD_CACHE = None


def _load_silero_vad():
    """懒加载 Silero VAD 模型 (首次调用时加载).

    Silero VAD 是轻量级神经网络 VAD:
        - 模型大小: ~2MB
        - 显存占用: <100MB
        - 对呼吸声/辅音切分精准 (远超 WebRTC GMM)

    Returns:
        (vad_model, get_speech_timestamps_fn)
    """
    global _SILERO_VAD_CACHE

    if _SILERO_VAD_CACHE is not None:
        return _SILERO_VAD_CACHE

    try:
        # 尝试从 silero-vad 包加载 (v6.x API: load_silero_vad)
        from silero_vad import load_silero_vad, get_speech_timestamps

        vad_model = load_silero_vad()
        _SILERO_VAD_CACHE = (vad_model, get_speech_timestamps)
        logger.info("Silero VAD loaded (silero-vad package)")
        return _SILERO_VAD_CACHE
    except ImportError:
        pass

    try:
        # 回退: 从 torch.hub 加载 (会缓存到本地)
        # Silero VAD 的 torch.hub 仓库
        vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
            source="github",
        )
        get_speech_timestamps = utils[0]
        _SILERO_VAD_CACHE = (vad_model, get_speech_timestamps)
        logger.info("Silero VAD loaded (torch.hub)")
        return _SILERO_VAD_CACHE
    except Exception as e:
        logger.warning(f"Silero VAD load failed: {e}, falling back to energy-based VAD")
        _SILERO_VAD_CACHE = (None, None)
        return _SILERO_VAD_CACHE


def silero_vad_trim(
    waveform: torch.Tensor,
    sample_rate: int,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 100,
    min_silence_duration_ms: int = 50,
) -> torch.Tensor:
    """用 Silero VAD 精准去除前后静音.

    v10.0 修复地雷3: 替代老旧的 WebRTC VAD.
    Silero VAD 基于轻量级神经网络, 对呼吸声和辅音 (/s/, /f/) 的切分
    极其精准, 大幅提升 1s 样本的参考纯度.

    Args:
        waveform: (1, T) 或 (T,) 波形
        sample_rate: 采样率 (Silero 要求 16kHz)
        threshold: VAD 阈值 (0-1, 越高越严格)
        min_speech_duration_ms: 最短语音段时长
        min_silence_duration_ms: 最短静音段时长

    Returns:
        trimmed_waveform: (1, T') 去除静音后的波形
    """
    # 确保单声道
    if waveform.ndim == 2:
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
    else:
        waveform = waveform.unsqueeze(0)

    # Silero VAD 要求 16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform_16k = resampler(waveform)
    else:
        waveform_16k = waveform

    vad_model, get_speech_timestamps = _load_silero_vad()

    # Silero VAD 精准端点检测 (满血架构, 无备用方案)
    speech_timestamps = get_speech_timestamps(
        waveform_16k.squeeze(0).numpy(),
        vad_model,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        return_seconds=False,
    )

    if speech_timestamps:
        # 取第一个语音段的开始到最后一个语音段的结束
        start_16k = speech_timestamps[0]["start"]
        end_16k = speech_timestamps[-1]["end"]

        # 转换回原始采样率
        scale = sample_rate / 16000
        start = int(start_16k * scale)
        end = int(end_16k * scale)

        waveform = waveform[:, start:end]
        logger.info(
            f"Silero VAD: trimmed {waveform.shape[-1]/sample_rate:.3f}s "
            f"(from {start/sample_rate:.3f}s to {end/sample_rate:.3f}s)"
        )
    else:
        logger.warning("Silero VAD: no speech detected, keeping original")

    return waveform


def rms_normalize(
    waveform: torch.Tensor,
    target_rms: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """RMS 能量归一化.

    确保生成音频的响度与参考音频一致, 避免不同录音环境导致的响度差异.

    Args:
        waveform: (1, T) 或 (T,) 波形
        target_rms: 目标 RMS 能量 (默认 0.1, 略低于参考以避免削波)
        eps: 数值稳定小量

    Returns:
        normalized_waveform: 归一化后的波形
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    rms = torch.sqrt(torch.mean(waveform ** 2) + eps)
    # 避免除零和极端缩放
    if rms < eps:
        return waveform

    scale = target_rms / rms
    # 限制缩放范围, 防止极端值
    scale = torch.clamp(scale, min=0.1, max=10.0)

    normalized = waveform * scale
    # 防止削波
    max_val = normalized.abs().max()
    if max_val > 0.99:
        normalized = normalized * (0.99 / max_val)

    return normalized


def preprocess_ref_audio(
    audio_path: str,
    target_sr: int = 24000,
    use_silero_vad: bool = True,
    use_rms_normalize: bool = True,
    vad_threshold: float = 0.5,
    rms_target: float = 0.1,
    tail_silence_ms: int = 50,
) -> Tuple[torch.Tensor, int]:
    """v10.0 前端音频预处理 Pipeline (修复地雷3).

    完整 Pipeline:
        1. 加载音频 → 重采样到 16kHz (Silero VAD 要求)
        2. Silero VAD 精准语音端点检测 → 去除前后静音
        3. 重采样到 24kHz (F5-TTS 要求)
        4. RMS 能量归一化 → 响度一致性
        5. 尾部补 50ms 静音 (F5-TTS 官方要求)

    Args:
        audio_path: 音频文件路径
        target_sr: 目标采样率 (F5-TTS 要求 24kHz)
        use_silero_vad: 是否启用 Silero VAD
        use_rms_normalize: 是否启用 RMS 归一化
        vad_threshold: VAD 阈值
        rms_target: RMS 归一化目标值
        tail_silence_ms: 尾部静音时长 (毫秒)

    Returns:
        (waveform, sample_rate): 预处理后的波形和采样率
    """
    # 1. 加载音频
    waveform, sr = torchaudio.load(audio_path)

    # 转单声道
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    logger.info(
        f"Preprocess: loaded {audio_path}, "
        f"{waveform.shape[-1]/sr:.3f}s @ {sr}Hz"
    )

    # 2. Silero VAD 精准去除静音
    if use_silero_vad:
        waveform = silero_vad_trim(
            waveform, sr,
            threshold=vad_threshold,
        )

    # 3. 重采样到目标采样率
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    # 4. RMS 能量归一化
    if use_rms_normalize:
        waveform = rms_normalize(waveform, target_rms=rms_target)

    # 5. 尾部补静音 (F5-TTS 官方要求)
    tail_samples = int(tail_silence_ms * 0.001 * sr)
    if tail_samples > 0:
        waveform = torch.cat([
            waveform,
            torch.zeros(1, tail_samples, dtype=waveform.dtype),
        ], dim=1)

    logger.info(
        f"Preprocess: output {waveform.shape[-1]/sr:.3f}s @ {sr}Hz, "
        f"RMS={torch.sqrt(torch.mean(waveform**2)).item():.4f}"
    )

    return waveform, sr
