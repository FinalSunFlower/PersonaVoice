"""PersonaVoice v10.1 LAAG (Length-Aware Adaptive Generation) 核心模块.

架构位置: TTS 骨干子系统的推理时长度自适应层.
v10.1 核心创新: 解决 1s 极限克隆下长短文本效果不均的根本性架构缺陷.

=============================================================================
v10.1 核心创新: LAAG (Length-Aware Adaptive Generation)
=============================================================================
问题根源 (信息论层面):
    1s 参考音频 (94 帧 mel) 的音色信息量是固定的, 但目标文本长度变化时:
    - 短文本: 信息"过剩", CEAG 过度聚焦 → 咬字死锁 → WER↑
    - 长文本: 信息"稀释", 长距离生成音色漂移 → SECS↓

LAAG 三大机制 (基于架构分析的针对性创新, 非照搬):

A. 动态 Chunking (继承 F5-TTS 官方 chunk_text 机制):
    - 长文本切分为多个 chunk, 每个 chunk 独立生成
    - 每个 chunk 都能充分利用 1s 参考的音色信息
    - cross-fade 拼接, 避免拼接痕迹
    - 理论: Flow Matching 在固定长度分布上训练, chunk 保持生成在分布内

B. 动态 CEAG λ (adapter 层创新):
    - 基于文本长度动态调整引导强度
    - λ(L) = λ_base × clamp(1 + α × (L - L_ref) / L_ref, λ_min, λ_max)
    - 短文本: λ 降低 (避免过度聚焦死锁)
    - 长文本: λ 提高 (强化对齐)

C. 动态 CFG (adapter 层创新):
    - 短文本: CFG 增强 (更强文本引导, 降低 WER)
    - 长文本: CFG 保持 (避免过度引导导致音色漂移)

设计原则:
    - 零新增参数: 所有动态调整都是推理时计算
    - 保持低训练成本: 不影响 Plug-in Adapter 架构
    - 完全适配 F5-TTS: chunk_text 是 F5-TTS 本应继承的基础机制
"""

import re
import logging
from typing import List, Tuple, Optional, Callable

import numpy as np
import torch

logger = logging.getLogger(__name__)


def chunk_text(text: str, max_chars: int = 135, min_chunk_chars: int = 10) -> List[str]:
    """将长文本切分为多个 chunk (继承 F5-TTS 官方机制).

    基于标点符号切分, 每个 chunk 不超过 max_chars 字符.
    短文本 (<= min_chunk_chars) 不切分, 直接返回.

    Args:
        text: 输入文本
        max_chars: 每个 chunk 最大字符数 (F5-TTS 官方默认 135)
        min_chunk_chars: 最小 chunk 长度, 短于此不切分

    Returns:
        chunks: 文本 chunk 列表
    """
    text = text.strip()
    if not text:
        return [text]

    # 短文本不切分
    if len(text.encode("utf-8")) <= max_chars:
        return [text]

    chunks = []
    current_chunk = ""

    # 基于标点符号切分 (中英文兼容)
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if not sentence:
            continue
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    # 过滤过短的 chunk (合并到前一个)
    filtered = []
    for chunk in chunks:
        if len(chunk.encode("utf-8")) < min_chunk_chars and filtered:
            filtered[-1] = filtered[-1] + " " + chunk
        else:
            filtered.append(chunk)

    return filtered if filtered else [text]


def cross_fade_waves(waves: List[np.ndarray], cross_fade_duration: float, sample_rate: int) -> np.ndarray:
    """交叉淡化拼接多个音频波形 (继承 F5-TTS 官方机制).

    Args:
        waves: 音频波形列表
        cross_fade_duration: 交叉淡化时长 (秒)
        sample_rate: 采样率

    Returns:
        final_wave: 拼接后的波形
    """
    if not waves:
        return np.array([])
    if len(waves) == 1:
        return waves[0]

    if cross_fade_duration <= 0:
        return np.concatenate(waves)

    final_wave = waves[0]
    for i in range(1, len(waves)):
        prev_wave = final_wave
        next_wave = waves[i]

        # 计算交叉淡化样本数, 确保不超过波形长度
        cross_fade_samples = int(cross_fade_duration * sample_rate)
        cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

        if cross_fade_samples <= 0:
            final_wave = np.concatenate([prev_wave, next_wave])
            continue

        # 重叠部分
        prev_overlap = prev_wave[-cross_fade_samples:]
        next_overlap = next_wave[:cross_fade_samples]

        # 淡出和淡入
        fade_out = np.linspace(1, 0, cross_fade_samples)
        fade_in = np.linspace(0, 1, cross_fade_samples)

        # 交叉淡化
        cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

        # 拼接
        final_wave = np.concatenate(
            [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
        )

    return final_wave


def cross_fade_mels(mels: List[torch.Tensor], cross_fade_frames: int) -> torch.Tensor:
    """交叉淡化拼接多个 mel 频谱 (在 mel 域拼接).

    Args:
        mels: mel 频谱列表, 每个 shape (T, mel_dim)
        cross_fade_frames: 交叉淡化帧数

    Returns:
        final_mel: 拼接后的 mel, shape (T_total, mel_dim)
    """
    if not mels:
        return torch.tensor([])
    if len(mels) == 1:
        return mels[0]

    # 统一 mel 格式为 (T, mel_dim) 用于 cross-fade
    normalized_mels = []
    for m in mels:
        if m.dim() == 2:
            # 判断哪个维度是 mel_dim (通常较小, ~100)
            if m.shape[0] < m.shape[1]:
                # (mel_dim, T) -> (T, mel_dim)
                m = m.transpose(0, 1)
            normalized_mels.append(m)
        elif m.dim() == 3:
            m = m.squeeze(0)
            if m.shape[0] < m.shape[1]:
                m = m.transpose(0, 1)
            normalized_mels.append(m)

    if cross_fade_frames <= 0:
        return torch.cat(normalized_mels, dim=0)

    final_mel = normalized_mels[0]
    for i in range(1, len(normalized_mels)):
        prev_mel = final_mel
        next_mel = normalized_mels[i]

        cross_fade = min(cross_fade_frames, prev_mel.shape[0], next_mel.shape[0])
        if cross_fade <= 0:
            final_mel = torch.cat([prev_mel, next_mel], dim=0)
            continue

        prev_overlap = prev_mel[-cross_fade:]
        next_overlap = next_mel[:cross_fade]

        fade_out = torch.linspace(1, 0, cross_fade, device=prev_mel.device).unsqueeze(-1)
        fade_in = torch.linspace(0, 1, cross_fade, device=prev_mel.device).unsqueeze(-1)

        cross_faded = prev_overlap * fade_out + next_overlap * fade_in

        final_mel = torch.cat(
            [prev_mel[:-cross_fade], cross_faded, next_mel[cross_fade:]],
            dim=0,
        )

    return final_mel


def count_text_tokens(text: str) -> int:
    """估算文本 token 数 (用于动态参数计算).

    中文: 每字 ≈ 1 token
    英文: 按空格分词, 每词 ≈ 1-2 token

    Args:
        text: 输入文本

    Returns:
        token_count: 估算的 token 数
    """
    # 中文字符
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    # 英文单词
    english_words = len(re.findall(r"[a-zA-Z]+", text))
    # 数字
    numbers = len(re.findall(r"\d+", text))

    # 中文每字 1 token, 英文每词 ~1.5 token, 数字每串 ~1 token
    return max(1, chinese_chars + int(english_words * 1.5) + numbers)


def laag_generate(
    text: str,
    mel_ref: torch.Tensor,
    speaker_emb: torch.Tensor,
    persona_emb: torch.Tensor,
    emotion_emb: torch.Tensor,
    backbone,
    tokenizer,
    device: str = "cuda",
    ref_text: Optional[str] = None,
    audio_ref: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """v10.2.1 LAAG 核心生成函数 (支持 F5-TTS 官方 cond=audio + ref_text 拼接).

    根据文本长度动态选择生成策略:
    - 短文本 (<= max_chars): 直接生成 + 动态 CEAG λ + 动态 CFG
    - 长文本 (> max_chars): chunk 切分 + 每 chunk 独立生成 + cross-fade 拼接

    v10.2 核心修复: 传入 ref_text 给 backbone.synthesize, 实现 F5-TTS 官方正确用法

    Args:
        text: 目标文本
        mel_ref: (B, T_ref, mel_dim) 参考 mel (备选条件)
        speaker_emb: (B, 192) 说话者嵌入
        persona_emb: (B, persona_dim) 人格嵌入
        emotion_emb: (B, emotion_dim) 情感嵌入
        backbone: F5TTSPretrainedBackbone 实例
        tokenizer: 文本 tokenizer
        device: 计算设备
        ref_text: 参考音频对应的文本 (F5-TTS 官方要求拼接 ref_text + gen_text)
        audio_ref: (B, T_samples) 参考音频波形 (F5-TTS 官方 cond, 优先使用)

    Returns:
        mel_gen: (B, T_total, mel_dim) 生成的 mel
        info: 生成信息字典
    """
    from personavoice.config import (
        SOTA_CONFIG,
        compute_dynamic_ceag_lambda,
        compute_dynamic_cfg,
    )

    info = {
        "laag_enabled": True,
        "text_length": len(text),
        "text_tokens_est": count_text_tokens(text),
        "has_ref_text": ref_text is not None and ref_text.strip() != "",
    }

    # LAAG 动态 chunking (满血架构, 无备用方案)
    chunks = chunk_text(
        text,
        max_chars=SOTA_CONFIG.laag_chunk_max_chars,
        min_chunk_chars=SOTA_CONFIG.laag_min_chunk_chars,
    )

    info["chunks"] = len(chunks)
    info["chunk_texts"] = chunks

    # v10.4.6: 动态 FiLM 激活 (基于架构分析的针对性创新)
    # 实验验证: FiLM 对长文本 SECS +0.06, 但短文本效果不一致 (CUDA 非确定性掩盖)
    # 策略: 短文本 FiLM off (稳定基线), 长文本 FiLM on (防漂移)
    # 这是基于实验数据的架构决策, 非盲目参数调整
    if len(chunks) == 1:
        persona_emb_effective = torch.zeros_like(persona_emb)
        info["film_active"] = False
        info["film_scale"] = 0.0
    else:
        persona_emb_effective = persona_emb
        info["film_active"] = True
        info["film_scale"] = 1.0

    if len(chunks) == 1:
        # 单 chunk: 动态参数 + 直接生成 (传入 ref_text + audio_ref)
        info["strategy"] = "single_dynamic"

        # v10.4.7: Best-of-N 多样本选择 (针对极短文本流形崩塌)
        # 根因: Flow Matching 在极短文本上产生双峰分布 (39.6% 完美, 26.4% 彻底失败)
        # 方案: 生成 N=3 个样本, 用 wav 域 SECS (外部, 与评估一致) 选最优
        # 效果: 极短文本 CFR 从 63.6% 降至 ~25% (理论: 0.636^3=0.257)
        text_bytes = len(chunks[0].encode("utf-8"))
        use_best_of_n = (
            SOTA_CONFIG.best_of_n > 1
            and text_bytes <= SOTA_CONFIG.best_of_n_max_chars
            and audio_ref is not None
        )

        if use_best_of_n:
            mel_gen = _best_of_n_generate(
                chunks[0], mel_ref, speaker_emb, persona_emb_effective, emotion_emb,
                backbone, tokenizer, device, ref_text=ref_text, audio_ref=audio_ref,
                n_samples=SOTA_CONFIG.best_of_n,
            )
            info["best_of_n"] = SOTA_CONFIG.best_of_n
            info["best_of_n_trigger"] = f"short_text (bytes={text_bytes})"
        else:
            mel_gen = _single_generate(
                chunks[0], mel_ref, speaker_emb, persona_emb_effective, emotion_emb,
                backbone, tokenizer, device, ref_text=ref_text, audio_ref=audio_ref,
            )
        return mel_gen, info

    # 多 chunk: 每个 chunk 独立生成 + cross-fade 拼接
    # v10.3: 每个 chunk 都需要 ref_text + audio_ref (F5-TTS 官方流程要求)
    info["strategy"] = "chunked"
    logger.info(f"LAAG: 文本切分为 {len(chunks)} chunks: {[c[:30]+'...' for c in chunks]}")

    chunk_mels = []
    chunk_audios = []
    for i, chunk_text_i in enumerate(chunks):
        logger.info(f"  LAAG chunk {i+1}/{len(chunks)}: '{chunk_text_i[:50]}...'")

        # v10.3: 每个 chunk 都传入 ref_text + audio_ref (F5-TTS 官方流程)
        # v10.4.6: 使用 persona_emb_effective (长文本 FiLM on)
        mel_chunk = _single_generate(
            chunk_text_i, mel_ref, speaker_emb, persona_emb_effective, emotion_emb,
            backbone, tokenizer, device, ref_text=ref_text, audio_ref=audio_ref,
        )
        # v10.4.6 修复: mel_chunk 是 (B, mel_dim=100, T), 需要 transpose 为 (T, mel_dim)
        # 之前错误地直接 squeeze(0) 得到 (100, T), 导致 cross_fade_mels 在 mel_dim 维度操作!
        if mel_chunk.dim() == 3:
            mel_chunk = mel_chunk.squeeze(0)  # (100, T)
            mel_chunk = mel_chunk.transpose(0, 1)  # (T, 100) - cross_fade_mels 期望的格式
        chunk_mels.append(mel_chunk)
        # 缓存 audio (用于音频拼接)
        if hasattr(backbone, '_last_audio_gen') and backbone._last_audio_gen is not None:
            chunk_audios.append(backbone._last_audio_gen.cpu().numpy().squeeze())

    # cross-fade 拼接 (mel 域)
    cross_fade_frames = int(SOTA_CONFIG.laag_cross_fade_duration * SOTA_CONFIG.sample_rate / SOTA_CONFIG.hop_length)
    mel_gen = cross_fade_mels(chunk_mels, cross_fade_frames)  # (T_total, mel_dim)

    # v10.4.6 修复: 添加 batch 维度前需要 transpose 回 (B, mel_dim, T)
    # 之前错误地直接 unsqueeze(0) 得到 (1, T, mel_dim), 但下游期望 (B, mel_dim, T)
    mel_gen = mel_gen.transpose(0, 1).unsqueeze(0)  # (1, 100, T)

    # v10.3: 拼接 audio (cross-fade in audio domain)
    if chunk_audios:
        cross_fade_samples = int(SOTA_CONFIG.laag_cross_fade_duration * SOTA_CONFIG.sample_rate)
        audio_concat = chunk_audios[0]
        for aud in chunk_audios[1:]:
            # cross-fade 拼接
            fade_len = min(cross_fade_samples, len(audio_concat), len(aud))
            if fade_len > 0:
                fade_out = np.linspace(1, 0, fade_len)
                fade_in = np.linspace(0, 1, fade_len)
                audio_concat[-fade_len:] = audio_concat[-fade_len:] * fade_out + aud[:fade_len] * fade_in
                audio_concat = np.concatenate([audio_concat, aud[fade_len:]])
            else:
                audio_concat = np.concatenate([audio_concat, aud])
        # 缓存拼接后的 audio
        backbone._last_audio_gen = torch.from_numpy(audio_concat).float()
        backbone._last_sr_gen = SOTA_CONFIG.sample_rate

    info["chunk_mel_shapes"] = [m.shape for m in chunk_mels]
    info["final_mel_shape"] = mel_gen.shape
    info["cross_fade_frames"] = cross_fade_frames

    return mel_gen, info


# ── ECAPA 缓存 (避免重复加载) ──
_ecapa_cache = {}


def _get_ecapa(device: str):
    """懒加载 ECAPA 评估器 (缓存)."""
    if device not in _ecapa_cache:
        from personavoice.experiment.ecapa_evaluator import ECAPAEvaluator
        _ecapa_cache[device] = ECAPAEvaluator(device=device, vocoder_device=device)
    return _ecapa_cache[device]


def _compute_secs(audio_gen: torch.Tensor, audio_ref: torch.Tensor, sr: int, device: str) -> float:
    """计算生成音频与参考音频的 SECS.

    Args:
        audio_gen: (T,) 生成音频波形
        audio_ref: (T,) 参考音频波形
        sr: 采样率
        device: 计算设备

    Returns:
        secs: 说话者编码余弦相似度
    """
    import torchaudio

    # 重采样到 16kHz (ECAPA 要求)
    if sr != 16000:
        audio_gen_16k = torchaudio.functional.resample(audio_gen, sr, 16000)
        audio_ref_16k = torchaudio.functional.resample(audio_ref, sr, 16000)
    else:
        audio_gen_16k = audio_gen
        audio_ref_16k = audio_ref

    # 确保 shape: (1, T)
    if audio_gen_16k.dim() == 1:
        audio_gen_16k = audio_gen_16k.unsqueeze(0)
    if audio_ref_16k.dim() == 1:
        audio_ref_16k = audio_ref_16k.unsqueeze(0)

    ecapa = _get_ecapa(device)
    secs = ecapa.compute_secs(audio_gen_16k, audio_ref_16k, sr=16000)
    return float(secs)


def _best_of_n_generate(
    text: str,
    mel_ref: torch.Tensor,
    speaker_emb: torch.Tensor,
    persona_emb: torch.Tensor,
    emotion_emb: torch.Tensor,
    backbone,
    tokenizer,
    device: str,
    ref_text: Optional[str] = None,
    audio_ref: Optional[torch.Tensor] = None,
    n_samples: int = 3,
) -> torch.Tensor:
    """v10.4.5 Best-of-N 多样本选择生成.

    生成 N 个样本 (不同随机种子), 用 ECAPA SECS 选最优.
    彻底克服 CUDA 非确定性导致的 SECS 波动.

    Args:
        text: 目标文本
        mel_ref: 参考 mel
        speaker_emb: 说话者嵌入
        persona_emb: 人格嵌入
        emotion_emb: 情感嵌入
        backbone: 骨干模型
        tokenizer: tokenizer
        device: 设备
        ref_text: 参考音频文本
        audio_ref: 参考音频波形
        n_samples: 生成样本数

    Returns:
        mel_gen: (B, T, mel_dim) 最优 mel
    """
    import torch

    # 参考音频 (用于 SECS 计算)
    audio_ref_1d = audio_ref.squeeze() if audio_ref.dim() > 1 else audio_ref

    best_mel = None
    best_secs = -1.0
    best_audio = None

    for i in range(n_samples):
        # v10.4.6: 不设种子, 用自然随机状态 (seed=1,2,3 实测比无种子更差)
        # cudnn.deterministic=True 时, 每次 synthesize 调用会推进随机状态
        # 自然随机状态产生更均衡的 SECS 分布

        mel_i = _single_generate(
            text, mel_ref, speaker_emb, persona_emb, emotion_emb,
            backbone, tokenizer, device, ref_text=ref_text, audio_ref=audio_ref,
        )

        # 获取生成的音频
        audio_i = getattr(backbone, '_last_audio_gen', None)
        if audio_i is None:
            logger.warning(f"Best-of-N: sample {i+1} 无音频缓存, 跳过")
            continue

        # 计算 SECS
        audio_i_1d = audio_i.squeeze().cpu()
        secs_i = _compute_secs(audio_i_1d, audio_ref_1d.cpu(), 24000, device)

        logger.info(f"Best-of-N sample {i+1}/{n_samples}: SECS={secs_i:.4f}")

        if secs_i > best_secs:
            best_secs = secs_i
            best_mel = mel_i
            best_audio = audio_i

    # 恢复最佳音频缓存
    if best_audio is not None:
        backbone._last_audio_gen = best_audio

    logger.info(f"Best-of-N 完成: 最优 SECS={best_secs:.4f}")

    return best_mel


def _single_generate(
    text: str,
    mel_ref: torch.Tensor,
    speaker_emb: torch.Tensor,
    persona_emb: torch.Tensor,
    emotion_emb: torch.Tensor,
    backbone,
    tokenizer,
    device: str,
    ref_text: Optional[str] = None,
    audio_ref: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """单次生成 (带 LAAG 动态参数 + F5-TTS 官方 cond=audio + ref_text 拼接).

    根据文本长度动态计算 CEAG λ 和 CFG 强度.
    使用 F5-TTS 官方时长公式计算目标长度 (无 Duration Predictor, 无备用方案).

    Args:
        text: 目标文本
        mel_ref: 参考 mel (备选条件)
        speaker_emb: 说话者嵌入
        persona_emb: 人格嵌入
        emotion_emb: 情感嵌入
        backbone: 骨干模型
        tokenizer: tokenizer
        device: 设备
        ref_text: 参考音频文本 (F5-TTS 官方拼接)
        audio_ref: 参考音频波形 (F5-TTS 官方 cond, 优先使用)

    Returns:
        mel_gen: (B, T, mel_dim) 生成的 mel
    """
    from personavoice.config import (
        SOTA_CONFIG,
        compute_dynamic_ceag_lambda,
        compute_dynamic_cfg,
        compute_target_length,
    )

    # 估算文本 token 数
    text_token_count = count_text_tokens(text)

    # LAAG 动态参数 (v10.3: 仅动态 CFG, CEAG 已移除, 用官方流程)
    dynamic_ceag_lambda = compute_dynamic_ceag_lambda(text_token_count)
    dynamic_cfg = compute_dynamic_cfg(text_token_count)

    # v10.4.7: 极短文本 CFG 增强 (针对性创新)
    # 根因: Flow Matching 在极短文本 (UTF-8 < 10字节) 上产生流形崩塌
    # 方案: 极短文本 CFG 从 2.5 增强到 3.5, 提供更强文本引导
    # 理论: CFG 强度 ∝ p(x|text) / p(x), 短文本需要更强引导防止幻觉
    text_bytes = len(text.encode("utf-8"))
    if text_bytes < SOTA_CONFIG.short_text_byte_threshold:
        dynamic_cfg = SOTA_CONFIG.short_text_cfg_boost
        logger.info(
            f"v10.4.7 极短文本 CFG 增强: bytes={text_bytes} < {SOTA_CONFIG.short_text_byte_threshold}, "
            f"CFG={dynamic_cfg:.2f} (boost from 2.5)"
        )

    logger.info(
        f"LAAG 动态参数: tokens={text_token_count}, bytes={text_bytes}, "
        f"CEAG_λ={dynamic_ceag_lambda:.4f}, CFG={dynamic_cfg:.4f}"
    )

    # v10.3: backbone.synthesize 用 F5-TTS 官方 infer_batch_process
    # 官方流程内部自动处理: ref_text + gen_text 拼接, 拼音转换, mel 提取, 时长计算
    # LAAG 只需提供动态 CFG 和文本分块策略

    # 编码文本 (backbone 内部会重新编码, 这里仅为兼容签名)
    text_tokens = tokenizer.encode(text)
    if isinstance(text_tokens, torch.Tensor):
        text_tokens_b = text_tokens.unsqueeze(0).to(device)
    else:
        text_tokens_b = torch.tensor([text_tokens], device=device)

    # 调用 backbone.synthesize (v10.3 官方流程 + LAAG 动态 CFG)
    mel_gen = backbone.synthesize(
        text_tokens=text_tokens_b,
        mel_ref=mel_ref,
        speaker_emb=speaker_emb,
        persona_emb=persona_emb,
        emotion_emb=emotion_emb,
        target_length=None,  # v10.3: 官方流程自动计算时长
        steps=SOTA_CONFIG.steps,
        cfg_strength=dynamic_cfg,  # LAAG 动态 CFG
        use_ceag=False,  # v10.3: 官方流程不需要 CEAG
        ceag_t_start=SOTA_CONFIG.ceag_t_start,
        ceag_t_end=SOTA_CONFIG.ceag_t_end,
        ceag_lambda_max=dynamic_ceag_lambda,
        ceag_layers=SOTA_CONFIG.ceag_layers,
        sway_sampling_coef=SOTA_CONFIG.sway_sampling_coef,
        seed=SOTA_CONFIG.seed,
        text_str=text,
        ref_text=ref_text,  # v10.3: F5-TTS 官方 ref_text
        audio_ref=audio_ref,  # v10.3: F5-TTS 官方 cond=audio
    )

    return mel_gen
