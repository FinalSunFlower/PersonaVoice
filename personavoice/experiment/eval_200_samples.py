"""PersonaVoice v9.1 200样本统计显著性评估 (SOTA P0 级任务).

架构位置: 实验模块中的大规模统计评估脚本, 基于 v9.1 精简配置在 200 个样本上
进行统计显著性验证, 解决 v7.7 报告"20样本标准差过大、缺乏 p-value"的学术软肋.

v9.1 精简架构 (基于 200 样本消融实验):
    - 保留: F5-TTS 预训练骨干 + FiLM + OES (env_weight=0.0) + IEAG
    - 移除: IBOP/AM-ODE/TD-CFG/SBM (消融实验证实无效)

核心目标 (对应 SOTA_VERIFICATION_REPORT.md 第 14 节):
    1. 暴力扩充测试集: 20 → 200 样本 (P0 级任务)
    2. 计算统计显著性: p-value, 95%置信区间, 效应量 (Cohen's d)
    3. 同时评估 F5-TTS Baseline 与 PersonaVoice v9.1 精简配置
    4. 按文本长度分层分析 (短/长文本)
    5. 为信息熵衰减定律 (Phase 4) 收集 (target_length, SECS) 数据点

v9.1 最佳配置 (基于 SOTA_VERIFICATION_REPORT.md 14.5 节):
    - 静态 CFG: cfg_strength=2.5 (替代无效的 TD-CFG)
    - IEAG: lambda_max=0.15, t∈[0.1, 0.4] (有效, WER 改善主要贡献者)
    - OES: env_weight=0.0 (纯净录音棚音质, SECS SOTA 配置)
    - 动态长度: seconds_per_word=0.30
    - ODE 步数: 64 (v9.0 SOTA 配置)
    - Sway Sampling: coef=-1.0 (F5-TTS 官方默认)

输出:
    - results/eval_200_samples.json: 完整 200 样本评估数据
    - results/eval_200_statistics.json: 统计显著性分析
    - results/eval_200_by_length.json: 按文本长度分层分析
    - results/entropy_decay_data.json: 信息熵衰减定律数据点
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torchaudio
from scipy import stats as scipy_stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from personavoice.tts_backbone.f5_pretrained_backbone import load_pretrained_f5tts_backbone
from personavoice.experiment.utils import load_tokenizer, setup_logger
from personavoice.experiment.ecapa_evaluator import ECAPAEvaluator
from personavoice.experiment.wer_evaluator import WEREvaluator


def estimate_target_length(text: str, seconds_per_word: float = 0.35) -> int:
    """动态目标长度估计 (v9.0 优化).

    根据文本词数估计目标音频长度 (帧数), 解决数据集 3s 截断导致的长文本 OOD 问题.

    Args:
        text: 目标文本
        seconds_per_word: 每词秒数 (v9.0 最佳配置: 0.35s/word, 10样本WER=0.2748)

    Returns:
        target_length: 目标 mel 帧数 (94帧/秒)
    """
    n_words = max(1, len(text.split()))
    target_sec = n_words * seconds_per_word + 1.0  # 加 1s 缓冲
    target_sec = max(1.0, min(target_sec, 30.0))  # 限制 1-30s
    return int(target_sec * 94)  # 94帧/秒 @ 24kHz/hop=256


def estimate_target_length_f5_official(
    text: str,
    ref_text: str,
    ref_mel_frames: int = 94,
    speed: float = 1.0,
) -> int:
    """F5-TTS官方语速推算 (v9.0 借鉴XTTS自回归时长自适应).

    XTTS自回归自动决定时长, F5-TTS需要预设.
    F5-TTS官方公式 (utils_infer.py:493):
        duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)

    这比简单的 n_words * 0.35 更准确, 因为它基于参考音频的实际语速.

    Args:
        text: 生成文本
        ref_text: 参考音频对应文本 (1s参考的文本内容)
        ref_mel_frames: 参考音频mel帧数 (1s = 94帧)
        speed: 语速因子 (1.0=正常, <1.0=慢, >1.0=快)

    Returns:
        target_length: 目标 mel 帧数
    """
    ref_text_len = max(1, len(ref_text.encode("utf-8")))
    gen_text_len = max(1, len(text.encode("utf-8")))
    # F5-TTS官方公式: 总时长 = 参考时长 + 参考语速推算的生成时长
    duration = ref_mel_frames + int(ref_mel_frames / ref_text_len * gen_text_len / speed)
    # 限制在合理范围 (1s-30s)
    duration = max(94, min(duration, 30 * 94))
    return duration

logger = setup_logger("eval_200_samples")


# =============================================================================
# 数据加载: 200样本
# =============================================================================

def load_200_test_samples(seed: int = 42) -> List[Dict]:
    """加载200个测试样本 (从387样本中随机抽取, 确保统计显著性).

    策略:
        - 从387样本中随机抽取200个 (无放回)
        - 固定seed保证可复现
        - 每个样本包含1s参考mel + 完整mel + 目标文本 + ECAPA嵌入

    Args:
        seed: 随机种子

    Returns:
        samples: 200个样本列表
    """
    data_path = PROJECT_ROOT / "data" / "libritts_processed" / "libritts_devclean_processed.pt"
    data = torch.load(str(data_path), weights_only=False)

    mel = data["tensors"]["mel"]  # (N, 100, T)
    wav = data["tensors"]["wav"]  # (N, T_samples) v10.4.6: 加载原始波形
    ecapa_emb = data["tensors"]["ecapa_emb"]  # (N, 192)
    texts = data["metadata"]["texts"]
    speaker_names = data["metadata"]["speaker_names"]
    sample_rate = data["metadata"].get("sample_rate", 24000)

    n_total = mel.shape[0]
    n_samples = min(200, n_total)
    logger.info(f"Loading {n_samples} samples from {n_total} total (seed={seed})")

    rng = np.random.RandomState(seed)
    indices = rng.choice(n_total, n_samples, replace=False)
    indices = sorted(indices.tolist())

    samples = []
    for idx in indices:
        idx = int(idx)
        mel_full = mel[idx]  # (100, T)
        if mel_full.dim() == 2:
            mel_full = mel_full.transpose(0, 1)  # (T, 100)

        # 1s参考 (94帧 @ 24kHz/hop=256)
        mel_ref = mel_full[:94, :]  # (94, 100)

        # v10.4.6: 提取 1s 参考音频波形 (F5-TTS 官方 cond=audio 要求)
        # 数据集 wav 是 16kHz, F5-TTS 要求 24kHz
        # 1s @ 16kHz = 16000 samples, 重采样到 24kHz = 24000 samples
        ref_samples_1s_16k = 16000  # 1s @ 16kHz
        wav_full = wav[idx]  # (T_samples,)
        if wav_full.dim() > 1:
            wav_full = wav_full.squeeze(0)
        audio_ref_1s_16k = wav_full[:ref_samples_1s_16k]  # (16000,) @ 16kHz

        # 重采样到 24kHz (F5-TTS 要求)
        import torchaudio as _ta
        audio_ref_1s = _ta.functional.resample(audio_ref_1s_16k, 16000, 24000)  # (24000,) @ 24kHz

        # v10.4.6: ref_text 由 Whisper ASR 转录得到 (与前端流程一致)
        # F5-TTS 官方要求 ref_text 与 ref_audio 内容一致
        # 数据集中 wav 和 text 是同一句话, 但 1s 截取只包含前几个词
        # 用 Whisper 转录 1s 截取得到准确的 ref_text
        target_text = texts[idx] if idx < len(texts) else ""

        samples.append({
            "mel_ref": mel_ref,
            "mel_full": mel_full,
            "audio_ref": audio_ref_1s,  # v10.4.6: 1s 参考波形 @ 24kHz
            "audio_ref_16k": audio_ref_1s_16k,  # v10.4.6: 1s 参考波形 @ 16kHz (Whisper 转录用)
            "target_text": target_text,
            "speaker_emb": ecapa_emb[idx],
            "speaker_name": speaker_names[idx] if idx < len(speaker_names) else "",
            "idx": idx,
        })

    logger.info(f"Loaded {len(samples)} test samples (ref=1.0s, ref_samples=94)")
    return samples


# =============================================================================
# PersonaVoice v9.1 评估
# =============================================================================

@torch.no_grad()
def evaluate_v91_sample(
    sample: Dict,
    backbone,
    tokenizer,
    ecapa_evaluator: ECAPAEvaluator,
    wer_evaluator: WEREvaluator,
    seconds_per_word: float = 0.30,
    cfg_strength: float = 3.0,
    steps: int = 96,
    device: str = "cuda",
    use_oes: bool = True,
    env_weight: float = 0.0,
    use_ceag: bool = True,
    ceag_lambda_max: float = 0.25,
    ceag_t_start: float = 0.05,
    ceag_t_end: float = 0.5,
    ceag_layers: Tuple[int, ...] = (-3, -2, -1),
    sway_sampling_coef: float = -1.0,
) -> Dict:
    """评估单个样本 (v10.1 满血配置: LAAG + CEAG).

    v10.1 配置 (满血架构, 无备用方案):
        - LAAG: 长度自适应生成 (动态 Chunking + 动态 CEAG λ + 动态 CFG + F5-TTS 官方时长公式)
        - CEAG: 交叉注意力熵引导 (升级自 IEAG, 带 padding mask)
        - OES env_scale=0.1: 渐进初始化 (解决 1111.mp3 SECS 灾难)
        - 静态 CFG=2.0, steps=32, Sway Sampling=-1.0
        - 移除: IBOP/AM-ODE/TD-CFG/SBM/Duration Predictor (消融实验证实无效或未训练)
    """
    mel_ref = sample["mel_ref"].to(device).float()
    target_text = sample["target_text"]
    speaker_emb = sample["speaker_emb"].to(device).float()
    # v10.4.6: 加载 1s 参考音频波形 (F5-TTS 官方 cond=audio 要求, 24kHz)
    audio_ref = sample.get("audio_ref")
    if audio_ref is not None:
        audio_ref = audio_ref.to(device).float()
    audio_ref_16k = sample.get("audio_ref_16k")

    # v10.4.6: 用 Whisper ASR 转录 1s 参考音频得到 ref_text (与前端流程一致)
    # F5-TTS 官方要求 ref_text 与 ref_audio 内容一致
    # 1s 截取只包含前几个词, 用 Whisper 转录得到准确内容
    global _asr_pipe_eval
    if '_asr_pipe_eval' not in globals() or _asr_pipe_eval is None:
        from transformers import pipeline as hf_pipeline
        _asr_pipe_eval = hf_pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32,
            device=device,
        )
    audio_ref_np = audio_ref_16k.numpy().astype("float32")
    asr_result = _asr_pipe_eval(audio_ref_np, generate_kwargs={"language": "english"})
    ref_text = asr_result["text"].strip()
    if not ref_text:
        ref_text = "Hello."  # 空转录兜底

    mel_ref_b = mel_ref.unsqueeze(0)
    speaker_emb_b = speaker_emb.unsqueeze(0)
    # v10.0 修复: persona_emb 应为 192 维 (会被 persona_proj 投影到 64)
    # emotion_emb 应为 4 维 (one-hot, 会被 emotion_proj 投影到 64)
    persona_emb_b = torch.zeros(1, 192, device=device)
    emotion_emb_b = torch.zeros(1, 4, device=device)

    n_words = len(target_text.split())

    # v9.1: OES 环境权重设置 (默认 0.0: 纯净录音棚音质, SOTA 配置)
    if use_oes and hasattr(backbone, "set_env_weight"):
        backbone.set_env_weight(env_weight)
    elif hasattr(backbone, "set_env_weight"):
        backbone.set_env_weight(0.0)

    # 编码文本
    text_tokens = tokenizer.encode(target_text)
    if isinstance(text_tokens, torch.Tensor):
        text_tokens_b = text_tokens.unsqueeze(0).to(device)
    else:
        text_tokens_b = torch.tensor([text_tokens], device=device)

    t_start = time.time()
    try:
        # v10.1 LAAG: 长度自适应生成 (满血架构, 无备用方案)
        # v10.4.6: 传入 audio_ref + ref_text (F5-TTS 官方 cond=audio 流程)
        mel_gen, laag_info = backbone.laag_synthesize(
            text_str=target_text,
            mel_ref=mel_ref_b,
            speaker_emb=speaker_emb_b,
            persona_emb=persona_emb_b,
            emotion_emb=emotion_emb_b,
            tokenizer=tokenizer,
            ref_text=ref_text,
            audio_ref=audio_ref,
        )
    except Exception as e:
        return {
            "secs": 0.0, "wer": 1.0, "time_ms": 0.0,
            "error": str(e), "target_length": 0,
            "text_len": n_words, "gen_length_sec": 0.0,
        }

    t_elapsed = (time.time() - t_start) * 1000

    # v10.4.6 修复: backbone.synthesize 返回 (B, n_mels=100, T) 格式
    # ECAPA compute_secs_from_mel 期望 (B, n_mels, T) - 无需转置!
    # 之前错误地做了 transpose(1,2) 变成 (B, T, 100), 导致短文本 SECS=0
    mel_gen_vocos = mel_gen  # (B, 100, T) - 直接使用

    # SECS评估
    try:
        secs = ecapa_evaluator.compute_secs_from_mel(
            mel_generated=mel_gen_vocos.cpu(),
            emb_reference=speaker_emb_b.cpu(),
            mel_sample_rate=24000,
            mel_n_mels=100,
            is_normalized=False,
        )
    except Exception:
        secs = 0.0

    # WER评估
    try:
        wav_gen, sr_wav = ecapa_evaluator.mel_to_waveform(
            mel_gen_vocos.cpu(), mel_sample_rate=24000, mel_n_mels=100, is_normalized=False
        )
        if wav_gen.dim() == 1:
            wav_gen = wav_gen.unsqueeze(0)
        if sr_wav != 16000:
            wav_gen = torchaudio.functional.resample(wav_gen, sr_wav, 16000)
        wer = wer_evaluator.compute_wer(wav_gen.squeeze(0), target_text, sr=16000)
    except Exception:
        wer = 1.0

    # 生成音频长度 (秒): mel 形状 (B, 100, T), T 是时间帧
    gen_length_sec = mel_gen_vocos.shape[-1] / 94.0  # 94帧/秒

    # v10.4.6: target_length 由 LAAG 内部 F5-TTS 官方公式计算, 这里仅记录实际生成长度
    actual_target_length = int(mel_gen_vocos.shape[-1])

    return {
        "secs": float(secs),
        "wer": float(wer),
        "time_ms": t_elapsed,
        "idx": sample.get("idx", -1),
        "text": target_text[:80],
        "text_len": n_words,
        "target_length": actual_target_length,
        "gen_length_sec": float(gen_length_sec),
        "speaker_name": sample.get("speaker_name", ""),
    }


# =============================================================================
# F5-TTS Baseline 评估
# =============================================================================

@torch.no_grad()
def evaluate_baseline_f5tts(
    sample: Dict,
    f5tts_obj,
    ecapa_evaluator: ECAPAEvaluator,
    wer_evaluator: WEREvaluator,
    device: str = "cuda",
) -> Dict:
    """评估F5-TTS官方原版基线 (不加载微调权重)."""
    import torchaudio
    from f5_tts.infer.utils_infer import infer_process

    mel_ref = sample["mel_ref"].cpu()  # (T_ref, 100)
    target_text = sample["target_text"]
    speaker_emb = sample["speaker_emb"].cpu()  # (192,)
    n_words = len(target_text.split())

    # mel_ref -> wav (作为F5-TTS参考)
    mel_ref_vocos = mel_ref.transpose(0, 1).unsqueeze(0)  # (1, 100, T_ref)
    try:
        ref_wav_16k, _ = ecapa_evaluator.mel_to_waveform(
            mel_ref_vocos, mel_sample_rate=24000, mel_n_mels=100, is_normalized=False
        )
        if ref_wav_16k.dim() == 1:
            ref_wav_16k = ref_wav_16k.unsqueeze(0)
    except Exception as e:
        return {
            "secs": 0.0, "wer": 1.0, "time_ms": 0.0,
            "error": str(e), "text_len": n_words, "gen_length_sec": 0.0,
        }

    # 保存参考wav到临时文件
    tmp_dir = PROJECT_ROOT / "tmp_eval200"
    tmp_dir.mkdir(exist_ok=True)
    ref_path = tmp_dir / f"ref_baseline_{sample.get('idx', 0)}.wav"
    torchaudio.save(str(ref_path), ref_wav_16k, 16000)

    t_start = time.time()
    try:
        ref_text = target_text.strip()
        if not ref_text.endswith("."):
            ref_text += ". "
        else:
            ref_text += " "

        wav_gen, sr_gen, _ = infer_process(
            ref_audio=str(ref_path),
            ref_text=ref_text,
            gen_text=target_text,
            model_obj=f5tts_obj.ema_model,
            vocoder=f5tts_obj.vocoder,
            mel_spec_type=f5tts_obj.mel_spec_type,
            show_info=lambda x: None,
            nfe_step=32,
            cfg_strength=2,
            sway_sampling_coef=-1,
            speed=1.0,
            device=f5tts_obj.device,
        )
        t_elapsed = (time.time() - t_start) * 1000

        if isinstance(wav_gen, np.ndarray):
            wav_gen = torch.from_numpy(wav_gen).float()
        if wav_gen.dim() == 1:
            wav_gen = wav_gen.unsqueeze(0)

        if sr_gen != 16000:
            wav_gen = torchaudio.functional.resample(wav_gen, sr_gen, 16000)

    except Exception as e:
        try:
            ref_path.unlink(missing_ok=True)
        except Exception:
            pass
        return {
            "secs": 0.0, "wer": 1.0, "time_ms": 0.0,
            "error": str(e), "text_len": n_words, "gen_length_sec": 0.0,
        }

    try:
        ref_path.unlink(missing_ok=True)
    except Exception:
        pass

    # ECAPA SECS
    secs = 0.0
    try:
        emb_gen = ecapa_evaluator.extract_embedding(wav_gen, sr=16000)
        secs = torch.nn.functional.cosine_similarity(
            emb_gen.squeeze(0), speaker_emb, dim=-1
        ).item()
    except Exception:
        secs = 0.0

    # WER
    wer = 1.0
    try:
        wer = wer_evaluator.compute_wer(wav_gen.squeeze(0), target_text, sr=16000)
    except Exception:
        wer = 1.0

    # 生成音频长度 (秒)
    gen_length_sec = wav_gen.shape[-1] / 16000.0

    return {
        "secs": float(secs),
        "wer": float(wer),
        "time_ms": t_elapsed,
        "idx": sample.get("idx", -1),
        "text_len": n_words,
        "gen_length_sec": float(gen_length_sec),
    }


# =============================================================================
# 统计显著性分析
# =============================================================================

def compute_statistics(scores: List[float]) -> Dict:
    """计算统计指标: 均值, 标准差, 95%CI, 标准误差."""
    arr = np.array(scores)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 0 else 0.0

    # 95% 置信区间 (t分布)
    if n > 1:
        t_val = scipy_stats.t.ppf(0.975, df=n-1)
        ci_low = mean - t_val * sem
        ci_high = mean + t_val * sem
    else:
        ci_low = ci_high = mean

    return {
        "mean": mean,
        "std": std,
        "sem": float(sem),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "n": n,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }


def compute_paired_t_test(
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05,
) -> Dict:
    """配对样本t检验 (PersonaVoice vs Baseline).

    H0: 两个系统性能无差异
    H1: PersonaVoice 优于 Baseline (单侧检验)

    Args:
        scores_a: Baseline 分数
        scores_b: PersonaVoice 分数
        alpha: 显著性水平

    Returns:
        统计检验结果
    """
    n = min(len(scores_a), len(scores_b))
    a = np.array(scores_a[:n])
    b = np.array(scores_b[:n])

    # 配对差异
    diff = b - a
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0

    # t统计量
    if std_diff > 0:
        t_stat = mean_diff / (std_diff / np.sqrt(n))
    else:
        t_stat = 0.0

    # 单侧p-value (H1: b > a)
    if n > 1:
        p_value_one_sided = float(scipy_stats.t.sf(t_stat, df=n-1))
    else:
        p_value_one_sided = 1.0

    # 双侧p-value
    if n > 1:
        p_value_two_sided = float(2 * scipy_stats.t.sf(abs(t_stat), df=n-1))
    else:
        p_value_two_sided = 1.0

    # Cohen's d 效应量
    if std_diff > 0:
        cohen_d = mean_diff / std_diff
    else:
        cohen_d = 0.0

    # 效应量解释
    if abs(cohen_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohen_d) < 0.5:
        effect_size = "small"
    elif abs(cohen_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"

    return {
        "n_pairs": n,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_statistic": float(t_stat),
        "p_value_one_sided": p_value_one_sided,
        "p_value_two_sided": p_value_two_sided,
        "cohen_d": float(cohen_d),
        "effect_size": effect_size,
        "significant_at_alpha": bool(p_value_one_sided < alpha),
        "alpha": alpha,
    }


def analyze_by_text_length(results: List[Dict], threshold: int = 8) -> Dict:
    """按文本长度分层分析 (短文本 vs 长文本).

    Args:
        results: 评估结果列表
        threshold: 长短文本分界 (词数)

    Returns:
        分层分析结果
    """
    short = [r for r in results if r["text_len"] < threshold]
    long = [r for r in results if r["text_len"] >= threshold]

    short_secs = [r["secs"] for r in short]
    short_wer = [r["wer"] for r in short]
    long_secs = [r["secs"] for r in long]
    long_wer = [r["wer"] for r in long]

    return {
        "threshold_words": threshold,
        "short_text": {
            "n_samples": len(short),
            "secs": compute_statistics(short_secs),
            "wer": compute_statistics(short_wer),
        },
        "long_text": {
            "n_samples": len(long),
            "secs": compute_statistics(long_secs),
            "wer": compute_statistics(long_wer),
        },
    }


def collect_entropy_decay_data(results: List[Dict]) -> Dict:
    """收集信息熵衰减定律数据点 (target_length, SECS).

    用于Phase 4: 拟合 SECS vs 生成音频长度 的衰减曲线,
    定义"时序漂移边界 (Temporal Drift Boundary)".

    Returns:
        data: 包含 (length, secs) 数据点, 可用于拟合
    """
    data_points = []
    for r in results:
        if r.get("gen_length_sec", 0) > 0:
            data_points.append({
                "gen_length_sec": r["gen_length_sec"],
                "secs": r["secs"],
                "wer": r["wer"],
                "text_len": r["text_len"],
                "idx": r.get("idx", -1),
            })

    # 按生成长度分桶计算平均SECS
    bins = [0, 3, 5, 7, 10, 15, 20, 30]
    bin_stats = []
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        bucket = [d for d in data_points if low <= d["gen_length_sec"] < high]
        if bucket:
            bin_stats.append({
                "length_range": [low, high],
                "n_samples": len(bucket),
                "secs_mean": float(np.mean([d["secs"] for d in bucket])),
                "secs_std": float(np.std([d["secs"] for d in bucket])) if len(bucket) > 1 else 0.0,
                "wer_mean": float(np.mean([d["wer"] for d in bucket])),
            })

    return {
        "data_points": data_points,
        "bin_stats": bin_stats,
        "n_total": len(data_points),
    }


# =============================================================================
# 主函数
# =============================================================================

def run_evaluation(
    n_samples: int = 200,
    skip_baseline: bool = False,
    skip_personavoice: bool = False,
    seed: int = 42,
):
    """运行200样本统计显著性评估.

    Args:
        n_samples: 样本数 (默认200)
        skip_baseline: 跳过F5-TTS基线评估
        skip_personavoice: 跳过PersonaVoice评估
        seed: 随机种子
    """
    logger.info("=" * 70)
    logger.info(f"PersonaVoice v9.1 {n_samples}-Sample Statistical Evaluation")
    logger.info("Goal: Statistical Significance + Information Entropy Decay")
    logger.info("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载200样本
    logger.info(f"\n[1/5] Loading {n_samples} test samples...")
    samples = load_200_test_samples(seed=seed)
    if len(samples) < n_samples:
        logger.warning(f"Only {len(samples)} samples available (requested {n_samples})")

    # 2. 加载tokenizer和评估器
    logger.info("\n[2/5] Loading tokenizer and evaluators...")
    tokenizer = load_tokenizer()
    ecapa_evaluator = ECAPAEvaluator(device="cpu")
    wer_evaluator = WEREvaluator(device="cpu", model_name="openai/whisper-large-v3-turbo")

    # 3. 评估F5-TTS基线
    baseline_results = []
    if not skip_baseline:
        logger.info("\n[3/5] Evaluating F5-TTS Baseline...")
        try:
            from f5_tts.api import F5TTS
            f5tts_obj = F5TTS()
            logger.info(f"  F5-TTS loaded on {f5tts_obj.device}")

            for i, sample in enumerate(samples):
                r = evaluate_baseline_f5tts(
                    sample, f5tts_obj, ecapa_evaluator, wer_evaluator, device=device
                )
                baseline_results.append(r)

                if (i + 1) % 20 == 0:
                    mean_secs = np.mean([r["secs"] for r in baseline_results])
                    mean_wer = np.mean([r["wer"] for r in baseline_results])
                    logger.info(f"  Baseline [{i+1}/{len(samples)}] "
                                f"SECS={mean_secs:.4f}, WER={mean_wer:.4f}")

            # 清理临时目录
            tmp_dir = PROJECT_ROOT / "tmp_eval200"
            if tmp_dir.exists():
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

        except Exception as e:
            logger.error(f"  F5-TTS baseline failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.info("\n[3/5] Skipping F5-TTS Baseline evaluation")

    # 4. 评估PersonaVoice v9.1
    pv_results = []
    if not skip_personavoice:
        logger.info("\n[4/5] Evaluating PersonaVoice v9.1 (streamlined config)...")
        backbone, _ = load_pretrained_f5tts_backbone(device=device)
        ckpt_path = PROJECT_ROOT / "checkpoints" / "sota_v50" / "best_model.pt"
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            backbone.load_state_dict(ckpt["backbone_state_dict"], strict=False)
            logger.info(f"  Loaded v5.0 weights from {ckpt_path}")
        backbone = backbone.to(device)
        backbone.eval()

        # v10.1: 从统一配置读取参数 (消除配置矛盾)
        from personavoice.config import SOTA_CONFIG, get_inference_kwargs
        inference_kwargs = get_inference_kwargs()

        # v10.1: 删除 Duration Predictor, 用 LAAG + F5-TTS 官方时长公式
        for i, sample in enumerate(samples):
            r = evaluate_v91_sample(
                sample, backbone, tokenizer, ecapa_evaluator, wer_evaluator,
                seconds_per_word=SOTA_CONFIG.seconds_per_char,
                cfg_strength=inference_kwargs["cfg_strength"],
                steps=inference_kwargs["steps"],
                device=device,
                use_oes=True, env_weight=SOTA_CONFIG.env_weight,
                use_ceag=inference_kwargs["use_ceag"],  # v10.0: CEAG (升级自 IEAG)
                ceag_lambda_max=inference_kwargs["ceag_lambda_max"],
                ceag_t_start=inference_kwargs["ceag_t_start"],
                ceag_t_end=inference_kwargs["ceag_t_end"],
                ceag_layers=inference_kwargs["ceag_layers"],
                sway_sampling_coef=inference_kwargs["sway_sampling_coef"],
            )
            pv_results.append(r)

            if (i + 1) % 20 == 0:
                mean_secs = np.mean([r["secs"] for r in pv_results])
                mean_wer = np.mean([r["wer"] for r in pv_results])
                logger.info(f"  PersonaVoice v10.0 [{i+1}/{len(samples)}] "
                            f"SECS={mean_secs:.4f}, WER={mean_wer:.4f}")
    else:
        logger.info("\n[4/5] Skipping PersonaVoice evaluation")

    # 5. 统计分析
    logger.info("\n[5/5] Computing statistics...")
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    # 完整结果
    full_results = {
        "metadata": {
            "n_samples": len(samples),
            "seed": seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "v10.0",
            "personavoice_config": {
                "steps": SOTA_CONFIG.steps,
                "cfg_strength": SOTA_CONFIG.cfg_strength,
                "use_oes": True,
                "env_weight": SOTA_CONFIG.env_weight,
                "oes_env_scale_init": SOTA_CONFIG.oes_env_scale_init,
                "use_ceag": SOTA_CONFIG.use_ceag,
                "ceag_lambda_max": SOTA_CONFIG.ceag_lambda_max,
                "ceag_t_start": SOTA_CONFIG.ceag_t_start,
                "ceag_t_end": SOTA_CONFIG.ceag_t_end,
                "ceag_layers": SOTA_CONFIG.ceag_layers,
                "use_laag": SOTA_CONFIG.use_laag,
                "seconds_per_char": SOTA_CONFIG.seconds_per_char,
                "sway_sampling_coef": SOTA_CONFIG.sway_sampling_coef,
                "removed_modules": ["IBOP", "AM-ODE", "TD-CFG", "SBM", "GRPO", "Duration Predictor"],
            },
        },
        "baseline_results": baseline_results,
        "personavoice_results": pv_results,
    }

    with open(results_dir / "eval_200_samples.json", "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    logger.info(f"  Full results saved to: {results_dir / 'eval_200_samples.json'}")

    # 统计显著性分析
    statistics = {}

    if baseline_results and pv_results:
        baseline_secs = [r["secs"] for r in baseline_results]
        baseline_wer = [r["wer"] for r in baseline_results]
        pv_secs = [r["secs"] for r in pv_results]
        pv_wer = [r["wer"] for r in pv_results]

        statistics = {
            "baseline": {
                "secs": compute_statistics(baseline_secs),
                "wer": compute_statistics(baseline_wer),
            },
            "personavoice": {
                "secs": compute_statistics(pv_secs),
                "wer": compute_statistics(pv_wer),
            },
            "paired_t_test_secs": compute_paired_t_test(baseline_secs, pv_secs),
            "paired_t_test_wer": compute_paired_t_test(baseline_wer, pv_wer),
        }

        # 按文本长度分层
        statistics["by_length_baseline"] = analyze_by_text_length(baseline_results)
        statistics["by_length_personavoice"] = analyze_by_text_length(pv_results)

    elif pv_results:
        pv_secs = [r["secs"] for r in pv_results]
        pv_wer = [r["wer"] for r in pv_results]
        statistics = {
            "personavoice": {
                "secs": compute_statistics(pv_secs),
                "wer": compute_statistics(pv_wer),
            },
            "by_length_personavoice": analyze_by_text_length(pv_results),
        }

    elif baseline_results:
        baseline_secs = [r["secs"] for r in baseline_results]
        baseline_wer = [r["wer"] for r in baseline_results]
        statistics = {
            "baseline": {
                "secs": compute_statistics(baseline_secs),
                "wer": compute_statistics(baseline_wer),
            },
            "by_length_baseline": analyze_by_text_length(baseline_results),
        }

    with open(results_dir / "eval_200_statistics.json", "w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    logger.info(f"  Statistics saved to: {results_dir / 'eval_200_statistics.json'}")

    # 信息熵衰减定律数据
    if pv_results:
        entropy_data = collect_entropy_decay_data(pv_results)
        with open(results_dir / "entropy_decay_data.json", "w", encoding="utf-8") as f:
            json.dump(entropy_data, f, indent=2, ensure_ascii=False)
        logger.info(f"  Entropy decay data saved to: {results_dir / 'entropy_decay_data.json'}")

    # 汇总打印
    logger.info("\n" + "=" * 70)
    logger.info(f"PersonaVoice v9.1 {n_samples}-Sample Evaluation Summary")
    logger.info("=" * 70)

    if statistics:
        if "baseline" in statistics:
            b = statistics["baseline"]
            logger.info(f"\nF5-TTS Baseline (n={b['secs']['n']}):")
            logger.info(f"  SECS = {b['secs']['mean']:.4f} ± {b['secs']['std']:.4f} "
                        f"(95% CI: [{b['secs']['ci_95_low']:.4f}, {b['secs']['ci_95_high']:.4f}])")
            logger.info(f"  WER  = {b['wer']['mean']:.4f} ± {b['wer']['std']:.4f} "
                        f"(95% CI: [{b['wer']['ci_95_low']:.4f}, {b['wer']['ci_95_high']:.4f}])")

        if "personavoice" in statistics:
            p = statistics["personavoice"]
            logger.info(f"\nPersonaVoice v9.1 (n={p['secs']['n']}):")
            logger.info(f"  SECS = {p['secs']['mean']:.4f} ± {p['secs']['std']:.4f} "
                        f"(95% CI: [{p['secs']['ci_95_low']:.4f}, {p['secs']['ci_95_high']:.4f}])")
            logger.info(f"  WER  = {p['wer']['mean']:.4f} ± {p['wer']['std']:.4f} "
                        f"(95% CI: [{p['wer']['ci_95_low']:.4f}, {p['wer']['ci_95_high']:.4f}])")

        if "paired_t_test_secs" in statistics:
            t_secs = statistics["paired_t_test_secs"]
            logger.info(f"\nStatistical Significance (SECS, paired t-test):")
            logger.info(f"  Mean diff = {t_secs['mean_diff']:.4f}")
            logger.info(f"  t-statistic = {t_secs['t_statistic']:.4f}")
            logger.info(f"  p-value (one-sided) = {t_secs['p_value_one_sided']:.6f}")
            logger.info(f"  Cohen's d = {t_secs['cohen_d']:.4f} ({t_secs['effect_size']})")
            logger.info(f"  Significant at α=0.05: {t_secs['significant_at_alpha']}")

        if "paired_t_test_wer" in statistics:
            t_wer = statistics["paired_t_test_wer"]
            logger.info(f"\nStatistical Significance (WER, paired t-test):")
            logger.info(f"  Mean diff = {t_wer['mean_diff']:.4f}")
            logger.info(f"  t-statistic = {t_wer['t_statistic']:.4f}")
            logger.info(f"  p-value (one-sided) = {t_wer['p_value_one_sided']:.6f}")
            logger.info(f"  Cohen's d = {t_wer['cohen_d']:.4f} ({t_wer['effect_size']})")

        # 分层分析
        if "by_length_personavoice" in statistics:
            bl = statistics["by_length_personavoice"]
            logger.info(f"\nPersonaVoice v9.1 by Text Length (threshold={bl['threshold_words']} words):")
            s = bl["short_text"]
            l = bl["long_text"]
            logger.info(f"  Short ({s['n_samples']}): SECS={s['secs']['mean']:.4f}±{s['secs']['std']:.4f}, "
                        f"WER={s['wer']['mean']:.4f}±{s['wer']['std']:.4f}")
            logger.info(f"  Long  ({l['n_samples']}): SECS={l['secs']['mean']:.4f}±{l['secs']['std']:.4f}, "
                        f"WER={l['wer']['mean']:.4f}±{l['wer']['std']:.4f}")

    logger.info("\n" + "=" * 70)
    logger.info("Evaluation complete.")
    logger.info("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="200-sample statistical evaluation")
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of test samples (default: 200)")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip F5-TTS baseline evaluation")
    parser.add_argument("--skip_personavoice", action="store_true",
                        help="Skip PersonaVoice evaluation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sample selection")
    args = parser.parse_args()

    run_evaluation(
        n_samples=args.n_samples,
        skip_baseline=args.skip_baseline,
        skip_personavoice=args.skip_personavoice,
        seed=args.seed,
    )
