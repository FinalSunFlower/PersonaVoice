"""验证 1111.mp3 完美克隆 - SECS SOTA 测试 (v10.4 满血架构, 无备用方案)"""
import os
# 设置离线模式, 避免连接 huggingface.co 超时 (模型已在本地缓存)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
# v10.4: 禁用 wandb (f5_tts.trainer 依赖 wandb, 与 speechbrain lazy import 冲突)
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import sys
import time
import torch
import torchaudio
import numpy as np
from pathlib import Path

# 添加项目路径 (兼容直接运行与 -m 运行)
# examples/clone_demo.py → 项目根目录 = 上两级
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# v10.4 关键: 先导入 f5_tts (避免 speechbrain lazy import 与 wandb/pydantic 冲突)
from f5_tts.api import F5TTS
from f5_tts.model.utils import list_str_to_idx, get_tokenizer

from personavoice.config import SOTA_CONFIG
from personavoice.tts_backbone.f5_pretrained_backbone import load_pretrained_f5tts_backbone


def compute_secs(ref_audio_path, gen_audio_path, device="cpu"):
    """计算 SECS (Speaker Encoder Cosine Similarity) - 满血架构, 无备用方案.

    使用项目自带的 EcapaEvaluator (已处理 speechbrain lazy import 问题).
    """
    import torchaudio
    from personavoice.experiment.ecapa_evaluator import ECAPAEvaluator

    evaluator = ECAPAEvaluator(device=device, vocoder_device=device)
    if not evaluator.available:
        raise RuntimeError("ECAPA evaluator 不可用")

    # 加载参考音频 (16kHz mono)
    ref_wav, ref_sr = torchaudio.load(ref_audio_path)
    if ref_sr != 16000:
        ref_wav = torchaudio.functional.resample(ref_wav, ref_sr, 16000)
    if ref_wav.shape[0] > 1:
        ref_wav = ref_wav.mean(dim=0, keepdim=True)

    # 加载生成音频 (16kHz mono)
    gen_wav, gen_sr = torchaudio.load(gen_audio_path)
    if gen_sr != 16000:
        gen_wav = torchaudio.functional.resample(gen_wav, gen_sr, 16000)
    if gen_wav.shape[0] > 1:
        gen_wav = gen_wav.mean(dim=0, keepdim=True)

    return evaluator.compute_secs(gen_wav, ref_wav, sr=16000)


def main():
    print("=" * 70)
    print("PersonaVoice v10.4.8 - 1111 完美克隆验证 (F5-TTS 官方流程)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. 加载参考音频 1111.mp3 (35KB, 项目内置 demo 参考)
    print("\n[1/3] 加载 1111.mp3...")
    ref_audio_path = str(PROJECT_ROOT / "1111.mp3")
    if not os.path.exists(ref_audio_path):
        print(f"  [ERROR] 文件不存在: {ref_audio_path}")
        return

    t0 = time.time()
    # v10.4: 直接加载音频 (与参数搜索一致, 不用 preprocess_ref_audio)
    # preprocess_ref_audio 会添加 50ms 尾部静音, 导致 SECS 下降
    audio_ref, sr_ref = torchaudio.load(ref_audio_path)
    if audio_ref.shape[0] > 1:
        audio_ref = audio_ref.mean(dim=0, keepdim=True)
    # 重采样到 24kHz (F5-TTS 要求)
    if sr_ref != SOTA_CONFIG.sample_rate:
        audio_ref = torchaudio.functional.resample(audio_ref, sr_ref, SOTA_CONFIG.sample_rate)
        sr_ref = SOTA_CONFIG.sample_rate
    print(f"  预处理耗时: {time.time()-t0:.2f}s")
    print(f"  audio_ref shape: {audio_ref.shape}, sr: {sr_ref}")

    # 提取 mel (用于签名兼容, 实际 v10.3 用 audio_ref)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr_ref,
        n_fft=1024,
        hop_length=SOTA_CONFIG.hop_length,
        win_length=1024,
        n_mels=SOTA_CONFIG.n_mels,
    )
    if audio_ref.ndim == 1:
        audio_ref = audio_ref.unsqueeze(0)
    mel_ref = mel_transform(audio_ref)  # (1, n_mels, T)
    mel_ref = torch.log(torch.clamp(mel_ref, min=1e-5))
    print(f"  mel_ref shape: {mel_ref.shape}")

    # 2. 加载 backbone
    print("\n[2/3] 加载 F5-TTS backbone...")
    t0 = time.time()
    backbone_result = load_pretrained_f5tts_backbone(device=device, use_film=True)
    if isinstance(backbone_result, tuple):
        backbone = backbone_result[0]
    else:
        backbone = backbone_result
    print(f"  Backbone 加载耗时: {time.time()-t0:.2f}s")
    print(f"  Backbone type: {type(backbone).__name__}")

    # 2.5 提取真实 speaker_emb (ECAPA 从 1111.mp3 提取, 满血架构)
    print("\n[2.5/3] 提取真实 speaker_emb (ECAPA) + 参考音频增强 + 综合评估器...")
    from personavoice.experiment.ecapa_evaluator import ECAPAEvaluator
    from personavoice.experiment.comprehensive_evaluator import ComprehensiveEvaluator
    ecapa = ECAPAEvaluator(device=device, vocoder_device=device)
    # 综合评估器 (SECS + WER + UTMOS + RTF + SIM-o)
    evaluator = ComprehensiveEvaluator(device=device, vocoder_device=device)
    # v10.4: 移除参考音频增强 (循环扩展伤害 SECS, 官方基线直接用原始 1s 音频)
    # 直接用原始 1s 音频提取 ECAPA 嵌入 (与 F5-TTS 官方基线一致)
    audio_ref_16k = torchaudio.functional.resample(audio_ref, sr_ref, 16000)
    if audio_ref_16k.ndim == 1:
        audio_ref_16k = audio_ref_16k.unsqueeze(0)
    # v10.4: 直接从原始 1s 音频提取 ECAPA 嵌入 (无增强, 与官方基线一致)
    speaker_emb_real = ecapa.extract_embedding(audio_ref_16k, sr=16000).to(device)
    if speaker_emb_real.dim() == 1:
        speaker_emb_real = speaker_emb_real.unsqueeze(0)
    print(f"  speaker_emb shape: {speaker_emb_real.shape}, norm: {speaker_emb_real.norm():.4f}")

    # 2.6 用 Whisper 转录 1111.mp3 得到 ref_text (F5-TTS 官方正确用法)
    print("\n[2.6/3] Whisper 转录参考音频得到 ref_text...")
    from transformers import pipeline as hf_pipeline
    import numpy as np
    # 直接传入 numpy 数组, 避免 ffmpeg 依赖
    asr_pipe = hf_pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        device=device,
    )
    audio_ref_np = audio_ref_16k.squeeze(0).cpu().numpy().astype(np.float32)
    asr_result = asr_pipe(audio_ref_np, generate_kwargs={"language": "chinese"})
    ref_text = asr_result["text"].strip()
    print(f"  ref_text: '{ref_text}'")

    # 3. 克隆合成 (v10.2: LAAG + F5-TTS 官方 ref_text 拼接)
    print("\n[3/3] 执行克隆合成...")
    test_texts = [
        "你好，这是一个完美的声音克隆测试。",
        "Hello, this is a perfect voice cloning test.",
        "今天天气真好，我们一起出去走走吧。",
        # v10.1 LAAG 长文本测试 (触发 chunk 机制)
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，可以设想，未来人工智能带来的科技产品，将会是人类智慧的容器。",
    ]

    os.makedirs("outputs", exist_ok=True)

    # 准备 tokenizer
    from f5_tts.model.utils import list_str_to_idx, get_tokenizer
    # 自动查找 F5-TTS vocab.txt 位置 (兼容不同安装方式)
    import f5_tts
    f5_pkg_dir = Path(f5_tts.__file__).resolve().parent
    vocab_candidates = [
        f5_pkg_dir / "infer" / "examples" / "vocab.txt",
        f5_pkg_dir.parent / "f5_tts" / "infer" / "examples" / "vocab.txt",
        PROJECT_ROOT / "f5_tts" / "infer" / "examples" / "vocab.txt",
    ]
    vocab_file = next((str(p) for p in vocab_candidates if p.exists()), None)
    if vocab_file is None:
        raise FileNotFoundError(
            "无法找到 F5-TTS vocab.txt, 请确认 f5-tts 已安装。候选路径: "
            + ", ".join(str(p) for p in vocab_candidates)
        )
    vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer="custom")

    for i, text in enumerate(test_texts):
        print(f"\n--- 测试 {i+1}/{len(test_texts)} ---")
        print(f"  文本: {text}")

        t0 = time.time()

        # 准备输入
        mel_ref_b = mel_ref.to(device).float()
        if mel_ref_b.ndim == 3:
            mel_ref_b = mel_ref_b.permute(0, 2, 1)  # (B, n_mels, T) -> (B, T, n_mels)

        # 用真实 ECAPA speaker_emb (满血架构, 无零向量降级)
        speaker_emb = speaker_emb_real
        # v10.4.6: persona_emb 由 LAAG 动态决定 (短文本=0, 长文本=speaker_emb)
        # 见 laag_generator.py 的动态 FiLM 机制
        persona_emb = speaker_emb.clone()  # 传递给 LAAG, 由 LAAG 决定是否使用
        emotion_emb = torch.zeros(1, 4, device=device)

        # v10.2.1 LAAG + F5-TTS 官方 cond=audio + ref_text 拼接 (满血架构)
        print(f"  [LAAG] 长度自适应生成 + cond=audio + ref_text 拼接")
        # audio_ref: (B, T_samples) 24kHz 音频波形 (F5-TTS 官方 cond)
        audio_ref_b = audio_ref.to(device).float()
        if audio_ref_b.ndim == 1:
            audio_ref_b = audio_ref_b.unsqueeze(0)
        mel_gen, laag_info = backbone.laag_synthesize(
            text_str=text,
            mel_ref=mel_ref_b,
            speaker_emb=speaker_emb,
            persona_emb=persona_emb,
            emotion_emb=emotion_emb,
            tokenizer=type("T", (), {"encode": staticmethod(lambda t: list_str_to_idx([t], vocab_char_map=vocab_char_map).squeeze(0))})(),
            ref_text=ref_text,
            audio_ref=audio_ref_b,
        )
        print(f"  [LAAG] 策略: {laag_info.get('strategy')}, chunks: {laag_info.get('chunks')}, has_ref_text: {laag_info.get('has_ref_text')}")

        # v10.3: 优先用官方流程生成的 audio (避免重复 vocoder 转换)
        gen_time = time.time() - t0
        if hasattr(backbone, '_last_audio_gen') and backbone._last_audio_gen is not None:
            audio_gen_np = backbone._last_audio_gen.cpu().numpy().squeeze()
            sr_gen = backbone._last_sr_gen
            print(f"  [v10.3] 使用官方流程生成的 audio: shape={audio_gen_np.shape}, sr={sr_gen}")
        else:
            # 备选: 用 vocoder 转换 mel (理论上不会走到这里)
            if isinstance(mel_gen, tuple):
                mel_gen = mel_gen[0]
            mel_gen_np = mel_gen.cpu().numpy()
            vocoder = backbone._f5tts.vocoder
            # v10.4.6 修复: mel_gen 是 (B, 100, T), vocoder.decode 期望 (B, n_mels, T)
            # 之前错误地 permute(0,2,1) 变成 (B, T, 100), 导致音质下降
            mel_for_vocoder = mel_gen.to(device)
            with torch.no_grad():
                audio_gen = vocoder.decode(mel_for_vocoder)
            audio_gen_np = audio_gen.cpu().numpy().squeeze()
            sr_gen = SOTA_CONFIG.sample_rate

        # 保存音频
        out_wav = f"outputs/1111_clone_test_{i+1}.wav"
        import soundfile as sf
        sf.write(out_wav, audio_gen_np, sr_gen)
        print(f"  生成音频 shape: {audio_gen_np.shape}")
        print(f"  保存到: {out_wav}")
        print(f"  合成耗时: {gen_time:.2f}s")

        # v10.3: 综合评估 (SECS + WER + UTMOS + RTF + SIM-o)
        gen_wav, gen_sr = torchaudio.load(out_wav)
        if gen_wav.shape[0] > 1:
            gen_wav = gen_wav.mean(dim=0, keepdim=True)
        if gen_sr != 16000:
            gen_wav = torchaudio.functional.resample(gen_wav, gen_sr, 16000)

        # 判断语言 (英文用英文评估, 中文用中文)
        lang = "english" if all(ord(c) < 128 for c in text if c.isalpha()) else "chinese"
        metrics = evaluator.evaluate_all(
            gen_wav=gen_wav,
            ref_wav=audio_ref_16k,
            reference_text=text,
            sr=16000,
            gen_time_sec=gen_time,
            language=lang,
        )
        # v10.4.6: 基于实验数据的合理目标 (F5 基线均值)
        # F5 基线 SECS 方差测试结果: 中文短1=0.51±0.06, 英文短=0.31±0.03
        # PV 目标: 匹配或超过 F5 基线均值
        is_long = laag_info.get('chunks', 1) > 1
        is_english = all(ord(c) < 128 for c in text if c.isalpha())
        if is_long:
            secs_target = 0.55  # 长文本 (chunked + FiLM) 目标 0.55
        elif is_english:
            secs_target = 0.30  # 英文短文本 (F5 基线均值 0.31)
        else:
            secs_target = 0.50  # 中文短文本 (F5 基线均值 0.51)

        print(f"  ┌─── 综合评估 ───")
        print(f"  │ SECS:   {metrics['secs']:.4f}  (目标 >= {secs_target:.2f}, F5基线)")
        print(f"  │ WER:    {metrics['wer']:.4f}  (目标 <= 0.10)")
        print(f"  │ UTMOS:  {metrics['utmos']:.4f}  (目标 >= 4.0)")
        print(f"  │ SIM-o:  {metrics['sim_o']:.4f}  (目标 >= 0.50)")
        print(f"  │ RTF:    {metrics['rtf']:.4f}  (目标 <= 1.0)")
        print(f"  │ 时长:   {metrics['audio_duration']:.2f}s")
        print(f"  │ ASR:    '{metrics['recognized_text'][:60]}'")
        print(f"  └────────────────")

    print("\n" + "=" * 70)
    print("验证完成! 检查 outputs/ 目录")
    print("=" * 70)


if __name__ == "__main__":
    main()
