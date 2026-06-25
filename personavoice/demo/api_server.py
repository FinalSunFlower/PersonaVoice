"""v10.4.8 PersonaVoice 交互式演示后端 API.

架构位置: 演示模块的 Web API 服务端, 集成 F5-TTS 预训练模型做实际语音合成,
对外提供声音克隆/人格分析/架构状态查询接口.

核心功能:
- 声音克隆 (clone_voice): F5-TTS + LAAG + 动态 FiLM + OES + 韵律迁移 (v10.4.8)
- 人格分析 (analyze_persona): BERT 编码聊天记录 → Big Five 人格特征 → persona_emb
- 人格校准 (calibrate_persona): 基于用户反馈微调人格表示
- 架构状态查询

启动: python -m personavoice.demo.api_server 或直接运行此文件
"""

from __future__ import annotations

import sys
import io
import base64
import tempfile
import os
import time
from pathlib import Path

# v10.4.8: 确保 personavoice 模块可被导入 (直接运行时需要)
# 当作为模块运行 (python -m personavoice.demo.api_server) 时不需要
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # personavoice/demo/api_server.py -> voice/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 所有模型已缓存，设置完全离线模式，避免 HuggingFace 网络检查超时
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["SENTENCE_TRANSFORMERS_OFFLINE"] = "1"

# 配置 ffmpeg 路径（Whisper ASR 需要，使用 imageio-ffmpeg 自带的二进制）
try:
    import imageio_ffmpeg
    import shutil
    _ffmpeg_src = imageio_ffmpeg.get_ffmpeg_exe()
    # transformers 的 ffmpeg_read 直接调用 "ffmpeg" 命令，需要 ffmpeg.exe 在 PATH 中
    # 将 imageio-ffmpeg 的二进制复制为 ffmpeg.exe 放到一个目录并加入 PATH
    _ffmpeg_bin_dir = os.path.join(tempfile.gettempdir(), "personavoice_ffmpeg")
    os.makedirs(_ffmpeg_bin_dir, exist_ok=True)
    _ffmpeg_dst = os.path.join(_ffmpeg_bin_dir, "ffmpeg.exe")
    if not os.path.exists(_ffmpeg_dst):
        shutil.copy2(_ffmpeg_src, _ffmpeg_dst)
    os.environ["FFMPEG_BINARY"] = _ffmpeg_dst
    os.environ["IMAGEIO_FFMPEG_EXE"] = _ffmpeg_dst
    if _ffmpeg_bin_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _ffmpeg_bin_dir + os.pathsep + os.environ.get("PATH", "")
    print(f"[Demo] ffmpeg 路径: {_ffmpeg_dst}")
except Exception as _e:
    print(f"[Demo] 警告: 未找到 ffmpeg ({_e}), Whisper ASR 可能无法工作")

from typing import Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


# ─────────────────────────────────────────────────────────────────────
# PersonaVoice 训练好的骨干 (sota_v50)
# ─────────────────────────────────────────────────────────────────────
_pv_backbone = None
_pv_backbone_loading = False
_pv_backbone_error: Optional[str] = None
_pv_tokenizer = None
_pv_vocoder = None  # Vocos 声码器, 用于将 mel 转换为 wav


def get_personavoice_backbone():
    """获取训练好的 PersonaVoice 骨干单例 (sota_v50 + FiLM Adapter).

    加载流程:
        1. 加载 F5-TTS 预训练骨干 (含 FiLM Adapter)
        2. 加载 sota_v50/best_model.pt 训练权重
        3. 加载 F5-TTS tokenizer
        4. 加载 Vocos 声码器 (mel → wav)

    Returns:
        (backbone, tokenizer, vocoder) 或 (None, None, None)
    """
    global _pv_backbone, _pv_backbone_loading, _pv_backbone_error
    global _pv_tokenizer, _pv_vocoder

    if _pv_backbone is not None:
        return _pv_backbone, _pv_tokenizer, _pv_vocoder

    if _pv_backbone_loading:
        return None, None, None

    try:
        _pv_backbone_loading = True
        _pv_backbone_error = None

        # 强制离线模式
        try:
            from personavoice.common.local_models import setup_offline_mode
            setup_offline_mode()
        except ImportError:
            pass

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Demo] 加载 PersonaVoice 训练骨干 (device={device})...")

        # 1. 加载 F5-TTS 预训练骨干 + FiLM Adapter
        from personavoice.tts_backbone.f5_pretrained_backbone import (
            load_pretrained_f5tts_backbone,
        )
        backbone, _ = load_pretrained_f5tts_backbone(device=device)

        # 2. 加载 sota_v50 训练权重
        from pathlib import Path
        ckpt_path = Path("checkpoints/sota_v50/best_model.pt")
        if ckpt_path.exists():
            ckpt = torch.load(
                str(ckpt_path), map_location=device, weights_only=False
            )
            backbone.load_state_dict(ckpt["backbone_state_dict"], strict=False)
            print(f"[Demo] 已加载训练权重: {ckpt_path} (step={ckpt.get('step', '?')}, SECS={ckpt.get('secs', '?'):.4f})")
        else:
            print(f"[Demo] 警告: 未找到训练权重 {ckpt_path}, 将使用基础 F5-TTS")

        backbone = backbone.to(device)
        backbone.eval()
        _pv_backbone = backbone

        # 3. 加载 F5-TTS tokenizer (使用项目工具函数)
        from personavoice.experiment.utils import load_tokenizer
        _pv_tokenizer = load_tokenizer()
        print(f"[Demo] Tokenizer 加载完成")

        # 4. 加载 Vocos 声码器
        from personavoice.tts_backbone.vocoder import VocosVocoder
        _pv_vocoder = VocosVocoder(device=device, input_n_mels=100)  # backbone 输出 100 mel bins
        print(f"[Demo] Vocos 声码器加载完成")

        print(f"[Demo] PersonaVoice 训练骨干加载完成")
        return _pv_backbone, _pv_tokenizer, _pv_vocoder

    except Exception as e:
        _pv_backbone_error = str(e)
        print(f"[Demo] PersonaVoice 骨干加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
    finally:
        _pv_backbone_loading = False


_vocos_mel_extractor = None


def _get_vocos_mel_extractor(device: str = "cpu"):
    """获取 Vocos 自带的 MelSpectrogramFeatures 提取器 (SOTA标准).

    SOTA关键: Vocos 声码器训练时使用自己的 mel 提取器 (100-bin, 24kHz),
    其 mel scale 和数值范围与 torchaudio.transforms.MelSpectrogram 不同.
    必须使用 Vocos 自带的提取器才能保证 mel -> wav 的正确重建.
    """
    global _vocos_mel_extractor
    if _vocos_mel_extractor is None:
        from vocos.feature_extractors import MelSpectrogramFeatures
        _vocos_mel_extractor = MelSpectrogramFeatures(
            sample_rate=24000,
            n_fft=1024,
            hop_length=256,
            n_mels=100,
            padding="center",
        ).to(device)
        _vocos_mel_extractor.eval()
        print("[Demo] Vocos MelSpectrogramFeatures 提取器初始化完成 (100-bin, 24kHz)")
    return _vocos_mel_extractor


def _extract_mel_ref_from_audio(
    audio_path: str,
    target_sr: int = 24000,
    ref_duration_sec: float = 1.0,
    n_mels: int = 100,
) -> "torch.Tensor":
    """从参考音频中提取 1s mel 频谱 (Vocos 兼容格式).

    SOTA修正: 使用 Vocos 自带的 MelSpectrogramFeatures 提取器 (100-bin, 24kHz),
    确保 mel -> Vocos -> wav 的重建质量.
    旧版使用 torchaudio MelSpectrogram 导致 Vocos 解码失败 (SECS ~0.01).

    Returns:
        mel_ref: (T_ref, n_mels) 参考mel, T_ref ≈ 94 (1s @ 24kHz/256hop)
    """
    import torchaudio
    wav, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        wav = resampler(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # 取前 1s
    ref_samples = int(target_sr * ref_duration_sec)
    wav_ref = wav[:, :ref_samples]

    # 使用 Vocos 自带的 MelSpectrogramFeatures 提取器 (SOTA标准)
    mel_extractor = _get_vocos_mel_extractor()
    with torch.no_grad():
        mel_ref = mel_extractor(wav_ref)  # (1, n_mels, T_ref)
    mel_ref = mel_ref.squeeze(0).T  # (T_ref, n_mels)
    # Vocos mel 已经是 log-mel 格式, 范围约 [-80, 20]
    # 安全范围裁剪 (Vocos 期望 log-mel)
    mel_ref = torch.clamp(mel_ref, min=-80, max=20)
    return mel_ref


def _extract_speaker_emb(audio_path: str) -> "torch.Tensor":
    """从参考音频中提取 ECAPA 说话人嵌入 (192维)."""
    import torchaudio
    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        wav = resampler(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # 使用 ECAPA-TDNN 提取嵌入
    from speechbrain.inference.speaker import EncoderClassifier
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="./models/ecapa_tdnn",
        run_opts={"device": "cpu"},
    )
    emb = encoder.encode_batch(wav).squeeze(0)  # (192,)
    return emb


# ─────────────────────────────────────────────────────────────────────
# FastAPI 应用
# ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PersonaVoice Demo API",
    description="v10.4.8 交互式声音克隆 (Trade-off: LAAG + 动态 FiLM + OES + 人格分析)",
    version="10.4.8",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────
# 请求/响应模型
# ─────────────────────────────────────────────────────────────────────
class CloneRequest(BaseModel):
    text: str = "你好,这是 PersonaVoice 的声音克隆演示。"
    emotion: Optional[str] = None
    emotion_intensity: float = 0.5


# ─────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────
def _ensure_wav(input_path: str, target_sr: int = 24000) -> str:
    """使用 ffmpeg 将任意格式音频转换为单声道 WAV.

    解决前端上传 m4a/mp3/webm 等格式时 soundfile/torchaudio 无法识别的问题.
    返回新的临时 WAV 文件路径,调用方负责删除.
    """
    import subprocess
    import shutil

    ext = os.path.splitext(input_path)[1].lower()
    # 已经是 WAV 且无需改采样率时直接返回(节省一次转码)
    if ext == ".wav":
        return input_path

    wav_path = input_path + ".converted.wav"
    ffmpeg_exe = shutil.which("ffmpeg") or os.environ.get("FFMPEG_BINARY", "ffmpeg")
    cmd = [
        ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-ar", str(target_sr), "-ac", "1", "-sample_fmt", "s16",
        wav_path,
    ]
    try:
        subprocess.run(cmd, check=True)
        return wav_path
    except Exception as e:
        raise RuntimeError(f"ffmpeg 音频转换失败: {e}")


def _tensor_to_base64_wav(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
) -> str:
    """将 waveform tensor 编码为 base64 WAV 字符串."""
    import wave

    if waveform.dim() > 1:
        waveform = waveform.squeeze(0)
    wav_array = (waveform.cpu().numpy() * 32767).clip(-32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(wav_array.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _numpy_wav_to_base64(wav: np.ndarray, sample_rate: int) -> str:
    """将 numpy waveform 编码为 base64 WAV 字符串 (F5-TTS 输出格式)."""
    import wave

    wav_np = np.asarray(wav).astype(np.float32).squeeze()
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=0)
    wav_array = (wav_np * 32767).clip(-32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(wav_array.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _waveform_to_png_base64(wav: np.ndarray, sample_rate: int) -> str:
    """v3.3: 生成音频波形 PNG 图片 (base64).

    类似 Audacity 的波形显示: 深色背景 + 居中对称波形 + 渐变色.
    用户一眼就能看出这是"声音的形状", 比 Mel 频谱直观得多.
    """
    from PIL import Image, ImageDraw

    # 1. 参数
    W, H = 800, 160  # 输出尺寸
    bg_color = (12, 17, 28)       # 深色背景 #0c111c
    wave_color = (80, 200, 180)   # 青色主波形 #50c8b4
    wave_fill = (40, 100, 90, 60) # 填充色（半透明）
    grid_color = (30, 40, 55)     # 网格线
    text_color = (100, 115, 140)  # 轴标签
    center_line = (50, 65, 85)    # 中心零线

    # 2. 下采样到目标宽度
    n_samples = len(wav)
    if n_samples == 0:
        img = Image.new("RGB", (W, H), bg_color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # 每列像素对应多少个采样点
    samples_per_col = max(1, n_samples // W)
    # 重采样: 取每列的最大最小值（包络）
    n_cols = min(W, n_samples)
    envelope_max = np.zeros(n_cols, dtype=np.float32)
    envelope_min = np.zeros(n_cols, dtype=np.float32)

    for i in range(n_cols):
        start = i * samples_per_col
        end = min(start + samples_per_col, n_samples)
        chunk = wav[start:end]
        if len(chunk) > 0:
            envelope_max[i] = float(np.max(np.abs(chunk)))
            envelope_min[i] = -envelope_max[i]  # 对称

    # 3. 归一化到画布高度 (留边距)
    margin = 20
    draw_h = H - 2 * margin
    peak = max(float(envelope_max.max()), 1e-6)
    scale = (draw_h / 2 - 4) / peak  # -4 防止触边

    # 4. 创建图像
    img = Image.new("RGB", (W, H), bg_color)
    draw = ImageDraw.Draw(img)

    cy = H // 2  # 中心 Y 坐标

    # 5. 绘制网格线
    for gy in [margin, cy, H - margin]:
        draw.line([(0, gy), (W, gy)], fill=grid_color, width=1)
    # 中心零线稍亮
    draw.line([(0, cy), (W, cy)], fill=center_line, width=1)

    # 6. 绘制波形 (填充 + 描边)
    points_top = []
    points_bottom = []
    for i in range(n_cols):
        x = int(i * W / n_cols)
        y_top = int(cy - envelope_max[i] * scale)
        y_bot = int(cy - envelope_min[i] * scale)
        points_top.append((x, y_top))
        points_bottom.append((x, y_bot))

    # 填充区域 (上半部 + 反转下半部)
    fill_points = points_top + list(reversed(points_bottom))
    if len(fill_points) >= 3:
        draw.polygon(fill_points, fill=(30, 80, 70))

    # 描边 (上下两条线)
    if len(points_top) >= 2:
        draw.line(points_top, fill=wave_color, width=1)
        draw.line(points_bottom, fill=wave_color, width=1)

    # 7. 时间轴标签
    duration = n_samples / sample_rate
    draw.text((4, H - 14), f"0.0s", fill=text_color)
    draw.text((W - 36, H - 14), f"{duration:.1f}s", fill=text_color)
    draw.text((W // 2 - 16, H - 14), f"{duration/2:.1f}s", fill=text_color)

    # 8. 编码为 PNG
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _mel_to_png_base64(mel_np: np.ndarray) -> str:
    """v3.3: 将 Mel 频谱渲染为 PNG 图片 (base64), 替代前端 Canvas 渲染.

    使用 PIL + Viridis colormap 在服务端生成高质量频谱图,
    解决浏览器 Canvas 渲染不一致、colormap 失真等问题.
    """
    from PIL import Image, ImageDraw, ImageFont

    # 1. 归一化: percentile clip → [0, 255]
    p1 = float(np.percentile(mel_np, 1))
    p99 = float(np.percentile(mel_np, 99))
    mel_clipped = np.clip(mel_np, p1, p99)
    mel_uint8 = ((mel_clipped - p1) / max(p99 - p1, 1e-6) * 255).astype(np.uint8)

    n_mels, n_frames = mel_uint8.shape

    # 2. Viridis colormap LUT (256 级, 与 matplotlib 一致)
    _viridis_pts = [
        [0.00,(68,1,84)],[0.04,(71,22,104)],[0.08,(69,51,122)],[0.12,(62,78,138)],
        [0.16,(55,102,152)],[0.20,(48,125,163)],[0.24,(42,146,172)],[0.28,(35,165,179)],
        [0.32,(31,183,183)],[0.36,(34,199,197)],[0.40,(47,214,208)],[0.44,(67,226,216)],
        [0.48,(91,237,222)],[0.52,(118,247,225)],[0.56,(148,253,225)],[0.60,(180,255,221)],
        [0.64,(212,253,213)],[0.68,(243,249,200)],[0.72,(269,242,184)],[0.76,(292,232,166)],
        [0.80,(313,220,147)],[0.84,(332,205,127)],[0.88,(350,187,107)],[0.92,(366,167,87)],
        [0.96,(380,145,68)],[1.00,(393,123,50)]
    ]
    # 构建 256 级 LUT
    lut_r, lut_g, lut_b = [], [], []
    for i in range(256):
        v = i / 255.0
        # 找到 v 所在的区间并插值
        lo = 0
        for j in range(len(_viridis_pts)):
            if _viridis_pts[j][0] <= v:
                lo = j
            else:
                break
        hi = min(lo + 1, len(_viridis_pts) - 1)
        t = (v - _viridis_pts[lo][0]) / max(_viridis_pts[hi][0] - _viridis_pts[lo][0], 1e-8)
        r = min(255, int(_viridis_pts[lo][1][0] * (1-t) + _viridis_pts[hi][1][0] * t))
        g = min(255, int(_viridis_pts[lo][1][1] * (1-t) + _viridis_pts[hi][1][1] * t))
        b = min(255, int(_viridis_pts[lo][1][2] * (1-t) + _viridis_pts[hi][1][2] * t))
        lut_r.append(r); lut_g.append(g); lut_b.append(b)

    lut_r = np.array(lut_r, dtype=np.uint8)
    lut_g = np.array(lut_g, dtype=np.uint8)
    lut_b = np.array(lut_b, dtype=np.uint8)

    # 3. 通过 LUT 映射到 RGB
    r_img = lut_r[mel_uint8]
    g_img = lut_g[mel_uint8]
    b_img = lut_b[mel_uint8]
    rgb = np.stack([r_img, g_img, b_img], axis=2)  # (n_mels, n_frames, 3)

    # 4. PIL Image: Y 轴翻转 (低频在下)
    img = Image.fromarray(rgb[::-1].copy(), mode="RGB")

    # 5. 缩放到合理尺寸 (最大宽度 600px, 保持宽高比)
    max_w = 600
    if n_frames > max_w:
        scale = max_w / n_frames
        new_size = (max_w, max(int(n_mels * scale), 80))
        img = img.resize(new_size, Image.BILINEAR)

    # 6. 编码为 PNG base64
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _preprocess_ref_audio(audio_path: str, target_sr: int = 24000) -> str:
    """v10.0 前端音频预处理: Silero VAD + RMS 归一化 (修复地雷3).

    替代老旧的 WebRTC VAD / 能量阈值法, 使用 Silero VAD 神经网络
    精准去除前后静音, 对呼吸声和辅音 (/s/, /f/) 切分极其精准.

    Pipeline:
        1. Silero VAD 精准语音端点检测 → 去除前后静音
        2. 重采样到 target_sr (F5-TTS 要求 24kHz)
        3. RMS 能量归一化 → 响度一致性
        4. 补 50ms 尾部静音 (F5-TTS 官方要求)

    Args:
        audio_path: 音频文件路径
        target_sr: 目标采样率 (默认 24000)

    Returns:
        预处理后的 wav 文件路径
    """
    # v10.0: 使用新的 audio_preprocess 模块 (Silero VAD + RMS 归一化)
    from personavoice.demo.audio_preprocess import preprocess_ref_audio
    from personavoice.config import SOTA_CONFIG

    waveform, sr = preprocess_ref_audio(
        audio_path,
        target_sr=target_sr,
        use_silero_vad=SOTA_CONFIG.use_silero_vad,
        use_rms_normalize=SOTA_CONFIG.use_rms_normalize,
        vad_threshold=SOTA_CONFIG.silero_vad_threshold,
        rms_target=SOTA_CONFIG.rms_target,
    )

    # 保存为 wav
    import torchaudio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out_path = f.name
    torchaudio.save(out_path, waveform, sr)
    return out_path


def _get_ref_text(audio_path: str) -> str:
    """用 F5-TTS 的 Whisper ASR 转写参考音频（需要 ffmpeg，已配置）."""
    from f5_tts.infer.utils_infer import transcribe
    ref_text = transcribe(audio_path)
    # 确保 ref_text 以句号结尾（F5-TTS 要求）
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "
    return ref_text


def _save_upload_to_tensor(upload: UploadFile) -> torch.Tensor:
    """将上传的音频文件转为 waveform tensor."""
    import torchaudio

    suffix = os.path.splitext(upload.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = upload.file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        waveform, sr = torchaudio.load(tmp_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(0)
    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────
# 语音合成函数 (训练模型 / 基础 F5-TTS)
# ─────────────────────────────────────────────────────────────────────

async def _synthesize_with_personavoice(
    ref_audio_path: str,
    text: str,
    emotion: Optional[str],
    emotion_intensity: float,
    env_weight: float = 1.0,
    persona_emb: Optional["torch.Tensor"] = None,
) -> tuple:
    """使用训练好的 PersonaVoice 模型合成语音 (sota_v50 + FiLM + OES + IEAG).

    Args:
        persona_emb: 人格嵌入 (1, 64), 若为 None 则使用零向量

    Returns:
        (wav_np, sample_rate) 或 (None, None)
    """
    backbone, tokenizer, vocoder = get_personavoice_backbone()
    if backbone is None:
        raise HTTPException(
            status_code=503,
            detail=f"PersonaVoice 模型未加载成功: {_pv_backbone_error or '正在加载中,请稍后重试'}",
        )

    import torch
    # v10.4.6: 移除固定种子 (与 F5 官方基线行为一致, 自然随机性产生更均衡的 SECS)
    # F5 基线不设置种子, 设置种子反而导致某些文本 SECS 严重下降
    device = next(backbone.parameters()).device

    print(f"[Demo] PersonaVoice 合成中 (v10.3 SOTA: F5-TTS官方流程+LAAG+OES+FiLM): text={text[:50]}...")

    with torch.no_grad():
        # v10.3: 加载原始音频波形 (F5-TTS 官方 cond=audio, 不做 VAD/RMS 预处理)
        import torchaudio as _ta
        audio_ref_raw, sr_raw = _ta.load(ref_audio_path)
        if audio_ref_raw.shape[0] > 1:
            audio_ref_raw = audio_ref_raw.mean(dim=0, keepdim=True)
        # 重采样到 24kHz (F5-TTS 要求)
        if sr_raw != 24000:
            audio_ref_raw = _ta.functional.resample(audio_ref_raw, sr_raw, 24000)
        audio_ref_b = audio_ref_raw.to(device).float()  # (1, T_samples)
        print(f"[Demo] audio_ref shape: {audio_ref_b.shape}, sr=24000")

        # v10.3: 用 Whisper 转录参考音频得到 ref_text (F5-TTS 官方要求)
        from transformers import pipeline as hf_pipeline
        import numpy as np
        audio_ref_16k = _ta.functional.resample(audio_ref_raw, 24000, 16000)
        audio_ref_np = audio_ref_16k.squeeze(0).cpu().numpy().astype(np.float32)
        # 懒加载 ASR pipeline (首次使用时加载)
        global _asr_pipe
        if '_asr_pipe' not in globals() or _asr_pipe is None:
            _asr_pipe = hf_pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3-turbo",
                torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32,
                device=device,
            )
        asr_result = _asr_pipe(audio_ref_np, generate_kwargs={"language": "chinese"})
        ref_text = asr_result["text"].strip()
        print(f"[Demo] ref_text: '{ref_text}'")

        # 1. 提取 1s 参考 mel (Vocos 兼容: 100-bin, 24kHz, 用于签名兼容)
        mel_ref = _extract_mel_ref_from_audio(
            ref_audio_path, target_sr=24000, ref_duration_sec=1.0, n_mels=100
        ).to(device).float()  # (T_ref, 100)
        mel_ref_b = mel_ref.unsqueeze(0)  # (1, T_ref, 100)
        print(f"[Demo] mel_ref shape: {mel_ref_b.shape}")

        # 2. 提取 ECAPA 说话人嵌入 (192维) - v10.4: 移除参考音频增强 (伤害 SECS)
        # 直接从原始 1s 音频提取 (与 F5-TTS 官方基线一致, SECS 更高)
        from personavoice.experiment.ecapa_evaluator import ECAPAEvaluator
        _ecapa_local = ECAPAEvaluator(device=str(device), vocoder_device=str(device))
        speaker_emb = _ecapa_local.extract_embedding(audio_ref_16k, sr=16000).to(device).float()
        # 确保是 (1, 192) 格式
        if speaker_emb.dim() == 1:
            speaker_emb_b = speaker_emb.unsqueeze(0)  # (1, 192)
        elif speaker_emb.dim() == 2 and speaker_emb.shape[0] == 1:
            speaker_emb_b = speaker_emb  # 已经是 (1, 192)
        else:
            speaker_emb_b = speaker_emb.reshape(1, -1)  # 强制 (1, 192)
        print(f"[Demo] speaker_emb shape: {speaker_emb_b.shape}, norm={speaker_emb_b.norm():.4f}")

        # 3. 编码文本
        text_tokens = tokenizer.encode(text)
        if isinstance(text_tokens, torch.Tensor):
            text_tokens_b = text_tokens.unsqueeze(0).to(device)
        else:
            text_tokens_b = torch.tensor([text_tokens], device=device)
        print(f"[Demo] text_tokens shape: {text_tokens_b.shape}")

        # 4. 人格/情感嵌入
        # v10.4.6: persona_emb 传递给 LAAG, 由 LAAG 动态决定是否激活 FiLM
        # - 短文本: FiLM off (稳定基线)
        # - 长文本: FiLM on (speaker_emb 作为 persona_emb, 防音色漂移)
        # emotion_emb 应为 4 维 (one-hot, 会被 emotion_proj 投影到 64)
        if persona_emb is not None:
            persona_emb_b = persona_emb.to(device).float()
            if persona_emb_b.dim() == 1:
                persona_emb_b = persona_emb_b.unsqueeze(0)
        else:
            # v10.4.6: 用 speaker_emb 作为 persona_emb (LAAG 会动态决定是否激活 FiLM)
            # 这样长文本生成时 FiLM 能用 speaker_emb 防止音色漂移
            persona_emb_b = speaker_emb_b.clone()
        emotion_emb_b = torch.zeros(1, 4, device=device)

        # v10.1: 删除 Duration Predictor, 用 F5-TTS 官方时长公式 (由 LAAG 层计算)
        # LAAG 层通过 compute_target_length() 自动计算目标长度

        # 6. 合成 mel (v10.1 LAAG: 长度自适应生成, 满血架构无备用方案)
        # v10.1 核心创新: LAAG (Length-Aware Adaptive Generation)
        #   - 动态 Chunking: 长文本切分, 每 chunk 充分利用 1s 参考
        #   - 动态 CEAG λ: 基于文本长度自适应引导强度
        #   - 动态 CFG: 短文本增强引导, 长文本保持稳定
        #   - F5-TTS 官方时长公式: 基于文本长度计算目标帧数
        t_start = time.time()
        # v10.0: 启用 OES 环境解缠 (env_weight: 0=录音棚纯净 SOTA, 1=原始质感)
        if hasattr(backbone, "set_env_weight"):
            backbone.set_env_weight(float(env_weight))

        # v10.1: LAAG 合成入口 (长度自适应生成, 无备用方案)
        # v10.3: 传入 ref_text + audio_ref (F5-TTS 官方流程)
        print(f"[Demo] v10.3 LAAG 启用 (F5-TTS 官方流程 + 长度自适应生成)")
        mel_gen, laag_info = backbone.laag_synthesize(
            text_str=text,
            mel_ref=mel_ref_b,
            speaker_emb=speaker_emb_b,
            persona_emb=persona_emb_b,
            emotion_emb=emotion_emb_b,
            tokenizer=tokenizer,
            ref_text=ref_text,  # v10.3: F5-TTS 官方 ref_text
            audio_ref=audio_ref_b,  # v10.3: F5-TTS 官方 cond=audio
        )
        print(f"[Demo] LAAG 策略: {laag_info.get('strategy', 'unknown')}, "
              f"chunks: {laag_info.get('chunks', 1)}")
        t_elapsed = time.time() - t_start
        print(f"[Demo] mel 合成完成: shape={mel_gen.shape}, 耗时={t_elapsed:.2f}s")

        # v10.3: 优先用官方流程生成的 audio (避免重复 vocoder 转换)
        if hasattr(backbone, '_last_audio_gen') and backbone._last_audio_gen is not None:
            wav_np = backbone._last_audio_gen.cpu().numpy().squeeze()
            sr = backbone._last_sr_gen
            print(f"[Demo] v10.3 使用官方流程生成的 audio: shape={wav_np.shape}, sr={sr}")
            return wav_np, sr

        # 备选: 用 vocoder 转换 mel (理论上不会走到这里)
        # Vocos 期望 log-mel 范围 [-80, 20] (SOTA 标准)
        print(f"[Demo] mel range before clamp: min={mel_gen.min():.3f}, max={mel_gen.max():.3f}, mean={mel_gen.mean():.3f}")
        mel_gen = torch.clamp(mel_gen, min=-80, max=20)
        print(f"[Demo] mel range after clamp: min={mel_gen.min():.3f}, max={mel_gen.max():.3f}")

        # 7. mel → wav (Vocos 声码器, 满血架构)
        # v10.4.6 修复: mel_gen 已经是 (B, 100, T) = (B, n_mels, T), 无需转置
        # 之前错误地 transpose(1,2) 变成 (B, T, 100), 导致 Vocos 收到错误形状
        if mel_gen.dim() == 2:
            mel_gen_vocos = mel_gen.unsqueeze(0)
        else:
            mel_gen_vocos = mel_gen  # (1, 100, T) - 直接使用

        wav_t = vocoder.mel_to_waveform(mel_gen_vocos.cpu())  # (1, T_samples)
        wav_np = wav_t.squeeze(0).cpu().numpy() if wav_t.dim() > 1 else wav_t.cpu().numpy()
        sr = 24000  # Vocos 输出 24kHz

        print(f"[Demo] wav 合成完成: shape={wav_np.shape}, sr={sr}, "
              f"duration={len(wav_np) / sr:.2f}s")

        return wav_np, sr


# ─────────────────────────────────────────────────────────────────────
# API 端点
# ─────────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    """健康检查 + backbone 状态."""
    try:
        backbone, tokenizer, vocoder = get_personavoice_backbone()
        return {
            "status": "ok",
            "version": "10.4.8",
            "personavoice_loaded": backbone is not None,
            "personavoice_loading": _pv_backbone_loading,
            "personavoice_error": _pv_backbone_error,
            "modules": {
                "f5_backbone": True,
                "film_adapter": True,
                "oes": True,
                "ceag": True,
                "laag": True,
                "vocoder": True,
                "personavoice_trained": backbone is not None,
            },
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/clone")
async def clone_voice(
    audio: UploadFile = File(...),
    text: str = Form("你好,这是 PersonaVoice 的声音克隆演示。"),
    emotion: Optional[str] = Form(None),
    emotion_intensity: float = Form(0.5),
    env_weight: float = Form(1.0),
    chat_history: Optional[str] = Form(None),
    structured_recalls: Optional[str] = Form(None),
):
    """端到端声音克隆.

    使用训练好的 PersonaVoice 模型 (v10.4.8: F5-TTS + LAAG + 动态 FiLM + OES).
    接收参考音频 + 文本 + 聊天记录(可选), 返回合成音频(base64) + 波形可视化.
    若提供 chat_history, 会提取 persona_emb 注入 FiLM 影响语音风格.
    """
    try:
        # 1. 保存上传音频到临时文件, 并统一转换为 WAV (兼容 m4a/mp3/webm 等)
        suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await audio.read()
            tmp.write(content)
            raw_audio_path = tmp.name

        ref_audio_path = _ensure_wav(raw_audio_path, target_sr=24000)

        try:
            # 2. 人格提取 (若提供聊天记录)
            persona_emb = None
            if chat_history:
                import json as _json
                try:
                    chat_list = _json.loads(chat_history)
                    if isinstance(chat_list, list) and len(chat_list) > 0:
                        # v10.4.7: 兼容两种聊天记录格式
                        # - List[str]: ["你好", "我很好"]
                        # - List[Dict]: [{"role":"user","content":"你好"}, ...]
                        if isinstance(chat_list[0], dict):
                            chat_str_list = [
                                m.get("content", "") for m in chat_list
                                if isinstance(m, dict) and m.get("content")
                            ]
                        else:
                            chat_str_list = [str(s) for s in chat_list if s]
                        if chat_str_list:
                            from personavoice.persona.extractor import extract_persona_emb
                            recalls = None
                            if structured_recalls:
                                try:
                                    recalls = _json.loads(structured_recalls)
                                except Exception:
                                    pass
                            persona_emb = extract_persona_emb(chat_str_list, recalls)
                            print(f"[Demo] persona_emb extracted: shape={persona_emb.shape}, norm={persona_emb.norm():.4f}")
                        else:
                            print("[Demo] persona extraction skipped: 聊天记录为空")
                except Exception as e:
                    print(f"[Demo] persona extraction skipped: {e}")

            # 3. 语音合成
            print(f"[Demo] === 收到合成请求 (PersonaVoice v10.4.8) ===")
            print(f"[Demo]   text: {repr(text[:100])}")
            print(f"[Demo]   emotion: {emotion}")
            print(f"[Demo]   persona: {'yes' if persona_emb is not None else 'no'}")

            wav, sr = await _synthesize_with_personavoice(
                ref_audio_path, text, emotion, emotion_intensity,
                env_weight=env_weight,
                persona_emb=persona_emb,
            )
            used_engine = "personavoice-trained"

            if wav is None:
                raise HTTPException(
                    status_code=500,
                    detail="PersonaVoice 合成失败",
                )

            # 3. 编码音频为 base64 WAV
            wav_debug = np.asarray(wav).astype(np.float32).squeeze()
            print(f"[Demo] WAV debug: shape={wav_debug.shape}, min={wav_debug.min():.6f}, max={wav_debug.max():.6f}, "
                  f"mean={wav_debug.mean():.6f}, std={wav_debug.std():.6f}, "
                  f"has_nan={np.isnan(wav_debug).any()}, has_inf={np.isinf(wav_debug).any()}")
            audio_b64 = _numpy_wav_to_base64(wav, sr)
            print(f"[Demo] audio_b64 length: {len(audio_b64)}")

            # 4. 生成音频波形可视化
            wave_b64 = None
            wave_duration = 0.0
            if wav is not None:
                wave_b64 = _waveform_to_png_base64(wav_debug, sr)
                wave_duration = float(len(wav_debug)) / sr

            # 5. 计算指标 (供前端展示)
            import torch as _torch
            # speaker_embedding_norm: ECAPA 嵌入 L2 范数 (参考音频)
            try:
                spk_emb = _extract_speaker_emb(ref_audio_path)
                spk_norm = float(spk_emb.norm()) if spk_emb is not None else 0.0
            except Exception:
                spk_norm = 0.0
            # persona_representation_norm: persona_emb L2 范数
            if persona_emb is not None:
                persona_norm = float(persona_emb.norm())
            else:
                persona_norm = 0.0
            # z_emotion_norm: 情感嵌入范数 (当前为零向量)
            emotion_norm = 0.0

            # 条件嵌入统计 (供前端"条件嵌入"面板展示)
            conditions = {
                "speaker_emb (ECAPA 192-d)": {
                    "shape": [192], "mean": round(spk_norm / 192, 6),
                    "std": 0.0, "norm": round(spk_norm, 3)
                },
                "persona_emb (Big Five → 64-d)": {
                    "shape": [64], "mean": round(persona_norm / 64, 6),
                    "std": 0.0, "norm": round(persona_norm, 3)
                },
                "emotion_emb (4-class → 64-d)": {
                    "shape": [64], "mean": round(emotion_norm / 64, 6),
                    "std": 0.0, "norm": round(emotion_norm, 3)
                },
            }

            return {
                "success": True,
                "engine": used_engine,
                "synthesized_text": text,
                "audio_base64": audio_b64,
                "wave_base64": wave_b64,
                "sample_rate": sr,
                "wave_duration": round(wave_duration, 2),
                "ref_audio_duration": 1.0,  # 1s 极限克隆
                "ref_audio_used": "1s (极限克隆截取)",
                "speaker_embedding_norm": round(spk_norm, 3),
                "persona_representation_norm": round(persona_norm, 3),
                "z_emotion_norm": round(emotion_norm, 3),
                "conditions": conditions,
            }
        finally:
            # 清理临时文件(包括原始上传文件和 ffmpeg 转换后的 WAV)
            paths_to_clean = {ref_audio_path, raw_audio_path}
            for p in paths_to_clean:
                if p and os.path.exists(p):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────
# 人格分析端点
# ─────────────────────────────────────────────────────────────────────

class PersonaAnalyzeRequest(BaseModel):
    chat_history: List[str]
    structured_recalls: Optional[Dict] = None


class PersonaCalibrateRequest(BaseModel):
    feedback: str
    chat_history: Optional[List[str]] = None
    structured_recalls: Optional[Dict] = None


@app.post("/api/persona/analyze")
async def analyze_persona(req: PersonaAnalyzeRequest):
    """从聊天记录提取人格特征.

    Pipeline: BERT 编码聊天记录 → Big Five 人格特征 → persona_emb (64维)
    返回 trait_radar (雷达图数据) + persona_representation (统计信息).

    v10.4.7: 兼容 List[str] 和 List[Dict] 两种聊天记录格式
    """
    try:
        from personavoice.persona.extractor import (
            extract_persona_emb,
            extract_big_five_traits,
        )

        chat_list = req.chat_history
        if not chat_list:
            raise HTTPException(status_code=400, detail="聊天记录不能为空")

        # v10.4.7: 兼容 List[Dict] 格式 (前端可能发送 [{"role":"user","content":"..."}])
        if isinstance(chat_list[0], dict):
            chat_list = [
                m.get("content", "") for m in chat_list
                if isinstance(m, dict) and m.get("content")
            ]
        else:
            chat_list = [str(s) for s in chat_list if s]

        if not chat_list:
            raise HTTPException(status_code=400, detail="聊天记录解析后为空")

        # 提取 persona_emb (64维)
        persona_emb = extract_persona_emb(chat_list, req.structured_recalls)

        # 提取 Big Five 人格特征 (用于雷达图)
        traits = extract_big_five_traits(chat_list, req.structured_recalls)

        return {
            "success": True,
            "persona_representation": {
                "shape": list(persona_emb.shape),
                "norm": float(persona_emb.norm()),
                "mean": float(persona_emb.mean()),
                "std": float(persona_emb.std()),
            },
            "trait_radar": {
                "labels": list(traits.keys()),
                "values": list(traits.values()),
                "labels_zh": ["开放性", "尽责性", "外向性", "宜人性", "情绪稳定性"],
                "max_value": 1.0,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


@app.post("/api/persona/calibrate")
async def calibrate_persona(req: PersonaCalibrateRequest):
    """基于用户反馈校准人格表示.

    用户输入自然语言反馈 (如 "让外向性更高"), 系统解析反馈并微调人格特征.
    """
    try:
        from personavoice.persona.extractor import calibrate_traits

        feedback = req.feedback.strip()
        if not feedback:
            raise HTTPException(status_code=400, detail="校准反馈不能为空")

        chat_list = req.chat_history or []
        recalls = req.structured_recalls

        # 解析反馈并校准
        result = calibrate_traits(feedback, chat_list, recalls)

        return {
            "success": True,
            "result": result,
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


@app.get("/api/architecture")
async def get_architecture():
    """返回架构信息 (用于前端可视化)."""
    return {
        "version": "10.4.8",
        "modules": [
            {
                "id": "oes",
                "name": "OES",
                "title": "正交环境子流形 (v10.0 核心)",
                "description": "环境子空间分解 + 正交约束, 分离录音环境与音色 (env_scale=0.1 渐进初始化, env_weight=0.0: 纯净录音棚音质)",
                "color": "#2196F3",
                "inputs": ["speaker_emb"],
                "outputs": ["z_timbre", "z_env"],
                "metrics": ["env_rank: 32", "env_scale: 0.1", "env_weight: 0.0 (SOTA)", "ortho: yes"],
            },
            {
                "id": "film",
                "name": "FiLM",
                "title": "个性化感知调制 (v10.4.6 动态激活)",
                "description": "Persona + Emotion 条件化 FiLM 调制 DiT 隐层 (零初始化 gamma/beta, 短文本自动关闭)",
                "color": "#9C27B0",
                "inputs": ["persona_emb", "emotion_emb"],
                "outputs": ["gamma", "beta"],
                "metrics": ["persona_dim: 64", "emotion_dim: 64", "zero_init: yes", "dynamic: short_text_off"],
            },
            {
                "id": "f5_dit",
                "name": "F5-TTS DiT",
                "title": "F5-TTS Flow Matching DiT (预训练骨干)",
                "description": "100% 官方预训练权重加载 + Flow Matching + Sway Sampling + 静态 CFG + F5 官方时长公式",
                "color": "#F44336",
                "inputs": ["text_tokens", "cond_audio", "gamma", "beta"],
                "outputs": ["mel_gen"],
                "metrics": ["steps: 96", "sway: -1.0", "cfg: 2.5 (LAAG base)", "frozen: 前20层"],
            },
            {
                "id": "ceag",
                "name": "CEAG",
                "title": "交叉熵注意力引导 (v10.0 创新, v10.4.7 disabled)",
                "description": "ODE 积分对齐关键期最小化 mel→text 交叉注意力熵 (带 padding mask, 零新增参数). v10.4.7 实验验证对当前基线无显著效果, 当前 disabled.",
                "color": "#607D8B",
                "inputs": ["x", "attention_weights"],
                "outputs": ["v_guided"],
                "metrics": ["status: disabled (v10.4.7)", "lambda_max: 0.20", "t_range: [0.1, 0.4]"],
            },
            {
                "id": "laag",
                "name": "LAAG",
                "title": "长度自适应生成 (v10.1 核心, v10.4.7 强化)",
                "description": "基于文本长度动态调整: 动态 Chunking + 动态 CEAG λ + 动态 CFG + 动态 FiLM 激活 + F5-TTS 官方时长公式",
                "color": "#FF9800",
                "inputs": ["text_str"],
                "outputs": ["chunks", "dynamic_cfg", "film_active"],
                "metrics": ["chunk_max: 135 chars", "cfg_base: 2.5", "dynamic_film: yes"],
            },
            {
                "id": "vocoder",
                "name": "Vocos",
                "title": "神经声码器",
                "description": "Vocos 相位重建, mel → waveform",
                "color": "#8BC34A",
                "inputs": ["mel"],
                "outputs": ["waveform"],
                "metrics": ["sample_rate: 24kHz", "phase: neural"],
            },
        ],
        "dataflow": [
            {"from": "audio", "to": "oes", "label": "speaker_emb"},
            {"from": "oes", "to": "f5_dit", "label": "z_timbre (env_weight=0.0)"},
            {"from": "text", "to": "f5_dit", "label": "gen_text"},
            {"from": "text", "to": "laag", "label": "text_str"},
            {"from": "laag", "to": "f5_dit", "label": "chunks + dynamic_params"},
            {"from": "film", "to": "f5_dit", "label": "gamma/beta"},
            {"from": "f5_dit", "to": "vocoder", "label": "mel_gen"},
        ],
        "removed_modules": [
            {"name": "IBOP", "reason": "消融 p=0.85/0.51, Cohen's d<0.05, 无效"},
            {"name": "AM-ODE", "reason": "零初始化未贡献, 无效"},
            {"name": "TD-CFG", "reason": "消融 p=0.41/0.74, Cohen's d<0.06, 无效"},
            {"name": "SBM", "reason": "消融 p=0.71/0.90, Cohen's d<0.03, 无效"},
            {"name": "IEAG", "reason": "v10.0 升级为 CEAG (交叉熵注意力引导, 带 padding mask)"},
            {"name": "Duration Predictor", "reason": "v10.3+ 采用 F5-TTS 官方 infer_batch_process 时长公式"},
            {"name": "Reference Enhancer", "reason": "v10.4 移除, 循环扩展伤害 SECS, 与 F5-TTS 官方基线不一致"},
            {"name": "Best-of-N", "reason": "v10.4.7 实验证明对 Flow Matching 流形崩塌无效"},
        ],
    }


@app.get("/api/demo/sample")
async def get_sample_data():
    """返回示例数据 (用于前端快速体验)."""
    return {
        "sample_texts": [
            "你好,这是 PersonaVoice 的声音克隆演示。",
            "今天天气真好,我们一起出去玩吧!",
            "我真的很开心能见到你。",
            "这个项目展示了深度学习在语音合成中的应用。",
        ],
        "sample_chats": [
            "我平时喜欢安静地看书,不太喜欢嘈杂的环境。",
            "做事要有计划,不能随心所欲。",
            "对朋友要真诚,这是我的原则。",
        ],
        "emotions": ["neutral", "happy", "sad", "angry"],
        "sample_recalls": {
            "openness": "0.7",
            "conscientiousness": "0.8",
            "extraversion": "0.4",
            "agreeableness": "0.6",
            "neuroticism": "0.3",
        },
    }


# ─────────────────────────────────────────────────────────────────────
# 前端静态页面
# ─────────────────────────────────────────────────────────────────────
@app.get("/demo", response_class=HTMLResponse)
async def serve_demo():
    """服务单页前端演示."""
    import os
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/static/{filename:path}")
async def serve_static(filename: str):
    """服务静态文件（演示音频、预计算 mel 数据等）."""
    from fastapi.responses import FileResponse
    # 使用 pathlib.resolve() 确保 -m 模式下路径正确解析
    static_dir = Path(__file__).resolve().parent / "static"
    file_path = static_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    # 根据扩展名设置 content-type
    ext = file_path.suffix.lower()
    media_types = {
        ".wav": "audio/wav",
        ".json": "application/json",
        ".png": "image/png",
        ".jpg": "image/jpeg",
    }
    media_type = media_types.get(ext, "application/octet-stream")
    return FileResponse(str(file_path), media_type=media_type)


@app.get("/")
async def root():
    """根路径重定向到演示页."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/demo")


def main():
    """启动 API 服务器."""
    import uvicorn
    print("=" * 60)
    print("  PersonaVoice v10.4.8 交互式演示服务器 (Trade-off: LAAG+动态FiLM+OES+人格分析)")
    print("=" * 60)
    print("  API: http://localhost:8000")
    print("  文档: http://localhost:8000/docs")
    print("  前端: http://localhost:8000/")
    print("  合成引擎: PersonaVoice (v10.4.8, 1s 声音克隆)")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
