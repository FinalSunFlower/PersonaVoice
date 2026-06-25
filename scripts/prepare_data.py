"""PersonaVoice v10.4.8 — LibriTTS 评估数据集准备脚本.

架构位置: 开源基础设施层, 在首次复现 200 样本评估之前执行一次,
下载并预处理 LibriTTS dev-clean 子集, 生成项目评估所需的
libritts_devclean_processed.pt 文件.

数据来源:
    - LibriTTS dev-clean (https://openslr.org/60/)
    - 6 个说话人, ~5.4 小时, 16kHz

输出:
    data/libritts_processed/libritts_devclean_processed.pt
        - tensors.mel: (N, 100, T)  mel 频谱
        - tensors.ecapa_emb: (N, 192)  ECAPA 说话人嵌入
        - tensors.wav: (N, T_samples)  原始波形 (24kHz)
        - metadata.texts: List[str]  文本内容
        - metadata.speaker_ids: List[int]  说话人 ID

使用方法:
    python scripts/prepare_data.py                  # 完整下载+预处理
    python scripts/prepare_data.py --skip-download  # 跳过下载, 仅预处理
    python scripts/prepare_data.py --n_samples 200  # 限制样本数 (测试用)
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# LibriTTS dev-clean 下载地址
LIBRITTS_DEVCLEAN_URL = "https://openslr.org/resources/60/dev-clean.tar.gz"

# 输出路径
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "LibriTTS"
PROCESSED_DIR = DATA_DIR / "libritts_processed"
PROCESSED_FILE = PROCESSED_DIR / "libritts_devclean_processed.pt"


def download_libritts():
    """下载 LibriTTS dev-clean 数据集 (~1.2GB)."""
    if (RAW_DIR / "dev-clean").exists():
        print(f"  ✓ LibriTTS dev-clean 已存在: {RAW_DIR / 'dev-clean'}")
        return

    archive_path = DATA_DIR / "dev-clean.tar.gz"
    if not archive_path.exists():
        print(f"  下载 LibriTTS dev-clean (~1.2GB) ...")
        print(f"  URL: {LIBRITTS_DEVCLEAN_URL}")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(
                LIBRITTS_DEVCLEAN_URL,
                str(archive_path),
            )
            print(f"  ✓ 下载完成: {archive_path}")
        except Exception as e:
            print(f"  ✗ 下载失败: {e}")
            print(f"  请手动下载: {LIBRITTS_DEVCLEAN_URL}")
            print(f"  并解压到: {DATA_DIR}")
            raise

    print(f"  解压 {archive_path} → {DATA_DIR}")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(str(DATA_DIR))
    print(f"  ✓ 解压完成: {RAW_DIR / 'dev-clean'}")


def collect_audio_files() -> List[Tuple[Path, str, int]]:
    """收集所有 (wav_path, text, speaker_id) 三元组.

    Returns:
        List of (wav_path, normalized_text, speaker_id)
    """
    samples = []
    dev_clean = RAW_DIR / "dev-clean"
    if not dev_clean.exists():
        raise FileNotFoundError(
            f"LibriTTS dev-clean 不存在: {dev_clean}\n"
            f"请先运行: python scripts/prepare_data.py"
        )

    for speaker_dir in sorted(dev_clean.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker_id = int(speaker_dir.name)
        for chapter_dir in sorted(speaker_dir.iterdir()):
            if not chapter_dir.is_dir():
                continue
            # 找到 .wav 和对应的 .normalized.txt
            wav_files = sorted(chapter_dir.glob("*.wav"))
            for wav_file in wav_files:
                base = wav_file.stem
                text_file = chapter_dir / f"{base}.normalized.txt"
                if not text_file.exists():
                    continue
                text = text_file.read_text(encoding="utf-8").strip()
                if not text:
                    continue
                samples.append((wav_file, text, speaker_id))

    return samples


def extract_mel(wav_path: Path, sample_rate: int = 24000, n_mels: int = 100) -> torch.Tensor:
    """提取 mel 频谱 (100-bin, 24kHz, 与 Vocos 兼容)."""
    wav, sr = torchaudio.load(str(wav_path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=n_mels,
    )
    mel = mel_transform(wav)  # (1, n_mels, T)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel.squeeze(0)  # (n_mels, T)


def extract_ecapa_embedding(
    wav_path: Path,
    ecapa_encoder,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """提取 ECAPA 说话人嵌入 (192-d)."""
    wav, sr = torchaudio.load(str(wav_path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    with torch.no_grad():
        emb = ecapa_encoder.encode_batch(wav).squeeze(0)  # (192,)
    return emb


def preprocess_data(n_samples: int = 0):
    """预处理 LibriTTS dev-clean → libritts_devclean_processed.pt."""
    samples = collect_audio_files()
    if n_samples > 0:
        samples = samples[:n_samples]
    print(f"  共 {len(samples)} 个样本待处理")

    # 加载 ECAPA 编码器
    print("  加载 ECAPA-TDNN 编码器 ...")
    from speechbrain.inference.speaker import EncoderClassifier
    ecapa_encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(PROJECT_ROOT / "models" / "ecapa_tdnn"),
        run_opts={"device": "cpu"},
    )

    mels, embeddings, wavs, texts, speaker_ids = [], [], [], [], []
    for i, (wav_path, text, sid) in enumerate(samples):
        if (i + 1) % 50 == 0:
            print(f"  处理中: {i+1}/{len(samples)}")
        try:
            mel = extract_mel(wav_path)
            emb = extract_ecapa_embedding(wav_path, ecapa_encoder)
            wav, sr = torchaudio.load(str(wav_path))
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != 24000:
                wav = torchaudio.functional.resample(wav, sr, 24000)
            mels.append(mel)
            embeddings.append(emb)
            wavs.append(wav.squeeze(0))
            texts.append(text)
            speaker_ids.append(sid)
        except Exception as e:
            print(f"  [WARN] 跳过 {wav_path.name}: {e}")

    # 保存
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "tensors": {
            "mel": torch.nn.utils.rnn.pad_sequence(mels, batch_first=True).permute(0, 2, 1),
            "ecapa_emb": torch.stack(embeddings),
            "wav": wavs,
        },
        "metadata": {
            "texts": texts,
            "speaker_ids": speaker_ids,
        },
    }
    torch.save(data, str(PROCESSED_FILE))
    print(f"  ✓ 保存到: {PROCESSED_FILE}")
    print(f"    mel: {data['tensors']['mel'].shape}")
    print(f"    ecapa_emb: {data['tensors']['ecapa_emb'].shape}")
    print(f"    wav: {len(data['tensors']['wav'])} 条")
    print(f"    texts: {len(data['metadata']['texts'])} 条")


def main():
    parser = argparse.ArgumentParser(
        description="PersonaVoice LibriTTS 评估数据集准备脚本",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="跳过下载步骤 (假设 LibriTTS 已存在)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=0,
        help="限制样本数 (测试用, 0=全部)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  PersonaVoice v10.4.8 — LibriTTS 数据集准备")
    print("=" * 60)

    if not args.skip_download:
        download_libritts()
    else:
        print("  [--skip-download] 跳过下载步骤")

    if not PROCESSED_FILE.exists():
        preprocess_data(n_samples=args.n_samples)
    else:
        print(f"  ✓ 已存在预处理文件: {PROCESSED_FILE}")
        print(f"    如需重新生成, 请删除后重跑")

    print("\n  ✓ 数据集准备完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
