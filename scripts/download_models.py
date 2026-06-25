"""PersonaVoice v10.4.8 — 预训练模型一键下载脚本.

架构位置: 开源基础设施层, 在首次运行 PersonaVoice 之前执行一次,
将所有 HuggingFace 模型预下载到本地缓存, 之后可完全离线运行.

包含的模型:
    1. SWivid/F5-TTS                   (~1.5GB, TTS 骨干)
    2. speechbrain/spkrec-ecapa-voxceleb (~80MB, 说话人编码器)
    3. charactr/vocos-mel-24khz        (~50MB, 神经声码器)
    4. bert-base-chinese               (~400MB, 人格文本编码器)
    5. openai/whisper-large-v3-turbo   (~1.5GB, WER 评估用 ASR)
    6. Qwen/Qwen2.5-0.5B               (~1GB, 可选, LLM 人格对话)
    7. sentence-transformers/all-MiniLM-L6-v2 (~90MB, 可选, 文本嵌入)

使用方法:
    python scripts/download_models.py              # 下载所有必需模型
    python scripts/download_models.py --optional   # 同时下载可选模型
    python scripts/download_models.py --check      # 仅检查缓存状态

注意:
    - 默认下载到 HuggingFace 默认缓存目录 (~/.cache/huggingface/hub)
    - 可通过环境变量 HF_HOME 自定义缓存目录
    - 国内用户可设置 HF_ENDPOINT=https://hf-mirror.com 使用镜像加速
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# 必需模型 (运行 PersonaVoice 主流程必需)
REQUIRED_MODELS: Dict[str, str] = {
    "f5_tts": "SWivid/F5-TTS",
    "ecapa_voxceleb": "speechbrain/spkrec-ecapa-voxceleb",
    "vocos_mel_24khz": "charactr/vocos-mel-24khz",
    "bert_chinese": "bert-base-chinese",
    "whisper_large_v3_turbo": "openai/whisper-large-v3-turbo",
}

# 可选模型 (实验脚本或扩展功能需要)
OPTIONAL_MODELS: Dict[str, str] = {
    "qwen2.5_0.5b": "Qwen/Qwen2.5-0.5B",
    "minilm_l6": "sentence-transformers/all-MiniLM-L6-v2",
}


def check_cache_status(models: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """检查模型缓存状态.

    Returns:
        (cached_keys, missing_keys)
    """
    try:
        from personavoice.common.local_models import is_model_cached
    except ImportError:
        # 项目未安装时, 简化检查
        from huggingface_hub import constants
        cache_dir = Path(constants.HF_HUB_CACHE)

        def is_model_cached(model_id: str) -> bool:
            cache_name = "models--" + model_id.replace("/", "--")
            snapshots = cache_dir / cache_name / "snapshots"
            return snapshots.exists() and any(snapshots.iterdir())

    cached, missing = [], []
    for key, model_id in models.items():
        if is_model_cached(model_id):
            cached.append(key)
        else:
            missing.append(key)
    return cached, missing


def download_model(model_id: str, key: str) -> bool:
    """下载单个模型.

    Returns:
        True if success, False otherwise.
    """
    print(f"\n[{key}] 下载 {model_id} ...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=model_id,
            resume_download=True,
            max_workers=4,
        )
        print(f"  ✓ {model_id} 下载完成")
        return True
    except Exception as e:
        print(f"  ✗ {model_id} 下载失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="PersonaVoice 预训练模型下载脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/download_models.py              # 下载所有必需模型
  python scripts/download_models.py --optional   # 同时下载可选模型
  python scripts/download_models.py --check      # 仅检查缓存状态
  python scripts/download_models.py --only f5_tts ecapa_voxceleb  # 仅下载指定模型
        """,
    )
    parser.add_argument(
        "--optional", action="store_true",
        help="同时下载可选模型 (Qwen2.5, MiniLM)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="仅检查缓存状态, 不下载",
    )
    parser.add_argument(
        "--only", nargs="+", default=None,
        help="仅下载指定的模型别名 (空格分隔)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  PersonaVoice v10.4.8 预训练模型下载脚本")
    print("=" * 60)

    # 选择要处理的模型
    if args.only:
        all_models = {**REQUIRED_MODELS, **OPTIONAL_MODELS}
        models = {k: all_models[k] for k in args.only if k in all_models}
        if not models:
            print(f"  [ERROR] 未找到指定的模型别名: {args.only}")
            print(f"  可用别名: {list(all_models.keys())}")
            return 1
    elif args.optional:
        models = {**REQUIRED_MODELS, **OPTIONAL_MODELS}
    else:
        models = REQUIRED_MODELS

    # 检查缓存状态
    cached, missing = check_cache_status(models)
    print(f"\n  缓存目录: {os.environ.get('HF_HOME', '~/.cache/huggingface')}")
    print(f"  已缓存: {len(cached)}/{len(models)}")
    print(f"  缺失:   {len(missing)}/{len(models)}")
    if missing:
        print(f"  缺失模型: {missing}")

    if args.check:
        print("\n[CHECK] 仅检查模式, 不执行下载")
        return 0

    if not missing:
        print("\n  ✓ 所有模型已缓存, 无需下载")
        return 0

    # 下载缺失模型
    print(f"\n  开始下载 {len(missing)} 个缺失模型...")
    success, failed = 0, 0
    for key in missing:
        model_id = models[key]
        if download_model(model_id, key):
            success += 1
        else:
            failed += 1

    # 汇总
    print("\n" + "=" * 60)
    print(f"  下载完成: {success} 成功, {failed} 失败")
    if failed:
        print(f"  失败模型, 请稍后重试或手动下载:")
        for key in missing:
            model_id = models[key]
            print(f"    huggingface-cli download {model_id}")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
