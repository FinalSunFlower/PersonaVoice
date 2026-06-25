"""本地模型路径管理器: 强制使用本地 HuggingFace 缓存, 避免联网.

架构位置: 全局基础设施层, 在所有模型加载之前调用, 确保离线运行环境.

本地缓存位置: C:\\Users\\<user>\\.cache\\huggingface\\hub
已缓存的模型:
    - Qwen/Qwen2.5-0.5B              (LLM / Tokenizer)
    - SWivid/F5-TTS                  (TTS 骨干)
    - speechbrain/spkrec-ecapa-voxceleb  (说话人编码器)
    - charactr/vocos-mel-24khz       (声码器)
    - bert-base-chinese              (中文 BERT)
    - openai/whisper-large-v3-turbo  (Whisper ASR)

使用方法:
    from personavoice.common.local_models import (
        get_local_model_path,
        setup_offline_mode,
        LOCAL_MODEL_PATHS,
    )

    # 1. 全局开启离线模式 (推荐在程序入口调用)
    setup_offline_mode()

    # 2. 获取本地模型路径
    path = get_local_model_path("qwen2.5_0.5b")
    tokenizer = AutoTokenizer.from_pretrained(path)
"""
import os
import logging
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)


# =========================================================================
# 本地模型路径映射
# =========================================================================
# 所有 HuggingFace 模型 ID 与本地缓存路径的映射
# 优先使用本地缓存, 若不存在则回退到在线模式 (但会打印警告)
LOCAL_MODEL_PATHS: Dict[str, str] = {
    # ── LLM / Tokenizer ──
    "qwen2.5_0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen3_0.6b": "Qwen/Qwen3-0.6B",
    "qwen3_4b": "Qwen/Qwen3-4B",
    # ── TTS Backbone ──
    "f5_tts": "SWivid/F5-TTS",
    # ── Speaker Encoder ──
    "ecapa_voxceleb": "speechbrain/spkrec-ecapa-voxceleb",
    # ── Vocoder ──
    "vocos_mel_24khz": "charactr/vocos-mel-24khz",
    # ── Text Encoder ──
    "bert_chinese": "bert-base-chinese",
    "minilm_l6": "sentence-transformers/all-MiniLM-L6-v2",
    # ── ASR ──
    "whisper_large_v3_turbo": "openai/whisper-large-v3-turbo",
}


# =========================================================================
# 离线模式配置
# =========================================================================
_OFFLINE_MODE_CONFIGURED = False


def setup_offline_mode() -> None:
    """全局开启 HuggingFace 离线模式.

    设置环境变量:
        - HF_HUB_OFFLINE=1     禁止网络请求
        - TRANSFORMERS_OFFLINE=1  transformers 库离线模式
        - HF_DATASETS_OFFLINE=1   datasets 库离线模式

    调用后所有 transformers / huggingface_hub 调用都只从本地缓存读取,
    不会发起任何网络请求, 避免连接超时.
    """
    global _OFFLINE_MODE_CONFIGURED

    if _OFFLINE_MODE_CONFIGURED:
        return

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    # 减少 transformers 日志噪音
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    _OFFLINE_MODE_CONFIGURED = True
    logger.info(
        "[local_models] HuggingFace 离线模式已启用 "
        "(HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1)"
    )


def get_hf_cache_dir() -> Path:
    """获取 HuggingFace 本地缓存目录.

    Returns:
        Path: 缓存目录路径 (如 C:\\Users\\<user>\\.cache\\huggingface\\hub)
    """
    # 优先使用环境变量
    cache_env = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache_env:
        cache_path = Path(cache_env)
        if cache_path.name == "hub":
            return cache_path
        return cache_path / "hub"

    # 默认位置: ~/.cache/huggingface/hub
    try:
        from huggingface_hub import constants
        return Path(constants.HF_HUB_CACHE)
    except ImportError:
        # 回退到默认位置
        home = Path.home()
        return home / ".cache" / "huggingface" / "hub"


def is_model_cached(model_id: str) -> bool:
    """检查模型是否已在本地缓存.

    Args:
        model_id: HuggingFace 模型 ID (如 "Qwen/Qwen2.5-0.5B")

    Returns:
        bool: 模型是否已缓存
    """
    cache_dir = get_hf_cache_dir()
    # HuggingFace 缓存目录格式: models--<org>--<name>
    # 例如: models--Qwen--Qwen2.5-0.5B
    cache_name = "models--" + model_id.replace("/", "--")
    model_cache_path = cache_dir / cache_name

    # 必须存在 snapshots 子目录且非空
    snapshots_dir = model_cache_path / "snapshots"
    if not snapshots_dir.exists():
        return False

    # 检查是否有至少一个 snapshot
    snapshots = list(snapshots_dir.iterdir()) if snapshots_dir.exists() else []
    return len(snapshots) > 0


def get_local_model_path(key: str, allow_network_fallback: bool = False) -> str:
    """获取本地模型路径.

    优先返回 HuggingFace 缓存中的模型路径, 若未缓存:
        - allow_network_fallback=True: 返回原始 model_id (允许联网下载)
        - allow_network_fallback=False: 抛出 FileNotFoundError

    Args:
        key: 模型别名 (如 "qwen2.5_0.5b", "f5_tts", "ecapa_voxceleb")
        allow_network_fallback: 未缓存时是否允许联网回退

    Returns:
        str: 模型路径 (可直接传给 AutoTokenizer.from_pretrained 等)

    Raises:
        KeyError: key 不在 LOCAL_MODEL_PATHS 中
        FileNotFoundError: 模型未缓存且不允许联网回退
    """
    if key not in LOCAL_MODEL_PATHS:
        raise KeyError(
            f"未知模型别名: '{key}'. 可用别名: {list(LOCAL_MODEL_PATHS.keys())}"
        )

    model_id = LOCAL_MODEL_PATHS[key]

    # 检查本地缓存
    if is_model_cached(model_id):
        # 返回 model_id, transformers 会自动从缓存加载
        # (配合 setup_offline_mode() 不会联网)
        logger.debug(f"[local_models] 使用本地缓存: {model_id}")
        return model_id

    # 未缓存
    if allow_network_fallback:
        logger.warning(
            f"[local_models] 模型 '{model_id}' 未在本地缓存, 将尝试联网下载. "
            f"建议预先下载: huggingface-cli download {model_id}"
        )
        return model_id

    raise FileNotFoundError(
        f"模型 '{model_id}' (别名 '{key}') 未在本地缓存. "
        f"请先下载: huggingface-cli download {model_id} "
        f"或设置 allow_network_fallback=True"
    )


def list_cached_models() -> Dict[str, bool]:
    """列出所有模型的缓存状态.

    Returns:
        Dict[别名, 是否已缓存]
    """
    return {key: is_model_cached(model_id) for key, model_id in LOCAL_MODEL_PATHS.items()}


def print_cache_status() -> None:
    """打印所有模型的缓存状态 (用于调试)."""
    print("\n" + "=" * 60)
    print("  HuggingFace 本地模型缓存状态")
    print("=" * 60)
    cache_dir = get_hf_cache_dir()
    print(f"  缓存目录: {cache_dir}")
    print(f"  离线模式: {'已启用' if _OFFLINE_MODE_CONFIGURED else '未启用'}")
    print()
    for key, model_id in LOCAL_MODEL_PATHS.items():
        cached = is_model_cached(model_id)
        status = "[已缓存]" if cached else "[未缓存]"
        print(f"  {status:10s} {key:25s} → {model_id}")
    print("=" * 60 + "\n")


# =========================================================================
# 自动初始化: 模块导入时自动启用离线模式
# =========================================================================
# 在导入本模块时自动启用离线模式, 避免后续所有 HF 调用联网
# 用户可通过 setup_offline_mode() 再次确认 (幂等)
setup_offline_mode()
