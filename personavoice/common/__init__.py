"""共享基础设施: 本地模型路径管理、离线模式配置.

架构位置: 全局基础设施层, 在所有模型加载之前调用, 确保离线运行环境.
"""

from personavoice.common.local_models import (
    LOCAL_MODEL_PATHS,
    setup_offline_mode,
    get_hf_cache_dir,
    is_model_cached,
    get_local_model_path,
    list_cached_models,
    print_cache_status,
)

__all__ = [
    "LOCAL_MODEL_PATHS",
    "setup_offline_mode",
    "get_hf_cache_dir",
    "is_model_cached",
    "get_local_model_path",
    "list_cached_models",
    "print_cache_status",
]
