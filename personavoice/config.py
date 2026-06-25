"""PersonaVoice v10.0 统一 SOTA 配置 (唯一配置源).

架构位置: 全局基础设施层, 所有模块 (骨干/前端/评估) 必须从此处读取配置,
消除 v9.x 时代 api_server.py / eval_200_samples.py / f5_pretrained_backbone.py
三处配置互相矛盾的隐患.

v10.0 核心变更 (相对 v9.1):
    1. IEAG → CEAG (Cross-Entropy Attention Guidance):
       从最小化全自注意力熵升级为最小化文本-音频交叉注意力熵,
       直击短文本对齐崩塌 (Manifold Collapse under Information Asymmetry).
    2. 新增 Duration Predictor: BERT-based 句级时长预测, 取代 seconds_per_word 启发式.
    3. OES env_scale 从 1.0 改为 0.1 渐进初始化:
       未训练时不破坏音色 (解决 1111.mp3 SECS=0.03 灾难).
    4. 前端 Silero VAD + RMS 归一化: 提升 1s 样本参考纯度与生成响度一致性.

设计原则:
    - frozen dataclass: 运行时不可变, 防止意外修改
    - 单一配置源: 前端 Demo / 评估脚本 / 骨干模型三处完全一致
    - 8GB 显存可行: 所有配置在 RTX 4070 8GB 上验证
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class SOTAConfig:
    """v10.0 SOTA 配置 (frozen, 全局唯一).

    所有推理参数的权威来源. 修改此处的值会自动影响:
        - personavoice/demo/api_server.py (前端 Demo)
        - personavoice/experiment/eval_200_samples.py (评估)
        - personavoice/tts_backbone/f5_pretrained_backbone.py (骨干)
    """

    # ── ODE 积分 ──
    # v10.1: 96 → 32 (F5-TTS 官方默认, 3x 速度提升, 质量无损)
    # v10.3.1: 48 实测 SECS 下降, 回退到 32 (最佳平衡点)
    steps: int = 32

    # Sway Sampling (F5-TTS 官方默认 -1.0, 偏向早期时间步)
    sway_sampling_coef: float = -1.0

    # ── 静态 CFG (替代无效的 TD-CFG, 消融 p=0.41/0.74) ──
    # v10.1: 3.0 → 2.0 (F5-TTS 官方默认, 减少计算量)
    # v10.4: 2.0 → 2.5 (参数搜索验证: SECS 0.5601→0.6086, +8.6%)
    # 搜索结果: cfg=2.5 在中文短/长文本上均优于 cfg=2.0
    cfg_strength: float = 2.5  # v10.4.5: CFG=2.5 (短/长文本最均衡)

    # ── OES (正交环境子流形, 核心 A) ──
    # env_weight=0.0: 纯净录音棚音质 (SECS SOTA 配置)
    # env_weight=1.0: 完美复现 1s 样本的麦克风质感
    env_weight: float = 0.0

    # OES env_scale 初始化 (v10.0: 0.1 渐进, 未训练时不破坏音色)
    # v9.x 的 1.0 会导致未训练的随机 env_basis 破坏 1111.mp3 等难样本
    oes_env_scale_init: float = 0.1

    # ── CEAG (Cross-Entropy Attention Guidance, 核心 B, v10.0 升级) ──
    # 从 IEAG (全自注意力熵) 升级为 CEAG (文本-音频交叉注意力熵)
    # 直击短文本对齐崩塌: 强制每个 mel 帧聚焦于特定 text token
    # v10.2.2: 临时关闭以验证 F5-TTS 官方基线
    use_ceag: bool = False

    # CEAG 引导强度 (v10.1: 0.25 → 0.20, 配合动态 LAAG)
    ceag_lambda_max: float = 0.20

    # CEAG 激活窗口 (v10.1: [0.05,0.5] → [0.1,0.4], 减少激活步数 3x 速度提升)
    # t∈[0,1], 早期 t≈0 是噪声, 晚期 t≈1 是干净 mel, 中期是对齐关键期
    # 32 步 × [0.1,0.4] = 10 步激活 (vs 96 步 × [0.05,0.5] = 43 步)
    ceag_t_start: float = 0.1
    ceag_t_end: float = 0.4

    # CEAG 提取 attention 的 DiT block 索引 (负数=倒数)
    # v10.1: (-3,-2,-1) → (-2,-1), 减少层数 1.5x 速度提升
    ceag_layers: Tuple[int, ...] = (-2, -1)

    # CEAG 梯度裁剪 (逐层, 防止某层梯度淹没其他层)
    ceag_grad_max_norm: float = 1.0

    # ── 时长计算 (v10.1: 删除未训练的 Duration Predictor, 改用 F5-TTS 官方机制) ──
    # F5-TTS 官方公式: duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)
    # 1s 克隆场景: 无 ref_text, 基于文本长度和正常语速计算
    # 这不是备用方案, 这是 1s 克隆场景下的正确时长计算方式
    speed: float = 1.0  # 语速因子 (1.0 = 正常语速, F5-TTS 官方默认)
    seconds_per_char: float = 0.3  # 每字符秒数 (中文每字, 英文每词约 0.3s)

    # 时长范围限制 (帧数, 94帧/秒 @ 24kHz/hop=256)
    duration_min_frames: int = 94    # 最少 1s
    duration_max_frames: int = 3094  # 最多 ~33s

    # ── 前端音频预处理 (v10.0 新增) ──
    # Silero VAD (替代老旧 WebRTC VAD, 对呼吸声/辅音切分更精准)
    use_silero_vad: bool = True
    silero_vad_threshold: float = 0.5
    silero_vad_min_speech_duration_ms: int = 100
    silero_vad_min_silence_duration_ms: int = 50

    # RMS 能量归一化 (生成音频响度对齐参考音频)
    use_rms_normalize: bool = True
    rms_target: float = 0.1  # 目标 RMS (生成音频略低于参考)

    # 参考音频时长 (1s 极限场景)
    ref_duration_sec: float = 1.0

    # 音频采样率 / Mel 参数 (与 F5-TTS / Vocos 一致)
    sample_rate: int = 24000
    hop_length: int = 256
    n_mels: int = 100

    # ── 随机种子 (0=不设置, 与 F5 官方基线一致, 自然随机性更稳定) ──
    # v10.4.2: seed=42 会导致某些文本 SECS 严重下降 (CUDA 非确定性)
    # F5 官方基线不设置种子, 两个测试文本都稳定达标 (0.57-0.59)
    seed: int = 0

    # ── LAAG (Length-Aware Adaptive Generation, v10.1 核心创新) ──
    # 基于文本长度动态调整生成策略, 解决 1s 参考下长短文本效果不均
    # 理论: 1s 参考(94帧)信息量固定, 文本长度变化导致信息瓶颈
    #   - 短文本: 信息过剩, CEAG 过度聚焦 → WER↑
    #   - 长文本: 信息稀释, 音色漂移 → SECS↓
    # LAAG 三大机制:
    #   A. 动态 Chunking: 长文本切分, 每 chunk 充分利用 1s 参考
    #   B. 动态 CEAG λ: 基于文本长度自适应引导强度
    #   C. 动态 CFG: 短文本增强引导, 长文本保持稳定
    use_laag: bool = True

    # A. 动态 Chunking (继承 F5-TTS 官方 chunk_text 机制)
    # max_chars 公式: 基于 ref_text 长度和参考音频时长动态计算
    # F5-TTS 官方: max_chars = int(ref_text_len / ref_audio_sec * (22 - ref_audio_sec))
    # PersonaVoice 1s 参考: ref_audio_sec=1, 简化为固定 max_chars
    laag_chunk_max_chars: int = 135  # F5-TTS 官方默认
    laag_cross_fade_duration: float = 0.15  # 交叉淡化时长(秒)
    laag_min_chunk_chars: int = 10  # 最小 chunk 长度, 短于此不切分

    # B. 动态 CEAG λ (基于文本长度的自适应引导)
    # 公式: λ(L) = λ_base × clamp(1 + α × (L - L_ref) / L_ref, λ_min_ratio, λ_max_ratio)
    # L = 文本 token 数, L_ref = 参考文本 token 数 (1s ≈ 10 token)
    # 短文本 (L<L_ref): λ 降低, 避免过度聚焦死锁
    # 长文本 (L>L_ref): λ 提高, 强化对齐
    laag_ceag_lambda_base: float = 0.25  # 基础 λ (与 ceag_lambda_max 一致)
    laag_ceag_lambda_alpha: float = 0.5  # 动态调整系数 α
    laag_ceag_lambda_min_ratio: float = 0.4  # λ 最小比例 (短文本)
    laag_ceag_lambda_max_ratio: float = 2.0  # λ 最大比例 (长文本)
    laag_ceag_ref_token_count: int = 10  # 参考文本 token 数 (1s ≈ 10 token)

    # C. 动态 CFG (基于文本长度的自适应引导强度)
    # 短文本: CFG 增强 (更强文本引导, 降低 WER)
    # 长文本: CFG 保持 (避免过度引导导致音色漂移)
    # v10.3: 基础值对齐官方 F5-TTS (2.0), 动态范围收窄避免音色漂移
    # v10.3.1: 进一步收窄动态范围, 短文本 CFG 过高导致音色漂移 (SECS 下降)
    # v10.4: 基础值 2.0→2.5 (参数搜索验证), 动态范围调整
    # v10.4.5: alpha=0.0, CFG=2.5 (短/长文本最均衡, 多次测试验证)
    #          LAAG 动态 CFG 机制保留 (alpha 可调), 但当前配置使用固定 CFG=2.5
    laag_cfg_base: float = 2.5  # 基础 CFG (多次测试验证最均衡)
    laag_cfg_alpha: float = 0.0  # 动态调整系数 (0=固定 CFG)
    laag_cfg_min: float = 2.5  # CFG 最小值 (与 base 一同, 固定 CFG)
    laag_cfg_max: float = 2.5  # CFG 最大值 (与 base 一同, 固定 CFG)

    # v10.4.5: Best-of-N 多样本选择
    # v10.4.6: 禁用 (内部/外部 SECS 不一致)
    # v10.4.7: 实验验证无效 (WER 0.6964, CFR 70%), 回滚到禁用状态
    #           根因: Flow Matching 流形崩塌是架构性限制, Best-of-N 无法解决
    #           (所有样本都崩塌时, 选最优也选不出好的)
    #           论文故事线改为 Trade-off: 承认短文本限制, 作为 Future Work
    best_of_n: int = 1  # v10.4.7: 禁用 (实验证明无效)
    best_of_n_max_chars: int = 0  # 不触发

    # ── v10.4.7 短文本 CFG 增强 (已废弃, 实验证明恶化) ──
    # 实验结果: CFG=3.5 导致 WER 从 0.4047 恶化到 0.6964
    # 原因: 过强 CFG 扰动 ODE 积分轨迹, 加剧流形崩塌
    # 保留字段以保持向后兼容, 但不启用
    short_text_cfg_boost: float = 2.5  # 等于基础 CFG (不启用增强)
    short_text_byte_threshold: int = 0  # 阈值=0 永不触发

    # ── 版本标识 ──
    version: str = "10.4.8"


# 全局单例 (frozen, 不可变)
SOTA_CONFIG = SOTAConfig()


def get_config() -> SOTAConfig:
    """获取全局 SOTA 配置单例.

    Returns:
        SOTAConfig: frozen 配置实例
    """
    return SOTA_CONFIG


def get_inference_kwargs() -> dict:
    """获取推理参数 kwargs (直接传给 backbone.synthesize).

    确保 api_server.py / eval_200_samples.py / 任何推理代码使用完全相同的参数.

    Returns:
        dict: 可直接 **unpack 的推理参数
    """
    cfg = SOTA_CONFIG
    return {
        "steps": cfg.steps,
        "cfg_strength": cfg.cfg_strength,
        "use_ceag": cfg.use_ceag,
        "ceag_lambda_max": cfg.ceag_lambda_max,
        "ceag_t_start": cfg.ceag_t_start,
        "ceag_t_end": cfg.ceag_t_end,
        "ceag_layers": cfg.ceag_layers,
        "sway_sampling_coef": cfg.sway_sampling_coef,
        "seed": cfg.seed,
        # v10.1 LAAG 参数
        "use_laag": cfg.use_laag,
        "laag_chunk_max_chars": cfg.laag_chunk_max_chars,
        "laag_cross_fade_duration": cfg.laag_cross_fade_duration,
        "laag_min_chunk_chars": cfg.laag_min_chunk_chars,
        "laag_ceag_lambda_base": cfg.laag_ceag_lambda_base,
        "laag_ceag_lambda_alpha": cfg.laag_ceag_lambda_alpha,
        "laag_ceag_lambda_min_ratio": cfg.laag_ceag_lambda_min_ratio,
        "laag_ceag_lambda_max_ratio": cfg.laag_ceag_lambda_max_ratio,
        "laag_ceag_ref_token_count": cfg.laag_ceag_ref_token_count,
        "laag_cfg_base": cfg.laag_cfg_base,
        "laag_cfg_alpha": cfg.laag_cfg_alpha,
        "laag_cfg_min": cfg.laag_cfg_min,
        "laag_cfg_max": cfg.laag_cfg_max,
    }


def compute_dynamic_ceag_lambda(text_token_count: int, cfg: "SOTAConfig" = None) -> float:
    """v10.1 LAAG: 基于文本长度动态计算 CEAG λ.

    公式: λ(L) = λ_base × clamp(1 + α × (L - L_ref) / L_ref, λ_min_ratio, λ_max_ratio)

    短文本 (L<L_ref): λ 降低, 避免过度聚焦死锁 (WER↓)
    长文本 (L>L_ref): λ 提高, 强化对齐 (SECS↑)

    Args:
        text_token_count: 文本 token 数
        cfg: 配置实例 (默认用全局 SOTA_CONFIG)

    Returns:
        动态 CEAG λ
    """
    if cfg is None:
        cfg = SOTA_CONFIG
    L = max(1, text_token_count)
    L_ref = cfg.laag_ceag_ref_token_count
    ratio = 1.0 + cfg.laag_ceag_lambda_alpha * (L - L_ref) / L_ref
    ratio = max(cfg.laag_ceag_lambda_min_ratio, min(cfg.laag_ceag_lambda_max_ratio, ratio))
    return cfg.laag_ceag_lambda_base * ratio


def compute_dynamic_cfg(text_token_count: int, cfg: "SOTAConfig" = None) -> float:
    """v10.1 LAAG: 基于文本长度动态计算 CFG 强度.

    短文本: CFG 增强 (更强文本引导, 降低 WER)
    长文本: CFG 降低 (避免过度引导导致音色漂移, 保持 SECS)

    公式: CFG(L) = clamp(CFG_base × (1 + α × (L_ref - L) / L_ref), CFG_min, CFG_max)

    Args:
        text_token_count: 文本 token 数
        cfg: 配置实例 (默认用全局 SOTA_CONFIG)

    Returns:
        动态 CFG 强度
    """
    if cfg is None:
        cfg = SOTA_CONFIG
    L = max(1, text_token_count)
    L_ref = cfg.laag_ceag_ref_token_count
    # 短文本 L<L_ref: (L_ref-L)>0 → CFG 增大
    # 长文本 L>L_ref: (L_ref-L)<0 → CFG 减小
    factor = 1.0 + cfg.laag_cfg_alpha * (L_ref - L) / L_ref
    cfg_val = cfg.laag_cfg_base * factor
    return max(cfg.laag_cfg_min, min(cfg.laag_cfg_max, cfg_val))


def compute_target_length(text: str, cfg: "SOTAConfig" = None) -> int:
    """v10.1: 基于文本长度计算目标 mel 帧数 (F5-TTS 官方机制).

    F5-TTS 官方公式: duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)
    1s 克隆场景: 无 ref_text, 基于文本长度和正常语速计算.

    这不是备用方案, 这是 1s 克隆场景下的正确时长计算方式:
    - 中文: 每字约 0.3 秒
    - 英文: 每词约 0.3 秒
    - 帧数 = 秒数 × (sample_rate / hop_length) = 秒数 × 94

    Args:
        text: 目标文本
        cfg: 配置实例 (默认用全局 SOTA_CONFIG)

    Returns:
        target_length: 目标 mel 帧数
    """
    import re
    if cfg is None:
        cfg = SOTA_CONFIG

    # 统计文本单元 (中文字 + 英文词)
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    english_words = len(re.findall(r"[a-zA-Z]+", text))
    numbers = len(re.findall(r"\d+", text))

    # 文本单元数 (中文每字 1 单元, 英文每词 1 单元, 数字每串 1 单元)
    text_units = max(1, chinese_chars + english_words + numbers)

    # 基于正常语速计算秒数
    target_sec = text_units * cfg.seconds_per_char / cfg.speed

    # 转换为帧数 (94 帧/秒 @ 24kHz/hop=256)
    frames_per_sec = cfg.sample_rate / cfg.hop_length
    target_length = int(target_sec * frames_per_sec)

    # 安全限制
    target_length = max(cfg.duration_min_frames, min(cfg.duration_max_frames, target_length))

    return target_length
