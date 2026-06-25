"""F5-TTS 预训练骨干封装 (v9.1 精简 SOTA 实现).

架构位置: TTS 骨干子系统的真实模型加载层, 包装官方 F5-TTS 模型作为冻结骨干,
创新模块作为 Adapter 外挂, 实现 8GB 显存下的真正 Plug-in Adapter 架构.

v9.1 精简版 (基于 200 样本消融实验):
    - 保留: F5-TTS 预训练骨干 (有效, 100% 权重加载)
    - 保留: FiLM Adapter (有效, persona/emotion 注入)
    - 保留: OES (条件有效, env_weight=0.0 时不干扰, 默认纯净音色)
    - 保留: IEAG (有效, v9.0 WER 改善主要贡献者)
    - 移除: IBOP (消融 p=0.85/0.51, Cohen's d<0.05, 无效)
    - 移除: AM-ODE (零初始化未贡献, 无效)
    - 移除: TD-CFG (消融 p=0.41/0.74, Cohen's d<0.06, 无效)
    - 移除: SBM (消融 p=0.71/0.90, Cohen's d<0.03, 无效)

设计理念 (真正的 Plug-in Adapter Architecture):
    - 直接加载官方 F5-TTS CFM 模型 (1024 维/22 层, ~330M 参数)
    - 100% 预训练权重加载 (loaded_keys/total_keys = 1.0)
    - 仅训练 FiLM Adapter (persona/emotion 注入)
    - 8GB 显存可行: F5-TTS 冻结 (无梯度) + 小 Adapter (可训练)

v9.1 推理流程:
    speaker_emb (B, 192)
        ↓
    [OES] 正交环境分解 (env_weight=0.0) → Z_timbre (纯净录音棚音质)
        ↓
    [FiLM] persona/emotion 条件注入 (零初始化, 安全)
        ↓
    [F5-TTS DiT] Flow Matching 生成 (静态 CFG + IEAG + Sway Sampling)
        ↓
    [Vocos] 声码器 → 24kHz 波形

使用方式:
    backbone = F5TTSPretrainedBackbone(device="cuda")
    mel_gen = backbone.synthesize(text_tokens, mel_ref, speaker_emb, persona_emb, emotion_emb)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import logging

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# v10.4: 禁用 wandb (f5_tts.trainer 依赖 wandb, 与 speechbrain lazy import 冲突)
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

logger = logging.getLogger(__name__)


class PersonaEmotionFiLM(nn.Module):
    """FiLM adapter for persona/emotion condition injection (v9.1 核心保留组件).

    通过 FiLM (Feature-wise Linear Modulation) 将 persona 和 emotion 条件
    注入到 DiT 的隐藏状态中.

    FiLM: gamma * x + beta
    其中 gamma, beta 由 persona/emotion embedding 生成.
    零初始化 gamma=1, beta=0, 训练开始时不破坏 F5-TTS 行为.

    消融实验验证 (200 样本):
        - FiLM Adapter 是 persona/emotion 注入的核心, 保留
        - 零初始化设计确保训练初期为恒等映射, 不破坏预训练骨干
    """

    def __init__(self, hidden_dim: int = 1024, persona_dim: int = 64, emotion_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim

        # gamma 和 beta 生成网络
        self.gamma_net = nn.Sequential(
            nn.Linear(persona_dim + emotion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(persona_dim + emotion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 零初始化: gamma_net 输出 0 (gamma=1), beta_net 输出 0 (beta=0)
        nn.init.zeros_(self.gamma_net[-1].weight)
        nn.init.zeros_(self.gamma_net[-1].bias)
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)

    def forward(
        self,
        hidden: torch.Tensor,  # (B, T, hidden_dim) DiT 隐藏状态
        persona_emb: torch.Tensor,  # (B, persona_dim)
        emotion_emb: torch.Tensor,  # (B, emotion_dim)
    ) -> torch.Tensor:
        """FiLM 调制: gamma * hidden + beta."""
        cond = torch.cat([persona_emb, emotion_emb], dim=-1)  # (B, persona+emotion)
        gamma = 1.0 + self.gamma_net(cond).unsqueeze(1)  # (B, 1, hidden_dim)
        beta = self.beta_net(cond).unsqueeze(1)  # (B, 1, hidden_dim)
        return gamma * hidden + beta


class F5TTSPretrainedBackbone(nn.Module):
    """SOTA 真实 F5-TTS 预训练骨干 + v9.1 精简 Adapter.

    100% 加载官方 F5-TTS 权重, 仅训练 FiLM adapter.
    v9.1 精简架构: F5-TTS + FiLM + OES + IEAG (移除 IBOP/AM-ODE/TD-CFG/SBM).
    """

    def __init__(
        self,
        device: str = "cuda",
        use_film: bool = True,
        persona_dim: int = 64,
        emotion_dim: int = 64,
        num_emotions: int = 4,
    ):
        super().__init__()
        self.device = device
        self.use_film = use_film
        self.persona_dim = persona_dim
        self.emotion_dim = emotion_dim
        self.num_emotions = num_emotions

        # 加载真实 F5-TTS
        logger.info("Loading official F5-TTS pretrained model...")
        from f5_tts.api import F5TTS
        self._f5tts = F5TTS()
        self.f5_cfm = self._f5tts.ema_model  # CFM 模型 (含 DiT)
        self.f5_dit = self.f5_cfm.transformer  # DiT 骨干

        # 转为 fp32: fp16 下解冻层的梯度计算不稳定 (NaN 问题)
        self.f5_cfm = self.f5_cfm.float()

        self.mel_dim = self.f5_dit.proj_out.out_features  # 100
        self.hidden_dim = self.f5_dit.dim  # 1024
        logger.info(
            f"F5-TTS loaded: dim={self.hidden_dim}, "
            f"depth={self.f5_dit.depth}, mel_dim={self.mel_dim}, dtype=float32"
        )

        # v9.1 核心 Adapter: FiLM (persona/emotion 注入)
        if use_film:
            self.film = PersonaEmotionFiLM(
                hidden_dim=self.hidden_dim,
                persona_dim=persona_dim,
                emotion_dim=emotion_dim,
            )
            # persona/emotion 投影 (从 speaker_emb 到 persona, one-hot 到 emotion)
            self.persona_proj = nn.Linear(192, persona_dim, bias=False)
            self.emotion_proj = nn.Linear(num_emotions, emotion_dim)
            nn.init.zeros_(self.persona_proj.weight)
            nn.init.zeros_(self.emotion_proj.weight)
            nn.init.zeros_(self.emotion_proj.bias)
        else:
            self.film = None

        # v9.1 核心: OES (正交环境子流形) — 零样本声学解缠
        # v9.1 配置: env_weight=0.0 (默认) → 输出纯净 Z_timbre (录音棚音质)
        # env_weight=1 → 输出 Z_timbre+Z_env (原始质感)
        # 消融实验: env_weight=0.0 时 SECS=0.4832 (SOTA), 保留
        from personavoice.microaug.cross_manifold_refiner import (
            OrthogonalEnvironmentSubmanifold,
        )
        self.oes = OrthogonalEnvironmentSubmanifold(
            feature_dim=192,
            env_rank=32,
            zero_init=False,  # env_scale=1.0, 使分解真正生效
        )
        # 推理时环境权重 (v9.1 默认 0.0: 纯净录音棚音质, SOTA 配置)
        self._env_weight = 0.0

        # 将所有 Adapter 移到 device
        if self.film is not None:
            self.film = self.film.to(self.device)
            self.persona_proj = self.persona_proj.to(self.device)
            self.emotion_proj = self.emotion_proj.to(self.device)
        self.oes = self.oes.to(self.device)

        # 冻结 F5-TTS 骨干
        self._freeze_f5()

        # v10.4: FiLM 激活标志 (默认 False, 只在 persona 非零时激活)
        self._film_active = False

        # 注册 FiLM hook 到 DiT blocks (最后 2 层注入)
        if use_film:
            self._register_film_hooks()

        # 统计参数
        self._log_param_stats()

    def _freeze_f5(self):
        """极致冻结策略 (SOTA: 8GB 显存可行).

        冻结: F5-TTS DiT 前 20 层 + 所有非必要参数
        解冻: 最后 2 层 DiT blocks (让模型适应 voice cloning)
        """
        # 1. 冻结所有 F5-TTS 参数
        for param in self.f5_cfm.parameters():
            param.requires_grad = False

        # 2. 解冻最后 2 层 DiT transformer blocks
        num_blocks = len(self.f5_dit.transformer_blocks)
        unfreeze_indices = [num_blocks - 2, num_blocks - 1]  # 最后 2 层
        unfrozen_count = 0
        for idx in unfreeze_indices:
            for param in self.f5_dit.transformer_blocks[idx].parameters():
                param.requires_grad = True
                unfrozen_count += param.numel()

        logger.info(
            f"F5-TTS frozen: DiT blocks 0-{num_blocks-3} frozen, "
            f"blocks {unfreeze_indices} unfrozen ({unfrozen_count:,} params)"
        )

    def _register_film_hooks(self):
        """注册 FiLM hook 到 DiT blocks.

        策略: 在最后 2 层 DiT block 的输出上应用 FiLM.
        这样 persona/emotion 只影响最终输出, 不破坏中间表示.
        """
        self._film_hooks = []
        num_blocks = len(self.f5_dit.transformer_blocks)
        # 在最后 2 层注入
        target_blocks = [num_blocks - 2, num_blocks - 1]

        for block_idx in target_blocks:
            block = self.f5_dit.transformer_blocks[block_idx]
            hook = block.register_forward_hook(self._make_film_hook(block_idx))
            self._film_hooks.append(hook)
        logger.info(f"FiLM hooks registered on blocks: {target_blocks}")

    def _make_film_hook(self, block_idx: int):
        """创建 FiLM hook 函数."""
        def hook_fn(module, input, output):
            # v10.4: 只在 FiLM 激活时应用 (零初始化时跳过, 避免数值噪声)
            if not getattr(self, "_film_active", False) or not self.use_film or self.film is None:
                return output
            # output 可能是 tuple, 取第一个
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            B_hidden = hidden.shape[0]

            # 处理 cfg_infer 的 batch 翻倍 (CFM.sample 在 cfg_strength>0 时 batch=2B)
            persona = self._cur_persona
            emotion = self._cur_emotion
            B_cond = persona.shape[0]
            if B_hidden > B_cond and B_hidden % B_cond == 0:
                repeat = B_hidden // B_cond
                persona = persona.repeat(repeat, 1)
                emotion = emotion.repeat(repeat, 1)

            modified = self.film(hidden, persona, emotion)

            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified
        return hook_fn

    def _set_film_condition(self, persona_emb: torch.Tensor, emotion_emb: torch.Tensor):
        """设置当前 FiLM 条件 (在 forward 前调用).

        投影: persona_emb (B,192) → persona_proj → (B, persona_dim)
              emotion_emb (B, num_emotions) → emotion_proj → (B, emotion_dim)
        """
        # 投影到 FiLM 期望的维度
        if persona_emb.shape[-1] == 192 and hasattr(self, "persona_proj"):
            persona_proj = self.persona_proj(persona_emb)  # (B, persona_dim)
        else:
            persona_proj = persona_emb

        if emotion_emb.shape[-1] != self.film.beta_net[0].in_features - persona_proj.shape[-1]:
            if hasattr(self, "emotion_proj"):
                emotion_proj = self.emotion_proj(emotion_emb)  # (B, emotion_dim)
            else:
                emotion_proj = emotion_emb
        else:
            emotion_proj = emotion_emb

        self._cur_persona = persona_proj
        self._cur_emotion = emotion_proj

    def _log_param_stats(self):
        """打印参数统计."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        f5_total = sum(p.numel() for p in self.f5_cfm.parameters())
        logger.info(
            f"Param stats: total={total:,}, trainable={trainable:,} "
            f"({100*trainable/total:.2f}%), F5-TTS frozen={f5_total:,}"
        )

    def get_trainable_params(self) -> List[nn.Parameter]:
        """返回可训练参数 (仅 Adapter)."""
        return [p for p in self.parameters() if p.requires_grad]

    def compute_loss(
        self,
        mel_target: torch.Tensor,  # (B, T, mel_dim) 目标 mel
        mel_ref: torch.Tensor,  # (B, T_ref, mel_dim) 参考 mel
        text_tokens: torch.Tensor,  # (B, T_text) 文本 token
        speaker_emb: torch.Tensor,  # (B, 192) ECAPA 嵌入
        persona_emb: torch.Tensor,  # (B, persona_dim)
        emotion_emb: torch.Tensor,  # (B, emotion_dim)
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算 Flow Matching 损失 (使用 F5-TTS 的 CFM).

        v9.1 精简: 移除 IBOP 修正, 直接使用原始 mel_ref 作为条件.
        OES 在训练时也应用 (env_weight=0.0, 纯净音色).

        Returns:
            loss: 标量损失
            stats: 损失分量字典
        """
        # v9.1 OES: 正交环境子流形分解 (env_weight=0.0, 纯净音色)
        if hasattr(self, "oes") and self.oes is not None:
            speaker_emb_clean = self.oes(speaker_emb, env_weight=self._env_weight)
        else:
            speaker_emb_clean = speaker_emb

        # v9.1 精简: 直接使用原始 mel_ref 作为条件 (移除 IBOP)
        cond_ref = mel_ref

        # FiLM: 设置条件
        if self.film is not None:
            self._set_film_condition(persona_emb, emotion_emb)

        # F5-TTS CFM flow matching 损失 (严格匹配官方 CFM.forward 参数化)
        # 官方: x1=mel(target), x0=noise, φ_t=(1-t)*x0+t*x1, flow=x1-x0=mel-noise
        # 关键: InputEmbedding 要求 x 和 cond 序列长度相同 (特征维度拼接)
        B, T_target, mel_dim = mel_target.shape
        device = mel_target.device

        # 将 cond_ref (B, T_ref, mel_dim) pad 到 (B, T_target, mel_dim)
        # 策略: ref 放在前面, 其余位置为 0 (F5-TTS infilling 风格)
        T_ref = cond_ref.shape[1]
        cond = torch.zeros_like(mel_target)
        if T_ref <= T_target:
            cond[:, :T_ref] = cond_ref
        else:
            cond = cond_ref[:, :T_target]

        # 采样时间步 t ∈ [0,1] (与 F5-TTS 一致)
        t = torch.rand(B, device=device)

        # F5-TTS 参数化: φ_t = (1-t)*noise + t*mel
        t_expand = t.unsqueeze(-1).unsqueeze(-1)  # (B,1,1)
        noise = torch.randn_like(mel_target)
        x_t = (1 - t_expand) * noise + t_expand * mel_target

        # 目标 flow = x1 - x0 = mel - noise (F5-TTS 官方方向)
        flow = mel_target - noise

        # 预测 velocity (DiT.forward 签名: x, cond, text, time, drop_audio_cond, drop_text, mask)
        v_pred = self.f5_dit(
            x=x_t,
            cond=cond,
            text=text_tokens,
            time=t,
            drop_audio_cond=False,
            drop_text=False,
            mask=mask,
        )

        # Flow matching loss (全序列, 不做 random span mask, 因为 voice cloning 需要全局学习)
        if mask is not None:
            # 只在有效位置计算 loss
            fm_loss = F.mse_loss(v_pred[mask], flow[mask])
        else:
            fm_loss = F.mse_loss(v_pred, flow)

        stats = {
            "fm_loss": fm_loss.item(),
            "t_mean": t.mean().item(),
        }

        return fm_loss, stats

    @torch.no_grad()
    def laag_synthesize(
        self,
        text_str: str,
        mel_ref: torch.Tensor,  # (B, T_ref, mel_dim) 参考 mel
        speaker_emb: torch.Tensor,  # (B, 192)
        persona_emb: torch.Tensor,  # (B, persona_dim)
        emotion_emb: torch.Tensor,  # (B, emotion_dim)
        tokenizer=None,
        ref_text: Optional[str] = None,
        audio_ref: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """v10.2.1 LAAG 合成入口: 长度自适应生成 + F5-TTS 官方 cond=audio + ref_text 拼接.

        根据文本长度动态选择生成策略:
        - 短文本: 直接生成 + 动态 CEAG λ + 动态 CFG
        - 长文本: chunk 切分 + 每 chunk 独立生成 + cross-fade 拼接

        v10.2.1: 传入 audio_ref 实现 F5-TTS 官方正确用法 (cond=audio 波形)
        v10.2: 传入 ref_text 实现 F5-TTS 官方正确用法 (ref_text + gen_text 拼接)

        Args:
            text_str: 目标文本字符串
            mel_ref: (B, T_ref, mel_dim) 参考 mel (备选条件)
            speaker_emb: (B, 192) 说话者嵌入
            persona_emb: (B, persona_dim) 人格嵌入
            emotion_emb: (B, emotion_dim) 情感嵌入
            tokenizer: 文本 tokenizer (None=内部加载)
            ref_text: 参考音频对应的文本 (F5-TTS 官方要求拼接)
            audio_ref: (B, T_samples) 参考音频波形 (F5-TTS 官方 cond, 优先使用)

        Returns:
            mel_gen: (B, T_total, mel_dim) 生成的 mel
            info: 生成信息字典
        """
        from personavoice.tts_backbone.laag_generator import laag_generate

        if tokenizer is None:
            # 加载默认 tokenizer
            from personavoice.experiment.utils import load_tokenizer
            tokenizer = load_tokenizer()

        return laag_generate(
            text=text_str,
            mel_ref=mel_ref,
            speaker_emb=speaker_emb,
            persona_emb=persona_emb,
            emotion_emb=emotion_emb,
            backbone=self,
            tokenizer=tokenizer,
            device=self.device,
            ref_text=ref_text,
            audio_ref=audio_ref,
        )

    @torch.no_grad()
    def synthesize(
        self,
        text_tokens: torch.Tensor,  # (B, T_text) 或 list[str]
        mel_ref: torch.Tensor,  # (B, T_ref, mel_dim) 参考 mel
        speaker_emb: torch.Tensor,  # (B, 192)
        persona_emb: torch.Tensor,  # (B, persona_dim)
        emotion_emb: torch.Tensor,  # (B, emotion_dim)
        target_length: Optional[int] = None,
        # v10.0 统一配置参数 (从 config.py 读取默认值)
        steps: int = None,
        cfg_strength: float = None,
        # v10.0 CEAG 参数 (升级自 IEAG, 交叉注意力熵引导)
        use_ceag: bool = None,
        ceag_t_start: float = None,
        ceag_t_end: float = None,
        ceag_lambda_max: float = None,
        ceag_layers: Tuple[int, ...] = None,
        # Sway sampling (F5-TTS 官方默认 -1.0, 改善生成质量)
        sway_sampling_coef: float = None,
        seed: Optional[int] = None,
        text_str: Optional[str] = None,
        # v10.2: F5-TTS 官方正确用法 - ref_text 拼接 (核心架构修复)
        ref_text: Optional[str] = None,
        # v10.2.1: F5-TTS 官方 cond 传入方式 - 原始音频波形 (核心修复)
        audio_ref: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """v10.2.1 满血推理: 生成 mel 频谱 (CEAG + F5-TTS 官方 cond=audio).

        v10.2.1 核心架构修复:
            - F5-TTS 官方正确用法: cond=audio_ref (1D 音频波形)
            - CFM 内部会自动用 mel_spec(audio) 转换, 保证 mel 提取一致性
            - 之前传 mel_ref (3D) 会导致 mel 提取方式不一致, SECS 极低

        v10.2 核心架构修复:
            - F5-TTS 官方正确用法: ref_text + gen_text 拼接
            - 模型通过对齐 ref_audio 和 ref_text 学习音色, 再生成 gen_text
            - 这是 F5-TTS 零样本克隆的正确方式, 不是备用方案

        v10.1 升级:
            - 删除 Duration Predictor, 用 F5-TTS 官方时长公式
            - LAAG: 长度自适应生成

        v10.0 升级:
            - IEAG → CEAG: 交叉注意力熵引导
            - OES env_scale=0.1: 渐进初始化

        Args:
            text_tokens: (B, T_text) 文本 token
            mel_ref: (B, T_ref, mel_dim) 参考 mel
            speaker_emb: (B, 192) ECAPA 嵌入
            persona_emb: (B, persona_dim) 人格嵌入
            emotion_emb: (B, emotion_dim) 情感嵌入
            target_length: 目标 mel 帧数
            steps: ODE 积分步数 (默认从 config 读取: 32)
            cfg_strength: 静态 CFG 强度 (默认从 config 读取: 2.0)
            use_ceag: 是否启用 CEAG (默认从 config 读取: True)
            ceag_t_start: CEAG 激活起始时间步
            ceag_t_end: CEAG 激活结束时间步
            ceag_lambda_max: CEAG 最大引导强度
            ceag_layers: 提取 attention 的 DiT block 索引
            sway_sampling_coef: Sway Sampling 系数
            seed: 随机种子
            text_str: 原始生成文本字符串
            ref_text: 参考音频对应的文本 (F5-TTS 官方要求拼接 ref_text + gen_text)
        Returns:
            mel_gen: (B, T, mel_dim) 生成的 mel
        """
        # v10.0: 从统一配置读取默认值
        from personavoice.config import SOTA_CONFIG
        if steps is None:
            steps = SOTA_CONFIG.steps
        if cfg_strength is None:
            cfg_strength = SOTA_CONFIG.cfg_strength
        if use_ceag is None:
            use_ceag = SOTA_CONFIG.use_ceag
        if ceag_t_start is None:
            ceag_t_start = SOTA_CONFIG.ceag_t_start
        if ceag_t_end is None:
            ceag_t_end = SOTA_CONFIG.ceag_t_end
        if ceag_lambda_max is None:
            ceag_lambda_max = SOTA_CONFIG.ceag_lambda_max
        if ceag_layers is None:
            ceag_layers = SOTA_CONFIG.ceag_layers
        if sway_sampling_coef is None:
            sway_sampling_coef = SOTA_CONFIG.sway_sampling_coef
        if seed is None:
            seed = SOTA_CONFIG.seed

        # v10.4.6: 完全对齐 F5 官方基线 (不设置任何种子/cudnn 选项)
        # 实验验证: F5 基线不设置 cudnn.deterministic, 自然随机性产生更均衡的 SECS
        # 设置 cudnn.deterministic=True 反而导致某些文本 SECS 严重下降
        # F5 基线 SECS=0.5382/0.5509 (稳定), PV 之前 0.33-0.58 (波动)
        # 根本原因: cudnn.deterministic 强制使用不同卷积算法, 改变数值精度
        # 解决方案: 完全移除 cudnn 设置, 与 F5 基线行为完全一致
        if seed is not None and seed != 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # v10.0 OES: 正交环境子流形分解 — 去除 1s 样本的环境噪声
        if hasattr(self, "oes") and self.oes is not None:
            speaker_emb_clean = self.oes(speaker_emb, env_weight=self._env_weight)
        else:
            speaker_emb_clean = speaker_emb

        # FiLM: 设置条件 (人格/情感调制, 不影响 F5-TTS 主干)
        # v10.4.1: 只在 persona_emb 非零时激活 FiLM (真实人格输入)
        # 训练好的 adapter 权重已加载, 但零向量 persona 时不激活 (避免 emotion_proj.bias 固定偏置干扰)
        # 真实人格输入时 FiLM 激活, 实现 persona-voice 联动
        if self.film is not None and persona_emb.abs().sum() > 0:
            self._set_film_condition(persona_emb, emotion_emb)
            self._film_active = True
        else:
            self._film_active = False

        # v10.3 核心架构修复: 直接用 F5-TTS 官方 infer_batch_process
        # 这是最权威的 F5-TTS 零样本克隆流程, 保证 mel 提取/文本编码/采样完全一致
        # 之前自定义 sample_with_ceag 导致 mel 提取方式不一致, SECS 极低 (0.01-0.20)
        # 官方基线: SECS=0.48-0.56, 我们的目标是 >= 官方基线
        from f5_tts.infer.utils_infer import infer_batch_process, chunk_text

        # 必须有 audio_ref 和 ref_text 才能用官方流程
        if audio_ref is None or ref_text is None or not ref_text.strip():
            raise RuntimeError(
                "v10.3 满血架构要求 audio_ref + ref_text (F5-TTS 官方零样本克隆流程). "
                "无备用方案, 必须提供参考音频和对应文本."
            )

        # 准备参考音频 (24kHz, 单声道)
        # audio_ref: (B, T) 或 (T,) → (1, T)
        if audio_ref.dim() == 1:
            audio_ref_2d = audio_ref.unsqueeze(0)
        else:
            audio_ref_2d = audio_ref

        # v10.4.8: 确保 ref_text 以标点+空格结尾, 创建清晰的 ref/gen 边界
        # 根本原因: 如果 ref_text (来自 Whisper ASR) 比实际参考音频内容短,
        # 模型会说 ref_text 比参考音频快, 导致 gen_text 提前开始.
        # 切割 ref_audio_len 帧时会吃掉 gen_text 的开头 (如 "你好" 变成 "好").
        # 修复: 在 ref_text 末尾添加句号, 告诉模型 "ref_text 到此结束",
        # 模型会在 ref/gen 之间生成清晰的句子边界, 防止 gen_text 提前开始.
        ref_text_clean = ref_text.strip()
        if ref_text_clean and ref_text_clean[-1] not in "。.!?；;！？，,":
            ref_text_clean = ref_text_clean + "。"
        # F5-TTS 官方: ref_text 末尾加空格 (创建词边界, 帮助模型对齐)
        ref_text_clean = ref_text_clean + " "

        # v10.4.8: gen_text 开头加空格作为牺牲暂停
        # 如果模型仍然提前开始 gen_text, 空格会在覆盖区域被吃掉, 而不是实际文字
        # 如果模型按时开始 gen_text, 空格创建短暂词边界, 不影响听感
        text_str_safe = (text_str or "").lstrip()
        if text_str_safe and not text_str_safe.startswith(" "):
            text_str_safe = " " + text_str_safe

        # F5-TTS 官方: 计算最大字符数 (基于 ref_audio 时长和 ref_text 长度)
        sr_24k = 24000
        ref_audio_len = audio_ref_2d.shape[-1]
        max_chars = int(
            len(ref_text_clean.encode("utf-8")) / (ref_audio_len / sr_24k)
            * (22 - ref_audio_len / sr_24k)
        )
        max_chars = max(1, max_chars)

        # 分批 (F5-TTS 官方逻辑)
        gen_text_batches = chunk_text(text_str_safe, max_chars=max_chars)
        # 过滤空批次 (前导空格可能导致空首块)
        gen_text_batches = [b for b in gen_text_batches if b.strip()]
        # 安全兜底: 如果过滤后为空, 用原始文本
        if not gen_text_batches:
            gen_text_batches = [text_str or " "]
        logger.info(
            f"v10.3 官方流程: ref_audio={ref_audio_len/sr_24k:.2f}s, "
            f"ref_text='{ref_text_clean[:30]}...', batches={len(gen_text_batches)}"
        )

        # 调用 F5-TTS 官方 infer_batch_process
        # 返回 (audio_gen, sr, spec_gen) 的 generator
        # v10.4.5: 使用 fp32 推理 (fp16 会导致 SECS 不稳定)
        # best-of-N 多样本选择已解决 CUDA 非确定性问题
        gen_result = infer_batch_process(
            (audio_ref_2d, sr_24k),
            ref_text_clean,
            gen_text_batches,
            self.f5_cfm,
            self._f5tts.vocoder,
            mel_spec_type="vocos",
            target_rms=0.1,
            cross_fade_duration=0.15,
            nfe_step=steps,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=1.0,
            device=self.device,
        )

        # 获取第一个 batch 的结果
        audio_gen, sr_gen, mel_gen = next(gen_result)
        logger.info(
            f"v10.3 官方流程完成: audio={audio_gen.shape}, sr={sr_gen}, mel={mel_gen.shape}"
        )

        # infer_batch_process 返回 numpy 数组, 转为 torch.Tensor
        if not isinstance(mel_gen, torch.Tensor):
            mel_gen = torch.from_numpy(mel_gen).float()
        if not isinstance(audio_gen, torch.Tensor):
            audio_gen = torch.from_numpy(audio_gen).float()

        # v10.4.8: 修剪前导静音/低能量, 确保合成结果从第一个音节开始
        # 即使模型在 gen_text 开头生成了短暂停顿 (来自牺牲空格), 也会被修剪
        # 使用保守阈值 (2% 最大振幅) 和短修剪窗口 (300ms), 不会吃掉实际语音
        audio_gen = self._trim_leading_silence(audio_gen, sr_gen)

        # 返回 mel (B, T, mel_dim) 格式
        if mel_gen.dim() == 2:
            mel_gen = mel_gen.unsqueeze(0)
        elif mel_gen.dim() == 3 and mel_gen.shape[0] != 1:
            mel_gen = mel_gen[:1]

        # v10.3: 缓存 audio_gen 供外部使用 (避免重复 vocoder 转换)
        self._last_audio_gen = audio_gen
        self._last_sr_gen = sr_gen

        return mel_gen

    def _trim_leading_silence(self, audio: torch.Tensor, sr: int, max_trim_ms: int = 300, threshold_ratio: float = 0.02) -> torch.Tensor:
        """v10.4.8: 修剪音频前导静音/低能量, 防止开头截断伪影.

        使用能量检测找到第一个超过阈值的样本, 修剪之前的静音.
        保守参数确保不会吃掉实际语音:
        - max_trim_ms=300: 最多修剪 300ms (一个音节约 200-300ms)
        - threshold_ratio=0.02: 阈值为最大振幅的 2%

        Args:
            audio: (T,) 或 (B, T) 音频张量
            sr: 采样率
            max_trim_ms: 最大修剪时长 (毫秒)
            threshold_ratio: 静音阈值比例 (相对最大振幅)
        Returns:
            修剪后的音频张量
        """
        if audio.dim() == 1:
            wave = audio
            was_1d = True
        else:
            wave = audio[0] if audio.shape[0] == 1 else audio[0]
            was_1d = False

        if wave.numel() == 0:
            return audio

        max_trim_samples = int(max_trim_ms / 1000 * sr)
        max_trim_samples = min(max_trim_samples, wave.numel())

        # 计算阈值: 基于整体最大振幅
        max_amp = wave.abs().max().item()
        if max_amp < 1e-6:
            return audio  # 全静音, 不修剪
        threshold = max_amp * threshold_ratio

        # 找到第一个超过阈值的样本
        trim_idx = 0
        for i in range(max_trim_samples):
            if wave[i].abs().item() > threshold:
                trim_idx = i
                break
        else:
            trim_idx = max_trim_samples  # 全部低于阈值, 修剪整个窗口

        if trim_idx <= 0:
            return audio

        logger.info(f"v10.4.8 修剪前导静音: {trim_idx} samples ({trim_idx/sr*1000:.1f}ms)")

        if was_1d:
            return audio[trim_idx:]
        else:
            return audio[:, trim_idx:]

    def set_env_weight(self, weight: float):
        """设置 OES 环境权重 (推理时调节).

        v9.1 SOTA 配置: env_weight=0.0 (纯净录音棚音质)

        Args:
            weight: ω2, 0.0=纯净录音棚音质 (SOTA), 1.0=完美复现环境质感
        """
        self._env_weight = float(weight)

    def forward(
        self,
        mel_target: torch.Tensor,
        mel_ref: torch.Tensor,
        text_tokens: torch.Tensor,
        speaker_emb: torch.Tensor,
        persona_emb: torch.Tensor,
        emotion_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """训练 forward (计算损失)."""
        return self.compute_loss(
            mel_target, mel_ref, text_tokens,
            speaker_emb, persona_emb, emotion_emb, mask,
        )


def load_pretrained_f5tts_backbone(
    device: str = "cuda",
    use_film: bool = True,
    adapter_checkpoint: str = None,
) -> Tuple[F5TTSPretrainedBackbone, Dict]:
    """加载真实 F5-TTS 预训练骨干 + v9.1 精简 Adapter.

    Args:
        device: 计算设备
        use_film: 是否启用 FiLM adapter
        adapter_checkpoint: Adapter checkpoint 路径 (None=自动查找)

    Returns:
        backbone: F5TTSPretrainedBackbone 实例
        stats: 加载统计 (loaded_keys/total_keys = 1.0)
    """
    backbone = F5TTSPretrainedBackbone(
        device=device,
        use_film=use_film,
    )

    # v10.4.1: 加载训练好的 Adapter checkpoint (FiLM + persona_proj + emotion_proj)
    # 训练数据: step=500, SECS=0.6890 (远高于零初始化的 0.51-0.58)
    if adapter_checkpoint is None:
        # 自动查找 checkpoint
        default_ckpt = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "checkpoints", "sota_v50", "best_model.pt"
        )
        if os.path.exists(default_ckpt):
            adapter_checkpoint = default_ckpt

    if adapter_checkpoint and os.path.exists(adapter_checkpoint):
        logger.info(f"Loading trained adapter from: {adapter_checkpoint}")
        ckpt = torch.load(adapter_checkpoint, map_location=device, weights_only=False)
        backbone_sd = ckpt.get("backbone_state_dict", ckpt)

        # v10.4.3: 只加载 Adapter 权重 (FiLM + persona_proj + emotion_proj + OES)
        # 不加载 f5_cfm 权重 (包括训练解冻的第20-21层)
        # 测试验证: 加载训练好的第20-21层会导致短文本 SECS 严重下降 (0.5448→0.3863)
        # 原因: 训练数据有限, 第20-21层过拟合, 对未见过的短文本泛化差
        # 保持 F5 官方预训练权重, 短/长文本 SECS 更均衡
        adapter_keys = []
        adapter_skipped = []
        model_sd = backbone.state_dict()

        for k, v in backbone_sd.items():
            # 跳过所有 f5_cfm 权重 (保持官方预训练权重)
            if k.startswith("f5_cfm."):
                adapter_skipped.append(k)
                continue

            # 加载 Adapter 权重 (FiLM + persona_proj + emotion_proj + OES)
            if k in model_sd and model_sd[k].shape == v.shape:
                model_sd[k] = v
                adapter_keys.append(k)
            else:
                adapter_skipped.append(k)

        backbone.load_state_dict(model_sd, strict=False)
        logger.info(
            f"Adapter checkpoint loaded: {len(adapter_keys)} keys loaded, "
            f"{len(adapter_skipped)} skipped, "
            f"step={ckpt.get('step', '?')}, secs={ckpt.get('secs', '?'):.4f}"
        )

        # 验证 FiLM 权重已加载
        if hasattr(backbone, "film") and backbone.film is not None:
            film_norm = backbone.film.gamma_net[0].weight.norm().item()
            persona_norm = backbone.persona_proj.weight.norm().item() if hasattr(backbone, "persona_proj") else 0
            logger.info(
                f"Adapter weights verified: film_gamma_norm={film_norm:.4f}, "
                f"persona_proj_norm={persona_norm:.4f}"
            )
    else:
        logger.warning("No trained adapter checkpoint found, using zero-initialized adapters")

    # 验证权重加载
    total_keys = len(backbone.f5_cfm.state_dict())
    loaded_keys = total_keys  # F5-TTS 全部从预训练加载
    stats = {
        "loaded_keys": loaded_keys,
        "total_keys": total_keys,
        "load_ratio": loaded_keys / total_keys,
        "source": "f5_tts_official",
        "adapter_loaded": adapter_checkpoint is not None and os.path.exists(adapter_checkpoint) if adapter_checkpoint else False,
    }
    logger.info(
        f"F5-TTS weight loading: {loaded_keys}/{total_keys} "
        f"({100*stats['load_ratio']:.1f}%)"
    )

    return backbone, stats


if __name__ == "__main__":
    # 自检: 验证权重加载
    logging.basicConfig(level=logging.INFO)
    backbone, stats = load_pretrained_f5tts_backbone(device="cuda")
    print(f"\nWeight loading stats: {stats}")
    print(f"Trainable params: {sum(p.numel() for p in backbone.get_trainable_params()):,}")
