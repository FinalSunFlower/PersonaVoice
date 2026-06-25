"""PersonaVoice v10.0 CEAG (交叉熵注意力引导) — 推理时 WER 降低.

架构位置: TTS 骨干子系统中的推理时优化组件.
v10.0: IEAG → CEAG 升级, 直击短文本对齐崩塌.

=============================================================================
v10.0 核心创新: CEAG (Cross-Entropy Attention Guidance)
=============================================================================
从 IEAG (全自注意力熵) 升级为 CEAG (文本-音频交叉注意力熵).

理论: 流形崩塌假说 (Manifold Collapse under Information Asymmetry)
    1s 参考音频的信息量不足以约束 DiT 的全注意力分布.
    在对齐关键期 (t∈[0.05, 0.5]), mel token 的注意力分散到所有 text token
    (高熵), 导致每个 mel 帧无法聚焦于特定音素 → 咬字模糊 → WER↑.

    CEAG 通过最小化 mel→text 交叉注意力熵, 强制每个 mel 帧聚焦于
    特定 text token, 恢复对齐流形:

        v_final = v_cfg - λ(t) · ∇_x H(A_{mel→text})

    其中 A_{mel→text} 是注意力矩阵中 mel 行、text 列的子矩阵.

=============================================================================
v10.0 关键修复: Padding Mask (地雷1)
=============================================================================
F5-TTS 的自注意力是 [text_tokens, mel_tokens] 拼接后的全局注意力.
Batch 推理时, text 和 mel 都有 Padding.

若不屏蔽 <PAD> token, CEAG 会把极大梯度施加在无意义的 Padding 区域,
导致生成特征瞬间爆炸 (NaN).

修复: 在 _compute_cross_attention_entropy 中引入 text_mask 和 mel_mask.
在 log_softmax 之前, 将 mask 外的注意力权重设为 -inf,
确保梯度只流向有效的语义对齐区域.

=============================================================================
v10.0 架构决策 (基于 200 样本消融实验):
=============================================================================
  消融实验数据 (ablation_200_statistics.json):
    - TD-CFG: p=0.41/0.74, Cohen's d=-0.058/0.023 → 无效, 已移除
    - SBM:    p=0.71/0.90, Cohen's d=-0.026/0.009 → 无效, 已移除
    - IBOP:   p=0.85/0.51, Cohen's d=0.014/0.047  → 无效, 已移除
    - IEAG:   v9.0 WER 改善的主要贡献者 (0.2405→0.1817) → 保留并升级为 CEAG
"""

import logging
import math
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


# =============================================================================
# v10.0 核心: CEAG (Cross-Entropy Attention Guidance)
# =============================================================================

class _CEAGAttentionExtractor:
    """临时替换 AttnProcessor, 手动计算 attention 并保存 weights 用于 CEAG.

    F5-TTS 使用 F.scaled_dot_product_attention (SDPA) 不返回 weights.
    在 CEAG 激活的时间步, 临时替换 processor 为手动实现,
    保留完整的计算图以支持 ∇_x H(A_{mel→text}) 的自动微分.
    """

    def __init__(self, original_processor):
        self.original = original_processor
        self.attention_weights = None  # (B, H, N, N)

    def __call__(self, attn, x, mask=None, rope=None):
        """手动计算 attention, 保存 weights."""
        from f5_tts.model.modules import apply_rotary_pos_emb

        batch_size = x.shape[0]
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # qk norm
        if attn.q_norm is not None:
            query = attn.q_norm(query)
        if attn.k_norm is not None:
            key = attn.k_norm(key)

        # apply RoPE (与原始 processor 一致)
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (
                (xpos_scale, xpos_scale ** -1.0) if xpos_scale is not None else (1.0, 1.0)
            )
            if self.original.pe_attn_head is not None:
                pn = self.original.pe_attn_head
                query[:, :pn, :, :] = apply_rotary_pos_emb(
                    query[:, :pn, :, :], freqs, q_xpos_scale
                )
                key[:, :pn, :, :] = apply_rotary_pos_emb(
                    key[:, :pn, :, :], freqs, k_xpos_scale
                )
            else:
                query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
                key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        # 手动计算 attention scores (保留梯度)
        scale = head_dim ** -0.5
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale  # (B,H,N,N)

        # 应用 mask (F5-TTS 的序列 mask)
        # f5_tts 1.0.0 AttnProcessor 没有 attn_mask_enabled 属性, 直接用 mask
        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(1)
            attn_mask = attn_mask.expand(
                batch_size, attn.heads, query.shape[-2], key.shape[-2]
            )
            scores = scores.masked_fill(~attn_mask, float('-inf'))

        # softmax 得到 attention weights
        attn_weights = torch.softmax(scores, dim=-1)  # (B, H, N, N)

        # 保存 weights 供 CEAG 使用
        self.attention_weights = attn_weights

        # 计算 output (与 SDPA 等价)
        out = torch.matmul(attn_weights, value)
        out = out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        return out


class CEAGGuidance:
    """Cross-Entropy Attention Guidance (CEAG) — v10.0 核心创新.

    在 ODE 积分对齐关键期 (t∈[t_start, t_end]), 通过最小化 mel→text
    交叉注意力熵引导速度场, 解决 1s 参考下注意力分散导致的咬字崩溃.

    v_final = v_cfg - λ(t) · ∇_x H(A_{mel→text})

    其中:
        A_{mel→text} = A[mel_positions, text_positions]  (交叉注意力子矩阵)
        H(A_{mt}) = -Σ_i Σ_j A_{mt}[i,j] · log(A_{mt}[i,j])  (香农熵)
        λ(t) = λ_max · cos(π/2 · (t - center) / half_width)  (余弦窗)

    v10.0 Padding Mask 修复 (地雷1):
        在计算熵之前, 用 text_mask 和 mel_mask 屏蔽 padding 位置,
        确保梯度只流向有效的语义对齐区域, 防止 NaN.

    使用方式:
        ceag = CEAGGuidance(dit, text_len, mel_len, text_mask, mel_mask)
        # 在采样循环中:
        if ceag.is_active(t):
            v = ceag.guided_forward(x, fn, t)
        else:
            v = fn(t, x)
    """

    def __init__(
        self,
        dit,
        text_len: int,
        mel_len: int,
        text_mask: torch.Tensor,
        mel_mask: torch.Tensor,
        active_layer_indices: Tuple[int, ...] = (-3, -2, -1),
        t_start: float = 0.05,
        t_end: float = 0.5,
        lambda_max: float = 0.25,
        grad_max_norm: float = 1.0,
    ):
        """初始化 CEAG.

        Args:
            dit: F5-TTS 的 DiT 模型
            text_len: 文本 token 数 (含 padding 的长度)
            mel_len: mel 帧数 (含 padding 的长度)
            text_mask: (B, text_len) True=有效文本 token
            mel_mask: (B, mel_len) True=有效 mel 帧
            active_layer_indices: 提取 attention 的 DiT block 索引 (负数=倒数)
            t_start: CEAG 激活起始时间步
            t_end: CEAG 激活结束时间步
            lambda_max: 最大引导强度
            grad_max_norm: 梯度裁剪最大范数 (逐层, 防止梯度爆炸)
        """
        self.dit = dit
        self.text_len = text_len
        self.mel_len = mel_len
        # text_mask: (B, text_len) → (B, 1, 1, text_len) for broadcasting
        self.text_mask = text_mask.bool().view(text_mask.shape[0], 1, 1, -1)
        # mel_mask: (B, mel_len) → (B, 1, mel_len, 1) for broadcasting
        self.mel_mask = mel_mask.bool().view(mel_mask.shape[0], 1, -1, 1)
        self.active_layer_indices = active_layer_indices
        self.t_start = t_start
        self.t_end = t_end
        self.lambda_max = lambda_max
        self.grad_max_norm = grad_max_norm

        self._extractors: Dict[int, _CEAGAttentionExtractor] = {}
        self._original_processors: Dict[int, object] = {}
        self._patched = False

    def _lambda_schedule(self, t: float) -> float:
        """CEAG 强度调度: 余弦窗函数, 在区间中心最大."""
        if t < self.t_start or t > self.t_end:
            return 0.0
        center = (self.t_start + self.t_end) / 2
        half_width = (self.t_end - self.t_start) / 2
        if half_width < 1e-6:
            return self.lambda_max
        window = math.cos(math.pi / 2 * (t - center) / half_width)
        return self.lambda_max * max(0.0, window)

    def is_active(self, t: float) -> bool:
        """判断当前时间步是否激活 CEAG."""
        return self.t_start <= t <= self.t_end

    def _install_extractors(self):
        """在目标 DiT block 上安装 attention 提取器."""
        if self._patched:
            return
        blocks = self.dit.transformer_blocks
        n_blocks = len(blocks)
        for idx in self.active_layer_indices:
            real_idx = idx if idx >= 0 else n_blocks + idx
            block = blocks[real_idx]
            attn = block.attn

            # 保存原始 processor
            self._original_processors[real_idx] = attn.processor

            # 安装提取器
            extractor = _CEAGAttentionExtractor(attn.processor)
            self._extractors[real_idx] = extractor
            attn.processor = extractor

        self._patched = True

    def _restore_processors(self):
        """恢复原始 AttnProcessor."""
        if not self._patched:
            return
        blocks = self.dit.transformer_blocks
        for real_idx, original_proc in self._original_processors.items():
            blocks[real_idx].attn.processor = original_proc
        self._extractors.clear()
        self._original_processors.clear()
        self._patched = False

    def _compute_cross_attention_entropy(
        self,
        x_batch_size: int,
    ) -> torch.Tensor:
        """计算 mel→text 交叉注意力的香农熵 (v10.0 核心, 带 Padding Mask).

        提取注意力矩阵中 mel 行、text 列的子矩阵 A_{mel→text},
        在屏蔽 padding 后计算香农熵.

        F5-TTS 序列结构: [text_tokens (T_text), mel_tokens (T_mel)]
        全局注意力 A: (B, H, N, N), N = T_text + T_mel
        交叉注意力: A_mt = A[:, :, T_text:T_text+T_mel, 0:T_text]
            即 mel token 对 text token 的注意力 (对齐矩阵)

        v10.0 Padding Mask 修复 (地雷1):
            1. 提取 A_mt 后, 用 text_mask 屏蔽 padding text 列
            2. 重新 log_softmax (仅在有效 text token 上归一化)
            3. 用 mel_mask 屏蔽 padding mel 行 (不参与熵计算)
            4. 梯度只流向有效对齐区域, 防止 NaN

        Args:
            x_batch_size: 原始 x 的 batch size (CFG 前)

        Returns:
            entropy: 标量, 所有层所有 head 的平均交叉注意力熵
        """
        total_entropy = 0.0
        count = 0
        eps = 1e-8
        T_text = self.text_len

        for extractor in self._extractors.values():
            if extractor.attention_weights is None:
                continue
            # attn_weights: (B_total, H, N, N)
            w = extractor.attention_weights

            # CFG 模式下 B_total = 2 * x_batch_size, 只取 cond 部分
            if w.shape[0] > x_batch_size:
                w = w[:x_batch_size]  # (B, H, N, N)

            B = w.shape[0]

            # === 提取 mel→text 交叉注意力子矩阵 ===
            # A_mt[b, h, i, j] = mel 帧 i 对 text token j 的注意力
            # 序列: [text(0..T_text-1), mel(T_text..T_text+T_mel-1)]
            # mel 行: [T_text, T_text+T_mel)
            # text 列: [0, T_text)
            # 使用实际 attention weights 的维度, 而不是 self.mel_len
            actual_T_mel = w.shape[-1] - T_text  # 实际 mel 长度
            A_mt = w[:, :, T_text:T_text + actual_T_mel, :T_text]  # (B, H, T_mel, T_text)

            # === v10.0 Padding Mask 修复 (地雷1) ===
            # text_mask: (B, 1, 1, T_text) → 屏蔽 padding text 列
            # mel_mask: (B, 1, T_mel, 1) → 屏蔽 padding mel 行
            text_mask = self.text_mask[:B]  # (B, 1, 1, T_text)
            # 使用实际 mel 长度的 mask
            mel_mask_full = self.mel_mask[:B]  # (B, 1, mel_len, 1)
            # 截取或填充到 actual_T_mel
            if mel_mask_full.shape[2] >= actual_T_mel:
                mel_mask = mel_mask_full[:, :, :actual_T_mel, :]  # (B, 1, T_mel, 1)
            else:
                # 填充 False (padding)
                pad_size = actual_T_mel - mel_mask_full.shape[2]
                mel_mask = F.pad(mel_mask_full, (0, 0, 0, pad_size), value=False)

            # 将 padding text 列的注意力设为极小值 (-1e9)
            # 这样 softmax 后 padding text 位置的权重 → 0
            # 注意: 不能用 -inf, 因为会导致梯度 NaN; 用大负数更安全
            A_mt_masked = A_mt.masked_fill(~text_mask, -1e9)

            # 重新在有效 text token 上做 log_softmax (数值稳定的归一化)
            # log_softmax 自动应用 Log-Sum-Exp trick, 防止溢出
            log_A_mt = F.log_softmax(A_mt_masked, dim=-1)  # (B, H, T_mel, T_text)
            A_mt_norm = log_A_mt.exp()  # 归一化后的注意力权重

            # 计算香农熵: H = -Σ_j A[j] · log(A[j])
            # 仅在有效 text token 上求和 (padding 列的 A→0, 贡献→0)
            entropy_per_mel = -(A_mt_norm * log_A_mt).sum(dim=-1)  # (B, H, T_mel)

            # 仅对有效 mel 帧计算熵 (屏蔽 padding mel 行)
            # mel_mask: (B, 1, T_mel, 1) → (B, 1, T_mel)
            mel_mask_2d = mel_mask.squeeze(-1)  # (B, 1, T_mel)
            # 用 mask 加权平均: padding mel 帧的熵不参与
            mask_sum = mel_mask_2d.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B, 1, 1)
            entropy_masked = entropy_per_mel * mel_mask_2d  # (B, H, T_mel)
            entropy_per_head = entropy_masked.sum(dim=-1) / mask_sum.squeeze(-1)  # (B, H)

            total_entropy = total_entropy + entropy_per_head.mean()
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=self.dit.time_embed.linear.weight.device)
        return total_entropy / count

    def guided_forward(
        self,
        x: torch.Tensor,
        fn: Callable,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """CEAG 引导的速度场计算.

        v_final = v_cfg - λ(t) · ∇_x H(A_{mel→text})

        Args:
            x: (B, T, D) 当前 ODE 状态
            fn: 速度场函数 fn(t, x) -> v
            t: 当前时间步

        Returns:
            v_final: (B, T, D) 引导后的速度场
        """
        t_val = t.item() if isinstance(t, torch.Tensor) else t
        lambda_t = self._lambda_schedule(t_val)

        if lambda_t < 1e-6:
            # 不需要引导, 直接返回
            return fn(t, x)

        x_batch_size = x.shape[0]

        # 1. 安装 attention 提取器
        self._install_extractors()

        try:
            # 2. 启用梯度追踪
            x_grad = x.detach().requires_grad_(True)

            # 3. 前向传播 (提取 attention map)
            v = fn(t, x_grad)

            # 4. 计算交叉注意力熵 (只用 cond 部分, 带 padding mask)
            entropy = self._compute_cross_attention_entropy(x_batch_size)

            # 5. 计算熵对 x 的梯度: ∇_x H(A_{mel→text})
            if entropy.requires_grad and x_grad.requires_grad:
                grad_entropy = torch.autograd.grad(
                    outputs=entropy,
                    inputs=x_grad,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )[0]

                if grad_entropy is not None:
                    # v10.0 梯度裁剪: 逐层防止梯度爆炸 (地雷1 防御)
                    grad_norm = grad_entropy.norm()
                    if grad_norm > self.grad_max_norm:
                        grad_entropy = grad_entropy * (self.grad_max_norm / (grad_norm + 1e-8))

                    # 6. 修正速度场: v_final = v - λ·∇_x H(A)
                    #    最小化熵 = 减去熵的梯度方向
                    v_final = v.detach() - lambda_t * grad_entropy.detach()
                else:
                    v_final = v.detach()
            else:
                v_final = v.detach()

        finally:
            # 7. 恢复原始 processor
            self._restore_processors()

        return v_final


# =============================================================================
# v10.0 采样器: 静态 CFG + CEAG + Sway Sampling (满血版)
# =============================================================================

def sample_with_ceag(
    cfm,
    cond: torch.Tensor,
    text: torch.Tensor,
    duration: int,
    mel_ref: torch.Tensor,
    steps: int = 96,
    # 静态 CFG 参数 (替代无效的 TD-CFG)
    cfg_strength: float = 3.0,
    # CEAG 参数 (v10.0 核心创新)
    use_ceag: bool = True,
    ceag_t_start: float = 0.05,
    ceag_t_end: float = 0.5,
    ceag_lambda_max: float = 0.25,
    ceag_layers: Tuple[int, ...] = (-3, -2, -1),
    # Sway sampling (F5-TTS 官方默认 -1.0)
    sway_sampling_coef: Optional[float] = -1.0,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """v10.0 满血采样器: 静态 CFG + CEAG + Sway Sampling.

    v10.0 架构决策 (基于消融实验):
        - 移除 TD-CFG (p=0.41/0.74, 无效) → 使用静态 CFG
        - 移除 SBM (p=0.71/0.90, 无效) → 纯 ODE 积分
        - IEAG → CEAG 升级 (交叉注意力熵, 直击短文本对齐)
        - 保留 Sway Sampling (F5-TTS 官方默认)

    Args:
        cfm: F5-TTS CFM 模型
        cond: (B, T_ref, mel_dim) 参考条件
        text: (B, T_text) 文本 token
        duration: 目标长度
        mel_ref: 参考 mel (未使用, 保留接口兼容)
        steps: ODE 积分步数 (默认 96, v10.0 SOTA)
        cfg_strength: 静态 CFG 强度 (默认 3.0, v10.0 SOTA)
        use_ceag: 是否启用 CEAG
        ceag_t_start: CEAG 激活起始时间步 (默认 0.05, v10.0 扩展)
        ceag_t_end: CEAG 激活结束时间步 (默认 0.5, v10.0 扩展)
        ceag_lambda_max: CEAG 最大引导强度 (默认 0.25, v10.0 SOTA)
        ceag_layers: 提取 attention 的 DiT block 索引 (默认 (-3,-2,-1))
        sway_sampling_coef: Sway Sampling 系数 (默认 -1.0)
        seed: 随机种子

    Returns:
        out: (B, T, mel_dim) 生成的 mel
        trajectory: (steps+1, B, T, mel_dim) 轨迹
    """
    from f5_tts.model.utils import lens_to_mask

    # 兼容性: f5_tts 1.0.0 没有 get_epss_timesteps, 自己实现
    def get_epss_timesteps(steps, device, dtype):
        return torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=dtype)

    cfm.eval()
    device = cfm.device
    dit = cfm.transformer

    # Handle cond input
    if cond.ndim == 2:
        cond = cfm.mel_spec(cond)
        cond = cond.permute(0, 2, 1)
    cond = cond.to(next(cfm.parameters()).dtype)

    batch, cond_seq_len = cond.shape[:2]

    # Handle text
    if isinstance(text, list):
        if hasattr(cfm, 'vocab_char_map') and cfm.vocab_char_map is not None:
            from f5_tts.model.utils import list_str_to_idx
            text = list_str_to_idx(text, cfm.vocab_char_map).to(device)
        else:
            from f5_tts.model.utils import list_str_to_tensor
            text = list_str_to_tensor(text).to(device)

    # Duration
    if isinstance(duration, int):
        duration = torch.full((batch,), duration, device=device, dtype=torch.long)

    duration = torch.maximum(
        torch.maximum((text != -1).sum(dim=-1), torch.full_like(duration, cond_seq_len)) + 1,
        duration
    )
    max_duration = duration.amax().item()

    # === 条件准备 (无 Prompt Padding, v10.0 精简) ===
    cond_mask = lens_to_mask(
        torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)
    )

    cond_padded = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
    cond_mask_padded = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
    cond_mask_3d = cond_mask_padded.unsqueeze(-1)

    step_cond = torch.where(cond_mask_3d, cond_padded, torch.zeros_like(cond_padded))

    cond = cond_padded
    cond_mask = cond_mask_3d

    if batch > 1:
        mask = lens_to_mask(duration)
    else:
        mask = None

    # === v10.0 核心: 构建 CEAG 所需的 Padding Mask ===
    # text_mask: (B, T_text) True=有效 text token
    # F5-TTS 用 -1 表示 padding text token
    text_padded_len = text.shape[1]
    text_mask = (text != -1)  # (B, T_text)

    # mel_mask: (B, T_mel) True=有效 mel 帧
    mel_mask = lens_to_mask(duration)  # (B, max_duration)

    # === CEAG 初始化 (v10.0 核心创新) ===
    ceag = CEAGGuidance(
        dit,
        text_len=text_padded_len,
        mel_len=max_duration,
        text_mask=text_mask,
        mel_mask=mel_mask,
        active_layer_indices=ceag_layers,
        t_start=ceag_t_start,
        t_end=ceag_t_end,
        lambda_max=ceag_lambda_max,
    ) if use_ceag else None

    def fn(t, x):
        """ODE 速度场函数 (静态 CFG, 无 TD-CFG)."""
        dit.clear_cache()

        if cfg_strength < 1e-5:
            pred = dit(
                x=x, cond=step_cond, text=text, time=t,
                mask=mask, drop_audio_cond=False, drop_text=False, cache=True,
            )
            return pred

        # 静态 CFG: pred + (pred - null_pred) * cfg_strength
        # f5_tts 1.0.0 不支持 cfg_infer, 改为两次调用
        pred = dit(
            x=x, cond=step_cond, text=text, time=t,
            mask=mask, drop_audio_cond=False, drop_text=False, cache=True,
        )
        null_pred = dit(
            x=x, cond=step_cond, text=text, time=t,
            mask=mask, drop_audio_cond=True, drop_text=True, cache=True,
        )
        return pred + (pred - null_pred) * cfg_strength

    # Noise initialization
    y0 = []
    for dur in duration:
        if seed is not None:
            torch.manual_seed(seed)
        y0.append(torch.randn(dur.item(), cfm.num_channels, device=device, dtype=step_cond.dtype))
    y0 = pad_sequence(y0, padding_value=0, batch_first=True)

    # Time steps (Sway Sampling, F5-TTS 官方默认)
    if sway_sampling_coef is not None:
        t_steps = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=step_cond.dtype)
        t_steps = t_steps + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_steps) - 1 + t_steps)
    else:
        t_steps = get_epss_timesteps(steps, device=device, dtype=step_cond.dtype)

    # === 纯 ODE 积分 + CEAG 引导 (无 SBM 随机微分) ===
    trajectory = [y0]
    x = y0

    for i in range(steps):
        t_curr = t_steps[i]
        t_next = t_steps[i + 1]
        dt = (t_next - t_curr).item()
        t_val = t_curr.item()

        # === v10.0 核心创新: CEAG 引导 ===
        if ceag is not None and ceag.is_active(t_val):
            # CEAG 引导的速度场 (带 padding mask 的交叉注意力熵引导)
            # 注意: CEAG 步需要梯度, 不能用 inference_mode
            v = ceag.guided_forward(x, fn, t_curr)
        else:
            # 普通静态 CFG 速度场 (v10.1: 用 inference_mode 加速)
            with torch.inference_mode():
                v = fn(t_curr, x)

        # 纯 ODE 积分 (无 SBM 噪声注入, v10.0 精简)
        x = x + dt * v

        trajectory.append(x)

    dit.clear_cache()

    trajectory = torch.stack(trajectory, dim=0)
    sampled = trajectory[-1]
    out = torch.where(cond_mask, cond, sampled)

    return out, trajectory


# 向后兼容别名 (v9.x 代码迁移)
sample_with_ieag = sample_with_ceag
IEAGGuidance = CEAGGuidance
