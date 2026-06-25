"""人格特征提取器: 聊天记录 → Big Five 人格 → persona_emb.

架构位置: 人格管线的核心实现层, 将非结构化聊天记录转化为结构化人格表示,
供 FiLM Adapter 消费以影响语音风格.

Pipeline:
    1. 聊天记录 → BERT 编码 → 句子级语义特征
    2. 句子级特征 → Big Five 人格特征 (OCEAN 模型)
    3. Big Five → persona_emb (64维, 通过确定性投影)

设计理念:
    - 使用预训练 BERT (bert-base-chinese, 已本地缓存) 提取语义特征
    - Big Five 特征提取基于心理学词典 + 语义相似度, 无需训练
    - persona_emb 投影使用固定随机矩阵 (确定性, 可复现)
    - 投影后 L2 归一化并缩放, 确保 FiLM Adapter 安全消费

Big Five 人格模型 (OCEAN):
    - Openness (开放性): 创造力、好奇心、尝试新事物
    - Conscientiousness (尽责性): 计划性、责任感、自律
    - Extraversion (外向性): 社交性、活跃度、积极情绪
    - Agreeableness (宜人性): 信任、利他、合作
    - Neuroticism (神经质): 焦虑、情绪不稳定性
"""

import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Big Five 人格词典 (中英文双语)
# ─────────────────────────────────────────────────────────────────────
# 每个维度的关键词列表, 用于基于词频的人格特征估计
BIG_FIVE_KEYWORDS: Dict[str, List[str]] = {
    "openness": [
        # 中文
        "创造", "创新", "新", "尝试", "好奇", "艺术", "想象", "探索", "学习",
        "兴趣", "灵感", "独特", "创意", "开放", "思考", "阅读", "音乐", "文化",
        # 英文
        "creative", "new", "try", "curious", "art", "imagine", "explore", "learn",
        "interest", "inspire", "unique", "open", "think", "read", "music", "culture",
    ],
    "conscientiousness": [
        "计划", "规律", "认真", "负责", "有序", "努力", "坚持", "目标", "完成",
        "安排", "准备", "准时", "规则", "纪律", "勤奋", "细心", "专注", "决心",
        "plan", "organized", "careful", "responsible", "orderly", "effort", "persist",
        "goal", "complete", "prepare", "punctual", "rule", "discipline", "diligent",
    ],
    "extraversion": [
        "社交", "朋友", "热闹", "开朗", "说话", "聚会", "活跃", "外向", "聊天",
        " party", "聚会", "表达", "分享", "互动", "热情", "自信", "乐观", "健谈",
        "social", "friend", "lively", "outgoing", "talk", "party", "active", "express",
        "share", "interact", "enthusiastic", "confident", "optimistic", "gregarious",
    ],
    "agreeableness": [
        "真诚", "善良", "帮助", "友好", "合作", "体贴", "信任", "宽容", "温柔",
        "关心", "理解", "支持", "尊重", "谦虚", "同情", "和谐", "感恩", "包容",
        "sincere", "kind", "help", "friendly", "cooperate", "considerate", "trust",
        "tolerant", "gentle", "care", "understand", "support", "respect", "humble",
    ],
    "neuroticism": [
        "焦虑", "紧张", "担心", "情绪", "压力", "不安", "害怕", "烦躁", "沮丧",
        "孤独", "疲惫", "纠结", "敏感", "忧虑", "失落", "愤怒", "悲伤", "恐惧",
        "anxious", "nervous", "worry", "emotional", "stress", "uneasy", "afraid",
        "irritable", "depressed", "lonely", "tired", "sensitive", "fear", "sad",
    ],
}

# ─────────────────────────────────────────────────────────────────────
# BERT 模型单例 (懒加载)
# ─────────────────────────────────────────────────────────────────────
_bert_model = None
_bert_tokenizer = None
_bert_device = "cpu"  # BERT 在 CPU 运行, 节省 GPU 显存给 TTS


def _get_bert_model():
    """获取 BERT 模型单例 (bert-base-chinese, 本地缓存)."""
    global _bert_model, _bert_tokenizer
    if _bert_model is not None:
        return _bert_model, _bert_tokenizer

    try:
        from transformers import AutoTokenizer, AutoModel
        logger.info("Loading BERT model: bert-base-chinese (local cache)")
        _bert_tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-chinese", local_files_only=True
        )
        _bert_model = AutoModel.from_pretrained(
            "bert-base-chinese", local_files_only=True
        )
        _bert_model.to(_bert_device)
        _bert_model.eval()
        logger.info(f"BERT loaded on {_bert_device}, params={sum(p.numel() for p in _bert_model.parameters()):,}")
    except Exception as e:
        logger.warning(f"BERT loading failed: {e}, falling back to keyword-only mode")
        _bert_model = None
        _bert_tokenizer = None

    return _bert_model, _bert_tokenizer


# ─────────────────────────────────────────────────────────────────────
# 确定性投影矩阵 (Big Five 5维 → persona_emb 64维)
# ─────────────────────────────────────────────────────────────────────
# 使用固定随机种子生成投影矩阵, 确保可复现
_PROJECTION_MATRIX = None
_PROJECTION_SEED = 42


def _get_projection_matrix():
    """获取 Big Five → persona_emb 的投影矩阵 (5 → 64)."""
    global _PROJECTION_MATRIX
    if _PROJECTION_MATRIX is not None:
        return _PROJECTION_MATRIX

    rng = np.random.RandomState(_PROJECTION_SEED)
    # 正交初始化的投影矩阵
    mat = rng.randn(5, 64).astype(np.float32)
    # QR 分解正交化
    q, _ = np.linalg.qr(mat.T)  # (64, 5)
    _PROJECTION_MATRIX = q.T  # (5, 64)
    return _PROJECTION_MATRIX


# ─────────────────────────────────────────────────────────────────────
# 核心函数
# ─────────────────────────────────────────────────────────────────────

def _compute_keyword_scores(chat_list: List[str]) -> Dict[str, float]:
    """基于关键词频率计算 Big Five 人格分数.

    Args:
        chat_list: 聊天记录列表, 每条一行

    Returns:
        Dict[str, float]: Big Five 分数 (0.0-1.0)
    """
    # 合并所有聊天记录
    full_text = " ".join(chat_list).lower()

    scores = {}
    for trait, keywords in BIG_FIVE_KEYWORDS.items():
        # 统计关键词出现次数
        count = 0
        for kw in keywords:
            count += full_text.count(kw.lower())
        # 归一化: 基于文本长度和关键词数量的期望
        text_len = max(1, len(full_text))
        # 使用 sigmoid 将计数映射到 [0, 1]
        # 期望: 每千字约 1-3 个关键词命中 → 分数约 0.5-0.7
        density = count / (text_len / 1000.0) if text_len > 0 else 0
        score = 1.0 / (1.0 + np.exp(-(density - 1.5)))  # sigmoid, 中心在 1.5
        scores[trait] = float(np.clip(score, 0.05, 0.95))

    return scores


def _compute_bert_scores(chat_list: List[str]) -> Dict[str, float]:
    """使用 BERT 语义相似度计算 Big Five 人格分数.

    对每个 Big Five 维度, 计算聊天记录与该维度描述的语义相似度.

    Args:
        chat_list: 聊天记录列表

    Returns:
        Dict[str, float]: Big Five 分数 (0.0-1.0)
    """
    model, tokenizer = _get_bert_model()
    if model is None:
        return {}

    # Big Five 维度描述句 (用于语义相似度计算)
    trait_descriptions = {
        "openness": "我喜欢尝试新事物, 对艺术和文化感兴趣, 充满好奇心和创造力.",
        "conscientiousness": "我做事有计划, 认真负责, 自律且有目标.",
        "extraversion": "我喜欢社交活动, 开朗活跃, 乐于与人交流.",
        "agreeableness": "我待人真诚友善, 乐于助人, 容易信任他人.",
        "neuroticism": "我容易焦虑紧张, 情绪波动较大, 经常感到压力.",
    }

    try:
        # 编码聊天记录
        chat_text = " [SEP] ".join(chat_list[:10])  # 最多取前 10 条
        inputs = tokenizer(
            chat_text, return_tensors="pt", max_length=512,
            truncation=True, padding=True
        ).to(_bert_device)

        with torch.no_grad():
            outputs = model(**inputs)
        chat_emb = outputs.last_hidden_state[:, 0, :].squeeze(0)  # [CLS] token (768,)

        # 编码每个 trait 描述并计算相似度
        scores = {}
        for trait, desc in trait_descriptions.items():
            inputs_t = tokenizer(
                desc, return_tensors="pt", max_length=128,
                truncation=True, padding=True
            ).to(_bert_device)
            with torch.no_grad():
                outputs_t = model(**inputs_t)
            trait_emb = outputs_t.last_hidden_state[:, 0, :].squeeze(0)  # (768,)

            # 余弦相似度
            sim = F.cosine_similarity(chat_emb.unsqueeze(0), trait_emb.unsqueeze(0)).item()
            # 映射到 [0.1, 0.9] (余弦相似度通常在 [0, 1] 之间)
            scores[trait] = float(np.clip(sim * 0.8 + 0.1, 0.05, 0.95))

        return scores
    except Exception as e:
        logger.warning(f"BERT scoring failed: {e}")
        return {}


def extract_big_five_traits(
    chat_list: List[str],
    structured_recalls: Optional[Dict] = None,
) -> Dict[str, float]:
    """提取 Big Five 人格特征.

    融合关键词分数和 BERT 语义分数, 若有结构化回忆则加权融合.

    Args:
        chat_list: 聊天记录列表
        structured_recalls: 结构化回忆 (Big Five 分数, 可选)

    Returns:
        Dict[str, float]: Big Five 分数, 键为 OCEAN 维度名
    """
    # 1. 关键词分数 (基础)
    kw_scores = _compute_keyword_scores(chat_list)

    # 2. BERT 语义分数 (增强)
    bert_scores = _compute_bert_scores(chat_list)

    # 3. 融合
    traits = {}
    for trait in BIG_FIVE_KEYWORDS.keys():
        kw = kw_scores.get(trait, 0.5)
        bert = bert_scores.get(trait, kw)  # BERT 失败时回退到关键词分数
        # 加权融合: BERT 0.6 + 关键词 0.4
        traits[trait] = float(0.6 * bert + 0.4 * kw)

    # 4. 结构化回忆覆盖 (若提供, 直接使用)
    if structured_recalls:
        for trait in traits:
            if trait in structured_recalls:
                try:
                    val = float(structured_recalls[trait])
                    # 结构化回忆权重 0.5
                    traits[trait] = float(0.5 * traits[trait] + 0.5 * np.clip(val, 0.0, 1.0))
                except (ValueError, TypeError):
                    pass

    return traits


def extract_persona_emb(
    chat_list: List[str],
    structured_recalls: Optional[Dict] = None,
) -> torch.Tensor:
    """从聊天记录提取 persona_emb (64维).

    Pipeline: 聊天记录 → Big Five 特征 → 确定性投影 → persona_emb

    Args:
        chat_list: 聊天记录列表
        structured_recalls: 结构化回忆 (可选)

    Returns:
        torch.Tensor: persona_emb (64,), L2 归一化 + 缩放
    """
    # 1. 提取 Big Five 特征
    traits = extract_big_five_traits(chat_list, structured_recalls)

    # 2. 转换为向量
    trait_vec = np.array([
        traits.get("openness", 0.5),
        traits.get("conscientiousness", 0.5),
        traits.get("extraversion", 0.5),
        traits.get("agreeableness", 0.5),
        traits.get("neuroticism", 0.5),
    ], dtype=np.float32)  # (5,)

    # 3. 中心化 (0.5 为中性)
    trait_vec_centered = trait_vec - 0.5  # [-0.5, 0.5]

    # 4. 投影到 64 维
    proj_matrix = _get_projection_matrix()  # (5, 64)
    persona_vec = trait_vec_centered @ proj_matrix  # (64,)

    # 5. L2 归一化
    norm = np.linalg.norm(persona_vec)
    if norm > 1e-8:
        persona_vec = persona_vec / norm

    # 6. 缩放 (FiLM Adapter 训练时 persona_emb=0, 使用小尺度避免破坏)
    # 尺度 0.1: 轻微影响语音风格, 不破坏 SOTA 音质
    persona_vec = persona_vec * 0.1

    return torch.from_numpy(persona_vec).float()


def calibrate_traits(
    feedback: str,
    chat_list: Optional[List[str]] = None,
    structured_recalls: Optional[Dict] = None,
) -> Dict:
    """基于用户自然语言反馈校准人格特征.

    解析反馈中的方向性关键词, 微调对应 Big Five 维度.

    Args:
        feedback: 用户反馈 (如 "让外向性更高", "减少神经质")
        chat_list: 聊天记录 (用于重新计算基础分数)
        structured_recalls: 结构化回忆

    Returns:
        Dict: 校准后的人格特征 + 校准详情
    """
    # 重新提取基础特征
    if chat_list:
        traits = extract_big_five_traits(chat_list, structured_recalls)
    else:
        traits = {t: 0.5 for t in BIG_FIVE_KEYWORDS.keys()}

    # 解析反馈方向
    feedback_lower = feedback.lower()
    adjustments = {}

    # 方向关键词
    increase_words = ["高", "多", "强", "增加", "提高", "提升", "更", "加", "more", "higher", "increase"]
    decrease_words = ["低", "少", "弱", "减少", "降低", "减", "less", "lower", "decrease"]

    is_increase = any(w in feedback_lower for w in increase_words)
    is_decrease = any(w in feedback_lower for w in decrease_words)

    # 维度关键词匹配
    dimension_map = {
        "openness": ["开放", "openness", "创造", "好奇", "新"],
        "conscientiousness": ["尽责", "责任", "计划", "conscientious", "认真", "自律"],
        "extraversion": ["外向", "extraversion", "社交", "开朗", "活跃"],
        "agreeableness": ["宜人", "agreeableness", "友善", "善良", "合作"],
        "neuroticism": ["神经质", "neuroticism", "焦虑", "情绪", "紧张"],
    }

    for trait, keywords in dimension_map.items():
        if any(kw in feedback_lower for kw in keywords):
            delta = 0.1 if is_increase else (-0.1 if is_decrease else 0.05)
            old_val = traits[trait]
            new_val = float(np.clip(old_val + delta, 0.05, 0.95))
            traits[trait] = new_val
            adjustments[trait] = {
                "old": round(old_val, 4),
                "new": round(new_val, 4),
                "delta": round(new_val - old_val, 4),
            }

    if not adjustments:
        # 未匹配到维度, 整体微调
        for trait in traits:
            old_val = traits[trait]
            delta = 0.05 if is_increase else (-0.05 if is_decrease else 0.0)
            new_val = float(np.clip(old_val + delta, 0.05, 0.95))
            traits[trait] = new_val
            adjustments[trait] = {
                "old": round(old_val, 4),
                "new": round(new_val, 4),
                "delta": round(new_val - old_val, 4),
            }

    return {
        "traits": traits,
        "adjustments": adjustments,
        "feedback": feedback,
    }
