"""Embedding 语义增强层：Qwen3-Embedding-0.6B

在12维规则引擎之上新增4个语义维度：
  13. 段落语义相似度 (替代TF-IDF)
  14. 论证线性度 (相邻段语义偏移方差)
  15. 语义回溯率 (新概念引入率)
  16. 跨段衔接自然度 (段首-上段尾相关性)

模型: Qwen/Qwen3-Embedding-0.6B
  - C-MTEB 71.02 | 32K上下文 | MRL支持
  - 600M参数 | ~1.2GB | Apache 2.0
"""

import os
import re
import math
import numpy as np
from typing import Any, Optional
from collections import Counter

# 国内用户：优先用 HF 镜像，避免下载失败
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False


# ============================================================
#  阈值定义
# ============================================================

EMBEDDING_THRESHOLDS = {
    "paragraph_semantic_similarity": {
        "ai_range": (0.70, 0.95),       # 段间语义过于相似 (cos)
        "gray_range": (0.50, 0.70),
        "human_range": (0.20, 0.50),    # 人类段间变化更大
        "description": "段落语义相似度",
        "weight": 1.0,
    },
    "argument_linearity": {
        "ai_range": (0.00001, 0.0002),  # AI：语义偏移方差极小（匀速运动）
        "gray_range": (0.0002, 0.0005),
        "human_range": (0.0005, 0.005), # 人类：偏移方差更大（变速运动）
        "description": "论证线性度",
        "weight": 0.8,
    },
    "semantic_backtrack_rate": {
        "ai_range": (0.0, 0.05),        # AI：几乎不回溯
        "gray_range": (0.05, 0.10),
        "human_range": (0.10, 0.30),    # 人类：会回头补充
        "description": "语义回溯率",
        "weight": 0.5,
    },
    "transition_naturalness": {
        "ai_range": (0.65, 0.95),       # AI：段间衔接过于丝滑
        "gray_range": (0.45, 0.65),
        "human_range": (0.15, 0.45),    # 人类：时断时续
        "description": "跨段衔接自然度",
        "weight": 0.6,
    },
}


# ============================================================
#  模型加载（懒加载单例）
# ============================================================

_MODEL: Optional[SentenceTransformer] = None
_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


def get_model() -> Optional[SentenceTransformer]:
    """懒加载 Qwen3-Embedding-0.6B"""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if not HAS_ST:
        raise ImportError(
            "需要安装 sentence-transformers: pip install sentence-transformers"
        )
    print(f"[embedding] Loading {_MODEL_NAME} ...")
    _MODEL = SentenceTransformer(_MODEL_NAME, trust_remote_code=True)
    print(f"[embedding] Model loaded.")
    return _MODEL


def _encode(texts: list[str], instruction: str = "") -> np.ndarray:
    """批量编码文本为向量

    Qwen3-Embedding 支持指令感知，通过 instruction 参数提升语义精度
    """
    model = get_model()
    if model is None:
        raise RuntimeError("Embedding model not loaded")

    if instruction and hasattr(model, 'encode'):
        # Qwen3-Embedding 的指令感知编码
        embeddings = model.encode(
            texts,
            prompt=instruction,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
    else:
        embeddings = model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
    return np.array(embeddings)


# ============================================================
#  四个语义维度计算
# ============================================================

def compute_paragraph_similarity(embeddings: list[np.ndarray]) -> list[float]:
    """段落间语义余弦相似度

    Args:
        embeddings: 每段的归一化向量列表

    Returns:
        相邻段之间的相似度列表（长度 = len(embeddings)-1）
    """
    if len(embeddings) < 2:
        return [0.0]
    similarities = []
    for i in range(len(embeddings) - 1):
        # 已归一化，直接点积即余弦相似度
        sim = float(np.dot(embeddings[i], embeddings[i + 1]))
        similarities.append(round(sim, 4))
    return similarities


def compute_argument_linearity(embeddings: list[np.ndarray]) -> float:
    """论证线性度：相邻段语义偏移量的方差

    AI的偏移像匀速运动（方差小），人类的偏移像变速运动（方差大）

    Args:
        embeddings: 每段的归一化向量列表

    Returns:
        语义偏移方差（越低越像AI）
    """
    if len(embeddings) < 3:
        return 0.0

    drifts = []
    for i in range(len(embeddings) - 1):
        drift = np.linalg.norm(embeddings[i + 1] - embeddings[i])
        drifts.append(drift)

    if not drifts:
        return 0.0

    variance = float(np.var(drifts))
    return round(variance, 6)


def compute_semantic_backtrack(paragraphs: list[str],
                               embeddings: list[np.ndarray]) -> float:
    """语义回溯率：后段中出现的前段未提及的新概念比例

    AI线性推进，每段均匀引入新概念；
    人类会回头补充前文提到过但当时未展开的概念。

    Returns:
        回溯率（越高越像人类）
    """
    if len(paragraphs) < 3:
        return 0.0

    # 提取每段的关键名词（长度>=2的汉字词）
    para_keywords = []
    for para in paragraphs:
        words = re.findall(r'[一-鿿]{2,}', para)
        keywords = set(w for w in words if len(w) >= 3)  # 3字以上视为概念
        para_keywords.append(keywords)

    backtrack_count = 0
    total_new_concepts = 0

    for i in range(2, len(paragraphs)):
        current_keywords = para_keywords[i]
        prev_all = set()
        for j in range(i):
            prev_all |= para_keywords[j]

        new_concepts = current_keywords - prev_all
        backtrack_concepts = current_keywords & prev_all
        total_new_concepts += len(new_concepts)

        # 如果当前段大量使用之前见过的概念 → 回溯行为
        if len(current_keywords) > 0:
            backtrack_ratio = len(backtrack_concepts) / len(current_keywords)
        else:
            backtrack_ratio = 0

        # 回溯率 = 当前段使用了之前见过的概念的比重
        if backtrack_ratio > 0.5:
            backtrack_count += 1

    if len(paragraphs) < 3:
        return 0.0

    return round(backtrack_count / (len(paragraphs) - 2), 4)


def compute_transition_naturalness(paragraphs: list[str],
                                   embeddings: list[np.ndarray]) -> list[float]:
    """跨段衔接自然度：每段段首句与上段段尾句的语义相关度

    AI的段间衔接过于丝滑（高相关度），人类的段间时断时续

    Returns:
        每对相邻段落的衔接自然度列表
    """
    if len(paragraphs) < 2:
        return [0.0]

    # 提取段首句和段尾句
    first_sentences = []
    last_sentences = []

    for para in paragraphs:
        sentences = re.split(r'[。！？；]', para)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            first_sentences.append(sentences[0])
            last_sentences.append(sentences[-1])
        else:
            first_sentences.append(para[:50])
            last_sentences.append(para[-50:])

    transitions = []
    for i in range(len(paragraphs) - 1):
        # 上段尾句 vs 下段首句
        pair = [last_sentences[i], first_sentences[i + 1]]
        try:
            emb = _encode(pair)
            sim = float(np.dot(emb[0], emb[1]))
            transitions.append(round(sim, 4))
        except Exception:
            transitions.append(0.5)  # 默认中性值

    return transitions


# ============================================================
#  综合扫描
# ============================================================

def _classify_embedding_dim(name: str, value: float) -> str:
    """判断 embedding 维度落在哪个区间"""
    thresholds = EMBEDDING_THRESHOLDS.get(name, {})
    if not thresholds:
        return "unknown"

    ai_low, ai_high = thresholds.get("ai_range", (0, 0))
    human_low, human_high = thresholds.get("human_range", (0, 0))

    if ai_low <= value <= ai_high:
        return "ai"
    if human_low <= value <= human_high:
        return "human"
    gray = thresholds.get("gray_range")
    if gray and gray[0] <= value <= gray[1]:
        return "gray"
    return "gray"


def _compute_gap_embedding(name: str, value: float) -> float:
    """计算与人类区间的差距"""
    thresholds = EMBEDDING_THRESHOLDS.get(name, {})
    human_range = thresholds.get("human_range", (0, 0))
    if not human_range:
        return 0.0
    human_low, human_high = human_range
    if value < human_low:
        return human_low - value
    elif value > human_high:
        return value - human_high
    return 0.0


def scan_semantic(paragraphs: list[str]) -> dict[str, Any]:
    """对整个文档执行语义层扫描

    Args:
        paragraphs: 段落文本列表（纯文本，不需要 Paragraph 对象）

    Returns:
        {
            "paragraph_similarities": [float, ...],     # 每对相邻段落的相似度
            "argument_linearity": float,                 # 论证线性度（总体）
            "semantic_backtrack_rate": float,            # 语义回溯率（总体）
            "transition_naturalness": [float, ...],      # 每对相邻段的衔接自然度
            "per_paragraph": [{...}, ...],               # 每段的语义诊断
        }
    """
    if len(paragraphs) < 2:
        return {
            "paragraph_similarities": [],
            "argument_linearity": 0.0,
            "semantic_backtrack_rate": 0.0,
            "transition_naturalness": [],
            "per_paragraph": [],
            "error": "需要至少2个段落",
        }

    # 批量编码所有段落
    try:
        embeddings = _encode(paragraphs)
        embeddings_list = [embeddings[i] for i in range(len(embeddings))]
    except Exception as e:
        return {"error": f"Embedding encoding failed: {e}"}

    # 1. 段落语义相似度
    para_sims = compute_paragraph_similarity(embeddings_list)

    # 2. 论证线性度
    linearity = compute_argument_linearity(embeddings_list)

    # 3. 语义回溯率
    backtrack = compute_semantic_backtrack(paragraphs, embeddings_list)

    # 4. 跨段衔接自然度
    transitions = compute_transition_naturalness(paragraphs, embeddings_list)

    # 逐段维度分类
    per_paragraph = []
    for i in range(len(paragraphs)):
        entry = {}
        # 段落相似度：当前段与下一段的相似度
        if i < len(para_sims):
            sim = para_sims[i]
            entry["paragraph_similarity"] = {
                "value": sim,
                "zone": _classify_embedding_dim("paragraph_semantic_similarity", sim),
                "gap": _compute_gap_embedding("paragraph_semantic_similarity", sim),
                "description": "段落语义相似度",
                "weight": 1.0,
            }
        else:
            entry["paragraph_similarity"] = {
                "value": 0.0, "zone": "unknown", "gap": 0.0,
                "description": "段落语义相似度", "weight": 1.0,
            }

        # 衔接自然度
        if i < len(transitions):
            tn = transitions[i]
            entry["transition_naturalness"] = {
                "value": tn,
                "zone": _classify_embedding_dim("transition_naturalness", tn),
                "gap": _compute_gap_embedding("transition_naturalness", tn),
                "description": "跨段衔接自然度",
                "weight": 0.6,
            }
        else:
            entry["transition_naturalness"] = {
                "value": 0.0, "zone": "unknown", "gap": 0.0,
                "description": "跨段衔接自然度", "weight": 0.6,
            }

        per_paragraph.append(entry)

    # 论证线性度是全局维度，附加到每段
    for entry in per_paragraph:
        entry["argument_linearity"] = {
            "value": linearity,
            "zone": _classify_embedding_dim("argument_linearity", linearity),
            "gap": _compute_gap_embedding("argument_linearity", linearity),
            "description": "论证线性度（全局）",
            "weight": 0.8,
        }
        entry["semantic_backtrack"] = {
            "value": backtrack,
            "zone": _classify_embedding_dim("semantic_backtrack_rate", backtrack),
            "gap": _compute_gap_embedding("semantic_backtrack_rate", backtrack),
            "description": "语义回溯率（全局）",
            "weight": 0.5,
        }

    return {
        "paragraph_similarities": para_sims,
        "argument_linearity": linearity,
        "semantic_backtrack_rate": backtrack,
        "transition_naturalness": transitions,
        "per_paragraph": per_paragraph,
    }
