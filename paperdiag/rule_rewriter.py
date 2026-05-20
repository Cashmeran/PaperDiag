"""规则改写引擎：确定性编辑操作，零模型依赖"""

import re
import json
import random
import jieba
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent / "data"


def _load_connector_data() -> dict:
    with open(DATA_DIR / "connector_blacklist.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _load_colloquial_dict() -> dict:
    with open(DATA_DIR / "colloquial_dict.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _load_antislop_data() -> dict:
    """加载AntiSlop负面清单 + AI深度痕迹模式"""
    path = DATA_DIR / "antislop_patterns_zh.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _split_sentences(text: str) -> list[str]:
    """中文分句，保留分隔符"""
    parts = re.split(r'([。！？；!?;])', text)
    sentences = []
    i = 0
    while i < len(parts) - 1:
        if parts[i].strip():
            sentences.append(parts[i].strip() + parts[i + 1])
        i += 2
    if i < len(parts) and parts[i].strip():
        sentences.append(parts[i].strip())
    return sentences


def _join_sentences(sentences: list[str]) -> str:
    """拼接句子"""
    return "".join(sentences)


# ============================================================
#  操作1: 删除/替换AI高频连接词
# ============================================================

def remove_connectors(text: str, intensity: float = 0.5, seed: int = 42) -> str:
    """按比例删除或替换AI高频连接词

    intensity: 0.0=不改, 0.5=改一半, 1.0=全改
    """
    rng = random.Random(seed)
    data = _load_connector_data()
    antislop = _load_antislop_data()
    result = text

    # 收集所有需要处理的AI标志词
    all_targets = []
    # 连接词黑名单（三级）
    for level, mult in [("high", 1.2), ("medium", 1.0), ("low", 0.5)]:
        for conn in data["levels"].get(level, []):
            all_targets.append((conn, intensity * mult))
    # AntiSlop负面清单（额外加强）
    neg_list = antislop.get("rewrite_negative_list", {})
    for category in ["transition", "evaluation", "verb", "analysis"]:
        for word in neg_list.get(category, []):
            if word not in [t[0] for t in all_targets]:
                all_targets.append((word, intensity * 0.8))

    suggestions = data.get("replacement_suggestions", {})

    for target, level_intensity in all_targets:
        if target in result and rng.random() < min(level_intensity, 1.0):
            if target in suggestions and suggestions[target] and rng.random() < 0.6:
                replacement = rng.choice(suggestions[target])
                if replacement:
                    result = result.replace(target, replacement, 1)
                else:
                    result = result.replace(target, "", 1)
            else:
                if rng.random() < 0.7:
                    result = result.replace(target, "", 1)

    return result


# ============================================================
#  操作2: 随机拆分/合并句子，调整句长方差
# ============================================================

def adjust_sentence_length(text: str, target_std: float = 14.0, seed: int = 42) -> str:
    """调整句长分布，目标是将句长方差从5-8的AI区间拉到12-20的人类区间"""
    rng = random.Random(seed)
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return text

    new_sentences = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        length = len(sent)

        # 长句(>35字)：有概率拆分为两句
        if length > 35 and rng.random() < 0.5:
            # 在逗号或顿号处拆分
            split_points = [m.start() for m in re.finditer(r'[，,、]', sent)]
            if split_points:
                mid = split_points[len(split_points) // 2]
                part1 = sent[:mid + 1].strip()
                part2 = sent[mid + 1:].strip()
                if len(part1) > 5 and len(part2) > 5:
                    # 给part2加句号（如果原句以句号结尾）
                    if part1.endswith("。"):
                        part1 = part1[:-1] + "，"
                    if not part2.endswith("。") and not part2.endswith("！") and not part2.endswith("？"):
                        part2 = part2.rstrip("，。") + "。"
                    new_sentences.append(part1)
                    new_sentences.append(part2)
                    continue
            new_sentences.append(sent)

        # 短句(<10字)：有概率与下一句合并
        elif length < 10 and rng.random() < 0.4:
            new_sentences.append(sent)  # 暂不合并，保持原样避免破坏语义

        else:
            new_sentences.append(sent)

    return _join_sentences(new_sentences)


# ============================================================
#  操作3: 术语口语化替换
# ============================================================

def colloquialize_terms(text: str, intensity: float = 0.3, seed: int = 42) -> str:
    """将部分学术术语替换为口语化表达

    intensity: 替换比例。建议0.2-0.4，太高会导致学术性丧失
    """
    rng = random.Random(seed)
    colloquial = _load_colloquial_dict()
    mappings = colloquial.get("mappings", {})
    result = text

    # 按intensity比例选择要替换的术语
    terms = list(mappings.keys())
    rng.shuffle(terms)
    num_to_replace = int(len(terms) * intensity)
    terms_to_replace = terms[:num_to_replace]

    for term in terms_to_replace:
        if term in result:
            alternatives = mappings[term]
            if alternatives:
                replacement = rng.choice(alternatives)
                if replacement:
                    result = result.replace(term, replacement, 1)

    return result


# ============================================================
#  操作4: 标点多样化
# ============================================================

def diversify_punctuation(text: str, seed: int = 42) -> str:
    """适度增加标点多样性，打破AI文本的标点规律"""
    rng = random.Random(seed)

    result = text

    # 将部分句号改为分号（在长句中）
    sentences = _split_sentences(result)
    new_sentences = []
    for sent in sentences:
        if len(sent) > 30 and sent.endswith("。") and rng.random() < 0.3:
            # 在句中随机将一处逗号改为分号
            if "，" in sent:
                commas = [m.start() for m in re.finditer(r'，', sent)]
                if commas:
                    idx = rng.choice(commas)
                    sent = sent[:idx] + "；" + sent[idx + 1:]
        new_sentences.append(sent)
    result = _join_sentences(new_sentences)

    # 偶尔将逗号改为顿号（在不影响语义的地方）
    if rng.random() < 0.3:
        # 在并列词之间
        result = re.sub(r'(\w)、(\w)', r'\1，\2', result)

    return result


# ============================================================
#  操作5: 注入短句打断节奏
# ============================================================

def inject_short_sentence(text: str, seed: int = 42) -> str:
    """在长句之间随机注入一个短句，打破句长均匀性"""
    rng = random.Random(seed)
    sentences = _split_sentences(text)
    if len(sentences) < 4:
        return text

    short_phrases = [
        "这很关键。",
        "说白了就是这样。",
        "值得深思。",
        "其实不然。",
        "这一点很重要。",
        "为什么？",
        "原因很简单。",
        "来看数据。",
        "反过来想。",
        "问题来了。",
    ]

    new_sentences = []
    for i, sent in enumerate(sentences):
        new_sentences.append(sent)
        # 在两个长句(>25字)之间以低概率插入短句
        if (len(sent) > 25
                and i + 1 < len(sentences)
                and len(sentences[i + 1]) > 25
                and rng.random() < 0.15):
            new_sentences.append(rng.choice(short_phrases))

    return _join_sentences(new_sentences)


# ============================================================
#  操作6: 被动改主动
# ============================================================

def passive_to_active(text: str, intensity: float = 0.3, seed: int = 42) -> str:
    """将部分被动语态改为主动语态"""
    rng = random.Random(seed)
    result = text

    # "被..." → 改为主语+动词
    passive_patterns = [
        (r'被(广泛)?应用于', '用于'),
        (r'被(普遍)?认为是', '公认是'),
        (r'被(人们)?视为', '看作'),
        (r'被(称之)?为', '称为'),
        (r'被(大量)?使用', '大量使用'),
        (r'被(广泛)?关注', '受到关注'),
    ]

    for pattern, replacement in passive_patterns:
        if rng.random() < intensity:
            result = re.sub(pattern, replacement, result, count=1)

    return result


# ============================================================
#  综合改写：根据诊断结果执行所有适用的操作
# ============================================================

def rewrite_paragraph(text: str,
                      diagnosis: dict[str, Any] | None = None,
                      intensity: float = 0.5,
                      seed: int = 42) -> dict[str, Any]:
    """综合改写单个段落

    Args:
        text: 原文
        diagnosis: 12维诊断结果（来自rules_engine.scan_paragraph）
        intensity: 改写强度 0-1
        seed: 随机种子（相同seed+相同输入=相同输出）

    Returns:
        {"text": 改写后文本, "operations": [执行的操作列表], "seed": seed}
    """
    rng = random.Random(seed)
    result = text
    operations = []

    if diagnosis is None:
        return {"text": text, "operations": [], "seed": seed}

    dims = diagnosis.get("dimensions", {})
    ranked_gaps = diagnosis.get("ranked_gaps", [])

    # 按排名差距排序，优先修改差距最大的维度
    gap_names = [g[0] for g in ranked_gaps]

    # 1. 如果连接词密度超标 → 删除/替换连接词
    conn = dims.get("connector_density", {})
    if conn.get("zone") == "ai" or "connector_density" in gap_names[:3]:
        op_intensity = min(intensity * 1.3, 1.0)
        result = remove_connectors(result, intensity=op_intensity, seed=seed)
        operations.append(f"删除/替换连接词 (强度:{op_intensity:.1f})")

    # 2. 如果句长方差在AI区间 → 调整句长
    sls = dims.get("sentence_length_std", {})
    if sls.get("zone") == "ai" or "sentence_length_std" in gap_names[:3]:
        result = adjust_sentence_length(result, seed=seed)
        operations.append("调整句长方差")

    # 3. 如果术语密度过高 → 口语化替换
    td = dims.get("term_density", {})
    if td.get("zone") == "ai" or "term_density" in gap_names[:4]:
        op_intensity = min(intensity * 0.5, 0.5)
        result = colloquialize_terms(result, intensity=op_intensity, seed=seed)
        operations.append(f"术语口语化替换 (强度:{op_intensity:.1f})")

    # 4. 标点熵在AI区间 → 标点多样化
    pe = dims.get("punctuation_entropy", {})
    if pe.get("zone") == "ai" or "punctuation_entropy" in gap_names[:5]:
        result = diversify_punctuation(result, seed=seed)
        operations.append("标点多样化")

    # 5. 随机注入短句（当多个维度异常时）
    if diagnosis.get("level") in ("red", "deep_red") and len(result) > 100:
        if rng.random() < 0.4:
            result = inject_short_sentence(result, seed=seed)
            operations.append("注入短句打断节奏")

    # 6. 被动改主动
    if rng.random() < intensity * 0.4:
        result = passive_to_active(result, intensity=intensity * 0.5, seed=seed)
        operations.append("被动改主动")

    return {"text": result, "operations": operations, "seed": seed}


def rewrite_document(paragraphs: list,
                     diagnoses: list[dict[str, Any]],
                     intensity: float = 0.5,
                     seed: int = 42) -> list[dict[str, Any]]:
    """对整个文档逐段改写

    Args:
        paragraphs: Paragraph对象列表
        diagnoses: 对应的诊断结果列表
        intensity: 改写强度
        seed: 随机种子

    Returns:
        [{"index": 0, "original": "...", "rewritten": "...", "operations": [...], "seed": 42}, ...]
    """
    results = []
    base_seed = seed

    for i, (para, diag) in enumerate(zip(paragraphs, diagnoses)):
        text = para.text if hasattr(para, 'text') else str(para)
        # 兼容 rule 级诊断 和 fused 级诊断
        level = diag.get("fused_level") or diag.get("level", "green")
        # 兼容两种gap来源
        if "ranked_gaps" not in diag and "rule_ranked_gaps" in diag:
            diag["ranked_gaps"] = diag["rule_ranked_gaps"]
        if "dimensions" not in diag and "rule_dimensions" in diag:
            diag["dimensions"] = diag["rule_dimensions"]

        # 绿色段落跳过
        if level == "green":
            results.append({
                "index": para.index if hasattr(para, 'index') else i,
                "original": text,
                "rewritten": text,
                "operations": ["已跳过（安全段落）"],
                "seed": base_seed + i,
            })
            continue

        # 按等级调整强度
        level_intensity = {
            "yellow": intensity * 0.6,
            "red": intensity,
            "deep_red": min(intensity * 1.3, 1.0),
        }.get(level, intensity)

        rw = rewrite_paragraph(text, diag, intensity=level_intensity, seed=base_seed + i)
        rw["index"] = para.index if hasattr(para, 'index') else i
        rw["original"] = text
        rw["rewritten"] = rw.pop("text", text)  # 统一key名

        results.append(rw)

    return results
