"""12维规则引擎：对中文文本进行AI特征统计扫描"""

import re
import json
import math
import jieba
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent / "data"


# ============================================================
#  维度定义：每个维度的 AI / 人类 / 灰色 阈值区间
# ============================================================

DIMENSION_THRESHOLDS = {
    "sentence_length_std": {
        "ai_range": (5, 8),
        "gray_range": (8, 12),
        "human_range": (12, 20),
        "description": "句长方差",
        "unit": "标准差",
        "weight": 1.2,
    },
    "connector_density": {
        "ai_range": (8, 15),
        "gray_range": (6, 8),
        "human_range": (2, 6),
        "description": "连接词密度",
        "unit": "个/千字",
        "weight": 1.1,
    },
    "info_density": {
        "ai_range": (0.65, 0.75),
        "gray_range": (0.55, 0.65),
        "human_range_low": (0.40, 0.55),
        "human_range_high": (0.75, 0.80),
        "description": "信息密度",
        "unit": "实义词占比",
        "weight": 0.9,
    },
    "term_density": {
        "ai_range": (6, 100),
        "gray_range": (4, 6),
        "human_range": (0, 4),
        "description": "术语密度",
        "unit": "个/100字",
        "weight": 0.8,
    },
    "paragraph_similarity": {
        "ai_range": (0.7, 0.9),
        "gray_range": (0.5, 0.7),
        "human_range": (0.2, 0.5),
        "description": "段落结构相似度",
        "unit": "余弦相似度",
        "weight": 1.0,
    },
    "hapax_ratio": {
        "ai_range": (0.25, 0.35),
        "gray_range": (0.35, 0.45),
        "human_range": (0.45, 0.65),
        "description": "Hapax Legomena比率",
        "unit": "仅出现一次词占比",
        "weight": 0.7,
    },
    "zipf_deviation": {
        "ai_range": (0.15, 0.50),
        "gray_range": (0.10, 0.15),
        "human_range": (0.0, 0.10),
        "description": "Zipf偏离度",
        "unit": "R²偏差",
        "weight": 0.6,
    },
    "bigram_repetition": {
        "ai_range": (0.08, 0.30),
        "gray_range": (0.04, 0.08),
        "human_range": (0.0, 0.04),
        "description": "Bigram重复率",
        "unit": "重复bigram占比",
        "weight": 0.8,
    },
    "trigram_repetition": {
        "ai_range": (0.03, 0.20),
        "gray_range": (0.01, 0.03),
        "human_range": (0.0, 0.01),
        "description": "Trigram重复率",
        "unit": "重复trigram占比",
        "weight": 0.6,
    },
    "punctuation_entropy": {
        "ai_range": (0.0, 1.5),
        "gray_range": (1.5, 2.0),
        "human_range": (2.0, 3.5),
        "description": "标点熵",
        "unit": "香农熵",
        "weight": 0.5,
    },
    "overall_entropy": {
        "ai_range": (3.0, 5.0),
        "gray_range": (5.0, 6.0),
        "human_range": (6.0, 8.0),
        "description": "整体字符熵",
        "unit": "香农熵",
        "weight": 0.5,
    },
    "slop_density": {
        "ai_range": (3, 20),
        "gray_range": (1, 3),
        "human_range": (0, 1),
        "description": "Slop词密度",
        "unit": "个/千字",
        "weight": 1.3,
    },
}


def load_connectors() -> dict:
    """加载连接词黑名单"""
    path = DATA_DIR / "connector_blacklist.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_slop_patterns() -> dict:
    """加载中文AI slop模式"""
    path = DATA_DIR / "slop_patterns_zh.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_stopwords() -> set:
    """加载基础停用词表"""
    # 内置常用中文停用词
    stopwords = {
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
        "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
        "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
        "所", "为", "所以", "因为", "但是", "然而", "如果", "虽然", "可以",
        "这个", "那个", "什么", "怎么", "哪", "吗", "啊", "吧", "呢", "哦",
        "的", "地", "得", "之", "与", "及", "或", "对", "等", "从", "到",
        "向", "往", "朝", "在", "当", "为", "以", "就", "才", "刚", "已",
        "将", "要", "能", "会", "可", "该", "应", "用", "把", "被", "让",
        "给", "经", "过", "于", "按", "照", "依", "据", "靠", "沿", "顺",
    }
    return stopwords


# ============================================================
#  单个维度计算函数
# ============================================================

def compute_sentence_length_std(text: str) -> float:
    """计算句长方差"""
    sentences = split_sentences(text)
    if len(sentences) < 3:
        return 0.0
    lengths = [len(s) for s in sentences]
    return float(np.std(lengths))


def compute_connector_density(text: str) -> float:
    """计算连接词密度（个/千字）"""
    data = load_connectors()
    all_connectors = []
    for level in ["high", "medium", "low"]:
        all_connectors.extend(data["levels"][level])

    count = 0
    for conn in all_connectors:
        count += text.count(conn)

    char_count = len(text)
    return (count / char_count) * 1000 if char_count > 0 else 0


def compute_info_density(text: str) -> float:
    """计算信息密度（实义词占比）"""
    stopwords = load_stopwords()
    words = list(jieba.cut(text))
    if not words:
        return 0.0
    content_words = [
        w for w in words
        if w.strip()
        and len(w.strip()) >= 2
        and w not in stopwords
        and not re.match(r'^[\d\s\W]+$', w)
    ]
    return len(content_words) / len(words) if words else 0


def compute_term_density(text: str) -> float:
    """计算术语密度（个/100字）。粗略估计：长度>=3的汉字序列且非停用词"""
    stopwords = load_stopwords()
    words = list(jieba.cut(text))
    term_count = 0
    for w in words:
        w = w.strip()
        if len(w) >= 3 and re.match(r'^[一-鿿]+$', w) and w not in stopwords:
            term_count += 1
    char_count = len(text.replace(" ", "").replace("\n", ""))
    return (term_count / char_count) * 100 if char_count > 0 else 0


def compute_paragraph_similarity(text: str, prev_text: str) -> float:
    """计算段落间语义相似度（优先用 embedding，失败回退 TF-IDF）"""
    if not text or not prev_text:
        return 0.0

    # 优先用 embedding 模型
    try:
        from .embedding_layer import get_model
        model = get_model()
        if model is not None:
            emb = model.encode([prev_text, text], normalize_embeddings=True)
            sim = float(emb[0] @ emb[1])
            return round(max(-1.0, min(1.0, sim)), 4)
    except Exception:
        pass

    # 回退 TF-IDF
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vectorizer = TfidfVectorizer(tokenizer=lambda x: list(jieba.cut(x)), max_features=100)
        tfidf = vectorizer.fit_transform([prev_text, text])
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return float(sim)
    except Exception:
        return 0.0


def compute_hapax_ratio(text: str) -> float:
    """计算Hapax Legomena比率（仅出现一次的词/总词数）"""
    words = [w.strip() for w in jieba.cut(text) if len(w.strip()) >= 2]
    if not words:
        return 0.0
    freq = Counter(words)
    hapax_count = sum(1 for v in freq.values() if v == 1)
    return hapax_count / len(words)


def compute_zipf_deviation(text: str) -> float:
    """计算Zipf分布偏离度（R²偏差）"""
    words = [w.strip() for w in jieba.cut(text) if len(w.strip()) >= 2]
    if len(words) < 20:
        return 0.0

    freq = Counter(words)
    # 按频率降序排列
    sorted_freq = sorted(freq.values(), reverse=True)
    ranks = np.arange(1, len(sorted_freq) + 1)

    # 取log
    log_ranks = np.log(ranks)
    log_freqs = np.log(sorted_freq)

    # 线性拟合
    if len(log_ranks) < 3:
        return 0.0
    coeffs = np.polyfit(log_ranks, log_freqs, 1)
    predicted = np.polyval(coeffs, log_ranks)

    # R²
    ss_res = np.sum((log_freqs - predicted) ** 2)
    ss_tot = np.sum((log_freqs - np.mean(log_freqs)) ** 2)

    if ss_tot == 0:
        return 0.0
    r_squared = 1 - ss_res / ss_tot
    return float(1.0 - r_squared)  # 偏离度 = 1 - R²


def compute_ngram_repetition(text: str, n: int = 2) -> float:
    """计算n-gram重复率"""
    words = [w.strip() for w in jieba.cut(text) if len(w.strip()) >= 2]
    if len(words) < n + 2:
        return 0.0

    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    freq = Counter(ngrams)
    repeated = sum(1 for v in freq.values() if v > 1)
    return repeated / len(ngrams)


def compute_punctuation_entropy(text: str) -> float:
    """计算标点符号的香农熵"""
    punct_pattern = re.compile(
        '[，。！？；：、""''（）《》【】…—·'
        ',.!?;:\"\'\\(\\)\\[\\]{}]'
    )
    puncts = punct_pattern.findall(text)
    if not puncts:
        return 0.0
    counter = Counter(puncts)
    total = len(puncts)
    entropy = -sum((c / total) * math.log2(c / total) for c in counter.values() if c > 0)
    return float(entropy)


def compute_overall_entropy(text: str) -> float:
    """计算字符级香农熵"""
    chars = list(text.replace(" ", "").replace("\n", ""))
    if not chars:
        return 0.0
    counter = Counter(chars)
    total = len(chars)
    entropy = -sum((c / total) * math.log2(c / total) for c in counter.values() if c > 0)
    return float(entropy)


def compute_slop_density(text: str) -> float:
    """计算Slop词密度（个/千字）"""
    data = load_slop_patterns()
    all_patterns = []
    for category in ["overused_phrases", "empty_modifiers", "ai_conclusion_starters"]:
        all_patterns.extend(data["patterns"].get(category, []))

    count = 0
    for pattern in all_patterns:
        count += text.count(pattern)

    char_count = len(text)
    return (count / char_count) * 1000 if char_count > 0 else 0


def _detect_human_markers(text: str) -> dict:
    """检测强人类写作标记——防止文学/口语文本被误判为AI"""
    markers = []
    score = 0

    # 1. 第一人称叙事（我/我们+认知/感官动词）
    if re.search(r'(?:我|我们|俺|咱)(?:觉得|认为|发现|注意到|猜测|怀疑|担心|希望|看见|看到|听到|闻到|感到|想起|记得|知道|明白)', text):
        markers.append("第一人称叙事")
        score += 1

    # 2. 口语化反问/感叹
    if re.search(r'(?:难道|为啥|怎么|何必|岂不|还真是|确实啊|吧|呢|嘛|啦)', text):
        markers.append("口语反问")
        score += 1

    # 3. 排比句式（文学特征）
    urns = re.findall(r'(?:是那样的|是那样的).{0,20}(?:是那样的)', text)
    if urns or re.search(r'(?:不是.{0,15}也不是.{0,15}而是)', text):
        markers.append("排比句式")
        score += 2

    # 4. 拟人/比喻（像...一样/如同/仿佛/像是...在...）
    if re.search(r'(?:像.{2,8}(?:一样|似的|般)|如同|仿佛|宛若|恰似|像是.{2,8}在)', text):
        markers.append("拟人比喻")
        score += 2

    # 5. 情感强度词
    if re.search(r'(?:不禁|簌簌|的确|确实|实在|真正|深深|多么|忍不住|不由得|不觉|不知不觉)', text):
        markers.append("情感强度词")
        score += 1

    # 6. 逗号密集+长句=文学/口语风格（不是AI的短句均匀）
    commas = text.count("，") + text.count(",")
    chars = len(text.replace(" ", "").replace("\n", ""))
    if commas > 0 and chars / max(commas, 1) < 20:
        markers.append("短句节奏")
        score += 1

    # 7. 散文叙事特征：动作描写连续（V着/V地/V下去/V上来）
    narrative_actions = len(re.findall(
        r'(?:着|地|下去|上来|起来|过去|过来|进去|进来|出去|出来)(?:[，。；]|$)',
        text
    ))
    if narrative_actions >= 3:
        markers.append("散文叙事")
        score += 2

    # 8. 第一人称 + 身体/情感词汇（我的泪/我的心/我的手）
    if re.search(r'我(?:的|的)?(?:泪|心|眼|手|脚|脸|背|头|身子)', text):
        markers.append("亲身感受")
        score += 2

    # 9. 经典文学标记（大量排比+比喻+情感词 组合，或强散文叙事）
    is_classical = (
        (score >= 4 and bool(re.search(r'(?:不是.{0,20}而是|像.{2,8}一样|满眼都是|.{2,4}得.{2,4})', text)))
        or (score >= 5)  # 多种人类特征叠加 → 基本确定是文学作品
    )

    return {
        "markers": markers,
        "score": score,
        "is_classical_literature": is_classical,
    }


# ============================================================
#  综合诊断
# ============================================================

def _classify_dimension(name: str, value: float) -> str:
    """判断单个维度落在哪个区间"""
    thresholds = DIMENSION_THRESHOLDS.get(name, {})
    if not thresholds:
        return "unknown"

    if name == "info_density":
        ai_low, ai_high = thresholds.get("ai_range", (0.65, 0.75))
        if ai_low <= value <= ai_high:
            return "ai"
        human_low = thresholds.get("human_range_low", (0.40, 0.55))
        human_high = thresholds.get("human_range_high", (0.75, 0.80))
        if human_low[0] <= value <= human_low[1] or human_high[0] <= value <= human_high[1]:
            return "human"
        return "gray"

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


def _compute_gap(name: str, value: float) -> float:
    """计算与人类区间的差距"""
    thresholds = DIMENSION_THRESHOLDS.get(name, {})
    human_range = thresholds.get("human_range")
    if not human_range:
        return 0.0
    human_low, human_high = human_range

    if value < human_low:
        return human_low - value
    elif value > human_high:
        return value - human_high
    return 0.0


def scan_paragraph(text: str, prev_text: str = "") -> dict[str, Any]:
    """对单个段落执行12维扫描"""
    results = {}

    # 1. 句长方差
    results["sentence_length_std"] = compute_sentence_length_std(text)

    # 2. 连接词密度
    results["connector_density"] = compute_connector_density(text)

    # 3. 信息密度
    results["info_density"] = compute_info_density(text)

    # 4. 术语密度
    results["term_density"] = compute_term_density(text)

    # 5. 段落相似度
    if prev_text:
        results["paragraph_similarity"] = compute_paragraph_similarity(text, prev_text)
    else:
        results["paragraph_similarity"] = 0.0

    # 6. Hapax比率
    results["hapax_ratio"] = compute_hapax_ratio(text)

    # 7. Zipf偏离
    results["zipf_deviation"] = compute_zipf_deviation(text)

    # 8. Bigram重复
    results["bigram_repetition"] = compute_ngram_repetition(text, n=2)

    # 9. Trigram重复
    results["trigram_repetition"] = compute_ngram_repetition(text, n=3)

    # 10. 标点熵
    results["punctuation_entropy"] = compute_punctuation_entropy(text)

    # 11. 整体熵
    results["overall_entropy"] = compute_overall_entropy(text)

    # 12. Slop密度
    results["slop_density"] = compute_slop_density(text)

    # 逐维分类 + 差距
    classified = {}
    for name, value in results.items():
        zone = _classify_dimension(name, value)
        gap = _compute_gap(name, value)
        classified[name] = {
            "value": round(value, 4),
            "zone": zone,
            "gap": round(gap, 4),
            "description": DIMENSION_THRESHOLDS.get(name, {}).get("description", name),
            "weight": DIMENSION_THRESHOLDS.get(name, {}).get("weight", 1.0),
        }

    # 综合分数（加权）
    ai_count = sum(1 for v in classified.values() if v["zone"] == "ai")
    gray_count = sum(1 for v in classified.values() if v["zone"] == "gray")
    human_count = sum(1 for v in classified.values() if v["zone"] == "human")

    total_weight = sum(v["weight"] for v in classified.values())
    ai_weight = sum(v["weight"] for k, v in classified.items() if v["zone"] == "ai")

    comprehensive_score = ai_weight / total_weight if total_weight > 0 else 0

    # 风险等级
    if ai_count >= 5:
        level = "deep_red"
    elif ai_count >= 3:
        level = "red"
    elif ai_count >= 2:
        level = "yellow"
    else:
        level = "green"

    # ---- 滑稽误判保护：检测强人类标记，降级 ----
    human_markers = _detect_human_markers(text)
    abs_guard = {"triggered": False, "reason": "", "original_level": level}

    if human_markers["is_classical_literature"] and level != "green":
        abs_guard["triggered"] = True
        abs_guard["reason"] = "检测到经典文学特征（排比/拟人/诗化语言），降级处理"
        level = {"deep_red": "red", "red": "yellow", "yellow": "green"}.get(level, level)
        comprehensive_score *= 0.5
    elif human_markers["score"] >= 4 and level != "green":
        abs_guard["triggered"] = True
        abs_guard["reason"] = f"检测到{human_markers['markers']}等强人类写作特征（≥4分）"
        level = {"deep_red": "red", "red": "yellow", "yellow": "green"}.get(level, level)
        comprehensive_score *= 0.6
    elif human_markers["score"] >= 3 and level in ("red", "deep_red"):
        abs_guard["triggered"] = True
        abs_guard["reason"] = f"检测到{human_markers['markers']}等强人类写作特征"
        level = "yellow"
        comprehensive_score *= 0.7
    elif human_markers["score"] >= 2 and level == "deep_red":
        abs_guard["triggered"] = True
        abs_guard["reason"] = f"检测到{human_markers['markers']}等人类写作特征"
        level = "red"
        comprehensive_score *= 0.85

    return {
        "dimensions": classified,
        "comprehensive_score": round(comprehensive_score, 4),
        "level": level,
        "ai_count": ai_count,
        "gray_count": gray_count,
        "human_count": human_count,
        "absurdity_guard": abs_guard,
        "ranked_gaps": sorted(
            [(k, v["gap"], v["zone"]) for k, v in classified.items() if v["zone"] == "ai"],
            key=lambda x: x[1], reverse=True,
        ),
    }


def scan_document(paragraphs: list) -> list[dict[str, Any]]:
    """对整个文档逐段扫描。paragraphs 是 Paragraph 对象列表"""
    results = []
    prev_text = ""
    for para in paragraphs:
        text = para.text if hasattr(para, 'text') else str(para)
        result = scan_paragraph(text, prev_text)
        result["paragraph_index"] = para.index if hasattr(para, 'index') else 0
        results.append(result)
        prev_text = text
    return results


def split_sentences(text: str) -> list[str]:
    """中文分句，返回句子列表（不含空句）"""
    sentences = re.split(r'[。！？；!?;]+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]


# === 模块级常量：正则模式 ===

_PARALLEL_STARTERS = [
    r'^.{0,4}?(通过|借助|依靠|经由|基于|将|把|对|由|因|按|沿|顺|朝|向|往)',
    r'^.{0,4}?(从|以|在)',
]

_ACADEMIC_FORMULAS = [
    (r'本文从.{2,20}(?:角度|维度|方面|层面|路径).{0,10}(?:分析|探讨|研究|考察|阐述|揭示)', 0.15, "学术开头"),
    (r'(?:通过|借助|经由|以).{2,20}(?:将|使|让|令|把)', 0.12, "手段-结果句式"),
    (r'(?:最终|总之|综上|由此).{0,10}(?:成就|可见|表明|说明|体现|造就)', 0.12, "学术结尾"),
    (r'(?:不仅|不只).{2,30}(?:而且|也|还|更|进而|同时)', 0.10, "递进句式"),
]

_AI_TEMPLATES = [
    (["首先", "其次", "最后"], 0.40, "首先/其次/最后三件套"),
    (["首先", "其次", "此外"], 0.35, "首先/其次/此外模板"),
    (["一方面", "另一方面"], 0.30, "一方面/另一方面模板"),
    (["从.*来看", "从.*来看", "从.*来看"], 0.35, "从X角度来看三次"),
    (["不仅", "而且", "同时"], 0.30, "不仅/而且/同时排比"),
]

_REF_STARTERS = re.compile(r'^\s*(?:参考文献|参考书目|References|Bibliography|引用文献|参考资料)')


# === 句子级统计特征（领域无关，零模型依赖） ===

def _sentence_stats(sent: str) -> dict[str, float]:
    """计算单个句子的领域无关统计特征

    Returns:
        {lexical_diversity, char_entropy, word_repetition, unique_char_ratio}
    """
    chars = sent.replace(" ", "").replace("\n", "")
    n = len(chars)
    if n < 5:
        return {"lexical_diversity": 1.0, "char_entropy": 0.0,
                "word_repetition": 0, "unique_char_ratio": 1.0}

    # 1. 字符级熵（低熵 = 字符分布集中 = AI倾向）
    from collections import Counter as _Counter
    char_freq = _Counter(chars)
    char_entropy = -sum((c / n) * math.log2(c / n) for c in char_freq.values())

    # 2. 词汇多样性（jieba分词后 unique/total，低多样性 = AI倾向）
    words = [w.strip() for w in jieba.cut(chars) if len(w.strip()) >= 1]
    unique_words = len(set(words))
    total_words = len(words)
    lexical_diversity = unique_words / total_words if total_words > 0 else 1.0

    # 3. 词重复度（某词出现3+次 = 强AI信号）
    word_freq = _Counter(words)
    word_repetition = sum(1 for v in word_freq.values() if v >= 3)

    # 4. 唯一字符占比
    unique_char_ratio = len(set(chars)) / n

    return {
        "lexical_diversity": round(lexical_diversity, 4),
        "char_entropy": round(char_entropy, 4),
        "word_repetition": word_repetition,
        "unique_char_ratio": round(unique_char_ratio, 4),
    }


# === 段内句子Embedding相似度（使用已加载的Qwen3-Embedding模型） ===

def _intra_paragraph_similarity(sentences: list[str]) -> float:
    """计算段内句子间的平均embedding相似度

    AI文本中同段句子语义高度相似（>0.75），人类写作句子间差异更大（<0.50）。

    Returns:
        平均相似度，如果模型不可用返回-1
    """
    if len(sentences) < 2:
        return -1.0
    try:
        from .embedding_layer import get_model
        model = get_model()
        if model is None:
            return -1.0
        embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
        sims = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sims.append(float(embeddings[i] @ embeddings[j]))
        return round(sum(sims) / len(sims), 4) if sims else -1.0
    except Exception:
        return -1.0


# === 人类参考向量池（embedding相似度检测） ===

_HUMAN_REF_EMBEDDINGS = None  # [np.ndarray]

def _human_similarity(text: str) -> float:
    """计算文本与人类参考池的最大embedding相似度"""
    pool = _build_human_reference_pool()
    if pool is None or len(text.strip()) < 20:
        return -1.0
    try:
        from .embedding_layer import get_model
        model = get_model()
        if model is None:
            return -1.0
        vec = model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        max_sim = max(float(vec @ ref) for ref in pool)
        return round(max_sim, 4)
    except Exception:
        return -1.0


def _human_sim_to_signal(sim: float) -> float:
    """人类相似度→AI信号（反比）"""
    if sim < 0:
        return -1.0
    if sim > 0.65:
        return 0.0
    elif sim > 0.50:
        return 0.04
    elif sim > 0.35:
        return 0.10
    else:
        return 0.16


def _build_human_reference_pool() -> list | None:
    """用已知人类文本构建embedding参考向量池"""
    global _HUMAN_REF_EMBEDDINGS
    if _HUMAN_REF_EMBEDDINGS is not None:
        return _HUMAN_REF_EMBEDDINGS

    human_texts = _load_human_references()  # 优先从CSL加载500篇
    if not human_texts:
        human_texts = [
            "月光如流水一般，静静地泻在这一片叶子和花上。薄薄的青雾浮起在荷塘里。",
            "我看见他戴着黑布小帽，穿着黑布大马褂，深青布棉袍，蹒跚地走到铁道边。",
            "秋天的后半夜，月亮下去了，太阳还没有出，只剩下一片乌蓝的天。",
            "这几天心里颇不宁静。今晚在院子里坐着乘凉，忽然想起日日走过的荷塘。",
            "关于唐诗中月意象的统计与分析。本文对《全唐诗》中出现的月亮意象进行了穷尽性的统计。",
            "做了三十年宋词研究，有一个感受越来越强烈：词这种文体在本质上是女性化的。",
            "做田野调查最怕的就是受访者不愿意说话。",
            "鲁迅的《朝花夕拾》有一种特别的声音。",
            "去年在西北大学参加了一个关于口述史的工作坊，收获很大。",
            "读了这么多年书，有一个心得：真正好的研究问题往往是简单的。",
        ]

    try:
        from .embedding_layer import get_model
        model = get_model()
        if model is None:
            return None

        # 优先从磁盘缓存加载（首次编码后存盘，后续秒加载）
        cache_path = Path(__file__).parent.parent / "output" / "calibration" / "human_embeddings.npy"
        if cache_path.exists():
            import numpy as _np
            cached = _np.load(cache_path)
            _HUMAN_REF_EMBEDDINGS = [cached[i] for i in range(len(cached))]
            return _HUMAN_REF_EMBEDDINGS

        # 首次编码
        embeddings = model.encode(human_texts, normalize_embeddings=True, show_progress_bar=False)
        _HUMAN_REF_EMBEDDINGS = [embeddings[i] for i in range(len(embeddings))]

        # 存盘
        try:
            import numpy as _np
            _np.save(cache_path, embeddings)
        except Exception:
            pass

        return _HUMAN_REF_EMBEDDINGS
    except Exception:
        return None

def _extract_sentence_features(sent: str, s_idx: int, sents: list[str],
                                 char_tfidf, high_connectors, med_connectors,
                                 low_connectors, slop_words,
                                 para_template_score, para_parallel_bonus,
                                 para_uniformity_bonus, length_uniformity_bonus,
                                 sent_stats: dict, para_level: str, bigram_bonus: float = 0.0, self_bonus: float = 0.0, human_bonus: float = 0.0) -> list[float]:
    """从单个句子提取13维特征向量（不做任何加权融合）

    特征维度（全部归一化到0-1区间）：
      0: lexical_diversity (词汇多样性)
      1: char_entropy (字符熵, 0-8归一化)
      2: word_repetition (词重复次数, 截断到5)
      3: unique_char_ratio (唯一字占比)
      4: char_ngram_sim (char n-gram相似度)
      5: high_connector_count (高级连接词命中数)
      6: med_connector_count (中级连接词命中数)
      7: low_connector_count (低级连接词命中数)
      8: slop_hit (是否命中slop, 0/1)
      9: para_template_score (段落模板分)
      10: parallel_starter (是否匹配排比开头, 0/1)
      11: formula_hit (是否命中学术套话, 0/1)
      12: length_uniformity (句长均匀度)
      13: position_score (句位: 首=1, 尾=0.6, 中=0)
      14: embedding_uniformity (段内同质化)
      15: para_level_score (段落基线: deep_red=1, red=0.7, yellow=0.4, green=0)
    """
    chars = len(sent)
    stats = sent_stats

    # 0: 词汇多样性
    f0 = stats.get("lexical_diversity", 1.0)

    # 1: 字符熵 (归一化到0-1, 典型范围0-8)
    f1 = min(stats.get("char_entropy", 4.0) / 8.0, 1.0)

    # 2: 词重复 (截断到5)
    f2 = min(stats.get("word_repetition", 0) / 5.0, 1.0)

    # 3: 唯一字占比
    f3 = stats.get("unique_char_ratio", 1.0)

    # 4: char n-gram相似度
    f4 = 0.0
    if char_tfidf is not None and chars >= 10:
        try:
            f4 = _char_ngram_similarity(sent, char_tfidf)
        except Exception:
            pass

    # 5-7: 连接词命中 (计数, 截断到3)
    f5 = min(sum(1 for c in high_connectors if c in sent) / 3.0, 1.0)
    f6 = min(sum(1 for c in med_connectors if c in sent) / 3.0, 1.0)
    f7 = min(sum(1 for c in low_connectors if c in sent) / 3.0, 1.0)

    # 8: slop命中
    f8 = 1.0 if sum(1 for s in slop_words if s in sent) >= 1 else 0.0

    # 9: 段落模板分
    f9 = min(para_template_score / 0.40, 1.0)

    # 10: 排比开头
    f10 = 0.0
    for pat in _PARALLEL_STARTERS:
        if re.match(pat, sent.strip()):
            f10 = 1.0
            break

    # 11: 学术套话
    f11 = 0.0
    for pattern, _, _ in _ACADEMIC_FORMULAS:
        if re.search(pattern, sent):
            f11 = 1.0
            break

    # 12: 句长均匀度
    f12 = min(length_uniformity_bonus / 0.10, 1.0)

    # 13: 句位
    is_first = (s_idx == 0)
    is_last = (s_idx == len(sents) - 1 and len(sents) > 1)
    f13 = 1.0 if is_first else (0.6 if is_last else 0.0)

    # 14: embedding同质化
    f14 = min(para_uniformity_bonus / 0.14, 1.0)

    # 15: 段落基线
    level_map = {"deep_red": 1.0, "red": 0.7, "yellow": 0.4, "green": 0.0}
    f15 = level_map.get(para_level, 0.0)

    # 16-18: 困惑度 + 人类相似度（归一化到0-1）
    f16 = min(bigram_bonus / 0.10, 1.0)
    f17 = min(self_bonus / 0.15, 1.0)
    f18 = min(human_bonus / 0.16, 1.0)

    return [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18]


# === 困惑度检测（词级bigram，本地计算，领域无关） ===

_WORD_BIGRAM_MODEL = None  # {word: {next_word: count}}

def _build_word_bigram_model() -> dict | None:
    """用AI参考文本构建词级bigram模型，用于计算似然度"""
    global _WORD_BIGRAM_MODEL
    if _WORD_BIGRAM_MODEL is not None:
        return _WORD_BIGRAM_MODEL

    # 磁盘缓存
    cache_path = Path(__file__).parent.parent / "output" / "calibration" / "bigram_model.pkl"
    if cache_path.exists():
        try:
            import pickle
            with open(cache_path, "rb") as f:
                _WORD_BIGRAM_MODEL = pickle.load(f)
            return _WORD_BIGRAM_MODEL
        except Exception:
            pass

    from collections import defaultdict
    ai_refs = _load_ai_references()
    if not ai_refs:
        return None

    model = defaultdict(lambda: defaultdict(int))
    for ref in ai_refs:
        words = [w.strip() for w in jieba.cut(ref) if len(w.strip()) >= 1]
        for i in range(len(words) - 1):
            model[words[i]][words[i+1]] += 1

    # 转换为概率（加平滑）
    for w1 in model:
        total = sum(model[w1].values()) + len(model[w1])
        for w2 in model[w1]:
            model[w1][w2] = (model[w1][w2] + 1) / total

    _WORD_BIGRAM_MODEL = dict(model)

    # 存盘
    try:
        import pickle
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(_WORD_BIGRAM_MODEL, f)
    except Exception:
        pass

    return _WORD_BIGRAM_MODEL


def _compute_bigram_perplexity(text: str, words: list[str] | None = None) -> float:
    """使用词级bigram模型计算文本的似然度。

    Args:
        text: 原始文本
        words: 可选，预分词结果（复用调用方的jieba分词）

    Returns: 平均转移概率（0-1，越高=越像AI参考文本）
    """
    model = _build_word_bigram_model()
    if model is None:
        return -1.0

    if words is None:
        words = [w.strip() for w in jieba.cut(text) if len(w.strip()) >= 1]
    if len(words) < 4:
        return -1.0

    total_prob = 0.0
    count = 0
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i+1]
        if w1 in model and w2 in model[w1]:
            total_prob += model[w1][w2]
            count += 1

    if count < 3:
        return -1.0

    avg_prob = total_prob / count
    return round(avg_prob, 6)


def _bigram_ppl_to_signal(avg_prob: float) -> float:
    """avg_prob → AI信号。高概率=更像AI参考文本"""
    if avg_prob < 0:
        return -1.0
    if avg_prob > 0.25:     # 极高似然 → 强AI
        return 0.10
    elif avg_prob > 0.12:   # 高似然 → 中AI
        return 0.06
    elif avg_prob > 0.05:   # 中似然 → 弱AI
        return 0.02
    else:
        return 0.0


def _compute_self_perplexity(text: str, words: list[str] | None = None) -> tuple[float, float]:
    """计算文本自困惑度——用前半段词分布预测后半段。

    Args:
        text: 原始文本
        words: 可选，预分词结果

    Returns: (自困惑度, 新词占比) 或 (-1, -1)
    """
    if words is None:
        words = [w.strip() for w in jieba.cut(text) if len(w.strip()) >= 1]
    n = len(words)
    if n < 20:
        return (-1.0, -1.0)

    mid = n // 2
    first_half = words[:mid]
    second_half = words[mid:]

    freq = Counter(first_half)
    vocab_size = len(freq)
    total_first = len(first_half)

    total_surprise = 0.0
    new_words = 0
    for w in second_half:
        if w in freq:
            prob = freq[w] / total_first
            total_surprise += -math.log2(max(prob, 1e-8))
        else:
            # 新词：按Laplace平滑给一个小概率
            prob_smooth = 1.0 / (total_first + vocab_size + 1)
            total_surprise += -math.log2(prob_smooth)
            new_words += 1

    avg_surprise = total_surprise / len(second_half)
    self_ppl = round(2 ** avg_surprise, 2)
    new_word_ratio = round(new_words / len(second_half), 3)
    return (self_ppl, new_word_ratio)


def _self_ppl_to_signal(self_ppl: float, new_word_ratio: float) -> float:
    """自困惑度 → AI信号。低自困惑 + 低新词 = 文本自洽 = AI特征"""
    if self_ppl < 0:
        return -1.0
    signal = 0.0
    if self_ppl < 6:
        signal += 0.10     # 极自洽 → 强AI
    elif self_ppl < 12:
        signal += 0.05     # 较自洽 → 中AI
    if new_word_ratio < 0.25:
        signal += 0.05     # 新词极少 → AI特征
    elif new_word_ratio < 0.40:
        signal += 0.02
    return signal


# === 参考文本加载 ===

_CSL_AI_REFS = None
_CSL_HUMAN_REFS = None

def _load_ai_references() -> list[str]:
    global _CSL_AI_REFS
    if _CSL_AI_REFS is not None:
        return _CSL_AI_REFS
    refs = []
    ref_file = Path(__file__).parent.parent / "output" / "calibration" / "ref_ai_csl.txt"
    if ref_file.exists():
        try:
            with open(ref_file, "r", encoding="utf-8") as f:
                blocks = f.read().split("\n---\n")
                refs = [b.strip() for b in blocks if len(b.strip()) > 50][:500]
        except Exception:
            pass
    if not refs:
        import sys as _sys
        print("[rules_engine] 未找到CSL参考文件，使用内置14篇AI参考文本", file=_sys.stderr)
        refs = [
            "深度学习作为机器学习领域的重要分支，在近年来取得了显著的研究进展。首先，从技术架构的角度来看，卷积神经网络通过局部连接和权重共享机制有效降低了模型参数量。其次，残差网络的提出解决了深层网络中的梯度消失问题。此外，注意力机制的出现进一步推动了自然语言处理领域的发展。",
            "数字经济作为一种新型经济形态，正在深刻改变传统经济运行的基本逻辑。从生产端来看，数据已经成为与土地、劳动力、资本并列的关键生产要素。从消费端来看，平台经济的兴起降低了交易双方的信息不对称程度。",
            "混合式教学模式融合了线上学习的灵活性与线下教学的互动性，成为后疫情时代教育改革的重要方向。从理论层面分析，混合式教学依托建构主义学习理论。从实践层面考察，翻转课堂有效提升了学生的参与度。",
            "肿瘤免疫治疗是近年来癌症治疗领域最具突破性的研究方向之一。首先，免疫检查点抑制剂通过阻断PD-1/PD-L1等免疫抑制信号通路，重新激活T细胞的抗肿瘤活性。其次，CAR-T细胞疗法通过基因工程改造患者自身的T细胞。",
            "民法典的颁布实施标志着我国民事权利保护进入了一个全新的发展阶段。从立法理念来看，民法典坚持以人民为中心的发展思想。从制度创新来看，人格权独立成编是民法典最大的亮点之一。",
            "数字化转型已经成为企业获取竞争优势的关键战略选择。在组织层面，数字化转型要求企业打破传统的科层制结构。在技术层面，大数据分析、云计算和人工智能等新兴技术的应用正在重新定义企业的运营模式。",
            "气候变化对全球生态系统的影响已经引起了国际社会的广泛关注。从观测数据来看，过去一个世纪全球平均气温上升了约1.1摄氏度。从影响层面来看，气候变化已经对农业生产、水资源分布、生物多样性和人类健康产生了深远的影响。",
            "人工智能的发展引发了关于意识本质和人类主体性的深刻哲学思考。从认识论的角度来看，机器学习的模式识别能力挑战了传统的主客二分认知框架。从伦理学的角度来看，算法决策的透明度和可解释性已经成为AI伦理研究的核心议题。",
            "平台经济的兴起正在深刻改变传统的社会结构和劳动关系。从社会分层理论来看，平台经济催生了新的职业群体。从劳动过程理论来看，算法管理正在替代传统的人力资源管理。",
            "网络文学作为一种新兴的文学形态，已经发展成为当代中国文化的重要组成部分。从创作主体来看，网络文学打破了传统文学精英化的创作壁垒。从传播方式来看，数字媒介的互动性和即时性深刻改变了文学的生产和消费模式。",
            "新型复合材料的研发对航空航天领域的技术进步具有重要的推动作用。首先，碳纤维增强复合材料的比强度和比模量远高于传统金属材料。其次，陶瓷基复合材料在高温环境下表现出优异的抗氧化性能。",
            "积极心理学的研究视角从传统的心理疾病治疗转向了人类优势美德的培养。从理论基础来看，积极心理学建立在人本主义心理学的传统之上。从研究方法来看，积极心理学综合运用了实验研究、纵向追踪和跨文化比较等多种研究范式。",
        ]
    _CSL_AI_REFS = refs
    return _CSL_AI_REFS

def _load_human_references() -> list[str]:
    global _CSL_HUMAN_REFS
    if _CSL_HUMAN_REFS is not None:
        return _CSL_HUMAN_REFS
    refs = []
    ref_file = Path(__file__).parent.parent / "output" / "calibration" / "ref_human_csl.txt"
    if ref_file.exists():
        try:
            with open(ref_file, "r", encoding="utf-8") as f:
                blocks = f.read().split("\n---\n")
                refs = [b.strip() for b in blocks if len(b.strip()) > 50][:500]
        except Exception:
            pass
    _CSL_HUMAN_REFS = refs if refs else []
    return _CSL_HUMAN_REFS


# === Char n-gram TF-IDF 参考模型（lyc8503方法的核心） ===

_CHAR_NGRAM_MODEL = None
_AI_REF_VECTORS = None  # 参考文本的TF-IDF矩阵

def _build_char_ngram_model():
    """用已知AI文本构建char n-gram TF-IDF参考模型 + 缓存向量矩阵

    返回 (vectorizer, ref_vectors) 或 None
    """
    global _CHAR_NGRAM_MODEL, _AI_REF_VECTORS
    if _CHAR_NGRAM_MODEL is not None:
        return _CHAR_NGRAM_MODEL

    ai_refs = _load_ai_references()
    if not ai_refs:
        return None

    # 优先从缓存加载
    cache_path = Path(__file__).parent.parent / "output" / "calibration" / "char_tfidf_model.pkl"
    if cache_path.exists():
        try:
            import pickle
            with open(cache_path, "rb") as f:
                saved = pickle.load(f)
            _CHAR_NGRAM_MODEL = saved["vectorizer"]
            _AI_REF_VECTORS = saved["vectors"]
            return _CHAR_NGRAM_MODEL
        except Exception:
            pass

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 4),
            max_features=5000,
        )
        ref_vectors = vectorizer.fit_transform(ai_refs)
        _CHAR_NGRAM_MODEL = vectorizer
        _AI_REF_VECTORS = ref_vectors

        # 存盘缓存
        try:
            import pickle
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump({"vectorizer": vectorizer, "vectors": ref_vectors}, f)
        except Exception:
            pass

        return vectorizer
    except Exception:
        return None


def scan_sentences(paragraphs: list) -> dict[str, Any]:
    """句子级AI检测 — 18信号 + ML模型集成"""
    all_sentences = []
    total_chars = 0

    # ML模型 + 校准参数
    ml_model = None
    ml_params = {"red_th": 0.55, "orange_th": 0.40, "yellow_th": 0.28,
                 "red_w": 1.0, "orange_w": 0.7, "yellow_w": 0.4}
    try:
        import pickle
        model_path = Path(__file__).parent.parent / "output" / "calibration" / "detector_nlpcc.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                ml_model = pickle.load(f)
        params_path = Path(__file__).parent.parent / "output" / "calibration" / "best_params.json"
        if params_path.exists():
            with open(params_path, "r", encoding="utf-8") as f:
                import json as _json
                saved = _json.load(f)
                ml_params = {"red_th": saved["red_threshold"], "orange_th": saved["orange_threshold"],
                            "yellow_th": saved["yellow_threshold"], "red_w": saved["red_weight"],
                            "orange_w": saved["orange_weight"], "yellow_w": saved["yellow_weight"]}
    except Exception:
        pass

    # 加载词表
    high_connectors, med_connectors, low_connectors = set(), set(), set()
    slop_words = set()
    try:
        with open(DATA_DIR / "connector_blacklist.json", "r", encoding="utf-8") as f:
            cdata = json.load(f)
        high_connectors.update(cdata["levels"].get("high", []))
        med_connectors.update(cdata["levels"].get("medium", []))
        low_connectors.update(cdata["levels"].get("low", []))
        with open(DATA_DIR / "slop_patterns_zh.json", "r", encoding="utf-8") as f:
            sdata = json.load(f)
        for cat in ["overused_phrases", "template_structures", "empty_modifiers",
                     "ai_conclusion_starters", "ai_transition_sets"]:
            for phrase in sdata["patterns"].get(cat, []):
                if isinstance(phrase, str) and len(phrase) <= 30:
                    slop_words.add(phrase)
    except Exception:
        pass

    char_tfidf = _build_char_ngram_model()

    in_references = False

    for p_idx, para in enumerate(paragraphs):
        text = para.text if hasattr(para, 'text') else str(para)
        if _REF_STARTERS.match(text.strip()):
            in_references = True

        sents = split_sentences(text)
        para_result = scan_paragraph(text)

        para_template_score = 0.0
        for patterns, weight, label in _AI_TEMPLATES:
            hits = sum(1 for p in patterns if re.search(p, text))
            if hits >= len(patterns) * 0.6:
                para_template_score += weight

        para_parallel_bonus = 0.0
        if len(sents) >= 3:
            starter_counts = {}
            for sent in sents:
                for pat in _PARALLEL_STARTERS:
                    m = re.match(pat, sent.strip())
                    if m:
                        w = m.group(1)
                        starter_counts[w] = starter_counts.get(w, 0) + 1
                        break
            for cnt in starter_counts.values():
                if cnt >= 3:
                    para_parallel_bonus = 0.20
                    break

        # 预分词一次，复用给 bigram + 自困惑度
        para_words = [w.strip() for w in jieba.cut(text) if len(w.strip()) >= 1]

        bigram_ppl = _compute_bigram_perplexity(text, words=para_words)
        bigram_bonus = _bigram_ppl_to_signal(bigram_ppl)
        self_ppl, new_word_ratio = _compute_self_perplexity(text, words=para_words)
        self_bonus = _self_ppl_to_signal(self_ppl, new_word_ratio)
        human_sim = _human_similarity(text)
        human_bonus = _human_sim_to_signal(human_sim)

        sent_stats = [_sentence_stats(s) for s in sents]
        intra_sim = _intra_paragraph_similarity(sents)
        para_uniformity_bonus = 0.0
        if intra_sim > 0.82:
            para_uniformity_bonus = 0.14
        elif intra_sim > 0.72:
            para_uniformity_bonus = 0.08

        length_uniformity_bonus = 0.0
        sent_lengths = [len(s) for s in sents if len(s) >= 10]
        length_std = 0.0
        if len(sent_lengths) >= 3:
            length_std = float(np.std(sent_lengths))
            if length_std < 5:
                length_uniformity_bonus = 0.10
            elif length_std < 8:
                length_uniformity_bonus = 0.06

        # 参考文献: 全部标绿
        if in_references:
            for sent in sents:
                chars = len(sent)
                if chars < 5:
                    continue
                total_chars += chars
                all_sentences.append({
                    "text": sent, "chars": chars, "zone": "green",
                    "score": 0.0, "ai_dim_count": 0,
                    "paragraph_index": p_idx,
                    "_reasons": ["参考文献(保护)"],
                })
            continue

        for s_idx, sent in enumerate(sents):
            chars = len(sent)
            if chars < 5:
                continue
            total_chars += chars

            sent_score = 0.0
            reasons = []
            stats = sent_stats[s_idx]

            # === 领域无关统计特征 ===
            if chars >= 15:
                if stats["lexical_diversity"] < 0.55:
                    sent_score += 0.10
                    reasons.append(f"词汇重复({stats['lexical_diversity']:.0%})")
                elif stats["lexical_diversity"] < 0.68:
                    sent_score += 0.05
                if stats["char_entropy"] < 3.0:
                    sent_score += 0.07
                    reasons.append(f"字熵低({stats['char_entropy']:.1f})")
                elif stats["char_entropy"] < 3.8:
                    sent_score += 0.03
                if stats["word_repetition"] >= 3:
                    sent_score += 0.10
                    reasons.append(f"词重复x{stats['word_repetition']}")
                elif stats["word_repetition"] >= 2:
                    sent_score += 0.05
                if stats["unique_char_ratio"] < 0.50:
                    sent_score += 0.07
                    reasons.append(f"字面单调({stats['unique_char_ratio']:.0%})")
                elif stats["unique_char_ratio"] < 0.62:
                    sent_score += 0.03

            # === 段内同质化 ===
            if para_uniformity_bonus > 0:
                sent_score += para_uniformity_bonus
                if s_idx == 0:
                    reasons.append(f"段内同质化({intra_sim:.0%})")

            # === char n-gram ===
            if char_tfidf is not None:
                try:
                    char_sim = _char_ngram_similarity(sent, char_tfidf)
                    if char_sim > 0.5:
                        sent_score += 0.40
                        reasons.append(f"char指纹{char_sim:.0%}")
                    elif char_sim > 0.3:
                        sent_score += 0.22
                        reasons.append(f"char指纹{char_sim:.0%}")
                except Exception:
                    pass

            # === 连接词 ===
            high_hits = sum(1 for c in high_connectors if c in sent)
            med_hits = sum(1 for c in med_connectors if c in sent)
            low_hits = sum(1 for c in low_connectors if c in sent)
            if high_hits >= 2:
                sent_score += 0.30; reasons.append(f"高连接词x{high_hits}")
            elif high_hits == 1:
                sent_score += 0.18; reasons.append("高连接词x1")
            elif med_hits >= 2:
                sent_score += 0.22; reasons.append(f"中连接词x{med_hits}")
            elif med_hits == 1:
                sent_score += 0.10
            elif low_hits >= 2:
                sent_score += 0.08
            elif low_hits == 1:
                sent_score += 0.04

            # === Slop ===
            slop_hits = sum(1 for s in slop_words if s in sent)
            if slop_hits >= 1:
                sent_score += 0.28
                reasons.append("Slop模式")

            # === 模板 ===
            if para_template_score > 0:
                sent_score += para_template_score * 0.4

            # === 排比 ===
            if para_parallel_bonus > 0:
                for pat in _PARALLEL_STARTERS:
                    if re.match(pat, sent.strip()):
                        sent_score += para_parallel_bonus
                        reasons.append("排比句式")
                        break

            # === 学术套话 ===
            for pattern, weight, label in _ACADEMIC_FORMULAS:
                if re.search(pattern, sent):
                    sent_score += weight
                    reasons.append(label)
                    break

            # === 句长均匀 ===
            if length_uniformity_bonus > 0 and chars >= 10:
                sent_score += length_uniformity_bonus
                if s_idx == 0:
                    reasons.append(f"句长均匀(std={length_std:.1f})")

            # === 困惑度 ===
            ppl_total = bigram_bonus + self_bonus
            if ppl_total > 0:
                sent_score += ppl_total
                if s_idx == 0:
                    parts = []
                    if bigram_bonus > 0:
                        parts.append(f"bigram(prob={bigram_ppl:.2f})")
                    if self_bonus > 0:
                        parts.append(f"自洽(self_ppl={self_ppl:.1f})")
                    reasons.append("低困惑度:" + ",".join(parts))

            # === 人类相似度 ===
            if human_bonus > 0:
                sent_score += human_bonus
                if s_idx == 0:
                    reasons.append(f"不像人类(sim={human_sim:.2f})")

            # === ML模型：主检测器 ===
            ml_prob = -1.0
            if ml_model is not None:
                features = _extract_sentence_features(
                    sent, s_idx, sents, char_tfidf,
                    high_connectors, med_connectors, low_connectors,
                    slop_words, para_template_score, para_parallel_bonus,
                    para_uniformity_bonus, length_uniformity_bonus,
                    sent_stats[s_idx], para_result["level"], bigram_bonus, self_bonus, human_bonus
                )
                ml_prob = float(ml_model.predict_proba([features])[0, 1])

            if ml_prob >= 0:
                # ML做主：用校准阈值判定
                if ml_prob >= ml_params["red_th"]:
                    zone = "red"
                elif ml_prob >= ml_params["orange_th"]:
                    zone = "orange"
                elif ml_prob >= ml_params["yellow_th"]:
                    zone = "yellow"
                else:
                    zone = "green"
                sent_score = float(ml_prob)
                if s_idx == 0:
                    reasons.append(f"ML({ml_prob:.0%})")
            else:
                # ML不可用：回退手工规则
                # === 辅助信号 ===
                if 15 <= chars <= 25:
                    sent_score += 0.03
                is_first = (s_idx == 0)
                is_last = (s_idx == len(sents) - 1 and len(sents) > 1)
                if is_first:
                    sent_score += 0.04
                    reasons.append("段首句")
                if is_last:
                    sent_score += 0.03
                para_level = para_result["level"]
                if para_level in ("red", "deep_red"):
                    sent_score += 0.18
                elif para_level == "yellow":
                    sent_score += 0.10
                sent_score += 0.03

                # === 手工判定 ===
                if sent_score >= 0.26:
                    zone = "red"
                elif sent_score >= 0.20:
                    zone = "orange"
                elif sent_score >= 0.10:
                    zone = "yellow"
                else:
                    zone = "green"

            all_sentences.append({
                "text": sent, "chars": chars, "zone": zone,
                "score": round(min(sent_score, 1.0), 4),
                "ai_dim_count": len(reasons),
                "paragraph_index": p_idx,
                "_reasons": reasons,
            })

    # === 上下文传染 ===
    for i in range(1, len(all_sentences) - 1):
        prev_z = all_sentences[i-1]["zone"]
        next_z = all_sentences[i+1]["zone"]
        cur = all_sentences[i]

        # 邻居最高等级
        neighbor_max = "green"
        for z in (prev_z, next_z):
            if z == "red" or (neighbor_max != "red" and z == "orange") or (neighbor_max == "green" and z == "yellow"):
                neighbor_max = z

        if cur["zone"] == "green":
            if neighbor_max == "red":
                cur["zone"] = "orange"; cur["score"] = max(cur["score"], 0.22)
            elif neighbor_max == "orange":
                cur["zone"] = "yellow"; cur["score"] = max(cur["score"], 0.10)
            elif neighbor_max == "yellow":
                cur["zone"] = "yellow"; cur["score"] = max(cur["score"], 0.08)
        elif cur["zone"] == "yellow" and neighbor_max == "red":
            cur["zone"] = "orange"; cur["score"] = max(cur["score"], 0.20)

    # === AI率计算（对齐商业平台：红+橙=AI嫌疑，黄+绿=安全）===
    ai_chars = 0.0
    for s in all_sentences:
        if s["zone"] in ("red", "orange"):
            ai_chars += s["chars"]

    ai_rate = round(ai_chars / total_chars, 4) if total_chars > 0 else 0

    return {
        "total_chars": total_chars, "ai_chars": round(ai_chars, 1), "ai_rate": ai_rate,
        "total_sentences": len(all_sentences),
        "ai_sentences": sum(1 for s in all_sentences if s["zone"] == "red"),
        "suspect_sentences": sum(1 for s in all_sentences if s["zone"] in ("red", "orange", "yellow")),
        "red_count": sum(1 for s in all_sentences if s["zone"] == "red"),
        "orange_count": sum(1 for s in all_sentences if s["zone"] == "orange"),
        "yellow_count": sum(1 for s in all_sentences if s["zone"] == "yellow"),
        "sentences": all_sentences,
    }


def _char_ngram_similarity(text: str, vectorizer) -> float:
    """计算文本与AI参考文本的char n-gram余弦相似度

    lyc8503方法：TF-IDF char n-gram + cosine similarity。
    ⚠️ 待校准阈值: scan_sentences中 >0.6=强证据, >0.4=中等
    """
    if vectorizer is None or _AI_REF_VECTORS is None or len(text) < 10:
        return 0.0
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        input_vec = vectorizer.transform([text])
        sims = cosine_similarity(input_vec, _AI_REF_VECTORS)
        return float(sims.max()) if sims.size > 0 else 0.0
    except Exception:
        return 0.0
