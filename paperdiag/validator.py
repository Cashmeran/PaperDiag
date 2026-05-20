"""后处理质检模块：10项artifact检查 + 语义保真度验证"""

import re
from typing import Any


# ============================================================
#  10项 artifact 检查
# ============================================================

def check_duplicate_words(text: str) -> list[str]:
    """检查重复词（如"的的"、"了了"）"""
    issues = []
    pattern = re.compile(r'([一-鿿])\1')
    matches = pattern.findall(text)
    for m in set(matches):
        issues.append(f"重复字符: {m}{m}")
    return issues


def check_broken_contractions(text: str) -> list[str]:
    """检查破碎的缩写/结构"""
    issues = []
    patterns = [
        (r'\w\s+n\'t', '英文缩写断裂'),
        (r'[一-鿿]\s+的\s+[一-鿿]', '可能的"的"字碎片'),
    ]
    for pattern, desc in patterns:
        if re.search(pattern, text):
            issues.append(desc)
    return issues


def check_dangling_connectors(text: str) -> list[str]:
    """检查悬空连接词（段尾以"因此、"等结尾）"""
    issues = []
    dangling = ["因此", "然而", "所以", "而且", "但是", "此外", "进而", "从而", "并且"]
    for conn in dangling:
        if text.rstrip().endswith(conn + "。"):
            pass  # 正常
        elif text.rstrip().endswith(conn + ",") or text.rstrip().endswith(conn + "，"):
            issues.append(f"悬空连接词（逗号后无内容）: ...{conn},")
        elif text.rstrip().endswith(conn) and not text.rstrip().endswith("。"):
            issues.append(f"悬空连接词（段尾无标点）: ...{conn}")
    return issues


def check_double_conjunctions(text: str) -> list[str]:
    """检查双连接词连用（"而且而且"、"但是但是"）"""
    issues = []
    conjunctions = ["而且", "但是", "所以", "因此", "然而", "此外", "并且", "进而"]
    for conj in conjunctions:
        if conj + conj in text:
            issues.append(f"双连接词: {conj}{conj}")
    return issues


def check_unclosed_parentheses(text: str) -> list[str]:
    """检查未闭合的括号"""
    issues = []
    pairs = [
        ("（", "）"),
        ("《", "》"),
        (""", """),
        (""", """),
        ("[", "]"),
    ]
    for op, cl in pairs:
        open_count = text.count(op)
        close_count = text.count(cl)
        if open_count != close_count:
            issues.append(f"括号不匹配: {op}({open_count}个) vs {cl}({close_count}个)")
    return issues


def check_empty_sentences(text: str) -> list[str]:
    """检查空句或过短句"""
    issues = []
    sentences = re.split(r'[。！？；]', text)
    for s in sentences:
        s = s.strip()
        if len(s) == 1 and s not in "好对错是否行":
            issues.append(f"过短句子: '{s}'")
        if len(s) == 0:
            continue
    return issues


def check_term_destruction(original: str, rewritten: str, whitelist: set) -> list[str]:
    """检查术语是否被破坏"""
    issues = []
    for term in whitelist:
        if term in original and term not in rewritten:
            issues.append(f"术语丢失: {term}")
    return issues


def check_citation_changes(original: str, rewritten: str) -> list[str]:
    """检查引文编号是否被改动"""
    issues = []
    orig_citations = set(re.findall(r'\[[\d,\s\-;]+\]', original))
    new_citations = set(re.findall(r'\[[\d,\s\-;]+\]', rewritten))
    missing = orig_citations - new_citations
    if missing:
        issues.append(f"引文编号丢失: {missing}")
    return issues


def check_number_changes(original: str, rewritten: str) -> list[str]:
    """检查数据/数字是否被改动"""
    issues = []
    orig_numbers = set(re.findall(r'\d+\.?\d*', original))
    new_numbers = set(re.findall(r'\d+\.?\d*', rewritten))
    missing = orig_numbers - new_numbers
    if missing:
        issues.append(f"数字丢失: {missing}")
    return issues


def check_overcorrection(original: str, rewritten: str) -> list[str]:
    """检查是否过度改写（形成新的均匀模式）"""
    issues = []

    # 检测所有"因此"是否都被改成了"所以"（新均匀模式）
    orig_因此 = original.count("因此")
    orig_所以 = original.count("所以")
    new_因此 = rewritten.count("因此")
    new_所以 = rewritten.count("所以")

    if orig_因此 > 2 and new_因此 == 0 and new_所以 > orig_所以 + 2:
        issues.append("过度改写: 所有'因此'都被替换（可能形成新均匀模式）")

    # 检测是否删除了所有连接词（矫枉过正）
    connectors = ["因此", "然而", "此外", "所以", "但是", "并且", "而且"]
    orig_conn = sum(1 for c in connectors if c in original)
    new_conn = sum(1 for c in connectors if c in rewritten)
    if orig_conn > 3 and new_conn == 0:
        issues.append("过度改写: 所有连接词被删除（可能破坏逻辑连贯性）")

    return issues


# ============================================================
#  语义保真度检查
# ============================================================

def check_semantic_fidelity(original: str, rewritten: str,
                            threshold: float = 0.85) -> dict[str, Any]:
    """用 embedding 余弦相似度做语义保真度检查（替代3-gram Jaccard）

    优先使用已加载的 embedding 模型，失败时回退到字符级相似度
    """
    try:
        from .embedding_layer import get_model
        model = get_model()
        if model is not None:
            emb = model.encode([original, rewritten], normalize_embeddings=True)
            similarity = float(emb[0] @ emb[1])
            # 浮点修正：归一化后余弦相似度 ∈ [-1, 1]
            similarity = max(-1.0, min(1.0, similarity))
            return {
                "similarity": round(similarity, 4),
                "method": "embedding_cosine",
                "pass": similarity >= threshold,
                "threshold": threshold,
            }
    except Exception:
        pass

    # 回退：字符级3-gram Jaccard
    def char_ngrams(s, n=3):
        s = s.replace(" ", "").replace("\n", "")
        return set(s[i:i + n] for i in range(len(s) - n + 1))

    orig_ngrams = char_ngrams(original)
    new_ngrams = char_ngrams(rewritten)

    if not orig_ngrams or not new_ngrams:
        return {"similarity": 0.0, "method": "char_jaccard", "pass": False}

    intersection = len(orig_ngrams & new_ngrams)
    union = len(orig_ngrams | new_ngrams)
    similarity = intersection / union if union > 0 else 0.0

    return {
        "similarity": round(similarity, 4),
        "method": "char_jaccard",
        "pass": similarity >= threshold,
        "threshold": threshold,
    }


def validate_rewrite(original: str, rewritten: str,
                     term_whitelist: set | None = None) -> dict[str, Any]:
    """对单段改写结果执行全面质检

    Returns:
        {
            "pass": True/False,
            "issues": [...],
            "semantic_fidelity": {...},
            "checks_performed": [...]
        }
    """
    all_issues = []
    checks = []

    # 1. 重复词
    issues = check_duplicate_words(rewritten)
    if issues:
        all_issues.extend(issues)
    checks.append("duplicate_words")

    # 2. 破碎缩写
    issues = check_broken_contractions(rewritten)
    if issues:
        all_issues.extend(issues)
    checks.append("broken_contractions")

    # 3. 悬空连接词
    issues = check_dangling_connectors(rewritten)
    if issues:
        all_issues.extend(issues)
    checks.append("dangling_connectors")

    # 4. 双连接词
    issues = check_double_conjunctions(rewritten)
    if issues:
        all_issues.extend(issues)
    checks.append("double_conjunctions")

    # 5. 未闭合括号
    issues = check_unclosed_parentheses(rewritten)
    if issues:
        all_issues.extend(issues)
    checks.append("unclosed_parentheses")

    # 6. 术语破坏
    if term_whitelist:
        issues = check_term_destruction(original, rewritten, term_whitelist)
        if issues:
            all_issues.extend(issues)
        checks.append("term_destruction")

    # 7. 引文变更
    issues = check_citation_changes(original, rewritten)
    if issues:
        all_issues.extend(issues)
    checks.append("citation_changes")

    # 8. 数字变更
    issues = check_number_changes(original, rewritten)
    if issues:
        all_issues.extend(issues)
    checks.append("number_changes")

    # 9. 空段落
    issues = check_empty_sentences(rewritten)
    if issues:
        all_issues.extend(issues)
    checks.append("empty_sentences")

    # 10. 过度改写
    issues = check_overcorrection(original, rewritten)
    if issues:
        all_issues.extend(issues)
    checks.append("overcorrection")

    # 语义保真度
    fidelity = check_semantic_fidelity(original, rewritten)
    checks.append("semantic_fidelity")

    # 综合判定：区分严重问题(critical)和警告(warning)
    critical_keywords = ["术语丢失", "引文编号丢失", "数字丢失", "括号不匹配"]
    critical = [i for i in all_issues if any(kw in i for kw in critical_keywords)]
    warnings = [i for i in all_issues if i not in critical]
    # 严重问题=0 且 语义保真度通过 → 通过
    pass_check = len(critical) == 0 and fidelity["pass"]

    return {
        "pass": pass_check,
        "issues": all_issues,
        "critical": critical,
        "warnings": warnings,
        "critical_count": len(critical),
        "warning_count": len(warnings),
        "semantic_fidelity": fidelity,
        "checks_performed": checks,
    }
