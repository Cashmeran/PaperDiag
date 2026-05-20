"""融合诊断层：规则12维 + Embedding4维 → 统一16维诊断报告

融合判定矩阵：
  - 绿色 (规则+语义均正常) → 无需改写
  - 黄色 (1-3个规则维度异常 或 语义轻度异常) → 触发LLM诊断
  - 红色 (4+个规则维度异常 或 语义严重异常) → LLM诊断+改写
  - 深红 (全部维度异常) → 跳过诊断，直接改写
"""

from typing import Any


def fuse_diagnosis(rule_results: list[dict[str, Any]],
                   semantic_results: dict[str, Any] | None = None
                   ) -> list[dict[str, Any]]:
    """融合规则引擎和 embedding 的诊断结果

    Args:
        rule_results: rules_engine.scan_document() 的输出
        semantic_results: embedding_layer.scan_semantic() 的输出

    Returns:
        融合后的逐段诊断列表
    """
    fused = []

    has_semantic = (
        semantic_results is not None
        and "error" not in (semantic_results or {})
        and semantic_results.get("per_paragraph")
    )

    for i, rule in enumerate(rule_results):
        paragraph_index = rule.get("paragraph_index", i)

        # 基础数据来自规则引擎
        entry = {
            "paragraph_index": paragraph_index,
            "rule_dimensions": rule.get("dimensions", {}),
            "rule_score": rule.get("comprehensive_score", 0),
            "rule_level": rule.get("level", "green"),
            "rule_ai_count": rule.get("ai_count", 0),
            "rule_gray_count": rule.get("gray_count", 0),
            "rule_ranked_gaps": rule.get("ranked_gaps", []),
        }

        # 合并 embedding 维度
        if has_semantic and i < len(semantic_results["per_paragraph"]):
            sem = semantic_results["per_paragraph"][i]
            entry["semantic_dimensions"] = sem

            # 统计 embedding 维度的 AI 数量
            sem_ai = sum(
                1 for k, v in sem.items()
                if isinstance(v, dict) and v.get("zone") == "ai"
            )
            entry["semantic_ai_count"] = sem_ai

            # 合并所有维度（16维）
            all_dims = {}
            all_dims.update(rule.get("dimensions", {}))
            for k, v in sem.items():
                all_dims[f"sem_{k}"] = v
            entry["all_dimensions"] = all_dims

            # 综合评分
            total_ai = rule.get("ai_count", 0) + sem_ai
            total_dims = len(all_dims)
            entry["total_dimensions"] = total_dims
            entry["combined_ai_count"] = total_ai
            entry["combined_score"] = round(total_ai / total_dims, 4) if total_dims > 0 else 0

            # 融合等级
            entry["fused_level"] = _fused_level(
                rule.get("level", "green"), rule.get("ai_count", 0), sem_ai
            )
        else:
            entry["semantic_dimensions"] = {}
            entry["all_dimensions"] = rule.get("dimensions", {})
            entry["total_dimensions"] = 12
            entry["combined_ai_count"] = rule.get("ai_count", 0)
            entry["combined_score"] = rule.get("comprehensive_score", 0)
            entry["fused_level"] = rule.get("level", "green")

        fused.append(entry)

    return fused


def _fused_level(rule_level: str, rule_ai: int, sem_ai: int) -> str:
    """融合等级判定"""
    combined = rule_ai + sem_ai

    if combined >= 10:
        return "deep_red"
    elif combined >= 6:
        return "red"
    elif combined >= 3:
        return "yellow"
    elif rule_level == "red" or rule_level == "deep_red":
        return "red"
    elif rule_level == "yellow":
        return "yellow"
    return "green"


def generate_gap_report(fused_result: dict[str, Any]) -> list[dict[str, Any]]:
    """生成排名差距报告——所有维度按偏离程度排序

    Returns:
        [{"dimension": "slop_density", "gap": 10.11, "zone": "ai", "layer": "rule"}, ...]
    """
    gaps = []

    # 规则维度
    for name, d in fused_result.get("rule_dimensions", {}).items():
        if d.get("zone") == "ai":
            gaps.append({
                "dimension": name,
                "description": d.get("description", name),
                "gap": d.get("gap", 0),
                "value": d.get("value", 0),
                "zone": d.get("zone"),
                "layer": "rule",
            })

    # Embedding 维度
    for name, d in fused_result.get("semantic_dimensions", {}).items():
        if isinstance(d, dict) and d.get("zone") == "ai":
            gaps.append({
                "dimension": name,
                "description": d.get("description", name),
                "gap": d.get("gap", 0),
                "value": d.get("value", 0),
                "zone": d.get("zone"),
                "layer": "embedding",
            })

    gaps.sort(key=lambda x: x["gap"], reverse=True)
    return gaps


def generate_rewrite_instructions(fused_result: dict[str, Any]) -> list[str]:
    """根据排名差距生成原子化改写指令"""
    gaps = generate_gap_report(fused_result)
    instructions = []
    instruction_templates = {
        "connector_density": "删除/替换3-5个AI高频连接词(如: 因此/然而/此外)",
        "sentence_length_std": "拆分1-2个长句(>35字), 或合并2个相邻短句(<10字), 让句长分布更不均匀",
        "info_density": "在1-2处注入口语化短句或主观评价, 打破信息密度的均匀性",
        "term_density": "将1-2个过于规整的学术术语替换为口语化表达",
        "slop_density": "删除或替换以下AI标志性短语(如: 综上所述/值得注意的是)",
        "paragraph_similarity": "打破相邻段落的语义结构, 在第N段插入一句离题评论或过渡性的自我质疑",
        "hapax_ratio": "增加用词的多样性, 避免重复使用相同的描述词",
        "zipf_deviation": "适当使用低频词或口语化表达, 让词频分布更接近自然语言",
        "bigram_repetition": "避免重复使用相同的短语搭配",
        "trigram_repetition": "打破固定的三词组合模式",
        "punctuation_entropy": "增加标点多样性, 将部分逗号改为分号, 或加入破折号、省略号",
        "overall_entropy": "引入更多样化的表达方式, 打破字词使用的规律性",
        "sem_paragraph_similarity": "让相邻段落的主题有所跳跃, 不要每段都完美承接上段",
        "sem_argument_linearity": "在论证中插入一个迂回或自我质疑的句子, 打破直线的A->B->C逻辑",
        "sem_transition_naturalness": "让段间过渡更自然, 不要每段都用特定连接词开头",
    }

    for g in gaps[:5]:  # 前5个最大差距
        dim = g["dimension"]
        if dim in instruction_templates:
            instructions.append(instruction_templates[dim])
        else:
            # 去掉 sem_ 前缀再查
            clean = dim.replace("sem_", "")
            if clean in instruction_templates:
                instructions.append(instruction_templates[clean])
            else:
                instructions.append(f"减少 {g['description']} 的AI特征 (当前: {g['value']}, 差距: {g['gap']:.2f})")

    return instructions
