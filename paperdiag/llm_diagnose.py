"""LLM定性诊断：通过Ollama调用Qwen3.5-4B做文本AI特征判断"""

import json
import re
from typing import Any, Optional

from .model_manager import chat, ensure_ollama_ready

DIAGNOSIS_SYSTEM = "你是学术写作风格分析专家。分析段落是否存在AI生成文本的特征。只输出JSON。"

DIAGNOSIS_PROMPT = """分析以下段落的AI生成文本特征：

段落：{paragraph_text}

统计数据：{stats_text}

输出JSON（不要其他内容）：
{{"naturalness":"low/medium/high","template_patterns":[],"mechanical_feel":true/false,"primary_issue":"最突出的AI特征","ai_fingerprint_type":"connector_heavy/sentence_uniform/template_structured/terminology_rigid/mixed/natural","rewrite_priority":0-10,"specific_advice":"1-2句改写建议"}}"""


def _build_stats_text(rule_dims: Optional[dict] = None,
                      sem_dims: Optional[dict] = None) -> str:
    lines = []
    if rule_dims:
        lines.append("规则层：")
        for name, d in rule_dims.items():
            if isinstance(d, dict) and d.get("zone") in ("ai", "gray"):
                lines.append(f"  {d.get('description', name)}: {d.get('value', '?')} [{d.get('zone')}]")
    if sem_dims:
        lines.append("语义层：")
        for name, d in sem_dims.items():
            if isinstance(d, dict) and d.get("zone") in ("ai", "gray"):
                lines.append(f"  {d.get('description', name)}: {d.get('value', '?')} [{d.get('zone')}]")
    return "\n".join(lines) if lines else "无统计数据"


def _parse_json_response(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {
        "naturalness": "unknown", "template_patterns": [],
        "mechanical_feel": True, "primary_issue": "解析失败",
        "ai_fingerprint_type": "unknown", "rewrite_priority": 5,
        "specific_advice": "诊断解析失败", "_raw": text,
    }


def diagnose_paragraph(paragraph_text: str,
                       rule_dims: Optional[dict] = None,
                       sem_dims: Optional[dict] = None,
                       temperature: float = 0.0) -> dict[str, Any]:
    """对单个段落执行LLM定性诊断

    Args:
        paragraph_text: 段落原文
        rule_dims: 规则引擎诊断维度
        sem_dims: Embedding层诊断维度
        temperature: LLM温度 (0.0=最确定)

    Returns:
        {naturalness, template_patterns, mechanical_feel, primary_issue,
         ai_fingerprint_type, rewrite_priority, specific_advice}
    """
    stats_text = _build_stats_text(rule_dims, sem_dims)
    text = paragraph_text[:1200]

    prompt = DIAGNOSIS_PROMPT.format(
        paragraph_text=text,
        stats_text=stats_text,
    )

    try:
        result = chat(
            messages=[
                {"role": "system", "content": DIAGNOSIS_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=200,
        )
        return _parse_json_response(result["content"].strip())
    except Exception as e:
        return {
            "naturalness": "unknown", "template_patterns": [],
            "mechanical_feel": True,
            "primary_issue": f"诊断失败: {e}",
            "ai_fingerprint_type": "unknown", "rewrite_priority": 5,
            "specific_advice": "LLM不可用，参考规则和语义诊断",
            "_error": str(e),
        }


def diagnose_document(paragraphs: list[str],
                      fused_diagnoses: list[dict[str, Any]],
                      only_high_risk: bool = True,
                      max_workers: int = 4) -> list[Optional[dict]]:
    """对文档中需诊断的段落执行LLM分析（并行）

    Args:
        paragraphs: 段落文本列表
        fused_diagnoses: 融合诊断结果
        only_high_risk: 仅对黄色及以上段落调用LLM
        max_workers: 并行线程数（默认4，充分利用CPU多核）
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ensure_ollama_ready()

    # 收集需要诊断的段落
    tasks = []
    for i, (para, fused) in enumerate(zip(paragraphs, fused_diagnoses)):
        level = fused.get("fused_level", fused.get("rule_level", "green"))
        if only_high_risk and level == "green":
            continue
        tasks.append((i, para, fused.get("rule_dimensions", {}),
                      fused.get("semantic_dimensions", {})))

    if not tasks:
        return [None] * len(paragraphs)

    # 并行执行
    results_map = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as executor:
        futures = {
            executor.submit(
                diagnose_paragraph, para, rule_dims=rd, sem_dims=sd
            ): idx
            for idx, para, rd, sd in tasks
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                r = future.result()
                r["paragraph_index"] = idx
                results_map[idx] = r
            except Exception as e:
                results_map[idx] = {
                    "naturalness": "unknown", "template_patterns": [],
                    "mechanical_feel": True,
                    "primary_issue": f"并行诊断失败: {e}",
                    "ai_fingerprint_type": "unknown", "rewrite_priority": 5,
                    "specific_advice": "", "_error": str(e),
                    "paragraph_index": idx,
                }

    # 组装结果（保持原顺序，绿色段落为None）
    return [results_map.get(i) for i in range(len(paragraphs))]
