"""LLM改写引擎：正向注入人类特征 + 约束编辑"""

from typing import Any, Optional
from .model_manager import chat, ensure_ollama_ready

INJECT_SYSTEM = """你是学术文本"人性化"助手。你的任务不是改写整段文字，而是在指定位置注入人类写作特有的元素。

## 你可以做的操作
1. 在指定句子后插入一句个人感受（"说实话""笔者注意到""有意思的是"）
2. 将指定句子改为反问句式
3. 在指定句子后追加一条不确定性说明（"不过这个结论还需要更多数据验证"）
4. 用更口语化的方式重述一个术语（括号内补充说明）
5. 打破对称/排比结构——改其中一句的句式
6. 在两个长句之间插入一个短句打断节奏

## 铁律
- 不动数据、引文编号、专业术语、专有名词
- 每次只做要求的操作，不自由发挥
- 不要删除或替换原文内容，只做添加和微调
- 输出改写后的完整段落"""

INJECT_USER = """## 诊断问题
{diagnosis_text}

## 原文
{original_text}

## 需要执行的操作
{operations}

只输出改写后的完整段落："""


# --- 正向注入操作模板 ---
INJECTION_TEMPLATES = {
    "add_personal_view": "在句{idx}后插入一句带个人视角的补充，如'笔者注意到''有意思的是''说实话'等开头",
    "add_rhetorical_question": "将句{idx}改为反问句式",
    "add_uncertainty": "在句{idx}后追加不确定性说明，如'不过这个结论还有待更多数据验证'",
    "break_parallel": "句{idx}和相邻句形成排比/对称结构，请修改其中一句的句式来打破它",
    "inject_short_sentence": "在句{idx}和句{idx2}之间插入一个短句（3-8字）打断节奏",
    "colloquialize_term": "在句{idx}中用括号加一个口语化解释",
}

# --- 原子化操作触发规则 ---
def _diagnosis_to_injections(sentence_diagnoses: list[dict]) -> list[str]:
    """从诊断结果生成正向注入操作列表"""
    ops = []
    for i, sd in enumerate(sentence_diagnoses):
        reasons = sd.get("_reasons", [])
        r_text = " ".join(reasons)

        if "排比" in r_text:
            ops.append(INJECTION_TEMPLATES["break_parallel"].format(idx=i+1))
        if "段首句" in r_text and i == 0 and len(sentence_diagnoses) > 3:
            ops.append(INJECTION_TEMPLATES["add_personal_view"].format(idx=1))
        if "Slop" in r_text or "学术" in r_text:
            ops.append(INJECTION_TEMPLATES["add_rhetorical_question"].format(idx=i+1))

    # 如果段落较长且AI特征明显，加入节奏打断
    if len(sentence_diagnoses) >= 4:
        mid = len(sentence_diagnoses) // 2
        any_ai = any(sd.get("zone", "green") != "green" for sd in sentence_diagnoses)
        if any_ai:
            ops.append(INJECTION_TEMPLATES["inject_short_sentence"].format(idx=mid, idx2=mid+1))

    # 尾部加不确定性（如果没触发其他操作）
    if not ops and any(sd.get("zone", "green") != "green" for sd in sentence_diagnoses):
        ops.append(INJECTION_TEMPLATES["add_uncertainty"].format(idx=len(sentence_diagnoses)))

    return ops[:3]  # 最多3个操作


def rewrite_paragraph_inject(paragraph_text: str,
                              sentence_diagnoses: list[dict] | None = None,
                              temperature: float = 0.3,
                              max_tokens: int = 800) -> dict[str, Any]:
    """正向注入改写：在诊断病灶处注入人类特征"""
    ensure_ollama_ready()

    ops = _diagnosis_to_injections(sentence_diagnoses or [])
    diagnosis_text = ", ".join(
        set(r for sd in (sentence_diagnoses or []) for r in sd.get("_reasons", [])[:2])
    ) or "无明显病灶"
    operations_text = "\n".join(f"  {i+1}. {op}" for i, op in enumerate(ops)) if ops else "轻度口语化调整"

    prompt = INJECT_USER.format(
        diagnosis_text=diagnosis_text,
        original_text=paragraph_text,
        operations=operations_text,
    )

    try:
        result = chat(
            messages=[
                {"role": "system", "content": INJECT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        rewritten = result["content"].strip()
        if rewritten.startswith("```"):
            rewritten = rewritten.split("\n", 1)[-1].strip().rstrip("```").strip()
        return {"text": rewritten, "temperature": temperature, "operations": ops}
    except Exception as e:
        return {"text": paragraph_text, "temperature": temperature, "error": str(e)}


# 保留旧版兼容
def rewrite_paragraph_llm(paragraph_text, fused_result=None, intensity=0.5,
                          temperature=0.3, max_tokens=1024):
    return rewrite_paragraph_inject(paragraph_text, None, temperature, max_tokens)


def rewrite_paragraph_multi_temp(paragraph_text, fused_result=None,
                                 sentence_diagnoses=None, intensity=0.5):
    return rewrite_paragraph_inject(paragraph_text, sentence_diagnoses, temperature=0.5)

from typing import Any, Optional
from .model_manager import chat, ensure_ollama_ready

REWRITE_SYSTEM = """你是学术论文语言润色专家。你的任务是对指定句子执行精准的外科手术——只修改导致AI检测的病灶，不动其他内容。

## 铁律
1. 只改指定句子，不动其他
2. 保留所有数据、引文编号([1][2,3]等)、专业术语、书名号内容、人名头衔
3. 每次只改被诊断的病灶，不自由发挥
4. 输出格式: 原文 → 改写后（附一句改动说明）

## 禁止使用的AI标志词（如果用了就白改了）
过渡: 此外/与此同时/综上所述/值得注意的是/基于此/首先其次最后/一方面另一方面/不仅而且
评价: 至关重要/不可忽视/具有重要意义/起到关键作用/具有良好的应用前景
动词: 促进/构建/体现/代表/作为/展现/呈现/推动/引领
分析: 从而/进而/由此/基于此

## 人性化指南
- 长短句交错，打破15-25字均匀句长
- 用具体有温度的词替换生硬表达
- 不用"首先其次总之"，用隐性过渡
- 适当注入不确定性("推测""可能""或许")
- 可加第一人称视角("我们注意到""有意思的是")"""

REWRITE_USER = """## 诊断病灶
{diagnosis_text}

## 需要修改的句子
{target_sentences}

## 修改要求
{instructions}

只输出改写后文本（不含诊断原文）："""


def _build_instructions(fused_result: Optional[dict] = None,
                        sentence_diagnoses: list[dict] | None = None) -> str:
    """从句子级诊断生成原子化改写指令"""
    if sentence_diagnoses:
        lines = []
        for sd in sentence_diagnoses:
            reasons = sd.get("_reasons", [])
            zone = sd.get("zone", "green")
            text = sd.get("text", "")
            if zone == "green":
                continue
            ops = []
            for r in reasons:
                if "连接词" in r:
                    ops.append("删连接词")
                elif "Slop" in r:
                    ops.append("去模板化表达")
                elif "排比" in r:
                    ops.append("打破排比结构")
                elif "学术" in r:
                    ops.append("换掉公式化句式")
                elif "困惑度" in r:
                    ops.append("增加用词不可预测性")
                elif "同质化" in r:
                    ops.append("多样化句式")
                elif "句长均匀" in r:
                    ops.append("打破句长均匀分布")
            if ops:
                lines.append(f"  [{zone}] \"{text[:60]}...\" → {', '.join(ops)}")
        if lines:
            return "逐句精准修改：\n" + "\n".join(lines)

    # fallback
    if fused_result is None:
        return "轻度同义词替换和句式微调，让表达更自然。"
    from .fusion import generate_rewrite_instructions
    insts = generate_rewrite_instructions(fused_result)
    return "\n".join(f"  {i}. {inst}" for i, inst in enumerate(insts, 1)) if insts else "轻度同义词替换和句式微调。"


def _get_targets(fused_result: Optional[dict]) -> tuple[str, str]:
    if fused_result is None:
        return "14", "6"
    dims = fused_result.get("rule_dimensions", {})
    sls = dims.get("sentence_length_std", {})
    conn = dims.get("connector_density", {})
    return (
        "14" if sls.get("zone") == "ai" else "保持",
        "6" if conn.get("zone") == "ai" else "保持",
    )


INTENSITY_LABELS = {
    0.0: "最保守（仅调整标点、删除多余连接词）",
    0.3: "轻度（调整句长分布，部分术语口语化）",
    0.5: "中度（显著改变句式，注入短句，改变论证节奏）",
    0.7: "较强（大幅重构段落结构）",
    1.0: "最激进（完全重写，仅保留核心信息）",
}


def rewrite_paragraph_llm(paragraph_text: str,
                          fused_result: Optional[dict] = None,
                          sentence_diagnoses: list[dict] | None = None,
                          intensity: float = 0.5,
                          temperature: float = 0.3,
                          max_tokens: int = 1024) -> dict[str, Any]:
    """用LLM对单个段落执行约束改写（原子化指令）"""
    instructions = _build_instructions(fused_result, sentence_diagnoses)

    # 诊断文本：汇总所有句子级病灶
    diag_lines = []
    if sentence_diagnoses:
        for sd in sentence_diagnoses:
            zone = sd.get("zone", "green")
            reasons = sd.get("_reasons", [])
            if zone != "green" and reasons:
                diag_lines.append(f"  [{zone}] {', '.join(reasons[:3])}")
    diagnosis_text = "\n".join(diag_lines) if diag_lines else "无特殊病灶"

    # 目标句子：只列出需要改的
    target_parts = []
    if sentence_diagnoses:
        for i, sd in enumerate(sentence_diagnoses):
            if sd.get("zone", "green") != "green":
                target_parts.append(f"  句{i+1}: {sd['text']}")
    target_sentences = "\n".join(target_parts) if target_parts else paragraph_text

    prompt = REWRITE_USER.format(
        diagnosis_text=diagnosis_text,
        target_sentences=target_sentences,
        instructions=instructions,
    )

    try:
        result = chat(
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        rewritten = result["content"].strip()
        if rewritten.startswith("```"):
            rewritten = rewritten.split("\n", 1)[-1].strip().rstrip("```").strip()

        return {
            "text": rewritten,
            "temperature": temperature,
            "instructions_given": instructions[:200],
        }
    except Exception as e:
        return {
            "text": paragraph_text,
            "temperature": temperature,
            "error": str(e),
        }


def rewrite_paragraph_multi_temp(paragraph_text: str,
                                 fused_result: Optional[dict] = None,
                                 sentence_diagnoses: list[dict] | None = None,
                                 intensity: float = 0.5) -> dict[str, Any]:
    """双温度策略改写：两次调用，选句长方差最大版本"""
    ensure_ollama_ready()

    rounds = [
        (0.3, "conservative"),
        (0.8, "diverse"),
    ]
    candidates = []
    for temp, label in rounds:
        r = rewrite_paragraph_llm(
            paragraph_text, fused_result=fused_result,
            sentence_diagnoses=sentence_diagnoses,
            intensity=intensity, temperature=temp,
        )
        r["round"] = label
        try:
            from .rules_engine import compute_sentence_length_std
            r["_sentence_std"] = compute_sentence_length_std(r["text"])
        except Exception:
            r["_sentence_std"] = 0
        candidates.append(r)

    valid = [c for c in candidates if c.get("text") and c["text"] != paragraph_text]
    if not valid:
        return candidates[0]

    valid.sort(key=lambda c: c.get("_sentence_std", 0), reverse=True)
    best = valid[0]
    best["selected_by"] = "max_sentence_std"
    best["candidates_count"] = len(valid)
    return best
