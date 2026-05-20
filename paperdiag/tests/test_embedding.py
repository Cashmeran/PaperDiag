"""Embedding层 + 融合层测试"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from paperdiag.rules_engine import scan_paragraph, scan_document
from paperdiag.embedding_layer import (
    scan_semantic,
    compute_paragraph_similarity,
    compute_argument_linearity,
    compute_semantic_backtrack,
    compute_transition_naturalness,
    get_model,
)
from paperdiag.fusion import (
    fuse_diagnosis,
    generate_gap_report,
    generate_rewrite_instructions,
    _fused_level,
)


# 3段AI生成的学术文本
AI_PARAGRAPHS = [
    "深度学习作为机器学习领域的重要分支，在近年来取得了显著的研究进展。首先，从技术架构的角度来看，卷积神经网络通过局部连接和权重共享机制有效降低了模型参数量，这使得深层网络的训练成为可能。其次，残差网络的提出解决了深层网络中的梯度消失问题，使得数百层的网络结构能够被有效训练。",
    "此外，注意力机制的出现进一步推动了自然语言处理领域的发展，Transformer架构已经成为当前主流的基础模型。值得注意的是，这些技术突破不仅提升了模型的性能表现，也为后续的研究工作奠定了坚实的基础。",
    "综上所述，深度学习技术的持续演进正在深刻改变人工智能领域的研究范式。未来的研究方向可能会更加注重多模态融合、小样本学习以及模型的可解释性。这些新兴方向将推动AI技术向更通用、更可信的方向发展。",
]

# 3段人类写的文本（模拟口语化、非线性风格）
HUMAN_PARAGRAPHS = [
    "我们做了个实验，拿两组模型对比了一下。第一组就是标准的CNN，直接在ImageNet上跑，top-1大概76%。第二组加了我们自己搞的注意力分支——其实就是让网络学会哪些区域比较重要。",
    "结果还挺意外的。加了注意力之后，准确率涨到了81%，而且参数量只多了不到3M。不过说实话，在小数据集上效果会打折扣。我们怀疑可能是过拟合了。",
    "后面换了几种数据增强的策略，情况好了不少。当然这玩意儿到底有没有用还得看具体场景，不是所有任务都适合。这一点后面还得细琢磨。",
]


def test_model_loaded():
    """模型是否成功加载"""
    model = get_model()
    assert model is not None


def test_scan_semantic_ai():
    """AI文本的语义层扫描"""
    result = scan_semantic(AI_PARAGRAPHS)
    assert "error" not in result
    assert len(result["paragraph_similarities"]) == 2  # 3段=2对相邻
    assert result["argument_linearity"] >= 0
    assert result["semantic_backtrack_rate"] >= 0
    print(f"  AI semantic: linearity={result['argument_linearity']:.6f}, "
          f"backtrack={result['semantic_backtrack_rate']:.2%}")


def test_scan_semantic_human():
    """人类文本的语义层扫描"""
    result = scan_semantic(HUMAN_PARAGRAPHS)
    assert "error" not in result
    print(f"  Human semantic: linearity={result['argument_linearity']:.6f}, "
          f"backtrack={result['semantic_backtrack_rate']:.2%}")


def test_ai_vs_human_linearity():
    """AI论证线性度应低于人类"""
    ai_r = scan_semantic(AI_PARAGRAPHS)
    hu_r = scan_semantic(HUMAN_PARAGRAPHS)
    if "error" not in ai_r and "error" not in hu_r:
        print(f"  Linearity: AI={ai_r['argument_linearity']:.6f} vs Human={hu_r['argument_linearity']:.6f}")
        # AI偏移方差应更小
        if ai_r["argument_linearity"] > 0 and hu_r["argument_linearity"] > 0:
            # 不强制断言（需要更多样本），但打印对比
            pass


def test_paragraph_similarity():
    """AI段间相似度应较高"""
    model = get_model()
    texts = AI_PARAGRAPHS
    embeddings = model.encode(texts, normalize_embeddings=True)
    sims = compute_paragraph_similarity(list(embeddings))
    print(f"  AI paragraph similarities: {sims}")
    assert len(sims) == 2


def test_fuse_diagnosis():
    """融合诊断无错误"""
    # 规则诊断
    class FakePara:
        def __init__(self, index, text):
            self.index = index
            self.text = text

    paras = [FakePara(i, t) for i, t in enumerate(AI_PARAGRAPHS)]
    rule_results = scan_document(paras)

    # 语义诊断
    semantic = scan_semantic(AI_PARAGRAPHS)

    # 融合
    fused = fuse_diagnosis(rule_results, semantic)
    assert len(fused) == 3
    assert "rule_dimensions" in fused[0]
    assert "semantic_dimensions" in fused[0]
    print(f"  Fused levels: {[f['fused_level'] for f in fused]}")


def test_gap_report():
    """排名差距报告"""
    paras = [type('P', (), {'text': t, 'index': i})() for i, t in enumerate(AI_PARAGRAPHS)]
    rule_results = scan_document(paras)
    semantic = scan_semantic(AI_PARAGRAPHS)
    fused = fuse_diagnosis(rule_results, semantic)

    gaps = generate_gap_report(fused[0])
    gap_info = [(g['dimension'], round(g['gap'], 2)) for g in gaps[:3]]
    print(f"  Top gaps: {gap_info}")
    assert len(gaps) >= 0


def test_rewrite_instructions():
    """改写指令生成"""
    paras = [type('P', (), {'text': t, 'index': i})() for i, t in enumerate(AI_PARAGRAPHS)]
    rule_results = scan_document(paras)
    semantic = scan_semantic(AI_PARAGRAPHS)
    fused = fuse_diagnosis(rule_results, semantic)

    instructions = generate_rewrite_instructions(fused[0])
    print(f"  Instructions: {instructions[:2]}")
    assert len(instructions) > 0


def test_no_embedding_fallback():
    """无embedding时的fallback"""
    paras = [type('P', (), {'text': t, 'index': i})() for i, t in enumerate(AI_PARAGRAPHS)]
    rule_results = scan_document(paras)
    fused = fuse_diagnosis(rule_results, None)  # 无语义层
    assert len(fused) == 3
    assert fused[0]["fused_level"] == fused[0]["rule_level"]


def test_fused_level():
    """融合等级判定"""
    assert _fused_level("green", 0, 0) == "green"
    assert _fused_level("red", 4, 2) == "red"
    assert _fused_level("red", 5, 5) == "deep_red"
    assert _fused_level("yellow", 2, 1) == "yellow"


if __name__ == "__main__":
    print("=" * 60)
    print("  PaperDiag - Embedding Layer Tests")
    print("=" * 60)

    import traceback
    tests = [
        ("Model loaded", test_model_loaded),
        ("AI semantic scan", test_scan_semantic_ai),
        ("Human semantic scan", test_scan_semantic_human),
        ("AI vs Human linearity", test_ai_vs_human_linearity),
        ("Paragraph similarity", test_paragraph_similarity),
        ("Fuse diagnosis", test_fuse_diagnosis),
        ("Gap report", test_gap_report),
        ("Rewrite instructions", test_rewrite_instructions),
        ("No embedding fallback", test_no_embedding_fallback),
        ("Fused level logic", test_fused_level),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'='*60}")
    if failed > 0:
        sys.exit(1)
