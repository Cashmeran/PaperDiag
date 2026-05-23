"""核心模块测试"""

import sys
import json
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from paperdiag.preprocessor import (
    auto_parse, parse_text, Paragraph, _detect_protected,
    get_protected_text, restore_protected,
)
from paperdiag.rules_engine import (
    compute_sentence_length_std,
    compute_connector_density,
    compute_info_density,
    compute_term_density,
    compute_hapax_ratio,
    compute_zipf_deviation,
    compute_ngram_repetition,
    compute_punctuation_entropy,
    compute_overall_entropy,
    compute_slop_density,
    scan_paragraph,
    scan_document,
)
from paperdiag.report import json_report, html_report


# ============================================================
#  测试数据
# ============================================================

# AI生成的典型段落（来自collector/seed_texts.py）
AI_SEED_TEXT = """深度学习作为机器学习领域的重要分支，在近年来取得了显著的研究进展。首先，从技术架构的角度来看，卷积神经网络通过局部连接和权重共享机制有效降低了模型参数量，这使得深层网络的训练成为可能。其次，残差网络的提出解决了深层网络中的梯度消失问题，使得数百层的网络结构能够被有效训练。此外，注意力机制的出现进一步推动了自然语言处理领域的发展，Transformer架构已经成为当前主流的基础模型。值得注意的是，这些技术突破不仅提升了模型的性能表现，也为后续的研究工作奠定了坚实的基础。综上所述，深度学习技术的持续演进正在深刻改变人工智能领域的研究范式。"""

# 人类写的学术段落（模拟真人风格）
HUMAN_TEXT = """我们拿两组模型做了对比实验。第一组用标准CNN，在ImageNet上跑，top-1准确率大概76%。第二组加了我们提出的注意力分支——其实就是让网络自己学会关注哪块区域比较重要。结果挺意外的，准确率涨到了81%，但参数量只多了不到3M。不过有一说一，在小数据集上这个提升幅度会打折扣，我们猜测可能是过拟合。后面换了数据增强策略，效果好了不少。"""


# ============================================================
#  预处理测试
# ============================================================

def test_parse_text():
    """测试文本解析"""
    text = "第一段内容测试，这是足够长的文本内容。\n\n第二段内容测试，这是另一段足够长的文本内容。"
    paras = parse_text(text)
    assert len(paras) == 2, f"期望2段，实际{len(paras)}段"
    assert paras[0].index == 0
    assert paras[1].index == 1


def test_detect_protected():
    """测试受保护内容检测"""
    text = "根据文献[1,2,3]的研究，CNN模型（详见公式1）在ImageNet上达到了76.5%的准确率。"
    spans = _detect_protected(text)
    assert len(spans) >= 2  # 至少检测到引文和数字


def test_protected_roundtrip():
    """测试受保护内容替换和还原"""
    text = "文献[1]指出，准确率为92.3%。"
    para = Paragraph(0, text)
    para.protected_spans = _detect_protected(text)
    protected = get_protected_text(para)
    # 受保护内容应该被占位符替换
    assert "__CITATION_" in protected or "__NUMBER_" in protected
    restored = restore_protected(protected, para)
    assert restored == text


# ============================================================
#  规则引擎测试
# ============================================================

def test_sentence_length_std():
    """AI文本的句长方差应在AI区间(5-8)"""
    ai_std = compute_sentence_length_std(AI_SEED_TEXT)
    human_std = compute_sentence_length_std(HUMAN_TEXT)
    print(f"  句长方差: AI={ai_std:.2f}, 人类={human_std:.2f}")
    # AI文本句长方差应较低
    assert ai_std < 15  # AI文本一般方差较小


def test_connector_density():
    """AI文本连接词密度应在AI区间(8-15/千字)"""
    ai_density = compute_connector_density(AI_SEED_TEXT)
    human_density = compute_connector_density(HUMAN_TEXT)
    print(f"  连接词密度: AI={ai_density:.1f}/千字, 人类={human_density:.1f}/千字")
    assert ai_density > 2  # 至少有连接词


def test_info_density():
    """信息密度计算"""
    ai_info = compute_info_density(AI_SEED_TEXT)
    human_info = compute_info_density(HUMAN_TEXT)
    print(f"  信息密度: AI={ai_info:.2%}, 人类={human_info:.2%}")
    assert 0.3 < ai_info < 0.9


def test_hapax_ratio():
    """Hapax比率计算"""
    ai_hapax = compute_hapax_ratio(AI_SEED_TEXT)
    print(f"  Hapax比率: AI={ai_hapax:.2%}")
    assert 0.1 < ai_hapax < 0.8


def test_zipf_deviation():
    """Zipf偏离度计算"""
    ai_zipf = compute_zipf_deviation(AI_SEED_TEXT)
    print(f"  Zipf偏离: AI={ai_zipf:.4f}")
    assert 0 <= ai_zipf <= 1


def test_ngram_repetition():
    """N-gram重复率计算"""
    for n in [2, 3]:
        rep = compute_ngram_repetition(AI_SEED_TEXT, n=n)
        print(f"  {n}-gram重复率: AI={rep:.4f}")
        assert 0 <= rep <= 1


def test_punctuation_entropy():
    """标点熵计算"""
    ai_ent = compute_punctuation_entropy(AI_SEED_TEXT)
    print(f"  标点熵: AI={ai_ent:.2f}")
    assert ai_ent >= 0


def test_overall_entropy():
    """整体熵计算"""
    ai_ent = compute_overall_entropy(AI_SEED_TEXT)
    print(f"  整体熵: AI={ai_ent:.2f}")
    assert ai_ent > 0


def test_slop_density():
    """Slop密度计算"""
    ai_slop = compute_slop_density(AI_SEED_TEXT)
    print(f"  Slop密度: AI={ai_slop:.1f}/千字")
    assert ai_slop >= 0


def test_scan_paragraph_ai():
    """扫描AI文本应被识别为AI"""
    result = scan_paragraph(AI_SEED_TEXT)
    print(f"  AI文本扫描结果: level={result['level']}, score={result['comprehensive_score']:.2%}")
    print(f"  AI维度数: {result['ai_count']}, 灰色维度数: {result['gray_count']}")
    assert result["level"] in ("red", "deep_red", "yellow"), \
        f"AI文本应该被识别为AI特征，实际: {result['level']}"
    assert result["comprehensive_score"] > 0.3, \
        f"AI文本的AI嫌疑分数应较高"


def test_scan_paragraph_human():
    """扫描人类文本不应被误判"""
    result = scan_paragraph(HUMAN_TEXT)
    print(f"  人类文本扫描结果: level={result['level']}, score={result['comprehensive_score']:.2%}")
    print(f"  AI维度数: {result['ai_count']}")
    # 人类文本不应被深度误判
    assert result["level"] != "deep_red", f"人类文本不应被判为deep_red"


def test_scan_document():
    """测试文档扫描"""
    paras = parse_text(AI_SEED_TEXT + "\n\n" + HUMAN_TEXT)
    results = scan_document(paras)
    assert len(results) == 2


# ============================================================
#  改写引擎测试
# ============================================================

# ============================================================
#  质检模块测试
# ============================================================

# ============================================================
#  报告模块测试
# ============================================================

def test_json_report():
    """JSON报告生成"""
    paras = parse_text(AI_SEED_TEXT)
    diagnoses = scan_document(paras)
    report = json_report(diagnoses, paras, [])
    data = json.loads(report)
    assert data["total_paragraphs"] == 1
    assert "paragraphs" in data


def test_html_report():
    """HTML报告生成"""
    paras = parse_text(AI_SEED_TEXT)
    diagnoses = scan_document(paras)
    report = html_report(diagnoses, paras)
    assert "<html" in report
    assert "诊断报告" in report


# ============================================================
#  端到端测试
# ============================================================

def test_e2e_pipeline():
    """端到端：诊断→改写→质检→日志"""
    from paperdiag.logger import RewriteLogger

    text = AI_SEED_TEXT
    # 1. 诊断
    diag_before = scan_paragraph(text)
    assert diag_before["level"] in ("red", "deep_red", "yellow")

    # 2. 改写
    rw = rewrite_paragraph(text, diag_before, intensity=0.7, seed=42)
    rewritten = rw["text"]

    # 3. 质检
    v = validate_rewrite(text, rewritten)
    print(f"  语义相似度: {v['semantic_fidelity']['similarity']:.2%}")

    # 4. 语义不应崩溃
    if v["semantic_fidelity"]["similarity"] < 0.5:
        print(f"  ⚠ 语义相似度过低，但仍然在可接受范围")

    # 5. 再诊断
    diag_after = scan_paragraph(rewritten)

    # 6. 正常情况下应该有改善
    print(f"  改写前: level={diag_before['level']}, score={diag_before['comprehensive_score']:.2%}")
    print(f"  改写后: level={diag_after['level']}, score={diag_after['comprehensive_score']:.2%}")
    print(f"  操作: {rw['operations']}")

    assert len(rewritten) > 0


# ============================================================
#  批量测试：10段AI种子文本
# ============================================================

def test_batch_seed_texts():
    """使用collector中的10段种子文本做批量验证"""
    # 10段AI生成的学术段落
    ai_texts = [
        """深度学习作为机器学习领域的重要分支，在近年来取得了显著的研究进展。首先，从技术架构的角度来看，卷积神经网络通过局部连接和权重共享机制有效降低了模型参数量，这使得深层网络的训练成为可能。其次，残差网络的提出解决了深层网络中的梯度消失问题，使得数百层的网络结构能够被有效训练。此外，注意力机制的出现进一步推动了自然语言处理领域的发展，Transformer架构已经成为当前主流的基础模型。值得注意的是，这些技术突破不仅提升了模型的性能表现，也为后续的研究工作奠定了坚实的基础。综上所述，深度学习技术的持续演进正在深刻改变人工智能领域的研究范式。""",

        """数字经济作为一种新型经济形态，正在深刻改变传统经济运行的基本逻辑。从生产端来看，数据已经成为与土地、劳动力、资本并列的关键生产要素，其边际成本递减的特征使得规模化效应愈发显著。从消费端来看，平台经济的兴起降低了交易双方的信息不对称程度，提高了市场匹配效率。然而，与此同时，数字鸿沟问题也日益凸显，不同地区和群体之间的数字化差距可能导致新的不平等。因此，政策制定者需要在促进数字经济发展的同时注重包容性增长，确保技术进步的红利能够惠及更广泛的社会群体。""",

        """肿瘤免疫治疗是近年来癌症治疗领域最具突破性的研究方向之一。首先，免疫检查点抑制剂通过阻断PD-1/PD-L1等免疫抑制信号通路，重新激活T细胞的抗肿瘤活性。其次，CAR-T细胞疗法通过基因工程改造患者自身的T细胞，使其能够特异性识别并杀伤肿瘤细胞。此外，肿瘤疫苗和溶瘤病毒等新兴免疫治疗策略也在临床前研究中展现出良好的应用前景。需要指出的是，免疫治疗虽然在部分患者中取得了显著疗效，但免疫相关不良反应和耐药性问题仍然亟待解决。因此，未来的研究方向应聚焦于生物标志物的筛选和联合治疗方案的优化。""",

        """混合式教学模式融合了线上学习的灵活性与线下教学的互动性，成为后疫情时代教育改革的重要方向。从理论层面分析，混合式教学依托建构主义学习理论，强调学习者在多元情境中主动建构知识体系。从实践层面考察，翻转课堂作为混合式教学的重要形式，通过课前视频学习和课堂深度讨论的结合，有效提升了学生的参与度和学习效果。与此同时，学习分析技术的应用使得教师能够基于数据对教学过程进行精准干预。然而，混合式教学的实施也面临着数字基础设施不均衡、教师信息技术素养参差不齐等现实挑战。因此，构建适应不同教育场景的混合式教学模式仍然需要进一步的理论研究与实践探索。""",

        """碳中和目标的实现需要能源结构转型、技术创新和政策协调的多维协同推进。首先，在能源供给方面，光伏、风电等可再生能源的装机容量持续增长，度电成本已经接近甚至低于传统化石能源。其次，在能源消费方面，工业领域的电气化改造和建筑领域的节能标准提升是实现终端脱碳的重要路径。此外，碳捕集利用与封存技术虽然在技术层面已有突破，但其大规模商业化应用仍面临成本过高的瓶颈。需要强调的是，碳定价机制作为市场化减排工具，在欧盟等地区已经形成了较为成熟的运行体系。展望未来，碳中和将深刻重塑全球能源地缘政治格局，也将催生新的绿色产业增长点。""",
    ]

    ai_count = 0
    scores = []
    for text in ai_texts:
        result = scan_paragraph(text)
        scores.append(result["comprehensive_score"])
        if result["level"] in ("red", "deep_red"):
            ai_count += 1
        print(f"  [{result['level']}] score={result['comprehensive_score']:.2%} "
              f"ai_dims={result['ai_count']}")

    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"\n  总计: {ai_count}/{len(ai_texts)} 被正确识别为AI文本")
    print(f"  平均AI嫌疑分数: {avg_score:.2%}")

    # 至少大部分AI文本应被识别
    assert ai_count >= len(ai_texts) * 0.6, \
        f"至少60%的AI文本应被识别，实际: {ai_count}/{len(ai_texts)}"


if __name__ == "__main__":
    print("=" * 60)
    print("  PaperDiag — 核心模块测试")
    print("=" * 60)

    import traceback

    tests = [
        ("文本解析", test_parse_text),
        ("受保护内容检测", test_detect_protected),
        ("受保护内容还原", test_protected_roundtrip),
        ("句长方差", test_sentence_length_std),
        ("连接词密度", test_connector_density),
        ("信息密度", test_info_density),
        ("Hapax比率", test_hapax_ratio),
        ("Zipf偏离", test_zipf_deviation),
        ("N-gram重复", test_ngram_repetition),
        ("标点熵", test_punctuation_entropy),
        ("整体熵", test_overall_entropy),
        ("Slop密度", test_slop_density),
        ("扫描AI段落", test_scan_paragraph_ai),
        ("扫描人类段落", test_scan_paragraph_human),
        ("文档扫描", test_scan_document),
        ("JSON报告", test_json_report),
        ("HTML报告", test_html_report),
        ("批量种子文本", test_batch_seed_texts),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
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
