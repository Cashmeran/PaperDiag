"""WebUI — Flask 应用，Warm Paper 风格"""

import sys, io, json, traceback
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from flask import Flask, render_template, request, jsonify

from .preprocessor import auto_parse
from .rules_engine import scan_document, scan_sentences
from .rule_rewriter import rewrite_document
from .fusion import fuse_diagnosis

app = Flask(__name__, template_folder="templates")
app.config["JSON_AS_ASCII"] = False
app.config["TEMPLATES_AUTO_RELOAD"] = True

try:
    from .embedding_layer import scan_semantic
    HAS_EMBEDDING = True
except ImportError:
    HAS_EMBEDDING = False
try:
    from .model_manager import is_backend_available
    HAS_LLM = True
except ImportError:
    HAS_LLM = False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/scan", methods=["POST"])
def api_scan():
    try:
        data = request.get_json()
        text = data.get("text", "")
        if len(text.strip()) < 20:
            return jsonify({"error": "文本太短"}), 400

        paragraphs = auto_parse(text)
        if not paragraphs:
            return jsonify({"error": "未检测到有效段落"}), 400

        # 句子级扫描（对齐知网4.0）
        sent_result = scan_sentences(paragraphs)

        # 段落级诊断
        diagnoses = scan_document(paragraphs)
        semantic = None
        if HAS_EMBEDDING:
            try:
                pts = [p.text if hasattr(p, 'text') else str(p) for p in paragraphs]
                semantic = scan_semantic(pts)
                if semantic and "error" in semantic:
                    semantic = None
            except Exception:
                pass
        fused = fuse_diagnosis(diagnoses, semantic)

        # 按段落组织句子
        para_sentences = {}
        for s in sent_result["sentences"]:
            pi = s["paragraph_index"]
            para_sentences.setdefault(pi, []).append(s)

        results = []
        for i, (f, p) in enumerate(zip(fused, paragraphs)):
            pt = p.text if hasattr(p, 'text') else str(p)
            sents = para_sentences.get(i, [])
            ai_count = sum(1 for s in sents if s["zone"] in ("red", "orange"))
            results.append({
                "index": i, "text": pt, "level": f.get("fused_level", f.get("rule_level", "green")),
                "score": round(f.get("combined_score", f.get("rule_score", 0)) * 100),
                "sentences": [
                    {"text": s["text"], "zone": s["zone"], "score": round(s["score"] * 100)}
                    for s in sents
                ],
                "ai_sentence_count": ai_count,
                "total_sentence_count": len(sents),
            })

        return jsonify({
            "ai_rate": round(sent_result["ai_rate"] * 100),
            "breakdown": {
                "severe": sent_result.get("red_count", 0),
                "moderate": sent_result.get("orange_count", 0),
                "mild": sent_result.get("yellow_count", 0),
            },
            "total_chars": sent_result["total_chars"],
            "ai_chars": sent_result["ai_chars"],
            "total_sentences": sent_result["total_sentences"],
            "ai_sentences": sent_result["ai_sentences"],
            "suspect_sentences": sent_result.get("suspect_sentences", 0),
            "paragraphs": results,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/fix", methods=["POST"])
def api_fix():
    try:
        data = request.get_json()
        text = data.get("text", "")
        intensity = data.get("intensity", 0.3)
        seed = data.get("seed", 42)
        if len(text.strip()) < 20:
            return jsonify({"error": "文本太短"}), 400

        paragraphs = auto_parse(text)
        if not paragraphs:
            return jsonify({"error": "未检测到有效段落"}), 400

        # 改写前AI率
        before = scan_sentences(paragraphs)

        diagnoses = scan_document(paragraphs)
        semantic = None
        if HAS_EMBEDDING:
            try:
                pts = [p.text if hasattr(p, 'text') else str(p) for p in paragraphs]
                semantic = scan_semantic(pts)
            except Exception:
                pass
        fused = fuse_diagnosis(diagnoses, semantic)
        rewrite_results = rewrite_document(paragraphs, fused, intensity=intensity, seed=seed)

        # 组织改写前的句子数据（按段落）
        before_sents = {}
        for s in before.get("sentences", []):
            pi = s.get("paragraph_index", 0)
            before_sents.setdefault(pi, []).append(s)

        rewritten_texts = []
        results = []
        for i, (p, rw) in enumerate(zip(paragraphs, rewrite_results)):
            orig = p.text if hasattr(p, 'text') else str(p)
            re = rw.get("rewritten", orig)
            rewritten_texts.append(re)
            results.append({
                "index": i, "original": orig, "rewritten": re,
                "operations": rw.get("operations", []), "changed": orig != re,
                "sentences_before": [
                    {"text": s["text"], "zone": s["zone"], "score": round(s["score"] * 100)}
                    for s in before_sents.get(i, [])
                ],
            })

        full_rewritten = "\n\n".join(rewritten_texts)

        # 改写后AI率 + 句子级扫描
        after_paras = auto_parse(full_rewritten)
        after = scan_sentences(after_paras) if after_paras else {"ai_rate": before["ai_rate"], "sentences": []}

        # 组织改写后的句子数据（按段落）
        after_sents = {}
        for s in after.get("sentences", []):
            pi = s.get("paragraph_index", 0)
            after_sents.setdefault(pi, []).append(s)

        for i, r in enumerate(results):
            r["sentences_after"] = [
                {"text": s["text"], "zone": s["zone"], "score": round(s["score"] * 100)}
                for s in after_sents.get(i, [])
            ]

        return jsonify({
            "full_text": full_rewritten,
            "ai_rate_before": round(before["ai_rate"] * 100),
            "ai_rate_after": round(after["ai_rate"] * 100),
            "ai_rate_change": round((before["ai_rate"] - after["ai_rate"]) * 100),
            "breakdown_before": {
                "severe": before.get("red_count", 0),
                "moderate": before.get("orange_count", 0),
                "mild": before.get("yellow_count", 0),
            },
            "breakdown_after": {
                "severe": after.get("red_count", 0),
                "moderate": after.get("orange_count", 0),
                "mild": after.get("yellow_count", 0),
            },
            "paragraphs": results,
            "modified_count": sum(1 for r in results if r["changed"]),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/parse-docx", methods=["POST"])
def api_parse_docx():
    try:
        file = request.files.get("file")
        if not file or not file.filename.lower().endswith(".docx"):
            return jsonify({"error": "仅支持 .docx"}), 400
        from docx import Document
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        doc = Document(tmp_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        os.unlink(tmp_path)
        if not paragraphs:
            return jsonify({"error": "文档无文字"}), 400
        return jsonify({"text": "\n\n".join(paragraphs), "paragraph_count": len(paragraphs)})
    except ImportError:
        return jsonify({"error": "未安装 python-docx"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/status")
def api_status():
    llm_ok = False
    if HAS_LLM:
        try:
            llm_ok = is_backend_available()
        except Exception:
            llm_ok = False
    return jsonify({"embedding": HAS_EMBEDDING, "llm_available": llm_ok})
