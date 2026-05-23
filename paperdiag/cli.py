"""CLI 命令行接口"""

import sys
import json
import datetime
import io
from pathlib import Path

# 强制UTF-8编码（Windows兼容）
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import click

from .preprocessor import auto_parse, Paragraph
from .rules_engine import scan_document
from .report import terminal_report, json_report, html_report
from .fusion import fuse_diagnosis, generate_rewrite_instructions

# 可选导入
try:
    from .embedding_layer import scan_semantic
    HAS_EMBEDDING = True
except ImportError:
    HAS_EMBEDDING = False

HAS_LLM = False
HAS_LLM_MODULES = False




@click.group()
@click.version_option(version="0.1.0", prog_name="paperdiag")
def cli():
    """PaperDiag — 中文学术论文降AIGC率开源工具

    诊断：逐段告诉你论文哪里读起来像AI
    改写：帮你改掉这些AI特征
    """
    pass


@cli.command()
@click.option("--port", "-p", default=5000, help="Web服务端口 (默认5000)")
def webui(port: int):
    """启动 Web 界面"""
    click.echo(f"[webui] Starting PaperDiag on http://localhost:{port}")
    from .webui import app
    app.run(host="0.0.0.0", port=port, debug=False)



@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--format", "-f", "output_format", type=click.Choice(["terminal", "json", "html"]),
              default="terminal", help="输出格式")
@click.option("--output", "-o", "output_file", type=click.Path(), default=None, help="输出文件路径")
@click.option("--embedding/--no-embedding", default=True,
              help="启用/禁用 Embedding 语义层 (默认: 启用)")
@click.option("--llm/--no-llm", default=False,
              help="启用/禁用 LLM 定性诊断")
def scan(input_path: str, output_format: str, output_file: str | None,
         embedding: bool, llm: bool):
    """扫描论文，生成AI特征诊断报告

    INPUT_PATH: 论文文件路径 (.docx / .txt)
    """
    click.echo(f"[scan] Reading: {input_path}")

    try:
        paragraphs = auto_parse(input_path)
    except Exception as e:
        click.echo(f"[error] Failed to read: {e}", err=True)
        sys.exit(1)

    if not paragraphs:
        click.echo("[error] No valid paragraphs detected", err=True)
        sys.exit(1)

    click.echo(f"   Detected {len(paragraphs)} paragraphs")
    click.echo(f"[scan] Running 12-dimension rule scan...")

    results = scan_document(paragraphs)

    # Embedding 语义层
    semantic = None
    if embedding and HAS_EMBEDDING:
        click.echo(f"[scan] Running Qwen3-Embedding semantic scan...")
        try:
            para_texts = [p.text if hasattr(p, 'text') else str(p) for p in paragraphs]
            semantic = scan_semantic(para_texts)
            if "error" in semantic:
                click.echo(f"   [!] Embedding scan skipped: {semantic['error']}")
                semantic = None
        except Exception as e:
            click.echo(f"   [!] Embedding scan failed: {e}")
            semantic = None

    # 融合
    fused = fuse_diagnosis(results, semantic)

    # LLM 定性诊断（仅对黄/红/深红段落，绿色跳过）
    llm_diagnoses = None
    llm_count = 0
    if llm and HAS_LLM and HAS_LLM_MODULES:
        high_risk = sum(1 for f in fused if f.get("fused_level", f.get("rule_level", "green")) != "green")
        click.echo(f"[scan] LLM diagnosis: {high_risk}/{len(fused)} paragraphs (green skipped)")
        try:
            ensure_ollama_ready()
            para_texts = [p.text if hasattr(p, 'text') else str(p) for p in paragraphs]
            llm_diagnoses = diagnose_document(para_texts, fused, only_high_risk=True)
            for i, ld in enumerate(llm_diagnoses):
                if ld and i < len(fused):
                    fused[i]["llm_diagnosis"] = {
                        "naturalness": ld.get("naturalness"),
                        "template_patterns": ld.get("template_patterns", []),
                        "mechanical_feel": ld.get("mechanical_feel"),
                        "primary_issue": ld.get("primary_issue"),
                        "specific_advice": ld.get("specific_advice"),
                        "rewrite_priority": ld.get("rewrite_priority"),
                    }
                    llm_count += 1
        except Exception as e:
            click.echo(f"   [!] LLM diagnosis failed: {e}")

    # 输出
    if output_format == "terminal":
        _terminal_fused_report(fused, paragraphs)
    elif output_format == "json":
        report_str = json.dumps(fused, ensure_ascii=False, indent=2)
        if output_file:
            Path(output_file).write_text(report_str, encoding="utf-8")
            click.echo(f"[ok] JSON report saved to: {output_file}")
        else:
            click.echo(report_str)
    elif output_format == "html":
        report_str = html_report(results, paragraphs)  # TODO: update html_report for fused
        out = output_file or "diagnosis_report.html"
        Path(out).write_text(report_str, encoding="utf-8")
        click.echo(f"[ok] HTML report saved to: {out}")

    # 统计
    ai_count = sum(1 for r in fused if r.get("fused_level", r["rule_level"]) in ("red", "deep_red"))
    if ai_count > 0:
        click.echo(f"\n[!] Found {ai_count}/{len(fused)} high-risk paragraphs")
    else:
        click.echo(f"\n[ok] All paragraphs safe")


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--target", "-t", type=click.Choice(["cnki", "weipu", "wanfang", "auto"]),
              default="auto", help="目标检测平台（暂未生效）")
@click.option("--intensity", "-i", type=click.FloatRange(0.0, 1.0), default=0.3,
              help="改写强度")
@click.option("--seed", "-s", type=int, default=42,
              help="随机种子")
@click.option("--output", "-o", "output_dir", type=click.Path(), default="./output",
              help="输出目录")
@click.option("--format", "-f", "output_format", type=click.Choice(["terminal", "json", "html"]),
              default="terminal", help="报告输出格式")
@click.option("--no-validate", is_flag=True, help="跳过后处理质检")
@click.option("--embedding/--no-embedding", default=True,
              help="启用/禁用 Embedding 语义层 (默认: 启用)")
@click.option("--llm/--no-llm", default=False,
              help="启用/禁用 LLM 改写 (需要下载模型，改写质量更高)")
def fix(input_path: str, target: str, intensity: float, seed: int,
        output_dir: str, output_format: str, no_validate: bool,
        embedding: bool, llm: bool):
    """扫描并改写论文，降低AIGC检测率

    INPUT_PATH: 论文文件路径 (.docx / .txt)
    """
    # 创建输出目录
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 记录器
    logger = RewriteLogger(output_dir)

    click.echo(f"[fix] Reading: {input_path}")
    try:
        paragraphs = auto_parse(input_path)
    except Exception as e:
        click.echo(f"[error] Failed to read: {e}", err=True)
        sys.exit(1)

    if not paragraphs:
        click.echo("[error] No valid paragraphs detected", err=True)
        sys.exit(1)

    click.echo(f"   Detected {len(paragraphs)} paragraphs")
    click.echo(f"   Intensity: {intensity:.0%} | Seed: {seed}")

    # 第一阶段：诊断
    click.echo(f"\n[1/3] Running diagnosis...")
    diagnoses = scan_document(paragraphs)

    # Embedding 语义层
    semantic = None
    if embedding and HAS_EMBEDDING:
        click.echo(f"   + Qwen3-Embedding semantic scan...")
        try:
            para_texts = [p.text if hasattr(p, 'text') else str(p) for p in paragraphs]
            semantic = scan_semantic(para_texts)
            if "error" in semantic:
                click.echo(f"   [!] Skipped: {semantic['error']}")
                semantic = None
        except Exception as e:
            click.echo(f"   [!] Failed: {e}")
            semantic = None

    # 融合诊断
    fused = fuse_diagnosis(diagnoses, semantic)

    ai_before = sum(1 for f in fused if f.get("fused_level", f["rule_level"]) in ("red", "deep_red"))
    click.echo(f"   Found {ai_before}/{len(paragraphs)} high-risk paragraphs")

    # 第二阶段：改写
    llm_used = False
    if llm and HAS_LLM and HAS_LLM_MODULES:
        click.echo(f"\n[2/3] Running LLM rewriting (Qwen3.5-4B via Ollama)...")
        try:
            ensure_ollama_ready()
            para_texts = [p.text if hasattr(p, 'text') else str(p) for p in paragraphs]
            rewrite_results = []
            for i, (para, fused) in enumerate(zip(paragraphs, fused)):
                level = fused.get("fused_level", fused.get("rule_level", "green"))
                # 文学保护：检测到经典文学特征 → 跳过LLM改写
                rule_diag = diagnoses[i] if i < len(diagnoses) else {}
                abs_guard = rule_diag.get("absurdity_guard", {})
                if abs_guard.get("is_classical_literature") or abs_guard.get("triggered"):
                    rewrite_results.append({
                        "index": i, "original": para_texts[i],
                        "rewritten": para_texts[i],
                        "operations": [f"文学保护: {abs_guard.get('reason', '')}"],
                        "seed": seed + i,
                    })
                    continue
                if level == "green":
                    rewrite_results.append({
                        "index": i, "original": para_texts[i],
                        "rewritten": para_texts[i],
                        "operations": ["已跳过（安全段落）"],
                        "seed": seed + i,
                    })
                else:
                    rw = rewrite_paragraph_multi_temp(
                        para_texts[i],
                        fused_result=fused,
                        intensity=intensity,
                    )
                    rewrite_results.append({
                        "index": i,
                        "original": para_texts[i],
                        "rewritten": rw.get("text", para_texts[i]),
                        "operations": [f"LLM改写({rw.get('selected_by', 'unknown')})",
                                       f"temp={rw.get('temperature', '?')}"],
                        "seed": seed + i,
                        "llm_round": rw.get("round"),
                    })
            llm_used = True
        except Exception as e:
            click.echo(f"   [!] LLM rewrite failed: {e}, falling back to rules")
            llm_used = False

    if not llm_used:
        click.echo(f"\n[2/3] Running rule-based rewriting...")
        rewrite_results = rewrite_document(paragraphs, fused, intensity=intensity, seed=seed)

    modified = sum(1 for rw in rewrite_results if rw.get("operations") and rw["operations"] != ["已跳过（安全段落）"])
    click.echo(f"   Rewrote {modified} paragraphs{' (LLM)' if llm_used else ' (rules)'}")

    # 第三阶段：质检
    if not no_validate:
        click.echo(f"\n[3/3] Running post-processing validation...")
        validation_results = []
        for i, (para, rw) in enumerate(zip(paragraphs, rewrite_results)):
            original_text = para.text if hasattr(para, 'text') else str(para)
            rewritten_text = rw.get("rewritten", rw.get("text", ""))

            if original_text != rewritten_text:
                v = validate_rewrite(original_text, rewritten_text)
                validation_results.append(v)
                if not v["pass"]:
                    click.echo(f"   [!] Paragraph #{i+1}: {v['critical_count']} errors, {v['warning_count']} warnings, "
                               f"语义相似度: {v['semantic_fidelity']['similarity']:.2%}")
            else:
                validation_results.append({"pass": True, "issues": [], "semantic_fidelity": {"similarity": 1.0}})

        failed = sum(1 for v in validation_results if not v["pass"])
        if failed > 0:
            click.echo(f"   [!] {failed} paragraphs have errors (warnings excluded)")
    else:
        validation_results = []

    # 重新诊断改写后
    click.echo(f"\n[report] Generating post-rewrite diagnosis...")
    rewritten_paras = []
    for para, rw in zip(paragraphs, rewrite_results):
        new_para = Paragraph(
            para.index if hasattr(para, 'index') else 0,
            rw.get("rewritten", rw.get("text", ""))
        )
        rewritten_paras.append(new_para)

    diagnoses_after = scan_document(rewritten_paras)
    ai_after = sum(1 for d in diagnoses_after if d["level"] in ("red", "deep_red"))

    # 记录日志
    for i, (para, rw, d_before, d_after) in enumerate(
            zip(paragraphs, rewrite_results, diagnoses, diagnoses_after)):
        logger.log_paragraph(
            index=para.index if hasattr(para, 'index') else i,
            original=para.text if hasattr(para, 'text') else str(para),
            rewritten=rw.get("rewritten", rw.get("text", "")),
            diagnosis_before=d_before,
            diagnosis_after=d_after,
            operations=rw.get("operations", []),
            validation=validation_results[i] if i < len(validation_results) else None,
        )

    # 保存改写结果
    _save_results(paragraphs, rewrite_results, out_path, input_path)

    # 生成报告
    if output_format == "terminal":
        click.echo(f"\n{'='*60}")
        click.echo(f"  Rewrite complete!")
        click.echo(f"  High-risk paragraphs: {ai_before} -> {ai_after}")
        click.echo(f"  Output saved to: {out_path.absolute()}")
        click.echo(f"{'='*60}")
    elif output_format == "html":
        report_str = html_report(diagnoses_after, rewritten_paras, rewrite_results)
        report_path = out_path / "diagnosis_report.html"
        report_path.write_text(report_str, encoding="utf-8")
        click.echo(f"[ok] HTML report saved to: {report_path}")
    elif output_format == "json":
        report_str = json_report(diagnoses_after, rewritten_paras, rewrite_results)
        report_path = out_path / "diagnosis_report.json"
        report_path.write_text(report_str, encoding="utf-8")
        click.echo(f"[ok] JSON report saved to: {report_path}")

    # 摘要
    summary = logger.get_summary()
    click.echo(f"\n[summary]")
    click.echo(f"   改写段落: {summary['modified_paragraphs']}/{summary['total_paragraphs']}")
    click.echo(f"   字数变化: {summary['char_change_ratio']:+.1%}")
    click.echo(f"   平均AI分数降幅: {summary['avg_score_reduction']:.1%}")


def _terminal_fused_report(fused_results: list[dict], paragraphs: list):
    """终端输出融合诊断报告（规则+Embedding）"""
    try:
        from rich.console import Console
        from rich.panel import Panel
        RICH = True
    except ImportError:
        RICH = False

    if RICH:
        console = Console()
        console.print()
        console.print(Panel.fit(
            "[bold cyan]PaperDiag[/bold cyan] - Fused Diagnosis (Rule 12D + Embedding 4D)",
            border_style="cyan",
        ))
        for i, (fr, para) in enumerate(zip(fused_results, paragraphs)):
            level = fr.get("fused_level", fr.get("rule_level", "green"))
            rule_ai = fr.get("rule_ai_count", 0)
            sem_ai = fr.get("semantic_ai_count", 0)
            score = fr.get("combined_score", 0)

            color = {"green": "green", "yellow": "yellow",
                     "red": "red", "deep_red": "red"}.get(level, "white")
            label = {"green": "SAFE", "yellow": "SUSPICIOUS",
                     "red": "HIGH RISK", "deep_red": "AI CONFIRMED"}.get(level, level)

            preview = (para.text[:120] if hasattr(para, 'text') else str(para)[:120])
            has_llm = bool(fr.get("llm_diagnosis"))
            llm_tag = " [LLM]" if has_llm else ""
            console.print(f"\n[bold {color}]Para #{i+1}[/bold {color}] [{color}]{label}{llm_tag}[/{color}]")
            console.print(f"  [dim]Score: {score:.0%} | Rule: {rule_ai}D | Emb: {sem_ai}D[/dim]")
            console.print(f"  [dim]{preview}...[/dim]")

            # 显示LLM诊断结果
            llm_diag = fr.get("llm_diagnosis")
            if llm_diag:
                nat = llm_diag.get("naturalness", "?")
                issue = llm_diag.get("primary_issue", "?")
                advice = llm_diag.get("specific_advice", "?")
                fp_type = llm_diag.get("ai_fingerprint_type", "?")
                console.print(f"  [bold cyan]LLM判断:[/bold cyan] 自然度={nat} | 指纹类型={fp_type}")
                console.print(f"  [bold cyan]LLM诊断:[/bold cyan] {issue}")
                if advice and advice != "?":
                    console.print(f"  [bold cyan]LLM建议:[/bold cyan] {advice}")

            instructions = generate_rewrite_instructions(fr)
            if instructions and level in ("red", "deep_red", "yellow"):
                console.print("  [bold]Suggestions:[/bold]")
                for idx, inst in enumerate(instructions[:3], 1):
                    console.print(f"    {idx}. {inst}")
        console.print()
    else:
        for i, (fr, para) in enumerate(zip(fused_results, paragraphs)):
            level = fr.get("fused_level", fr.get("rule_level", "green"))
            print(f"\n[Para #{i+1}] {level} (Score: {fr.get('combined_score', 0):.0%})")
            for idx, inst in enumerate(generate_rewrite_instructions(fr)[:3], 1):
                print(f"  {idx}. {inst}")


