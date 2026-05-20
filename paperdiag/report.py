"""诊断报告生成模块：终端彩色 / JSON / HTML"""

import json
import datetime
from typing import Any

# 检测 Rich 是否可用
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.layout import Layout
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# ============================================================
#  终端彩色报告（Rich）
# ============================================================

ZONE_COLORS = {
    "ai": "red",
    "gray": "yellow",
    "human": "green",
    "unknown": "white",
}

LEVEL_LABELS = {
    "green": "安全 ✓",
    "yellow": "可疑 ⚡",
    "red": "高风险 ⚠",
    "deep_red": "确认AI 🚫",
}


def _zone_bar(value: float, ai_range: tuple, human_range: tuple,
              gray_range: tuple | None = None) -> str:
    """生成简单的ASCII区间指示器"""
    ai_low, ai_high = ai_range
    human_low, human_high = human_range

    if value < ai_low:
        return "⬇过低"
    elif ai_low <= value <= ai_high:
        return "🔴 AI区间"
    elif gray_range and gray_range[0] <= value <= gray_range[1]:
        return "🟡 灰色区间"
    elif human_low <= value <= human_high:
        return "🟢 人类区间"
    elif value > human_high:
        return "⬆过高"
    return "⚪ 未知"


def terminal_report(results: list[dict[str, Any]],
                    paragraphs: list) -> None:
    """输出终端彩色诊断报告"""
    if not RICH_AVAILABLE:
        _plain_report(results, paragraphs)
        return

    console = Console()

    # 标题
    console.print()
    console.print(Panel.fit(
        "[bold cyan]PaperDiag[/bold cyan] — 中文学术论文 AI 特征诊断报告",
        border_style="cyan",
    ))
    console.print(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    console.print(f"段落总数: {len(results)}")
    console.print()

    # 总体概览
    ai_paras = sum(1 for r in results if r["level"] in ("red", "deep_red"))
    gray_paras = sum(1 for r in results if r["level"] == "yellow")
    safe_paras = sum(1 for r in results if r["level"] == "green")

    overview = Table(title="总体概览", box=box.ROUNDED)
    overview.add_column("等级", style="bold")
    overview.add_column("段落数")
    overview.add_column("占比")
    overview.add_row("🚫 高风险(AI)", str(ai_paras),
                     f"{ai_paras/len(results)*100:.1f}%" if results else "0%")
    overview.add_row("⚡ 可疑", str(gray_paras),
                     f"{gray_paras/len(results)*100:.1f}%" if results else "0%")
    overview.add_row("✓ 安全", str(safe_paras),
                     f"{safe_paras/len(results)*100:.1f}%" if results else "0%")
    console.print(overview)
    console.print()

    # 逐段详细报告
    for i, (result, para) in enumerate(zip(results, paragraphs)):
        text = para.text if hasattr(para, 'text') else str(para)

        level = result["level"]
        score = result["comprehensive_score"]
        level_label = LEVEL_LABELS.get(level, level)
        color = {"green": "green", "yellow": "yellow",
                 "red": "red", "deep_red": "red"}.get(level, "white")

        # 段落标题
        preview = text[:80] + "..." if len(text) > 80 else text
        console.print(f"[bold {color}]段落 #{i+1}[/bold {color}] "
                      f"[{color}]{level_label}[/{color}] "
                      f"(AI嫌疑: {score:.0%})")
        console.print(f"  [dim]{preview}[/dim]")

        # 异常维度
        dims = result.get("dimensions", {})
        ai_dims = [(k, v) for k, v in dims.items() if v["zone"] == "ai"]
        gray_dims = [(k, v) for k, v in dims.items() if v["zone"] == "gray"]

        if ai_dims:
            console.print("  [red]🔴 AI异常维度:[/red]")
            for name, d in ai_dims:
                console.print(f"    {d['description']}: [red]{d['value']}[/red] "
                              f"(差距: {d['gap']})")

        if gray_dims and level != "green":
            console.print("  [yellow]🟡 灰色区间维度:[/yellow]")
            for name, d in gray_dims:
                console.print(f"    {d['description']}: [yellow]{d['value']}[/yellow]")

        # 排名差距
        ranked = result.get("ranked_gaps", [])
        if ranked:
            top3 = ranked[:3]
            console.print("  [bold]📋 优先修复:[/bold]")
            for j, (name, gap, _) in enumerate(top3, 1):
                desc = dims.get(name, {}).get("description", name)
                console.print(f"    {j}. {desc}: 差距 {gap:.2f}")

        console.print()

    # 尾部
    console.print("[dim]─── 诊断完成 ───[/dim]")
    console.print()


def _plain_report(results: list[dict[str, Any]], paragraphs: list) -> None:
    """纯文本报告（无Rich依赖）"""
    print()
    print("=" * 60)
    print("  PaperDiag — AI 特征诊断报告")
    print("=" * 60)
    print(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"段落总数: {len(results)}")
    print()

    for i, (result, para) in enumerate(zip(results, paragraphs)):
        text = para.text if hasattr(para, 'text') else str(para)
        level = result["level"]
        score = result["comprehensive_score"]
        label = LEVEL_LABELS.get(level, level)

        print(f"[段落 #{i+1}] {label} (AI嫌疑: {score:.0%})")
        preview = text[:100] + "..." if len(text) > 100 else text
        print(f"  {preview}")

        dims = result.get("dimensions", {})
        ai_dims = [(k, v) for k, v in dims.items() if v["zone"] == "ai"]
        if ai_dims:
            print("  AI异常维度:")
            for name, d in ai_dims:
                print(f"    {d['description']}: {d['value']} (差距: {d['gap']})")
        print()

    print("─── 诊断完成 ───")


# ============================================================
#  JSON 报告
# ============================================================

def json_report(results: list[dict[str, Any]],
                paragraphs: list,
                rewrite_results: list[dict[str, Any]] | None = None) -> str:
    """生成JSON格式诊断报告"""
    report = {
        "generated_at": datetime.datetime.now().isoformat(),
        "total_paragraphs": len(results),
        "summary": {
            "high_risk": sum(1 for r in results if r["level"] in ("red", "deep_red")),
            "suspicious": sum(1 for r in results if r["level"] == "yellow"),
            "safe": sum(1 for r in results if r["level"] == "green"),
        },
        "paragraphs": [],
    }

    for i, (result, para) in enumerate(zip(results, paragraphs)):
        text = para.text if hasattr(para, 'text') else str(para)
        entry = {
            "index": i,
            "text_preview": text[:200],
            "level": result["level"],
            "comprehensive_score": result["comprehensive_score"],
            "dimensions": result.get("dimensions", {}),
            "ranked_gaps": [
                {"dimension": name, "gap": gap, "zone": zone}
                for name, gap, zone in result.get("ranked_gaps", [])
            ],
        }
        if rewrite_results and i < len(rewrite_results):
            rw = rewrite_results[i]
            entry["rewritten"] = {
                "text_preview": rw.get("rewritten", rw.get("text", ""))[:200],
                "operations": rw.get("operations", []),
            }
        report["paragraphs"].append(entry)

    return json.dumps(report, ensure_ascii=False, indent=2)


# ============================================================
#  HTML 报告
# ============================================================

def html_report(results: list[dict[str, Any]],
                paragraphs: list,
                rewrite_results: list[dict[str, Any]] | None = None) -> str:
    """生成HTML格式诊断报告"""
    import datetime as dt

    level_badges = {
        "green": '<span style="background:#4caf50;color:white;padding:2px 8px;border-radius:4px">安全</span>',
        "yellow": '<span style="background:#ff9800;color:white;padding:2px 8px;border-radius:4px">可疑</span>',
        "red": '<span style="background:#f44336;color:white;padding:2px 8px;border-radius:4px">高风险</span>',
        "deep_red": '<span style="background:#b71c1c;color:white;padding:2px 8px;border-radius:4px">确认AI</span>',
    }

    rows_html = ""
    for i, (result, para) in enumerate(zip(results, paragraphs)):
        text = para.text if hasattr(para, 'text') else str(para)
        level = result["level"]
        score = result["comprehensive_score"]
        dims = result.get("dimensions", {})

        ai_dims_html = ""
        for name, d in dims.items():
            if d["zone"] == "ai":
                ai_dims_html += (
                    f'<tr><td>{d["description"]}</td>'
                    f'<td style="color:red"><b>{d["value"]}</b></td>'
                    f'<td>{d["gap"]}</td></tr>'
                )

        dims_table = ""
        if ai_dims_html:
            dims_table = (
                '<table style="width:100%;border-collapse:collapse;margin-top:8px">'
                '<tr style="background:#f5f5f5"><th>维度</th><th>数值</th><th>差距</th></tr>'
                f'{ai_dims_html}</table>'
            )

        rw_text = ""
        if rewrite_results and i < len(rewrite_results):
            rw = rewrite_results[i]
            rw_preview = rw.get("rewritten", rw.get("text", ""))[:300]
            ops = ", ".join(rw.get("operations", []))
            rw_text = (
                f'<div style="background:#e8f5e9;padding:10px;margin-top:8px;border-radius:4px">'
                f'<b>改写后:</b> {rw_preview}...<br>'
                f'<small>操作: {ops}</small></div>'
            )

        rows_html += f"""
        <div style="border:1px solid #ddd;padding:16px;margin:12px 0;border-radius:8px">
            <h3>段落 #{i+1} {level_badges.get(level, '')} <small>AI嫌疑: {score:.0%}</small></h3>
            <p style="color:#666">{text[:200]}...</p>
            {dims_table}
            {rw_text}
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>PaperDiag — 诊断报告</title>
<style>
body {{ font-family: -apple-system, "Microsoft YaHei", sans-serif; max-width:900px; margin:40px auto; padding:0 20px; color:#333 }}
h1 {{ color:#1a73e8 }}
h3 {{ margin:0 0 8px 0 }}
small {{ color:#999 }}
</style>
</head>
<body>
<h1>📄 PaperDiag — AI 特征诊断报告</h1>
<p>生成时间: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 段落总数: {len(results)}</p>
{rows_html}
<p style="color:#999;text-align:center;margin-top:40px">—— 报告结束 ——</p>
</body>
</html>"""
