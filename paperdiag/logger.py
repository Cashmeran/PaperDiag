"""改写日志记录模块：JSONL格式持久化"""

import json
import datetime
from pathlib import Path
from typing import Any


class RewriteLogger:
    """记录每轮改写的详细信息"""

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "rewrite_log.jsonl"
        self.records: list[dict[str, Any]] = []

    def log_paragraph(self,
                      index: int,
                      original: str,
                      rewritten: str,
                      diagnosis_before: dict[str, Any],
                      diagnosis_after: dict[str, Any] | None = None,
                      operations: list[str] | None = None,
                      validation: dict[str, Any] | None = None) -> dict[str, Any]:
        """记录单段改写"""
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "paragraph_index": index,
            "original": original,
            "original_length": len(original),
            "rewritten": rewritten,
            "rewritten_length": len(rewritten),
            "diagnosis_before": {
                "level": diagnosis_before.get("level"),
                "comprehensive_score": diagnosis_before.get("comprehensive_score"),
                "ai_count": diagnosis_before.get("ai_count"),
            },
            "diagnosis_after": {
                "level": diagnosis_after.get("level") if diagnosis_after else None,
                "comprehensive_score": diagnosis_after.get("comprehensive_score") if diagnosis_after else None,
                "ai_count": diagnosis_after.get("ai_count") if diagnosis_after else None,
            },
            "operations": operations or [],
            "validation": validation,
        }
        self.records.append(record)

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return record

    def get_summary(self) -> dict[str, Any]:
        """获取改写会话摘要"""
        if not self.records:
            return {"total": 0, "message": "无改写记录"}

        total_original_chars = sum(r["original_length"] for r in self.records)
        total_rewritten_chars = sum(r["rewritten_length"] for r in self.records)
        modified_paras = sum(1 for r in self.records
                            if r["original"] != r["rewritten"])

        # 诊断分数变化
        score_changes = []
        for r in self.records:
            before = r["diagnosis_before"].get("comprehensive_score", 0) or 0
            after = r["diagnosis_after"].get("comprehensive_score", 0) or 0
            if before > 0:
                score_changes.append((before - after) / before)

        return {
            "total_paragraphs": len(self.records),
            "modified_paragraphs": modified_paras,
            "total_original_chars": total_original_chars,
            "total_rewritten_chars": total_rewritten_chars,
            "char_change_ratio": round(
                (total_rewritten_chars - total_original_chars) / total_original_chars, 4
            ) if total_original_chars > 0 else 0,
            "avg_score_reduction": round(sum(score_changes) / len(score_changes), 4)
            if score_changes else 0,
            "operations_summary": _count_operations(self.records),
        }

    def load_previous(self) -> list[dict[str, Any]]:
        """加载之前的改写记录"""
        if not self.log_file.exists():
            return []
        records = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


def _count_operations(records: list[dict[str, Any]]) -> dict[str, int]:
    """统计各操作的使用次数"""
    counts = {}
    for r in records:
        for op in r.get("operations", []):
            # 提取操作类型（去除参数）
            op_type = op.split("(")[0] if "(" in op else op
            counts[op_type] = counts.get(op_type, 0) + 1
    return counts
