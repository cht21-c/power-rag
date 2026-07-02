"""入库报告数据类"""
import json, time, os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class StepReport:
    step: str
    total: int = 0
    success: int = 0
    failed: int = 0
    failed_items: List[dict] = field(default_factory=list)
    elapsed_sec: float = 0.0

    def add_failure(self, item: str, reason: str):
        self.failed += 1
        self.failed_items.append({"item": item, "reason": reason})


@dataclass
class IngestReport:
    pipeline: str = "ingestion"
    started_at: str = ""
    steps: List[StepReport] = field(default_factory=list)
    total_docs: int = 0
    success_docs: int = 0
    failed_docs: int = 0
    total_chunks: int = 0
    elapsed_sec: float = 0.0

    def add_step(self, name: str) -> StepReport:
        s = StepReport(step=name)
        self.steps.append(s)
        return s

    def save(self, log_dir: Optional[str] = None):
        if log_dir is None:
            project_root = Path(__file__).resolve().parent.parent
            log_dir = str(project_root / "logs" / "ingest")
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        fpath = os.path.join(log_dir, f"ingest_{ts}.json")
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        return fpath

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  入库流水线报告",
            "=" * 60,
            f"  文档总数: {self.total_docs}  成功: {self.success_docs}  失败: {self.failed_docs}",
            f"  生成 chunk: {self.total_chunks}  总耗时: {self.elapsed_sec:.1f}s",
            "-" * 60,
        ]
        for s in self.steps:
            status = "OK" if s.failed == 0 else f"{s.failed} ERRORS"
            lines.append(f"  [{status:>10s}] {s.step:20s}  "
                         f"total={s.total} success={s.success}  {s.elapsed_sec:.1f}s")
            for fi in s.failed_items:
                lines.append(f"           FAIL: {fi['item'][:60]}")
                lines.append(f"                  {fi['reason'][:80]}")
        lines.append("=" * 60)
        return "\n".join(lines)
