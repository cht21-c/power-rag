#!/usr/bin/env python3
"""
审计日志查询工具

支持按 trace_id 或时间范围检索全链路审计记录。

用法:
    # 按 trace_id 查询完整请求链路
    python scripts/query_audit_log.py --trace-id abc123

    # 按时间范围查询
    python scripts/query_audit_log.py --from 2026-07-01 --to 2026-07-02

    # 按用户查询
    python scripts/query_audit_log.py --user-id admin01 --from 2026-07-01

    # 只看错误
    python scripts/query_audit_log.py --status error --from 2026-07-01
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path


def find_log_dir() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    log_dir = project_root / "logs" / "audit"
    if not log_dir.exists():
        print(f"ERROR: Audit log directory not found: {log_dir}", file=sys.stderr)
        sys.exit(1)
    return log_dir


def date_range(start_str: str, end_str: str) -> list:
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    dates = []
    d = start
    while d <= end:
        dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return dates


def query_logs(log_dir: Path, trace_id: str = "",
               user_id: str = "", status: str = "",
               dates: list = None) -> list:
    results = []
    if dates is None:
        dates = [datetime.now().strftime("%Y-%m-%d")]

    for date_str in dates:
        fpath = log_dir / f"{date_str}.jsonl"
        if not fpath.exists():
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if trace_id and rec.get("trace_id") != trace_id:
                    continue
                if user_id and rec.get("user_id") != user_id:
                    continue
                if status and rec.get("status") != status:
                    continue
                results.append(rec)
    return results


def format_timeline(records: list) -> str:
    """格式化输出一次请求的全链路时间线。"""
    if not records:
        return "No records found."

    records.sort(key=lambda r: r.get("timestamp", ""))
    trace_id = records[0].get("trace_id", "unknown")
    user_id = records[0].get("user_id", "unknown")

    lines = [
        "=" * 70,
        f"  Trace ID: {trace_id} | User: {user_id}",
        "=" * 70,
    ]

    for r in records:
        node = r.get("node_name", "?")
        status = r.get("status", "?")
        latency = r.get("latency_ms", 0)
        route = r.get("route_decision", "")
        ts = r.get("timestamp", "")[-12:]  # just time part

        status_icon = "OK" if status == "success" else "!!"
        route_info = f" -> [{route}]" if route else ""
        lines.append(
            f"  [{ts}] {status_icon:2s}  {node:20s}  {latency:8.1f}ms{route_info}"
        )

        in_sum = r.get("input_summary", "")
        out_sum = r.get("output_summary", "")
        if in_sum and len(in_sum) > 120:
            lines.append(f"         in : {in_sum[:120]}...")
        elif in_sum:
            lines.append(f"         in : {in_sum}")
        if out_sum and len(out_sum) > 120:
            lines.append(f"         out: {out_sum[:120]}...")
        elif out_sum:
            lines.append(f"         out: {out_sum}")

    total_latency = sum(r.get("latency_ms", 0) for r in records)
    lines.append("-" * 70)
    lines.append(f"  Total latency: {total_latency:.1f}ms  |  Nodes: {len(records)}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="审计日志查询工具")
    parser.add_argument("--trace-id", help="按 trace_id 查询")
    parser.add_argument("--user-id", help="按用户 ID 查询")
    parser.add_argument("--status", help="按状态过滤 (success/error)")
    parser.add_argument("--from", dest="from_date", help="开始日期 YYYY-MM-DD")
    parser.add_argument("--to", dest="to_date", help="结束日期 YYYY-MM-DD")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    args = parser.parse_args()

    log_dir = find_log_dir()

    dates = None
    if args.from_date or args.to_date:
        start = args.from_date or datetime.now().strftime("%Y-%m-%d")
        end = args.to_date or start
        dates = date_range(start, end)
    else:
        dates = [datetime.now().strftime("%Y-%m-%d")]

    results = query_logs(
        log_dir,
        trace_id=args.trace_id or "",
        user_id=args.user_id or "",
        status=args.status or "",
        dates=dates,
    )

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        if args.trace_id:
            print(format_timeline(results))
        else:
            print(f"Found {len(results)} records:")
            for r in results:
                print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
