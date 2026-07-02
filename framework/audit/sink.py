"""
审计日志落地方 — AuditSink 抽象 + FileAuditSink + DBAuditSink(预留)

设计原则：
- 日志写入失败不能阻塞主流程（fail-open）
- 抽象接口支持切换后端
"""

import abc
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AuditSink(abc.ABC):
    """审计日志落地方抽象。"""
    @abc.abstractmethod
    def write(self, record: Dict[str, Any]) -> None:
        ...


class FileAuditSink(AuditSink):
    """按天切割的 JSON Lines 文件落地。”

    文件路径: logs/audit/YYYY-MM-DD.jsonl
    每条记录一行 JSON，便于 grep / jq 查询。
    """

    def __init__(self, log_dir: Optional[str] = None):
        if log_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            log_dir = str(project_root / "logs" / "audit")
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  # 线程安全写入

    def write(self, record: Dict[str, Any]) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        file_path = self._log_dir / f"{today}.jsonl"
        line = json.dumps(record, ensure_ascii=False) + "\n"
        try:
            with self._lock:
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception as e:
            logger.error("Audit log write failed (fail-open): %s", e)


class DBAuditSink(AuditSink):
    """数据库落地写（预留）。

    后续实现 SQLite / MySQL 写入，接口与 FileAuditSink 一致。
    """

    def __init__(self, db_url: str):
        self._db_url = db_url
        logger.info("DBAuditSink reserved (not yet implemented): %s", db_url)

    def write(self, record: Dict[str, Any]) -> None:
        # 预留实现
        logger.warning("DBAuditSink.write() not implemented, falling back to FileAuditSink")
