"""
结构化内容检测与规则分块 — 弥补 SemanticChunker 对模板化记录的盲区。

核心理念：
    语义分块器（chonkie.SemanticChunker）对连续散文段落效果很好，
    但遇到"名称 | 等级 | IP | 建议"这类高度模板化的短记录行时，
    句法骨架（"漏洞""建议""高危"）对 embedding 向量的贡献权重远超
    具体实体（"Unix"/"nginx"），导致不同类别的记录被判为相似、合并进同一 chunk。

    本模块在分块前先检测文本是否为结构化记录，
    结构化内容走确定性规则分块（按类别或记录数切分），
    非结构化内容保持原有语义分块路径不变。
"""

import logging
import re
from typing import List

from config import (
    STRUCTURED_ROW_FIELD_COUNT,
    STRUCTURED_BLOCK_THRESHOLD,
    MAX_RECORDS_PER_CHUNK,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 结构化行模式 — 根据配置的字段数动态构建正则
# ---------------------------------------------------------------------------
# 每个字段的长度范围: 2-60 字符（覆盖中文名称、英文简写、IP 地址等）
# 字段间用 | 分隔，首尾允许可选空格
_FIELD_PATTERN = r"\s*.{2,60}\s*"
_PIPE_FIELD = r"\|" + _FIELD_PATTERN


def _build_row_pattern(field_count: int) -> re.Pattern:
    """根据字段数动态构建管道分隔记录行的正则模式。

    Args:
        field_count: 期望的字段数量（如 4 字段: 名称|等级|IP|建议）。

    Returns:
        编译好的正则 Pattern。
    """
    if field_count < 2:
        raise ValueError(f"field_count must be >= 2, got {field_count}")
    # 第一个字段 + (field_count-1) 个管道分隔字段
    full = _FIELD_PATTERN + _PIPE_FIELD * (field_count - 1)
    return re.compile(r"^" + full + r"$")


# 模块加载时编译，由 config 的字段数决定
try:
    STRUCTURED_ROW_PATTERN = _build_row_pattern(STRUCTURED_ROW_FIELD_COUNT)
except ValueError as e:
    logger.warning("Invalid STRUCTURED_ROW_FIELD_COUNT, falling back to 4: %s", e)
    STRUCTURED_ROW_PATTERN = _build_row_pattern(4)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_structured_block(text: str, threshold: float = STRUCTURED_BLOCK_THRESHOLD) -> bool:
    """判断一段文本是否主要由结构化记录行组成。

    对每一非空行检测是否匹配管道分隔记录格式。
    命中比例超过 threshold 才判定为结构化块，
    避免正文中偶尔出现一两行像表格的内容被误判。

    Args:
        text: 待检测的文本块。
        threshold: 命中结构化模式的行数占比阈值（0.0-1.0），
                   默认从 config.STRUCTURED_BLOCK_THRESHOLD 读取。

    Returns:
        True 表示应走规则分块，False 表示应走语义分块。
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return False

    hit = sum(1 for l in lines if STRUCTURED_ROW_PATTERN.match(l))
    ratio = hit / len(lines)
    logger.debug(
        "structured_detect: %d/%d lines matched pattern (%.1f%%), threshold=%.1f",
        hit, len(lines), ratio * 100, threshold * 100,
    )
    return ratio >= threshold


def chunk_structured_records(
    text: str,
    max_records: int = MAX_RECORDS_PER_CHUNK,
) -> List[str]:
    """按记录数强制切分，不依赖相似度判断。

    边界确定可预测：每 max_records 条记录为一个 chunk。

    Args:
        text: 结构化记录文本（每行一条记录）。
        max_records: 每个 chunk 最多容纳的记录行数。

    Returns:
        切分后的 chunk 文本列表。
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return []

    chunks = []
    for i in range(0, len(lines), max_records):
        chunks.append("\n".join(lines[i: i + max_records]))

    logger.debug(
        "structured_records: %d lines → %d chunks (max %d/chunk)",
        len(lines), len(chunks), max_records,
    )
    return chunks


def chunk_by_category(
    text: str,
    max_records: int = MAX_RECORDS_PER_CHUNK,
) -> List[str]:
    """按记录名称字段里的类别前缀分组切分。

    策略：
    1. 取每行第一个 | 前的字段作为名称字段
    2. 从名称字段提取类别前缀（第一个 "-" 之前的内容）
    3. 不同类别的记录强制分入不同 chunk
    4. 同一类别下若记录过多，按 max_records 二次切分

    若所有记录都无法解析出类别（全部归入"其他"），
    退回到 chunk_structured_records 按记录数切分。

    Args:
        text: 结构化记录文本（每行一条记录）。
        max_records: 同一类别下每个 chunk 最多容纳的记录数。

    Returns:
        按类别分组切分后的 chunk 文本列表。
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return []

    # 按类别前缀分组
    groups: dict[str, list[str]] = {}
    for line in lines:
        name_field = line.split("|")[0].strip() if "|" in line else line.strip()
        if "-" in name_field:
            category = name_field.split("-")[0].strip()
        else:
            category = "其他"
        groups.setdefault(category, []).append(line)

    # 全部是"其他" → 退回按记录数切分
    if len(groups) == 1 and "其他" in groups:
        logger.debug("chunk_by_category: no category signal found, falling back to record-count chunking")
        return chunk_structured_records(text, max_records)

    # 每类别独立输出，若某类别记录过多则二次切分
    result = []
    for rows in groups.values():
        if len(rows) <= max_records:
            result.append("\n".join(rows))
        else:
            for i in range(0, len(rows), max_records):
                result.append("\n".join(rows[i: i + max_records]))

    logger.debug(
        "chunk_by_category: %d lines → %d groups → %d chunks",
        len(lines), len(groups), len(result),
    )
    return result
