"""
图纸语义匹配 — 用 LLM 语义比对替代 SQL LIKE。

全量取图纸库名称列表，交给 LLM 做语义匹配，
解决用户口语化表达（"那个泵的图""接线的东西"）
无法通过字符串匹配命中的问题。
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is on path for framework/ imports
_PROJECT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from framework.llm_utils.robust_parser import DrawingMatchResult, robust_parse

logger = logging.getLogger(__name__)


def match_drawing_by_llm(
    llm,
    drawing_entity: str,
    original_query: str,
) -> tuple[list[int], str]:
    """使用 LLM 语义匹配图纸库中的图纸。

    从 mock_db 取全量图纸名列表，编号后传给 LLM 做语义比对。
    不依赖任何关键词或字符串匹配。

    Args:
        llm: ChatOpenAI 实例（已绑定工具）。
        drawing_entity: LLM 从用户问题中提取的设备/图纸名称。
        original_query: 用户原始问题。

    Returns:
        (matched_ids: list[int], reasoning: str)
        - matched_ids: 匹配到的图纸 ID 列表（0-indexed from DB）
        - reasoning: LLM 给出的判断理由
    """
    from mock_db import get_all_drawing_names
    from .prompts import DRAWING_MATCH_PROMPT

    all_drawings = get_all_drawing_names()

    if not all_drawings:
        logger.warning("[drawing_match] No drawings in database")
        return ([], "图纸库为空，无可用图纸进行匹配。")

    # 构建编号列表
    lines = []
    id_map: dict[int, int] = {}  # 编号 → 数据库 id
    for idx, d in enumerate(all_drawings):
        num = idx + 1  # 1-indexed for LLM readability
        lines.append(f"{num}. {d['drawing_name']}")
        id_map[num] = d["id"]

    drawing_list = "\n".join(lines)

    prompt = DRAWING_MATCH_PROMPT.format(
        drawing_entity=drawing_entity,
        original_query=original_query,
        drawing_list=drawing_list,
    )

    logger.info("[drawing_match] Matching entity=%r against %d drawings",
                 drawing_entity, len(all_drawings))

    try:
        response = llm.invoke(prompt)
        result, _ = robust_parse(llm, response.content, DrawingMatchResult)

        # 编号 → 数据库 id
        db_ids = [id_map.get(n, n) for n in result.matched_ids]

        logger.info("[drawing_match] Result: matched_ids=%s, reasoning=%s",
                     db_ids, result.reasoning[:80])
        return (db_ids, result.reasoning)

    except Exception as e:
        logger.error("[drawing_match] LLM match failed: %s", e)
        return ([], f"LLM 匹配失败: {e}")


def resolve_matched_drawings(matched_ids: list[int]) -> list[dict]:
    """根据匹配到的数据库 ID 列表，查询完整图纸信息。

    Args:
        matched_ids: 数据库 drawing.id 列表。

    Returns:
        完整图纸信息列表。
    """
    from mock_db import query_all_drawings

    if not matched_ids:
        return []

    all_drawings = query_all_drawings()
    id_to_drawing = {d["id"]: d for d in all_drawings}

    results = []
    for db_id in matched_ids:
        if db_id in id_to_drawing:
            results.append(id_to_drawing[db_id])

    return results
