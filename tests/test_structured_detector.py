"""
测试 structured_detector 模块 — 覆盖 5 个回归场景。
"""
import sys
from pathlib import Path

# Ensure camera_sdk_agent is on sys.path
_PROJ = Path(__file__).resolve().parent.parent / "camera_sdk_agent"
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

import pytest
from ingestion.structured_detector import (
    is_structured_block,
    chunk_structured_records,
    chunk_by_category,
    STRUCTURED_ROW_PATTERN,
)


# ============================================================================
# 测试数据构造
# ============================================================================

VULN_REPORT_TEXT = """\
Unix操作系统不支持的版本检测 | 高危 | 172.168.16.5 | 升级到当前支持的Unix操作系统版本
nginx 0.6.x < 1.20.11 字节内存覆盖RCE | 紧急 | 172.18.248.101 | 建议升级到最新版本
Python不支持的版本检测 | 紧急 | 172.18.248.102 | 建议更新Python版本
PHP多个低版本漏洞 | 高危 | 172.18.248.101 | 建议升级PHP版本
Apache多个版本漏洞 | 中危 | 172.18.248.105 | 建议升级Apache版本
OpenSSL多个低版本漏洞 | 高危 | 172.18.248.106 | 建议升级OpenSSL版本\
"""

# With category prefix ("Unix - ...", "nginx - ...") for chunk_by_category
VULN_REPORT_WITH_CATEGORIES = """\
Unix操作系统 - 不支持的版本检测 | 高危 | 172.168.16.5 | 升级到当前支持的Unix操作系统版本
Unix操作系统 - 权限提权漏洞 | 高危 | 172.168.16.6 | 升级内核版本
nginx - 目录遍历漏洞 | 中危 | 10.20.1.8 | 建议升级版本
nginx - 字节内存覆盖RCE | 紧急 | 10.20.1.9 | 建议升级到最新版本
Python - 不支持的版本检测 | 紧急 | 172.18.248.102 | 建议更新Python版本
Python - 依赖库漏洞 | 中危 | 172.18.248.103 | 升级依赖库\
"""

PROSE_TEXT = """\
循环水泵是电厂热力循环中的关键设备，其主要作用是将凝汽器中的凝结水
加压后送入除氧器。循环水泵的正常运行直接关系到机组的安全和经济性。

巡检时应注意检查泵体振动、轴承温度、密封泄漏情况以及润滑油的油位和
油质。如发现异常应及时报告并采取相应措施。

维护周期建议每季度进行一次全面检查，包括叶轮磨损情况、密封环间隙、
联轴器对中精度等项目。\
"""

MIXED_TEXT = """\
根据本次安全评估报告，我们对全厂信息系统进行了全面扫描。

扫描共发现以下主要问题需要关注。

Unix操作系统不支持的版本检测 | 高危 | 172.168.16.5 | 升级到当前支持的Unix操作系统版本

除上述漏洞外，整体安全态势处于可控范围内。建议运维团队按照整改建议
逐项落实修复措施，并在修复完成后进行复测验证。

总体来看，网络安全防护体系基本健全，但仍需持续关注新出现的威胁情报。\
"""


# ============================================================================
# 测试 1: 纯结构化记录块 — 应判定为 True
# ============================================================================

def test_is_structured_block_pure_vuln():
    """纯漏洞报告结构化记录应判定为结构化块"""
    # 所有 6 行都是管道分隔记录
    assert is_structured_block(VULN_REPORT_TEXT, threshold=0.6) is True


def test_is_structured_block_pure_vuln_lower_threshold():
    """即使是较低阈值也应判定为结构化"""
    assert is_structured_block(VULN_REPORT_TEXT, threshold=0.3) is True


def test_pattern_matches_all_vuln_lines():
    """每行漏洞记录都应匹配结构化模式"""
    lines = [l.strip() for l in VULN_REPORT_TEXT.split("\n") if l.strip()]
    for line in lines:
        assert STRUCTURED_ROW_PATTERN.match(line), f"Should match: {line[:50]}"


# ============================================================================
# 测试 2: 纯散文段落 — 应判定为 False
# ============================================================================

def test_is_structured_block_pure_prose():
    """纯散文段落不应判定为结构化块"""
    assert is_structured_block(PROSE_TEXT, threshold=0.6) is False


def test_prose_lines_dont_match_pattern():
    """散文段落不应匹配结构化模式"""
    lines = [l.strip() for l in PROSE_TEXT.split("\n") if l.strip()]
    for line in lines:
        assert not STRUCTURED_ROW_PATTERN.match(line), f"Should NOT match: {line[:50]}"


# ============================================================================
# 测试 3: 混合内容 — 命中率不足 threshold，整体仍走语义分块
# ============================================================================

def test_is_structured_block_mixed_content():
    """混合文本（散文为主，偶尔一行表格式内容）不应判定为结构化"""
    lines = [l.strip() for l in MIXED_TEXT.split("\n") if l.strip()]
    structured_lines = sum(1 for l in lines if STRUCTURED_ROW_PATTERN.match(l))
    total = len(lines)
    # 确认有结构化行但占比很低
    assert structured_lines > 0, "Should have at least one structured line"
    ratio = structured_lines / total
    assert ratio < 0.5, f"Structured ratio {ratio} should be < 0.5 for mixed content"
    # threshold 0.6 不应触发
    assert is_structured_block(MIXED_TEXT, threshold=0.6) is False


# ============================================================================
# 测试 4: chunk_by_category — 有/无类别信号的两种行为
# ============================================================================

def test_chunk_by_category_with_signal():
    """有类别信号时应按类别分组切分"""
    chunks = chunk_by_category(VULN_REPORT_WITH_CATEGORIES)
    assert len(chunks) >= 3, f"Expected >= 3 category groups, got {len(chunks)}"

    # 验证不同类别的记录不在同一个 chunk 中
    for chunk in chunks:
        first_names = [l.split("|")[0].strip() for l in chunk.split("\n")]
        # 同一 chunk 内所有记录的类别前缀应相同
        categories = set()
        for name in first_names:
            cat = name.split("-")[0].strip() if "-" in name else name
            categories.add(cat)
        assert len(categories) == 1, f"Chunk contains mixed categories: {categories}"


def test_chunk_by_category_no_signal_fallback():
    """无类别信号时应退回 chunk_structured_records 按记录数切分"""
    # 无 "-" 的记录，全部归入"其他"
    chunks = chunk_by_category(VULN_REPORT_TEXT, max_records=3)
    assert len(chunks) > 0
    # 每条 chunk 不应超过 max_records 行
    for chunk in chunks:
        lines = [l for l in chunk.split("\n") if l.strip()]
        assert len(lines) <= 3, f"Chunk has {len(lines)} lines, max 3"


# ============================================================================
# 测试 5: chunk_structured_records 按记录数切分
# ============================================================================

def test_chunk_structured_records():
    """按记录数切分，每条 chunk 不超过限制"""
    chunks = chunk_structured_records(VULN_REPORT_TEXT, max_records=2)
    # 6 行 → 3 个 chunk，每 chunk 2 行
    assert len(chunks) == 3
    for chunk in chunks:
        lines = [l for l in chunk.split("\n") if l.strip()]
        assert len(lines) == 2


def test_chunk_structured_records_single_chunk():
    """记录数少于 max_records 时返回单个 chunk"""
    short_text = "Unix漏洞 | 高危 | 10.0.0.1 | 升级"
    chunks = chunk_structured_records(short_text, max_records=10)
    assert len(chunks) == 1
    assert "Unix漏洞" in chunks[0]


# ============================================================================
# 测试 6: 空文本边界
# ============================================================================

def test_empty_text_not_structured():
    """空文本不应判定为结构化"""
    assert is_structured_block("", threshold=0.6) is False
    assert is_structured_block("   \n\n   ", threshold=0.6) is False


def test_empty_text_chunking():
    """空文本切分应返回空列表"""
    assert chunk_structured_records("") == []
    assert chunk_by_category("") == []
