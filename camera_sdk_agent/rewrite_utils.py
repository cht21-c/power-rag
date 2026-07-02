"""
问题重写检测工具 —— 无重依赖（可在测试中独立导入）。

从 agent/graph.py 提取，避免为了测代词检测而引入 langgraph 等重型依赖。
"""

_PRONOUN_PATTERNS = ["这个", "那个", "它", "该设备", "那这个", "那那个", "这些", "那些",
                     "该模块", "该文件", "此设备", "此图", "上一步", "刚才"]


def needs_rewrite(query: str, messages: list) -> bool:
    """判断当前问题是否需要指代消解重写。"""
    if len(messages) < 2:
        return False
    if len(query) <= 8:
        return True
    if any(p in query for p in _PRONOUN_PATTERNS):
        return True
    return False


def build_conversation_summary(messages: list, max_turns: int = 3) -> str:
    """从对话历史构建摘要。"""
    lines = []
    recent = messages[-max_turns * 2:]
    for m in recent:
        role = "用户" if m.get("role") == "user" else "助手"
        content = m.get("content", "")[:200]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
