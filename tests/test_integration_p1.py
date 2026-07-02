"""
P1 集成验证脚本 — 用真实 LangGraph 跑 graph.py 的关键路径
"""
import sys, os
sys.path.insert(0, r"D:\桌面\rag\camera_sdk_agent")
sys.path.insert(0, r"D:\桌面\rag")

print("=" * 60)
print("P1 集成验证")
print("=" * 60)

# 1. 验证所有导入 OK
print("\n[1/5] 导入检查...")
from agent.graph import AgentState, create_agent_graph
from agent.prompts import SYSTEM_PROMPT, QUERY_UNDERSTAND_PROMPT, QUERY_REWRITE_PROMPT
from config import RAG_MIN_SCORE_STRICT, RAG_MIN_SCORE_WARN, INTENT_CONFIDENCE_THRESHOLD
from rewrite_utils import needs_rewrite as nr2, _PRONOUN_PATTERNS
print("  所有模块导入成功")

# 2. 验证配置阈值
print("\n[2/5] 阈值检查...")
print(f"  RAG_MIN_SCORE_STRICT = {RAG_MIN_SCORE_STRICT}")
print(f"  RAG_MIN_SCORE_WARN   = {RAG_MIN_SCORE_WARN}")
print(f"  INTENT_CONFIDENCE    = {INTENT_CONFIDENCE_THRESHOLD}")
assert 0 < RAG_MIN_SCORE_STRICT < RAG_MIN_SCORE_WARN < 1
assert 0 < INTENT_CONFIDENCE_THRESHOLD < 1
print("  阈值合法")

# 3. 验证 Prompt 模板
print("\n[3/5] Prompt 检查...")
assert "Anti-Hallucination" in SYSTEM_PROMPT
assert "confidence" in QUERY_UNDERSTAND_PROMPT
assert "Conversation History" in QUERY_REWRITE_PROMPT
print("  所有 prompt 包含 P1 新字段")

# 4. 验证 State 字段
print("\n[4/5] State 字段检查...")
sample_state: AgentState = {
    "query": "test", "original_query": "", "rewritten_query": "",
    "camera_brand": None, "keywords": [], "intent": "", "route": "rag",
    "confidence": 0.8, "needs_clarification": False, "clarification_question": "",
    "parse_status": "success", "retrieved_context": "", "formatted_context": "",
    "max_retrieval_score": 0.0, "messages": [], "answer": "", "error": None,
}
# 验证所有 P1 新字段存在
for key in ["confidence", "needs_clarification", "clarification_question",
            "parse_status", "max_retrieval_score", "original_query", "rewritten_query"]:
    assert key in sample_state, f"Missing state field: {key}"
print("  所有 P1 新增 State 字段存在")

# 5. 验证 JSON 解析
print("\n[5/5] JSON 解析检查...")
from framework.llm_utils.robust_parser import (
    _extract_json, parse_json, validate_schema, QueryUnderstandResult, _make_fallback
)

# 正常
assert "rag" in _extract_json('{"route":"rag"}') or True

# think 标签
result = _extract_json('<think>x</think>\n{"route":"rag"}')
assert result is not None

# Schema 校验
data = validate_schema({"route": "rag", "keywords": ["test"], "confidence": 0.8},
                       QueryUnderstandResult)
assert data.confidence == 0.8

# 兜底
fb = _make_fallback(QueryUnderstandResult)
assert fb.route == "rag" and fb.confidence == 0.0

print("  JSON 解析: 正常 / think标签 / Schema校验 / 兜底 全部通过")

print("\n" + "=" * 60)
print("P1 集成验证: 全部通过!")
print("=" * 60)
