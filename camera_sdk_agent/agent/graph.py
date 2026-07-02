"""
LangGraph state graph for the Power Plant RAG Agent (P1 质量加固版).

变更:
- query_understand: 问题重写 + schema校验 + 置信度反问
- llm_generate: 分数阈值拒答
- mysql_query: 模糊匹配置信度标注

Flow:
    query_understand → route_decision → (mysql_query | rag_retrieve) → llm_generate → END
    新增分支: query_understand → (clarification_return) [低置信反问，不调用LLM生成]
"""

import json, logging, re, sys
from datetime import date
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler

# P2: 模型路由 + 记忆压缩 + 工具注册
try:
    from infra_ai.model_router.model_router import ModelRouter, create_default_router
except ImportError:
    ModelRouter = None; create_default_router = None
try:
    from framework.memory.compressor import ConversationCompressor
except ImportError:
    ConversationCompressor = None
try:
    from framework.tool_registry.registry import get_registered_tools
except ImportError:
    get_registered_tools = None

from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    RAG_MIN_SCORE_STRICT, RAG_MIN_SCORE_WARN, INTENT_CONFIDENCE_THRESHOLD,
    VERIFY_ENABLED, LLM_MAX_CONTEXT_TOKENS,
)
from .prompts import SYSTEM_PROMPT, QUERY_UNDERSTAND_PROMPT, QUERY_REWRITE_PROMPT
from .tools import rag_search, format_context_for_llm, set_retriever, ocr_image
from .drawing_match import match_drawing_by_llm, resolve_matched_drawings
from rewrite_utils import needs_rewrite, build_conversation_summary, _PRONOUN_PATTERNS

# P1: 鲁棒JSON解析
try:
    from framework.llm_utils.robust_parser import robust_parse, QueryUnderstandResult
except ImportError:
    QueryUnderstandResult = None
    robust_parse = None

# P0: 审计日志
try:
    from framework.audit.tracer import get_tracer, audit_node
except ImportError:
    def audit_node(*a, **kw): return lambda f: f
    def get_tracer(): return None

logger = logging.getLogger(__name__)

# P2: 模块级持有 ModelRouter 和 Compressor（由 create_agent_graph 注入）
_model_router = None
_compressor = None

# ============================================================================
# Streaming callback handlers (unchanged from P0)
# ============================================================================

class _StreamToStdout(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        sys.stdout.write(token); sys.stdout.flush()

class _StreamToChainlit(BaseCallbackHandler):
    def __init__(self):
        super().__init__(); self._msg: Any = None; self._loop: Any = None
    def set_message(self, msg: Any, loop: Any) -> None:
        self._msg = msg; self._loop = loop
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self._msg is not None and self._loop is not None:
            import asyncio
            asyncio.run_coroutine_threadsafe(self._msg.stream_token(token), self._loop)

# ============================================================================
# State definition — P1: 新增字段
# ============================================================================

class AgentState(TypedDict):
    query: str
    original_query: str               # P1: 重写前的原始问题
    rewritten_query: str              # P1: 重写后的问题
    camera_brand: Optional[str]
    intent: str
    route: str
    confidence: float                 # P1: 意图置信度 0-1
    understood_intent: str            # P2: LLM 复述的意图（排查首选字段）
    route_source: str                 # P2: 路由来源（"llm"/"llm_low_confidence"/"verify_divergence"）
    needs_clarification: bool         # P1: 是否需要反问用户
    clarification_question: str       # P1: 反问文本
    clarify_options: list[str]        # P2: ambiguous 时的澄清方向
    parse_status: str                 # P1: JSON解析状态 (success/retry_success/fallback)
    retrieved_context: str
    formatted_context: str
    max_retrieval_score: float        # P1: 检索最高分（用于阈值判断）
    drawing_match_level: str          # P2: 图纸匹配级别
    drawing_match_reasoning: str      # P2: LLM 图纸匹配的判断理由
    verify_first_judgment: str        # P2: 二次验证原始判断
    verify_second_judgment: str       # P2: 二次验证复核判断
    messages: List[Dict[str, str]]
    answer: str
    error: Optional[str]

# ============================================================================
# P1: 代词检测 + 问题重写
# ============================================================================



def _rewrite_query(llm, query: str, messages: list) -> str:
    """调用LLM进行指代消解重写。"""
    try:
        history = build_conversation_summary(messages)
        prompt = QUERY_REWRITE_PROMPT.format(conversation_history=history, query=query)
        response = llm.invoke(prompt)
        rewritten = response.content.strip()
        if rewritten and len(rewritten) > 2:
            logger.info("[query_rewrite] '%s' → '%s'", query[:40], rewritten[:80])
            return rewritten
    except Exception as e:
        logger.warning("[query_rewrite] Failed: %s", e)
    return query

# ============================================================================
# P1: JSON解析兼容层
# ============================================================================

def _parse_classification(llm, raw: str) -> tuple:
    """解析LLM意图分类输出，返回 (parsed_dict, parse_status)。"""
    if robust_parse and QueryUnderstandResult:
        result, status = robust_parse(llm, raw, QueryUnderstandResult)
        return (result.model_dump(), status)
    else:
        # 回退到旧的 _robust_json_parse
        parsed = _robust_json_parse(raw)
        return (parsed, "success" if parsed.get("intent") else "fallback")

def _robust_json_parse(raw: str) -> dict:
    """旧版JSON解析（pydantic不可用时的回退）。"""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m: raw = m.group(1).strip()
    m = re.search(r"\{[\s\S]*?\}", raw)
    if m:
        try: return json.loads(m.group())
        except json.JSONDecodeError: pass
    result = {"route": "rag", "keywords": [], "intent": "other", "confidence": 0.0}
    if re.search(r"mysql|图纸|drawing|blueprint|图档|下载|调取", raw, re.IGNORECASE):
        result["route"] = "mysql"; result["intent"] = "drawing_lookup"
    return result

# ============================================================================
# Node: query_understand — P2 LLM 语义路由（纯 LLM，无关键词/正则）
# ============================================================================

def _build_history_context(state: AgentState) -> str:
    """构建传给路由 LLM 的历史上下文（含上一轮澄清状态）。"""
    messages = state.get("messages", [])
    summary = build_conversation_summary(messages) if messages else ""

    # 如果上一轮是 clarify 状态，附加已有理解信息
    extra = []
    prev_intent = state.get("understood_intent", "")
    if prev_intent:
        extra.append(f"上一轮理解的意图：{prev_intent}")
    prev_options = state.get("clarify_options", [])
    if prev_options:
        extra.append(f"上一轮给出的澄清选项：{'；'.join(prev_options)}")
    prev_clarify = state.get("clarification_question", "")
    if prev_clarify:
        extra.append(f"上一轮反问内容：{prev_clarify}")

    if extra:
        summary = summary + "\n" + "\n".join(extra) if summary else "\n".join(extra)

    return f"\n\n## 对话历史\n{summary}" if summary else ""


def _verify_drawing_route(llm, query: str, understood_intent: str) -> tuple[str, str]:
    """二次验证：用不同角度重新判断用户是否在找图纸。

    Returns:
        (verdict: "是"|"否"|"不确定", raw_response: str)
    """
    from .prompts import VERIFY_PROMPT
    try:
        prompt = VERIFY_PROMPT.format(query=query, understood_intent=understood_intent)
        response = llm.invoke(prompt)
        raw = response.content.strip()
        if "是" == raw[:1]:
            return ("是", raw)
        elif "否" == raw[:1]:
            return ("否", raw)
        else:
            return ("不确定", raw)
    except Exception as e:
        logger.error("[verify] Failed: %s", e)
        return ("不确定", str(e))


def _create_query_understand_node(llm: ChatOpenAI):
    def query_understand(state: AgentState) -> AgentState:
        query = state.get("query", "")
        messages = state.get("messages", [])
        state["original_query"] = query
        state.setdefault("confidence", 0.0)
        state.setdefault("needs_clarification", False)
        state.setdefault("clarification_question", "")
        state.setdefault("clarify_options", [])
        state.setdefault("parse_status", "success")
        state.setdefault("understood_intent", "")
        state.setdefault("route_source", "")
        state.setdefault("verify_first_judgment", "")
        state.setdefault("verify_second_judgment", "")

        # ── P1: 问题重写（指代消解） ──
        if needs_rewrite(query, messages):
            logger.info("[query_understand] Rewriting pronoun reference...")
            rewritten = _rewrite_query(llm, query, messages)
            if rewritten and rewritten != query:
                state["rewritten_query"] = rewritten
                query = rewritten
                state["query"] = rewritten
        else:
            state["rewritten_query"] = ""

        logger.info("[query_understand] Analyzing: %s", query[:100])

        # ── 构建历史上下文（含上一轮澄清状态） ──
        history_context = _build_history_context(state)

        # ── 单一入口：LLM 语义路由 ──
        try:
            from framework.llm_utils.robust_parser import QueryUnderstandResultV2, robust_parse

            prompt = QUERY_UNDERSTAND_PROMPT.format(query=query, history_context=history_context)
            response = llm.invoke(prompt)
            parsed, parse_status = robust_parse(llm, response.content, QueryUnderstandResultV2)

            state["parse_status"] = parse_status
            state["understood_intent"] = parsed.understood_intent
            state["confidence"] = parsed.confidence
            state["clarify_options"] = parsed.clarify_options

            # 路由映射
            route = parsed.route  # "drawing" | "knowledge_qa" | "ambiguous"

            # ── ambiguous 或低置信度 → clarify ──
            if route == "ambiguous" or parsed.confidence < INTENT_CONFIDENCE_THRESHOLD:
                state["route"] = "clarify"
                state["route_source"] = "llm_low_confidence"
                state["needs_clarification"] = True
                options = parsed.clarify_options or ["查询技术文档/操作规程", "查找工程图纸"]
                state["clarification_question"] = (
                    f"我不太确定您的问题方向，可能的方向有：\n"
                    f"1) {options[0] if len(options) > 0 else '查询技术文档'}\n"
                    f"2) {options[1] if len(options) > 1 else '查找工程图纸'}\n\n"
                    f"请回复帮助我确认。"
                )
                state["answer"] = state["clarification_question"]
                state.setdefault("messages", [])
                state["messages"].append({"role": "user", "content": query})
                state["intent"] = "other"
                logger.info("[query_understand] → clarify (route=%s, confidence=%.2f)", route, parsed.confidence)
                return state

            if route == "drawing":
                state["route"] = "mysql"
                state["intent"] = "drawing_lookup"
                state["route_source"] = "llm"

                # ── 二次验证（P2） ──
                if VERIFY_ENABLED and parsed.confidence >= INTENT_CONFIDENCE_THRESHOLD:
                    v_result, v_raw = _verify_drawing_route(
                        llm, query, parsed.understood_intent
                    )
                    state["verify_first_judgment"] = parsed.understood_intent
                    state["verify_second_judgment"] = v_raw
                    if v_result in ("否", "不确定"):
                        logger.warning("[query_understand] Verify divergence: first=%s, second=%s",
                                       parsed.understood_intent, v_result)
                        state["route"] = "clarify"
                        state["route_source"] = "verify_divergence"
                        state["needs_clarification"] = True
                        state["clarification_question"] = (
                            f"您的意思是想要查找图纸/接线图等图形资料吗？请确认。"
                        )
                        state["answer"] = state["clarification_question"]
                        state.setdefault("messages", [])
                        state["messages"].append({"role": "user", "content": query})
                        return state
            else:
                state["route"] = "rag"
                state["intent"] = parsed.understood_intent
                state["route_source"] = "llm"

        except Exception as e:
            logger.warning("[query_understand] LLM parse failed: %s, falling back to rag", e)
            state.setdefault("camera_brand", None)
            state["intent"] = "other"
            state["route"] = "rag"
            state["confidence"] = 0.0
            state["route_source"] = "llm"
            state["parse_status"] = "fallback"
            state["understood_intent"] = f"解析失败: {e}"

        state.setdefault("messages", [])
        state["messages"].append({"role": "user", "content": query})
        logger.info("[query_understand] route=%s, source=%s, confidence=%.2f, intent=%s",
                     state["route"], state["route_source"], state.get("confidence", 0),
                     state.get("understood_intent", "")[:60])
        return state

    return query_understand

# ============================================================================
# Node: route_decision (unchanged)
# ============================================================================

def route_decision_node(state: AgentState) -> AgentState:
    route = state.get("route", "rag")
    logger.info("[route_decision] Routing to: %s", route)
    return state

# ============================================================================
# P1: 条件路由 —— 低置信反问直接返回，不进入 mysql/rag
# ============================================================================

def _route_condition(state: AgentState) -> str:
    """处理三种情况：clarify | mysql | rag"""
    route = state.get("route", "rag")
    if route == "clarify" or state.get("needs_clarification"):
        return "clarification_return"
    return "mysql_query" if route == "mysql" else "rag_retrieve"

def clarification_return_node(state: AgentState) -> AgentState:
    """反问节点：直接把反问文本作为答案返回，不调用LLM。"""
    logger.info("[clarification] Returning clarification question")
    state["answer"] = state.get("clarification_question", "请明确您的问题方向。")
    return state

# ============================================================================
# Node: mysql_query — P2: LLM 语义图纸匹配
# ============================================================================

def _create_mysql_query_node(llm_for_match=None):
    def mysql_query(state: AgentState) -> AgentState:
        query = state.get("query", "")
        drawing_entity = state.get("understood_intent", query)
        logger.info("[mysql_query] LLM matching entity=%r for query=%s", drawing_entity, query[:60])

        try:
            from .drawing_match import match_drawing_by_llm, resolve_matched_drawings

            matched_ids, reasoning = match_drawing_by_llm(
                llm=llm_for_match,
                drawing_entity=drawing_entity,
                original_query=query,
            )
            state["drawing_match_reasoning"] = reasoning

            if matched_ids:
                results = resolve_matched_drawings(matched_ids)
                state["drawing_match_level"] = "llm_exact" if len(results) == 1 else "llm_multi"

                if len(results) == 1:
                    # 恰好一个 → 正常返回
                    r = results[0]
                    context = (
                        f"查询到以下图纸记录：\n\n"
                        f"1. 图纸名称：{r['drawing_name']}\n"
                        f"   文件名：{r['file_name']}\n"
                        f"   下载地址：{r['url']}\n"
                        f"   所属设备：{r['equipment_id']}\n\n"
                        f"匹配理由：{reasoning}"
                    )
                    state["max_retrieval_score"] = 0.9
                else:
                    # 多个候选 → 设置 clarify，不走 llm_generate
                    lines = [
                        f"⚠️ LLM 判断以下 {len(results)} 条图纸可能符合您的需求，请确认：\n"
                    ]
                    for i, r in enumerate(results, 1):
                        lines.append(
                            f"{i}. 图纸名称：{r['drawing_name']}\n"
                            f"   文件名：{r['file_name']}\n"
                            f"   下载地址：{r['url']}\n"
                            f"   所属设备：{r['equipment_id']}"
                        )
                    lines.append(f"\n匹配理由：{reasoning}")
                    lines.append("\n请回复编号或精确的图纸名称。")
                    context = "\n\n".join(lines)
                    state["max_retrieval_score"] = 0.5
                    # 多候选走 clarify 而不是正常答案生成
                    state["needs_clarification"] = True
                    state["clarification_question"] = context
                    state["answer"] = context
            else:
                # 无匹配 → fallback RAG
                logger.info("[mysql_query] LLM returned no match, falling back to RAG")
                state["drawing_match_level"] = "llm_miss"
                try:
                    rag_json = rag_search.invoke({
                        "query": query,
                        "camera_brand": state.get("camera_brand"),
                        "top_k": 5,
                    })
                    context = format_context_for_llm(rag_json)
                    try:
                        rag_data = json.loads(rag_json)
                        state["max_retrieval_score"] = rag_data.get("max_score", 0)
                    except Exception:
                        state["max_retrieval_score"] = 0
                except Exception:
                    context = "未找到相关图纸，也未检索到相关技术文档。请确认查询内容。"
                    state["max_retrieval_score"] = 0

            state["retrieved_context"] = json.dumps(
                {
                    "source": "mysql",
                    "results": results if matched_ids else [],
                    "drawing_match_level": state["drawing_match_level"],
                    "drawing_match_reasoning": reasoning,
                },
                ensure_ascii=False,
            )
            state["formatted_context"] = context
        except Exception as e:
            logger.error("[mysql_query] Error: %s", e)
            state["retrieved_context"] = json.dumps({"error": str(e)})
            state["formatted_context"] = f"图纸查询失败: {e}"
            state["max_retrieval_score"] = 0
            state["drawing_match_level"] = "error"
            state["drawing_match_reasoning"] = str(e)
        return state
    return mysql_query

# ============================================================================
# Node: rag_retrieve — P1: 记录最高分
# ============================================================================

def rag_retrieve_node(state: AgentState) -> AgentState:
    query = state.get("query", "")
    camera_brand = state.get("camera_brand")
    logger.info("[rag_retrieve] Searching: %s", query[:100])
    try:
        retrieved_json = rag_search.invoke({
            "query": query, "camera_brand": camera_brand, "top_k": 5,
        })
        state["retrieved_context"] = retrieved_json
        state["formatted_context"] = format_context_for_llm(retrieved_json)
        try:
            data = json.loads(retrieved_json)
            state["max_retrieval_score"] = data.get("max_score", 0)
            logger.info("[rag_retrieve] Retrieved %d chunks, max_score=%.4f",
                         data.get("count", 0), state["max_retrieval_score"])
        except Exception:
            state["max_retrieval_score"] = 0
    except Exception as e:
        logger.error("[rag_retrieve] Error: %s", e)
        state["retrieved_context"] = json.dumps({"error": str(e), "results": []})
        state["formatted_context"] = "Retrieval failed."
        state["max_retrieval_score"] = 0
    return state

# ============================================================================
# Node: llm_generate — P1: 分数阈值拒答
# ============================================================================

def _create_llm_generate_node(llm: ChatOpenAI):
    def _estimate_tokens(text: str) -> int:
        chinese = len(re.findall(r'[一-鿿]', text))
        return int(chinese * 1.5 + (len(text) - chinese) * 0.3)

    def _trim_messages_by_token(messages, system_prompt, context, max_tokens=LLM_MAX_CONTEXT_TOKENS):
        """截断消息历史，始终保留最后一条（当前用户问题）。"""
        overhead = _estimate_tokens(system_prompt) + _estimate_tokens(context)
        budget = max(0, max_tokens - overhead)
        kept, used = [], 0
        for i, msg in enumerate(reversed(messages)):
            t = _estimate_tokens(msg.get("content", ""))
            if used + t > budget:
                # 确保至少保留最后一条用户消息
                if i == 0 or (not kept):
                    kept.insert(0, msg)
                break
            kept.insert(0, msg); used += t
        return kept

    def llm_generate(state: AgentState) -> AgentState:
        query = state.get("query", "")
        context = state.get("formatted_context", "")
        max_score = state.get("max_retrieval_score", 0)
        current_date_str = date.today().isoformat()

        # ── 早退：mysql_query 已设置 clarify（多候选歧义） ──
        if state.get("needs_clarification") and state.get("answer"):
            logger.info("[llm_generate] Skipped — clarify answer already set by mysql_query")
            state.setdefault("messages", [])
            state["messages"].append({"role": "assistant", "content": state["answer"]})
            return state

        # P2: 记忆压缩
        messages = state.get("messages", [])
        state["_compressed"] = False
        if _compressor and _compressor.needs_compression(messages):
            logger.info("[llm_generate] Triggering memory compression")
            try:
                messages = _compressor.compress(messages)
                state["messages"] = messages
                state["_compressed"] = True
            except Exception as e:
                logger.error("[llm_generate] Compression failed: %s", e)
                messages = _compressor.hard_truncate(messages)
                state["messages"] = messages

        logger.info("[llm_generate] route=%s, max_score=%.4f", state.get("route", "rag"), max_score)

        # ── P1 Goal 1: 空上下文拒答 ──
        NO_CONTEXT_SIGNALS = ["Retrieval failed", "No relevant", "no_result"]
        context_empty = not context.strip() or any(s in context for s in NO_CONTEXT_SIGNALS)
        if context_empty:
            state["answer"] = (
                f"抱歉，知识库中未找到与「{query[:30]}」相关的内容。\n\n"
                "**建议：** 请尝试更换关键词，或联系技术支持。"
            )
            state["error"] = "no_context"
            return state

        # P2: 使用 ModelRouter（容错降级）
        _router = _model_router if _model_router else None

        # ── P1 Goal 1: 分数阈值拒答（不调用LLM） ──
        if max_score < RAG_MIN_SCORE_STRICT:
            state["answer"] = (
                f"⚠️  未检索到可靠依据\n\n"
                f"关于「{query[:40]}」，当前知识库中未找到匹配度足够高的参考资料"
                f"（最高相似度 {max_score:.4f}，低于阈值 {RAG_MIN_SCORE_STRICT}）。\n\n"
                f"**建议：**\n"
                f"- 重新描述您的问题，使用更具体的关键词\n"
                f"- 如有图纸编号或设备编号，请提供\n"
                f"- 联系运维部门获取最新技术资料"
            )
            state["error"] = "low_score_reject"
            logger.info("[llm_generate] Score %.4f < %.4f, rejecting", max_score, RAG_MIN_SCORE_STRICT)
            return state

        # ── P1 Goal 1: 低置信标注 ──
        if max_score < RAG_MIN_SCORE_WARN:
            prefix = "以下信息置信度较低，请以现场实际情况为准\n\n"
            context = prefix + context

        # ── 正常生成 ──
        try:
            system_msg = SYSTEM_PROMPT.format(current_date=current_date_str, retrieved_context=context)
            messages = state.get("messages", [])
            recent_messages = _trim_messages_by_token(messages, system_msg, context)

            if _router:
                try:
                    answer_text = _router.generate([
                        {"role": "system", "content": system_msg},
                        *recent_messages,
                    ])
                    state["answer"] = answer_text
                    state.setdefault("messages", [])
                    state["messages"].append({"role": "assistant", "content": answer_text})
                    logger.info("[llm_generate] Answer via ModelRouter (%d chars)", len(answer_text))
                    return state
                except Exception as e:
                    logger.error("[llm_generate] ModelRouter failed: %s", e)
                    state["answer"] = f"❌ 模型服务暂时不可用：{e}"
                    state["error"] = str(e)
                    return state

            # Fallback: 直接调用 LLM
            response = llm.invoke([
                {"role": "system", "content": system_msg},
                *recent_messages,
            ])

            # Tool call handling (unchanged)
            max_tool_rounds = 3
            for _ in range(max_tool_rounds):
                if not (hasattr(response, 'tool_calls') and response.tool_calls):
                    break
                for tc in response.tool_calls:
                    tc_name = tc.get('name', '')
                    if tc_name == 'ocr_image':
                        tool_result = ocr_image.invoke(tc.get('args', {}))
                        recent_messages.append({
                            "role": "assistant", "content": response.content or "",
                            "tool_calls": response.tool_calls,
                        })
                        recent_messages.append({
                            "role": "tool", "tool_call_id": tc.get('id', ''),
                            "content": tool_result,
                        })
                response = llm.invoke([
                    {"role": "system", "content": system_msg},
                    *recent_messages,
                ])

            state["answer"] = response.content
            state.setdefault("messages", [])
            state["messages"].append({"role": "assistant", "content": response.content})
            logger.info("[llm_generate] Answer generated (%d chars)", len(state["answer"]))
        except Exception as e:
            logger.error("[llm_generate] Error: %s", e)
            state["answer"] = f"Sorry, an error occurred: {e}"
            state["error"] = str(e)
        return state
    return llm_generate

# ============================================================================
# Graph builder — P1: 新增 clarification_return 节点
# ============================================================================

def create_agent_graph(retriever, llm: Optional[ChatOpenAI] = None,
                       model: str = DEEPSEEK_MODEL, api_key: str = DEEPSEEK_API_KEY,
                       streaming_callback: Optional[BaseCallbackHandler] = None,
                       model_router=None, compressor=None):
    set_retriever(retriever)

    # P2: 全局引用（供 llm_generate 使用）
    global _model_router, _compressor
    _model_router = model_router
    _compressor = compressor

    if llm is None:
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required.")
        kwargs: Dict[str, Any] = dict(model=model, api_key=api_key, base_url=DEEPSEEK_BASE_URL,
                                        temperature=0.2, max_tokens=2048)
        if streaming_callback is not None:
            kwargs["streaming"] = True; kwargs["callbacks"] = [streaming_callback]
        llm = ChatOpenAI(**kwargs)

    # P2: 从注册表动态拉取工具（替代硬编码 bind_tools([ocr_image])）
    tool_list = [ocr_image]  # fallback
    if get_registered_tools:
        from config import ENABLED_TOOLS
        tool_list = get_registered_tools(ENABLED_TOOLS)
        logger.info("[graph] Bound %d tools from registry", len(tool_list))
    llm = llm.bind_tools(tool_list)

    query_understand_fn = _create_query_understand_node(llm)
    mysql_query_fn = _create_mysql_query_node(llm)
    llm_generate_fn = _create_llm_generate_node(llm)

    # P0 审计包装
    tracer = get_tracer()
    _make_audit = lambda name: audit_node(name) if tracer else lambda f: f
    query_understand_fn = _make_audit("query_understand")(query_understand_fn)
    route_decision_fn_audited = _make_audit("route_decision")(route_decision_node)
    mysql_query_fn = _make_audit("mysql_query")(mysql_query_fn)
    rag_retrieve_fn_audited = _make_audit("rag_retrieve")(rag_retrieve_node)
    llm_generate_fn = _make_audit("llm_generate")(llm_generate_fn)
    clarification_fn = _make_audit("clarification_return")(clarification_return_node)

    workflow = StateGraph(AgentState)
    workflow.add_node("query_understand", query_understand_fn)
    workflow.add_node("route_decision", route_decision_fn_audited)
    workflow.add_node("mysql_query", mysql_query_fn)
    workflow.add_node("rag_retrieve", rag_retrieve_fn_audited)
    workflow.add_node("llm_generate", llm_generate_fn)
    workflow.add_node("clarification_return", clarification_fn)

    workflow.set_entry_point("query_understand")

    # P1: query_understand 直接到 route_decision，由 route_decision 做三路分支
    workflow.add_edge("query_understand", "route_decision")
    workflow.add_conditional_edges("route_decision", _route_condition, {
        "clarification_return": "clarification_return",
        "mysql_query": "mysql_query",
        "rag_retrieve": "rag_retrieve",
    })

    workflow.add_edge("clarification_return", END)
    workflow.add_edge("mysql_query", "llm_generate")
    workflow.add_edge("rag_retrieve", "llm_generate")
    workflow.add_edge("llm_generate", END)

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    logger.info("LangGraph agent compiled (P1: dual route + clarification + score guard + rewrite)")
    return graph
