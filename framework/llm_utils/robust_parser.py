"""
鲁棒 JSON 解析器 —— Schema 校验 + 单次重试 + 保守兜底

等保 P1 质量加固：解决 LLM 输出格式不稳定导致的路由失败问题。
"""

import json, logging, re
from typing import Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

class QueryUnderstandResult(BaseModel):
    """旧版 schema（P1），保留兼容。"""
    intent: str = "other"
    keywords: list[str] = []
    route: str = "rag"
    confidence: float = 0.0


class QueryUnderstandResultV2(BaseModel):
    """新版 schema（P2 LLM 语义路由）：多步推理 + 实体提取 + 澄清选项。"""
    understood_intent: str = ""
    route: str = "rag"  # "drawing" | "knowledge_qa" | "ambiguous"
    confidence: float = 0.0
    drawing_entity: Optional[str] = None
    clarify_options: list[str] = []


class DrawingMatchResult(BaseModel):
    """LLM 图纸语义匹配输出。"""
    matched_ids: list[int] = []
    reasoning: str = ""

_FALLBACK = QueryUnderstandResult(intent="other", keywords=[], route="rag", confidence=0.0)

def _extract_json(raw: str) -> Optional[str]:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m:
        raw = m.group(1).strip()
    raw = raw.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    raw = raw.replace("\uff1a", ":").replace("\uff0c", ",")
    m = re.search(r"\{[\s\S]*?\}", raw)
    return m.group() if m else None

def parse_json(raw: str) -> Dict[str, Any]:
    extracted = _extract_json(raw)
    if extracted is None:
        raise ValueError("No JSON object found in LLM output")
    return json.loads(extracted)

def validate_schema(data: Dict[str, Any], schema_cls: Type[T]) -> T:
    return schema_cls(**data)

def _build_retry_prompt(raw_output: str, schema_cls: Type[BaseModel]) -> str:
    fields = list(schema_cls.model_fields.keys())
    field_strs = []
    for f in fields:
        ann = str(schema_cls.model_fields[f].annotation)
        field_strs.append(f'  "{f}": {ann}')
    return (
        "你上次返回的 JSON 格式不对，请严格按照以下 schema 重新输出：\n\n"
        "Schema: {\n" + ",\n".join(field_strs) + "\n}\n\n"
        f"你上次的输出：\n{raw_output[:500]}\n\n"
        "请只返回一个合法的 JSON 对象，不要包含其他文字。"
    )

def robust_parse(llm, raw_output: str, schema_cls: Type[T],
                 retry_prompt: Optional[str] = None) -> tuple:
    try:
        data = parse_json(raw_output)
        result = validate_schema(data, schema_cls)
        return (result, "success")
    except Exception as e:
        logger.warning("[robust_parse] First attempt failed: %s", e)
    try:
        prompt = retry_prompt or _build_retry_prompt(raw_output, schema_cls)
        retry_response = llm.invoke(prompt)
        data = parse_json(retry_response.content)
        result = validate_schema(data, schema_cls)
        logger.info("[robust_parse] Retry succeeded")
        return (result, "retry_success")
    except Exception as e:
        logger.error("[robust_parse] Retry failed: %s, using fallback", e)
    fallback = _make_fallback(schema_cls)
    return (fallback, "fallback")

def _make_fallback(schema_cls: Type[T]) -> T:
    if schema_cls is QueryUnderstandResult:
        return _FALLBACK
    kwargs = {}
    for name, field in schema_cls.model_fields.items():
        if field.default not in (None, ...):
            kwargs[name] = field.default
        elif field.annotation in (str,):
            kwargs[name] = ""
        elif field.annotation in (list,):
            kwargs[name] = []
        elif field.annotation in (float,):
            kwargs[name] = 0.0
        elif field.annotation in (int,):
            kwargs[name] = 0
        else:
            kwargs[name] = None
    return schema_cls(**kwargs)
