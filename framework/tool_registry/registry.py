"""工具注册表 + @register_tool 装饰器"""
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_registry: Dict[str, dict] = {}
_tool_objects: Dict[str, Any] = {}


def register_tool(name: str, description: str):
    """装饰器：将函数注册为 LangChain 兼容的工具。

    用法:
        @register_tool(name="ocr_image", description="OCR识别图片文字")
        def ocr_image(image_path: str) -> str:
            ...

    在 graph 中拉取所有工具:
        tools = get_registered_tools()
        llm = llm.bind_tools(tools)
    """
    def decorator(func: Callable):
        _registry[name] = {"name": name, "description": description, "func": func.__name__}
        _tool_objects[name] = func
        logger.info("[ToolRegistry] Registered tool: %s", name)
        return func
    return decorator


def get_registered_tools(enabled: Optional[List[str]] = None) -> List[Any]:
    """返回所有已注册的 LangChain 工具对象。

    Args:
        enabled: 启用的工具名单；None = 全部启用。
    """
    from langchain_core.tools import tool as langchain_tool
    result = []
    for name, obj in _tool_objects.items():
        if enabled is not None and name not in enabled:
            continue
        # 包装为 LangChain tool（如果尚未包装）
        if hasattr(obj, 'name') and hasattr(obj, 'description') and callable(getattr(obj, 'func', None)):
            result.append(obj)
        elif hasattr(obj, 'name') and hasattr(obj, 'description'):
            # Already a LangChain tool — pass through
            result.append(obj)
        else:
            wrapped = _wrap_as_tool(name, _registry[name]["description"], obj)
            result.append(wrapped)
    return result


def _wrap_as_tool(name: str, description: str, func: Callable):
    # 如果已经是 tool，直接返回
    if hasattr(func, 'name') and hasattr(func, 'description') and hasattr(func, 'invoke'):
        return func
    """将普通函数包装为 LangChain tool。"""
    from langchain_core.tools import tool
    @tool(name, description=description)
    def _wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error("[ToolRegistry] Tool '%s' failed: %s", name, e)
            return f"工具 '{name}' 执行失败: {e}"
    return _wrapped


def list_tools() -> List[dict]:
    """列出所有已注册工具的信息。"""
    return [{"name": k, **v} for k, v in _registry.items()]
