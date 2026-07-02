"""
LangGraph tool definitions for the Camera SDK RAG Agent.

The primary tool is `rag_search` which wraps the hybrid retriever
and returns formatted context for the LLM.
"""

import json
import logging
from datetime import date
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

# P2: 工具注册
try:
    from framework.tool_registry.registry import register_tool
except ImportError:
    def register_tool(name, description):
        return lambda f: f

logger = logging.getLogger(__name__)

# Global retriever reference — set by the graph builder at startup.
_retriever = None


def set_retriever(retriever):
    """Set the global retriever instance for tool use.

    Args:
        retriever: HybridRetriever instance.
    """
    global _retriever
    _retriever = retriever


@tool
def rag_search(
    query: str,
    camera_brand: Optional[str] = None,
    top_k: int = 5,
) -> str:
    """Search the Camera SDK documentation using hybrid retrieval.

    This tool searches through indexed SDK documentation chunks using
    a combination of vector similarity and BM25 keyword matching.

    Args:
        query: The search query (natural language, e.g. "How to set exposure time?").
        camera_brand: Optional camera brand filter (e.g. "Basler", "HikVision").
        top_k: Number of top results to return (default 5).

    Returns:
        A JSON string containing the top-k retrieved chunks with metadata.
    """
    global _retriever

    if _retriever is None:
        return json.dumps({
            "error": "Retriever not initialized. Please run document ingestion first.",
            "results": [],
        })

    try:
        results = _retriever.retrieve(
            query=query,
            camera_brand=camera_brand,
            top_k=top_k,
        )

        # Format results for LLM consumption
        formatted: List[Dict[str, Any]] = []
        for r in results:
            formatted.append({
                "function_name": r.get("function_name", "unknown"),
                "category": r.get("category", "general"),
                "camera_brand": r.get("camera_brand", "unknown"),
                "sdk_version": r.get("sdk_version", "unknown"),
                "source_file": r.get("source_file", ""),
                "text": r.get("text", ""),
                "page_number": r.get("page_number", 0),
                "relevance_score": r.get("score", r.get("rrf_score", 0)),
            })

        max_score = max((r.get("relevance_score", 0) for r in formatted), default=0)
        return json.dumps({
            "count": len(formatted),
            "max_score": max_score,
            "results": formatted,
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error("rag_search tool error: %s", e)
        return json.dumps({
            "error": str(e),
            "results": [],
        })


def format_context_for_llm(results_json: str) -> str:
    """Convert the JSON results from rag_search into a readable context string.

    Args:
        results_json: JSON string from rag_search tool.

    Returns:
        Formatted context string for the LLM prompt.
    """
    try:
        data = json.loads(results_json)
    except json.JSONDecodeError:
        return "No context available (error parsing retrieval results)."

    results = data.get("results", [])
    if not results:
        return "No relevant SDK documentation chunks found."

    lines: List[str] = []
    for i, r in enumerate(results, 1):
        # Extract short filename from full path
        import os
        full_path = r.get('source_file', 'N/A')
        filename = os.path.basename(full_path) if full_path != 'N/A' else 'N/A'

        lines.append(f"--- Chunk {i} (citation: [REF{i}]) ---")
        lines.append(f"File: {filename}")
        lines.append(f"Full Path: {full_path}")
        lines.append(f"Page: {r.get('page_number', 'N/A')}")
        lines.append(f"Function: {r.get('function_name', 'N/A')}")
        lines.append(f"Category: {r.get('category', 'N/A')} | Score: {r.get('relevance_score', 0):.4f}")
        lines.append(f"Brand: {r.get('camera_brand', 'N/A')} (SDK v{r.get('sdk_version', '?')})")
        lines.append("")
        lines.append(r.get("text", ""))
        lines.append("")

    return "\n".join(lines)


@tool
def get_current_date() -> str:
    """Return today's date as a string (YYYY-MM-DD)."""
    return date.today().isoformat()


@tool
@register_tool(name="ocr_image", description="对本地图片进行OCR文字识别，返回图片中所有可识别的文字")
def ocr_image(image_path: str) -> str:
    """对本地图片或URL图片进行OCR文字识别，返回图片中所有可识别的文字内容。

    适用场景：
    - 识别扫描文档、截图、照片中的文字
    - 提取表格、发票、证件中的文字信息
    - 识别中文、英文及混合文字内容

    参数：
        image_path: 图片的本地文件路径（如 D:/data/img.jpg）或 HTTP URL

    返回：
        按行排列的识别文字，格式为：
        第1行: <文字内容> (置信度: 0.xx)
        第2行: <文字内容> (置信度: 0.xx)
        ...
        若无文字则返回："未检测到文字内容"
    """
    import os as _os
    from pathlib import Path as _Path

    # Reject URLs — only accept local file paths
    if image_path.startswith(("http://", "https://")):
        return "错误：ocr_image 仅支持本地文件路径，不支持 URL。请提供本地图片路径。"

    # Reject non-existent paths
    if not _os.path.exists(image_path):
        return f"错误：文件不存在 — {image_path}。请确认路径正确。"

    # Reject non-image files by extension
    ext = _Path(image_path).suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"):
        return f"错误：不支持的文件格式 {ext}。支持: png, jpg, bmp, tiff。"

    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR()
    result, _ = engine(image_path)
    if not result:
        return "未检测到文字内容"
    lines = [
        f"第{i+1}行: {item[1]} (置信度: {item[2]:.2f})"
        for i, item in enumerate(result)
    ]
    return "\n".join(lines)
