"""
Chainlit chat UI for Camera SDK RAG Agent.

Usage:
    chainlit run chat_ui.py
"""

import os as _os
_os.environ["HF_HUB_OFFLINE"] = "1"
_os.environ["TRANSFORMERS_OFFLINE"] = "1"
_os.environ["HF_DATASETS_OFFLINE"] = "1"

import logging
import time
from pathlib import Path

import chainlit as cl

from config import (
    validate_config,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    SDK_DOCS_DIR,
)

# 等保三级: 认证鉴权
try:
    from framework.auth.middleware import authenticate_session, set_current_user, AuthError
except ImportError:
    authenticate_session = None
    set_current_user = None
    AuthError = Exception
from ingestion.embedder import Embedder
from ingestion.parsers import parse_document
from ingestion.chunker import chunk_document
from store.qdrant_store import QdrantStore
from retrieval.retriever import HybridRetriever
import asyncio

from agent.graph import create_agent_graph, AgentState, _StreamToChainlit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent settings
# ---------------------------------------------------------------------------

BRANDS = ["全部品牌", "dahua", "haikang", "basler", "hikrobot"]


@cl.on_chat_start
async def on_chat_start():
    """Initialize agent components once per session."""
    # 等保三级: 认证校验
    api_key = _os.environ.get("API_KEY", "")
    if authenticate_session:
        try:
            user = authenticate_session(api_key)
            set_current_user(user["user_id"], user["role"])
            logger.info("Authenticated: %s (%s)", user["user_id"], user["role"])
        except AuthError as e:
            await cl.Message(
                content=f"❌ 认证失败: {e}\n\n请设置有效的 API_KEY 环境变量后重试。"
            ).send()
            return
    else:
        if set_current_user:
            set_current_user("guest", "operator")

    validate_config()

    store = QdrantStore()
    embedder = Embedder()
    retriever = HybridRetriever(store, embedder)

    # Streaming callback for Chainlit — one per session
    stream_handler = _StreamToChainlit()

    graph = create_agent_graph(
        retriever=retriever,
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        streaming_callback=stream_handler,
    )

    cl.user_session.set("graph", graph)
    cl.user_session.set("store", store)
    cl.user_session.set("embedder", embedder)
    cl.user_session.set("retriever", retriever)
    cl.user_session.set("stream_handler", stream_handler)
    cl.user_session.set("camera_brand", None)
    cl.user_session.set("messages", [])  # Persistent conversation history

    settings = await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="brand",
                label="相机品牌过滤",
                values=BRANDS,
                initial_index=0,
            ),
            cl.input_widget.Select(
                id="ingest_brand",
                label="上传文件归属品牌",
                values=[b for b in BRANDS if b != "全部品牌"],
                initial_index=0,
            ),
        ]
    ).send()

    point_count = store.count_points()
    await cl.Message(
        content=f"👋 **Camera SDK RAG Agent 已就绪！**\n\n"
                f"📊 向量库: {point_count} chunks\n\n"
                "📎 **上传文件**: 拖拽/粘贴 PDF/MD 文件到对话框，自动摄取到向量库\n\n"
                "💬 **提问**:\n"
                "- 如何初始化海康相机？\n"
                "- 大华相机怎么抓图？\n"
                "\n底部 Settings 可切换品牌过滤和上传品牌。"
    ).send()


# ---------------------------------------------------------------------------
# Settings change
# ---------------------------------------------------------------------------


@cl.on_settings_update
async def on_settings_update(settings):
    brand = settings.get("brand", "全部品牌")
    if brand == "全部品牌":
        cl.user_session.set("camera_brand", None)
    else:
        cl.user_session.set("camera_brand", brand)
    # ingest_brand is read on file upload, nothing to do here


# ---------------------------------------------------------------------------
# File upload → ingestion
# ---------------------------------------------------------------------------


async def _handle_file_upload(file_element: cl.File) -> str:
    """Save an uploaded file to sdk_docs/<brand>/ and run ingestion.

    Args:
        file_element: Chainlit File element from uploaded message.

    Returns:
        Status message string.
    """
    supported = {".pdf", ".md", ".html", ".htm", ".txt"}
    ext = Path(file_element.name).suffix.lower()
    if ext not in supported:
        return f"⚠ 不支持的文件格式: {ext}（支持: {', '.join(supported)}）"

    # Read upload brand from settings or default to "unknown"
    brand = cl.user_session.get("ingest_brand", "unknown")
    if not brand or brand == "全部品牌":
        brand = "unknown"

    # Ensure brand directory exists
    brand_dir = SDK_DOCS_DIR / brand
    brand_dir.mkdir(parents=True, exist_ok=True)

    # Clean filename to avoid issues
    safe_name = Path(file_element.name).name
    dest_path = brand_dir / safe_name

    # Avoid overwrite: append (1), (2) etc.
    counter = 1
    while dest_path.exists():
        stem = Path(safe_name).stem
        dest_path = brand_dir / f"{stem} ({counter}){ext}"
        counter += 1

    # Save file from Chainlit temp location
    try:
        file_bytes = file_element.content
        dest_path.write_bytes(file_bytes)
    except Exception as e:
        # Fallback: copy from path if content is not available
        if file_element.path and Path(file_element.path).exists():
            import shutil
            shutil.copy2(file_element.path, str(dest_path))
        else:
            return f"❌ 无法读取文件: {e}"

    logger.info("Saved uploaded file: %s (%d bytes)", dest_path, dest_path.stat().st_size)

    # Run ingestion for this single file
    store = cl.user_session.get("store")
    embedder = cl.user_session.get("embedder")

    if store is None or embedder is None:
        return "❌ Agent 组件未初始化，请刷新页面"

    t_start = time.time()

    try:
        # Parse
        parsed = parse_document(dest_path)
        # Chunk
        chunks = chunk_document(parsed)
        if not chunks:
            return f"⚠ 文件解析成功但未生成任何 chunk: {safe_name}"

        # Embed
        texts = [c.text for c in chunks]
        vectors = embedder.encode(texts, show_progress=False)

        # Upsert
        store.create_collection(force_recreate=False)
        inserted = store.insert_chunks(chunks, vectors)

        elapsed = time.time() - t_start
        result = (
            f"✅ **摄取成功！**\n"
            f"- 文件: `{safe_name}`\n"
            f"- 品牌: `{brand}`\n"
            f"- 新增 chunks: {inserted}\n"
            f"- 耗时: {elapsed:.1f}s"
        )
    except Exception as e:
        logger.exception("Ingestion failed for %s", dest_path)
        result = f"❌ 摄取失败: {e}"

    # Refresh BM25 index
    retriever = cl.user_session.get("retriever")
    if retriever:
        try:
            retriever.refresh_bm25()
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Message handler
# ---------------------------------------------------------------------------


@cl.on_message
async def on_message(message: cl.Message):
    graph = cl.user_session.get("graph")
    if graph is None:
        await cl.Message(content="Agent 未初始化，请刷新页面。").send()
        return

    # --- File upload handling ---
    if message.elements:
        results: list = []
        for elem in message.elements:
            if isinstance(elem, cl.File):
                status_msg = cl.Message(content=f"📥 正在摄取 `{elem.name}` ...")
                await status_msg.send()
                result = await _handle_file_upload(elem)
                status_msg.content = result
                await status_msg.update()
                results.append(result)

        # If user also typed a question alongside the file, answer it
        if not message.content or not message.content.strip():
            return
        # Otherwise fall through to normal Q&A

    # --- Normal Q&A ---
    brand = cl.user_session.get("camera_brand", None)
    messages = cl.user_session.get("messages", [])

    state: AgentState = {
        "query": message.content,
        "camera_brand": brand,
        "intent": "",
        "retrieved_context": "",
        "formatted_context": "",
        "messages": messages,
        "answer": "",
        "error": None,
    }

    msg = cl.Message(content="")
    await msg.send()

    # Wire streaming callback to this Chainlit message
    stream_handler = cl.user_session.get("stream_handler")
    if stream_handler is not None:
        stream_handler.set_message(msg, asyncio.get_running_loop())

    thread = {"configurable": {"thread_id": cl.user_session.get("id")}}

    try:
        result = graph.invoke(state, config=thread)

        # Persist updated message history back to session
        cl.user_session.set("messages", result.get("messages", messages))

        sources = _parse_sources(result.get("formatted_context", ""))

        # Append source footer after the streamed answer
        if sources:
            src_lines = "\n".join(f"- {s}" for s in sources[:5])
            await msg.stream_token(f"\n\n---\n**📚 检索来源 (Top 5):**\n{src_lines}")
    except Exception as e:
        await msg.stream_token(f"\n\n❌ 出错：{e}")

    await msg.update()


def _parse_sources(context: str) -> list:
    """Extract source info lines from formatted context."""
    sources = []
    for line in context.split("\n"):
        if line.startswith("File:") or line.startswith("Page:"):
            sources.append(line.strip())
    return sources
