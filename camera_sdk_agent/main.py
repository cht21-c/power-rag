"""
Interactive Q&A entry point for the Camera SDK RAG Agent.

Supports:
- Interactive mode: python main.py
- Ingest mode:     python main.py --ingest
- Brand filter:    python main.py --brand Basler
- One-shot query:  python main.py --query "How to set exposure?"
- Verbose logging: python main.py --verbose

Configuration is read from environment variables (see config.py).
"""

import argparse
import logging
import os as _os
import sys
from typing import Optional

# Force fully offline — must run before any import that touches huggingface_hub
_os.environ["HF_HUB_OFFLINE"] = "1"
_os.environ["TRANSFORMERS_OFFLINE"] = "1"
_os.environ["HF_DATASETS_OFFLINE"] = "1"

from config import (
    print_config,
    validate_config,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
)

# 等保三级: 认证鉴权
try:
    from framework.auth.middleware import require_auth, set_current_user, get_current_user
except ImportError:
    def require_auth(*a, **kw):
        return lambda f: f
    def set_current_user(*a):
        pass
    def get_current_user():
        return None
from ingest_pipeline import run_ingestion, setup_logging
from ingestion.embedder import Embedder
from store.qdrant_store import QdrantStore
from retrieval.retriever import HybridRetriever
from agent.graph import create_agent_graph, AgentState, _StreamToStdout

logger = logging.getLogger(__name__)


# ============================================================================
# Bootstrap: wire up the retrieval + agent stack
# ============================================================================


def bootstrap_agent(camera_brand: Optional[str] = None, stream: bool = True):
    """Initialize all components and return a compiled LangGraph agent.

    Args:
        camera_brand: Optional brand filter baked into the initial state.
        stream: If True, enable token-by-token streaming to stdout.

    Returns:
        Tuple of (compiled_graph, default_state_dict).
    """
    # 1. Validate config
    validate_config()

    # 2. Initialize components
    store = QdrantStore()

    # Quick check: is there data?
    point_count = store.count_points()
    if point_count == 0:
        print("\n⚠  WARNING: Qdrant collection is empty!")
        print("   Run ingestion first: python main.py --ingest\n")

    embedder = Embedder()
    retriever = HybridRetriever(store, embedder)

    # 3. Build graph — with streaming callback for CLI
    stream_handler = _StreamToStdout() if stream else None
    graph = create_agent_graph(
        retriever=retriever,
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        streaming_callback=stream_handler,
    )

    default_state: AgentState = {
        "query": "",
        "camera_brand": camera_brand,
        "intent": "",
        "retrieved_context": "",
        "formatted_context": "",
        "messages": [],
        "answer": "",
        "error": None,
    }

    return graph, default_state


# ============================================================================
# Interactive loop
# ============================================================================


def run_interactive(camera_brand: Optional[str] = None):
    """Run the agent in interactive CLI mode.

    Args:
        camera_brand: Optional brand filter applied to all queries.
    """
    print("\n" + "=" * 60)
    print("  Camera SDK RAG Agent")
    print("=" * 60)
    print_config()
    print("=" * 60)

    print("\nInitializing agent components ...")
    graph, base_state = bootstrap_agent(camera_brand=camera_brand)

    print("\n✓ Agent ready! Type your question or /help for commands.\n")

    # Use a thread ID for checkpointer continuity
    thread = {"configurable": {"thread_id": "interactive-session"}}

    # Accumulate conversation history across turns
    session_messages: list = []

    while True:
        try:
            user_input = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            _handle_command(user_input, base_state)
            continue

        # Build state for this turn — carry forward accumulated messages
        state = dict(base_state)
        state["query"] = user_input
        state["messages"] = session_messages

        print()

        try:
            result = graph.invoke(state, config=thread)
            print()  # newline after streamed output
            # Persist updated messages for next turn
            session_messages = result.get("messages", session_messages)
        except Exception as e:
            logger.exception("Error during agent invocation")
            print(f"\n❌ Error: {e}\n")


def _handle_command(cmd: str, state: dict):
    """Handle interactive slash commands."""
    parts = cmd.split(maxsplit=1)
    command = parts[0].lower()

    if command == "/help":
        print("""
Commands:
  /help          - Show this help
  /brand BRAND   - Set camera brand filter (e.g. /brand Basler)
  /brand clear   - Clear brand filter
  /config        - Show current configuration
  /exit or /quit - Exit the agent
        """)
    elif command == "/brand":
        if len(parts) > 1:
            brand = parts[1].strip()
            if brand.lower() == "clear":
                state["camera_brand"] = None
                print("✓ Brand filter cleared")
            else:
                state["camera_brand"] = brand
                print(f"✓ Brand filter set to: {brand}")
        else:
            print(f"Current brand filter: {state.get('camera_brand') or 'none'}")
    elif command == "/config":
        print_config()
    elif command in ("/exit", "/quit"):
        print("Goodbye!")
        sys.exit(0)
    else:
        print(f"Unknown command: {command}. Type /help for available commands.")


# ============================================================================
# One-shot query
# ============================================================================


def run_query(query: str, camera_brand: Optional[str] = None):
    """Run a single query and print the answer.

    Args:
        query: The user's question.
        camera_brand: Optional brand filter.
    """
    print("Initializing agent ...")
    graph, state = bootstrap_agent(camera_brand=camera_brand)
    state["query"] = query

    thread = {"configurable": {"thread_id": "oneshot"}}
    print(f"Query: {query}\n")
    result = graph.invoke(state, config=thread)
    print(result["answer"])


# ============================================================================
# CLI entry point
# ============================================================================


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Camera SDK RAG Agent - Interactive Q&A",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                          # Interactive mode
    python main.py --ingest                 # Run document ingestion first
    python main.py --brand Basler           # Filter by brand
    python main.py --query "How to init?"   # One-shot query
    python main.py --ingest --query "..."   # Ingest then query
        """,
    )
    parser.add_argument("--ingest", action="store_true",
                        help="Run document ingestion before starting")
    parser.add_argument("--ingest-only", action="store_true",
                        help="Run ingestion and exit (don't start Q&A)")
    parser.add_argument("--brand", type=str, default=None,
                        help="Camera brand filter (e.g. Basler, HikVision)")
    parser.add_argument("--query", type=str, default=None,
                        help="One-shot query (non-interactive)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--force-recreate", action="store_true",
                        help="Force recreate Qdrant collection on ingest")

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    # Handle ingestion (等保三级: 需要 admin 权限)
    if args.ingest or args.ingest_only:
        try:
            from framework.auth.middleware import get_key_store, AuthError
            store = get_key_store()
            api_key = _os.environ.get("API_KEY", "")
            if api_key:
                record = store.verify(api_key)
                if record is None:
                    print("ERROR: Invalid API_KEY. Ingest requires valid admin credentials.")
                    sys.exit(1)
                if record.role != "admin":
                    print(f"ERROR: Role '{record.role}' not authorized for ingest. Admin required.")
                    sys.exit(1)
                set_current_user(record.user_id, record.role)
            elif not store.list_keys():
                set_current_user("guest", "operator")
            else:
                print("ERROR: API_KEY env var required for ingest.")
                sys.exit(1)
        except ImportError:
            pass

        logger.info("Running document ingestion ...")
        run_ingestion(force_recreate=args.force_recreate)
        if args.ingest_only:
            return

    # Validate API key
    try:
        validate_config()
    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        print("   Set DEEPSEEK_API_KEY in your environment or .env file.")
        sys.exit(1)

    # Run query mode or interactive mode
    if args.query:
        run_query(args.query, camera_brand=args.brand)
    else:
        run_interactive(camera_brand=args.brand)


if __name__ == "__main__":
    main()
