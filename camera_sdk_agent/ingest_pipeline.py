"""
One-click ingestion pipeline for Camera SDK documentation.

Workflow:
  1. Discover all documents under sdk_docs/
  2. Parse each document (PDF/MD/HTML/TXT)
  3. Chunk into function-granular pieces (512 tokens, 64 overlap)
  4. Generate BGE-M3 embeddings
  5. Upsert chunks + vectors into Qdrant

Usage:
    python ingest_pipeline.py
    python ingest_pipeline.py --force-recreate
    python ingest_pipeline.py --docs-dir /path/to/docs
"""

import argparse
import logging
import os as _os
import sys
import time

# Force fully offline — must run before any import that touches huggingface_hub
_os.environ["HF_HUB_OFFLINE"] = "1"
_os.environ["TRANSFORMERS_OFFLINE"] = "1"
_os.environ["HF_DATASETS_OFFLINE"] = "1"
from pathlib import Path

from ingestion.parsers import discover_documents, parse_document
from ingestion.chunker import chunk_document
from ingestion.embedder import Embedder
from store.qdrant_store import QdrantStore
from config import (
    SDK_DOCS_DIR,
    CHUNK_SIZE_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    QDRANT_COLLECTION_NAME,
)

# ============================================================================
# Logging setup
# ============================================================================


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the ingestion pipeline.

    Args:
        verbose: If True, set log level to DEBUG.
    """
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)-7s] %(name)s - %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")


# ============================================================================
# Main pipeline
# ============================================================================


def run_ingestion(
    docs_dir: Path = SDK_DOCS_DIR,
    force_recreate: bool = False,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
    dry_run: bool = False,
) -> dict:
    """Execute the full ingestion pipeline.

    Args:
        docs_dir: Root directory containing SDK documentation files.
        force_recreate: If True, drop and recreate the Qdrant collection.
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Overlap between consecutive chunks in tokens.
        dry_run: If True, parse and chunk but skip embedding and Qdrant insert.

    Returns:
        Statistics dictionary with keys:
            files_found, files_parsed, total_chunks, elapsed_seconds
    """
    logger = logging.getLogger(__name__)
    stats = {
        "files_found": 0,
        "files_parsed": 0,
        "total_chunks": 0,
        "elapsed_seconds": 0.0,
    }

    t_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Discover documents
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 1/5: Discovering documents under %s ...", docs_dir)
    files = discover_documents(docs_dir)
    stats["files_found"] = len(files)

    if not files:
        logger.warning("No supported documents found in %s", docs_dir)
        logger.warning("Supported formats: PDF, MD, HTML, TXT")
        logger.warning("Expected structure: sdk_docs/<brand>/<version>/*.pdf")
        return stats

    logger.info("Found %d document(s):", len(files))
    for f in files:
        logger.info("  - %s", f)

    # ------------------------------------------------------------------
    # Step 2: Parse documents
    # ------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("Step 2/5: Parsing documents ...")

    parsed_docs = []
    for file_path in files:
        try:
            parsed = parse_document(file_path)
            parsed_docs.append(parsed)
            logger.info("  Parsed: %s (brand=%s, version=%s, %d chars)",
                         file_path.name,
                         parsed["metadata"].get("camera_brand", "?"),
                         parsed["metadata"].get("sdk_version", "?"),
                         len(str(parsed.get("text", ""))))
        except Exception as e:
            logger.error("  Failed to parse %s: %s", file_path, e)

    stats["files_parsed"] = len(parsed_docs)

    # ------------------------------------------------------------------
    # Step 2.5: Initialize embedder (shared between chunking & embedding)
    # ------------------------------------------------------------------
    embedder = Embedder()

    # ------------------------------------------------------------------
    # Step 3: Semantic chunking
    # ------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("Step 3/5: Semantic chunking (chonkie + BGE-M3, max=%d tokens, threshold=0.5) ...",
                 chunk_size)

    all_chunks = []
    for parsed in parsed_docs:
        try:
            chunks = chunk_document(parsed, embedder, chunk_size=chunk_size)
            all_chunks.extend(chunks)
            source = parsed.get("metadata", {}).get("source_file", "?")
            logger.info("  %s → %d chunks", Path(str(source)).name, len(chunks))
        except Exception as e:
            logger.error("  Chunking failed for %s: %s",
                          parsed.get("metadata", {}).get("source_file", "?"), e)

    stats["total_chunks"] = len(all_chunks)
    logger.info("Total chunks: %d", len(all_chunks))

    # Show chunk distribution by category
    category_counts: dict = {}
    for c in all_chunks:
        cat = c.metadata.category
        category_counts[cat] = category_counts.get(cat, 0) + 1
    logger.info("Chunks by category: %s", dict(sorted(category_counts.items())))

    # Show chunk method distribution (semantic vs rule_based_structured)
    method_counts: dict = {}
    for c in all_chunks:
        method = getattr(c.metadata, "chunk_method", "semantic")
        method_counts[method] = method_counts.get(method, 0) + 1
    logger.info("Chunks by method: %s", dict(sorted(method_counts.items())))
    stats["chunk_methods"] = method_counts

    if dry_run:
        logger.info("Dry run complete — skipping embedding and Qdrant insert.")
        stats["elapsed_seconds"] = round(time.time() - t_start, 2)
        return stats

    # ------------------------------------------------------------------
    # Step 4: Generate embeddings (reuse embedder from step 2.5)
    # ------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("Step 4/5: Generating BGE-M3 embeddings ...")

    texts = [c.text for c in all_chunks]
    t_embed_start = time.time()
    vectors = embedder.encode(texts)
    t_embed = time.time() - t_embed_start
    logger.info("Generated %d embeddings in %.1fs (%.0f texts/s)",
                 len(vectors), t_embed, len(vectors) / max(t_embed, 0.001))

    # ------------------------------------------------------------------
    # Step 5: Upsert to Qdrant
    # ------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("Step 5/5: Upserting to Qdrant (collection: %s) ...",
                 QDRANT_COLLECTION_NAME)

    store = QdrantStore()
    store.create_collection(force_recreate=force_recreate)
    inserted = store.insert_chunks(all_chunks, vectors)
    stats["inserted_points"] = inserted

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    stats["elapsed_seconds"] = round(time.time() - t_start, 2)
    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("  Files found:    %d", stats["files_found"])
    logger.info("  Files parsed:   %d", stats["files_parsed"])
    logger.info("  Total chunks:   %d", stats["total_chunks"])
    logger.info("  Chunk methods:  %s", stats.get("chunk_methods", {}))
    logger.info("  Qdrant points:  %d", stats.get("inserted_points", 0))
    logger.info("  Elapsed time:   %.1fs", stats["elapsed_seconds"])
    logger.info("=" * 60)

    return stats


# ============================================================================
# CLI entry point
# ============================================================================


def main():
    """CLI entry point for the ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Camera SDK Documentation Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ingest_pipeline.py
    python ingest_pipeline.py --force-recreate
    python ingest_pipeline.py --docs-dir ./my_sdk_docs
    python ingest_pipeline.py --dry-run --verbose
        """,
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=SDK_DOCS_DIR,
        help="Directory containing SDK documentation (default: sdk_docs/)",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Drop and recreate the Qdrant collection before ingestion",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk only; skip embedding and Qdrant insert",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging",
    )

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    logger = logging.getLogger(__name__)

    try:
        stats = run_ingestion(
            docs_dir=args.docs_dir,
            force_recreate=args.force_recreate,
            dry_run=args.dry_run,
        )

        if stats["files_found"] == 0:
            logger.warning("No documents found. Place SDK docs in: %s", args.docs_dir)
            sys.exit(1)

    except Exception as e:
        logger.exception("Ingestion failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
