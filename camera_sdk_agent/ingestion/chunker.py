"""
Semantic chunker for power plant documentation.

Uses chonkie SemanticChunker with BGE-M3 embeddings to split text
at natural semantic boundaries (similarity_threshold=0.5).
Adds 15% token overlap between consecutive chunks.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .parsers import parse_document

logger = logging.getLogger(__name__)

# ============================================================================
# Data types — power plant metadata format
# ============================================================================


@dataclass
class ChunkMetadata:
    """Metadata attached to each chunk (power plant domain)."""

    file_name: str = ""
    url: str = ""
    category: str = "其他"
    equipment_id: str = ""
    source_file: str = ""
    chunk_index: int = 0
    page_number: int = 0
    chunk_method: str = "semantic"  # "semantic" | "rule_based_structured"


@dataclass
class Chunk:
    """Represents a single chunk of documentation text."""

    text: str
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)

    def to_dict(self) -> Dict[str, object]:
        """Serialize chunk to dictionary for downstream use."""
        return {
            "text": self.text,
            "metadata": {
                "file_name": self.metadata.file_name,
                "url": self.metadata.url,
                "category": self.metadata.category,
                "equipment_id": self.metadata.equipment_id,
                "source_file": self.metadata.source_file,
                "chunk_index": self.metadata.chunk_index,
                "page_number": self.metadata.page_number,
                "chunk_method": self.metadata.chunk_method,
            },
        }


# ============================================================================
# Power plant category keywords
# ============================================================================

_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "管路": ["管路", "管道", "阀门", "泵", "抽汽", "给水", "循环水", "蒸汽管"],
    "电气": ["电气", "开关", "变压器", "母线", "继电器", "接线", "配电", "电缆"],
    "设备": ["机组", "汽机", "锅炉", "风机", "凝汽器", "汽轮机", "发电机", "加热器"],
    "规程": ["规程", "操作", "巡检", "制度", "流程", "维护", "检修", "运行"],
}


def _infer_category(text: str, source_file: str) -> str:
    """Infer the power plant category from chunk text and filename.

    Args:
        text: Chunk text.
        source_file: Source file path for additional hints.

    Returns:
        Category string ("管路", "电气", "设备", "规程", "其他").
    """
    text_lower = text.lower()
    file_lower = source_file.lower()

    scores: Dict[str, int] = {}
    for category, keywords in _CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        score += sum(2 for kw in keywords if kw in file_lower)
        scores[category] = score

    best = max(scores, key=scores.get)
    return best if scores.get(best, 0) > 0 else "其他"


# ============================================================================
# Semantic chunking core
# ============================================================================


class _EmbeddingCallable:
    """Thin wrapper around our BGE-M3 Embedder, making it a callable
    compatible with chonkie's embedding_model parameter."""

    def __init__(self, embedder):
        self._embedder = embedder

    def __call__(self, sentences: List[str]) -> np.ndarray:
        return self._embedder.encode(sentences, show_progress=False)


def _create_semantic_chunker(embedder, max_chunk_size: int, similarity_threshold: float):
    """Create a chonkie SemanticChunker reusing our BGE-M3 model.

    Passes the model name so chonkie can load from cache (no re-download).

    Args:
        embedder: Embedder instance from ingestion.embedder.
        max_chunk_size: Maximum chunk size in tokens.
        similarity_threshold: Cosine similarity below which to split.

    Returns:
        chonkie.SemanticChunker instance.
    """
    from chonkie import SemanticChunker
    return SemanticChunker(
        embedding_model=embedder.model_name,
        max_chunk_size=max_chunk_size,
        similarity_threshold=similarity_threshold,
    )


def _apply_overlap(chunks: List[str], overlap_ratio: float = 0.15) -> List[str]:
    """Add overlap from previous chunk tail to each subsequent chunk.

    Uses a character-based estimate: overlap_length = ceil(len(chunk) * overlap_ratio).

    Args:
        chunks: List of chunk text strings (from semantic chunker).
        overlap_ratio: Fraction of previous chunk to prepend (default 0.15).

    Returns:
        List of chunk text strings with overlap applied.
    """
    if len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        n_overlap = max(1, int(len(prev) * overlap_ratio))
        overlap_text = prev[-n_overlap:] if len(prev) > n_overlap else prev
        result.append(overlap_text + "\n\n" + chunks[i])

    return result


# ============================================================================
# Public API
# ============================================================================


def _chunk_one_text(
    text: str,
    chunker,
    overlap_ratio: float,
) -> List[str]:
    """Run semantic chunking on a single text blob.

    Args:
        text: Text to chunk.
        chunker: chonkie SemanticChunker instance.
        overlap_ratio: Fraction of previous chunk tail to include as overlap.

    Returns:
        List of chunk text strings.
    """
    if not text or not text.strip() or len(text.strip()) < 10:
        return []

    # chonkie returns List[Chunk] objects, each with .text attribute
    try:
        raw_chunks = chunker.chunk(text)
        texts = [c.text for c in raw_chunks if c.text.strip()]
    except Exception as e:
        logger.warning("Semantic chunker failed, falling back to full text: %s", e)
        return [text]

    # Apply overlap
    return _apply_overlap(texts, overlap_ratio)


def chunk_document(
    parsed_doc: Dict[str, object],
    embedder,
    chunk_size: int = 512,
    similarity_threshold: float = 0.5,
    overlap_ratio: float = 0.15,
    base_url: str = "",
) -> List[Chunk]:
    """Split a parsed document into semantic chunks.

    Uses chonkie SemanticChunker with BGE-M3 embeddings for boundary detection.
    For PDFs: chunks each page independently with correct page_number.
    For MD/HTML/TXT: chunks the full text with page_number=0.

    Args:
        parsed_doc: Output of parse_document() {"text", "metadata", "pages"}.
        embedder: Embedder instance (reused from project BGE-M3).
        chunk_size: Max chunk size in tokens (default 512).
        similarity_threshold: Cosine similarity threshold for splitting (default 0.5).
        overlap_ratio: Fraction of prev chunk tail to prepend (default 0.15).
        base_url: Optional base URL for constructing file URLs.

    Returns:
        List of Chunk objects with power plant metadata.
    """
    text = _safe_str(parsed_doc.get("text", ""))
    file_meta = _safe_dict(parsed_doc.get("metadata", {}))
    source_file = _safe_str(file_meta.get("source_file", ""))
    camera_brand = _safe_str(file_meta.get("camera_brand", "unknown"))
    pages = parsed_doc.get("pages", [])

    # Derive new-format metadata fields
    file_name = Path_safe(source_file).name if source_file else "unknown"
    url = _build_url(base_url, file_name) if base_url else file_name
    equipment_id = _extract_equipment_id(file_name)

    if not text.strip():
        logger.warning("Empty document: %s", source_file)
        return []

    # ── 结构化检测：模板化记录 → 规则分块 ──
    from ingestion.structured_detector import is_structured_block, chunk_by_category

    if is_structured_block(text):
        logger.info("  Detected structured content → rule-based chunking for %s", source_file)
        raw_chunks = chunk_by_category(text)
        chunk_method = "rule_based_structured"
    else:
        # 原有语义分块路径不变
        sem_chunker = _create_semantic_chunker(embedder, chunk_size, similarity_threshold)
        if pages:
            raw_chunks = []
            for page_num, page_text in pages:
                page_chunks = _chunk_one_text(page_text, sem_chunker, overlap_ratio)
                for t in page_chunks:
                    raw_chunks.append((t, page_num))
        else:
            page_chunks = _chunk_one_text(text, sem_chunker, overlap_ratio)
            raw_chunks = [(t, 0) for t in page_chunks]
        chunk_method = "semantic"

    # ── 统一生成 Chunk 对象 ──
    all_chunks: List[Chunk] = []
    chunk_counter = [0]

    if chunk_method == "rule_based_structured":
        # 结构化路径：无需重叠，无需逐页循环（已按类别分组）
        for t in raw_chunks:
            category = _infer_category(t, source_file)
            all_chunks.append(
                Chunk(
                    text=t.strip(),
                    metadata=ChunkMetadata(
                        file_name=file_name,
                        url=url,
                        category=category,
                        equipment_id=equipment_id,
                        source_file=source_file,
                        chunk_index=chunk_counter[0],
                        page_number=0,
                        chunk_method=chunk_method,
                    ),
                )
            )
            chunk_counter[0] += 1
    else:
        # 语义路径：原有逻辑保留
        for item in raw_chunks:
            t, page_num = item
            category = _infer_category(t, source_file)
            all_chunks.append(
                Chunk(
                    text=t.strip(),
                    metadata=ChunkMetadata(
                        file_name=file_name,
                        url=url,
                        category=category,
                        equipment_id=equipment_id,
                        source_file=source_file,
                        chunk_index=chunk_counter[0],
                        page_number=page_num,
                        chunk_method=chunk_method,
                    ),
                )
            )
            chunk_counter[0] += 1

    logger.debug("Semantic chunked %s → %d chunks", source_file, len(all_chunks))
    return all_chunks


# ============================================================================
# Helpers
# ============================================================================


def _safe_str(value: object) -> str:
    if value is None:
        return ""
    return str(value) if not isinstance(value, str) else value


def _safe_dict(value: object) -> Dict[str, object]:
    if value is None:
        return {}
    return value if isinstance(value, dict) else {}


def Path_safe(path_str: str):
    """Return a pathlib.Path, handling empty strings."""
    from pathlib import Path
    return Path(path_str) if path_str else Path()


def _build_url(base_url: str, file_name: str) -> str:
    """Build a URL from base_url and file_name."""
    base = base_url.rstrip("/")
    return f"{base}/{file_name}" if base else file_name


def _extract_equipment_id(file_name: str) -> str:
    """Try to extract equipment ID from filename patterns like '5号机组_xxx'."""
    m = re.search(r"(\d+号机组|机组\d+)", file_name)
    return m.group(1) if m else ""
