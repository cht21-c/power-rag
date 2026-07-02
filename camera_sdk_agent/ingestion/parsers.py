"""
Document parsers for Camera SDK documentation.

Supported formats:
- PDF: via PyMuPDF (fitz)
- Markdown: native reading with frontmatter extraction
- HTML: via BeautifulSoup with text extraction
- Plain text: native reading (fallback)
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# OCR lazy singleton — shared across all PDF pages
# ============================================================================

_ocr_instance = None


def _get_ocr():
    """Return a lazily-initialized RapidOCR instance (ONNX Runtime, CPU)."""
    global _ocr_instance
    if _ocr_instance is None:
        from rapidocr_onnxruntime import RapidOCR
        logger.info("Initializing RapidOCR (ONNX Runtime) ...")
        _ocr_instance = RapidOCR()
    return _ocr_instance

# ============================================================================
# Public API
# ============================================================================


def discover_documents(docs_dir: Path) -> List[Path]:
    """Recursively discover all supported documents under `docs_dir`.

    Args:
        docs_dir: Root directory for SDK documentation.

    Returns:
        List of file paths sorted by name.
    """
    supported_extensions = {".pdf", ".md", ".html", ".htm", ".txt"}
    files: List[Path] = []

    if not docs_dir.exists():
        logger.warning("Document directory does not exist: %s", docs_dir)
        return files

    for ext in supported_extensions:
        files.extend(docs_dir.rglob(f"*{ext}"))

    return sorted(files, key=lambda p: p.name)


def parse_document(file_path: Path) -> Dict[str, object]:
    """Parse a single document and return structured content.

    Args:
        file_path: Path to the document.

    Returns:
        Dictionary with keys:
            - "text": Full plain-text content
            - "metadata": Dict with file-level metadata (source_file, format,
              camera_brand, sdk_version, etc.)
            - "pages": List of (page_num, page_text) for PDFs; empty list otherwise.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        parsed = _parse_pdf(file_path)
    elif suffix in (".md", ".markdown"):
        parsed = _parse_markdown(file_path)
    elif suffix in (".html", ".htm"):
        parsed = _parse_html(file_path)
    else:
        parsed = _parse_text(file_path)

    # Enrich with file-level metadata
    metadata = _extract_file_metadata(file_path)
    return {
        "text": parsed["text"],
        "metadata": {**metadata, **parsed.get("extra_meta", {})},
        "pages": parsed.get("pages", []),
    }


# ============================================================================
# Per-format parsers
# ============================================================================


def _parse_pdf(file_path: Path) -> Dict[str, object]:
    """Parse a PDF file using PyMuPDF, tracking page numbers.

    Args:
        file_path: Path to PDF file.

    Returns:
        Dictionary with "text", "pages" (list of (page_num, page_text)),
        and optional "extra_meta".
    """
    try:
        import fitz  # pymupdf
    except ImportError:
        logger.error("PyMuPDF not installed. Install with: pip install pymupdf")
        raise

    doc = fitz.open(str(file_path))
    full_text: List[str] = []
    pages: List[tuple] = []  # [(page_num, page_text), ...]

    for page_num, page in enumerate(doc):
        text = page.get_text("text")

        # OCR fallback: scanned pages with little or no extractable text
        if len(text.strip()) < 200:
            logger.info("  Page %d: low text (%d chars), running OCR ...",
                         page_num + 1, len(text.strip()))
            try:
                pix = page.get_pixmap(dpi=300)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                ocr_result, _ = _get_ocr()(img)
                if ocr_result:
                    text = "\n".join(line[1] for line in ocr_result)
                    logger.info("  Page %d: OCR extracted %d chars", page_num + 1, len(text))
                else:
                    logger.warning("  Page %d: OCR returned no text", page_num + 1)
            except Exception as e:
                logger.error("  Page %d: OCR failed (%s), using raw text", page_num + 1, e)

        full_text.append(text)
        if text.strip():
            # PyMuPDF page numbers are 0-indexed; convert to 1-indexed
            pages.append((page_num + 1, text.strip()))

    doc.close()
    return {"text": "\n\n".join(full_text), "pages": pages, "extra_meta": {}}


def _parse_markdown(file_path: Path) -> Dict[str, object]:
    """Parse a Markdown file, extracting frontmatter if present.

    Args:
        file_path: Path to Markdown file.

    Returns:
        Dictionary with "text" and "extra_meta" from YAML frontmatter.
    """
    raw_text = file_path.read_text(encoding="utf-8", errors="replace")
    extra_meta: Dict[str, str] = {}

    # Try to extract YAML frontmatter (--- ... ---)
    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw_text, re.DOTALL)
    if frontmatter_match:
        frontmatter_text = frontmatter_match.group(1)
        raw_text = raw_text[frontmatter_match.end():]
        # Simple key: value extraction (not a full YAML parser)
        for line in frontmatter_text.split("\n"):
            kv = re.match(r"^\s*(\w[\w_-]*)\s*:\s*(.+?)\s*$", line)
            if kv:
                extra_meta[kv.group(1)] = kv.group(2).strip("\"'")

    return {"text": raw_text, "pages": [], "extra_meta": extra_meta}


def _parse_html(file_path: Path) -> Dict[str, object]:
    """Parse an HTML file, extracting readable text.

    Args:
        file_path: Path to HTML file.

    Returns:
        Dictionary with "text".
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("BeautifulSoup not installed. Install with: pip install beautifulsoup4")
        raise

    raw_html = file_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw_html, "html.parser")

    # Remove script / style tags
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return {"text": text, "pages": [], "extra_meta": {}}


def _parse_text(file_path: Path) -> Dict[str, object]:
    """Parse a plain-text file (fallback).

    Args:
        file_path: Path to text file.

    Returns:
        Dictionary with "text".
    """
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return {"text": text, "pages": [], "extra_meta": {}}


# ============================================================================
# Metadata extraction
# ============================================================================


def _extract_file_metadata(file_path: Path) -> Dict[str, str]:
    """Extract metadata from file path conventions.

    Expected directory structure:
        sdk_docs/<camera_brand>/<sdk_version_or_misc>/*.pdf|md|html

    Examples:
        sdk_docs/Basler/2.3.0/programming_guide.pdf
          → camera_brand="Basler", sdk_version="2.3.0"
        sdk_docs/HikVision/user_manual_en.md
          → camera_brand="HikVision", sdk_version="unknown"

    Args:
        file_path: Path to the document.

    Returns:
        Metadata dictionary.
    """
    camera_brand = "unknown"
    sdk_version = "unknown"

    # Try to extract brand from parent directory name
    parts = file_path.relative_to(file_path.anchor).parts if file_path.is_absolute() else file_path.parts

    # Walk up from the file to find brand/version
    # Assume: .../sdk_docs/<brand>/...
    sdk_docs_idx = None
    for i, part in enumerate(parts):
        if part.lower() in ("sdk_docs", "sdk-docs", "docs"):
            sdk_docs_idx = i
            break

    if sdk_docs_idx is not None and sdk_docs_idx + 1 < len(parts):
        camera_brand = parts[sdk_docs_idx + 1]

    # Try to find version pattern in remaining path components
    version_pattern = re.compile(r"^(?:v|ver\.?\s*)?(\d+\.\d+(?:\.\d+)?)$")
    for part in parts:
        m = version_pattern.match(part)
        if m:
            sdk_version = m.group(1)
            break

    # Also try to extract version from filename
    if sdk_version == "unknown":
        m = version_pattern.search(file_path.stem)
        if m:
            sdk_version = m.group(1)

    return {
        "camera_brand": camera_brand,
        "sdk_version": sdk_version,
        "source_file": str(file_path),
        "format": file_path.suffix.lower().lstrip("."),
    }
