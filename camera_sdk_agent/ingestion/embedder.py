"""
BGE-M3 Embedding model wrapper.

Uses sentence-transformers to load BAAI/bge-m3 locally.
Provides batch embedding with configurable device, batch size, and FP16 support.
"""

import logging
import os
from typing import List, Optional

import numpy as np

# Force fully offline mode — never contact HuggingFace
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from config import EMBED_MODEL_NAME, EMBED_DEVICE, EMBED_BATCH_SIZE

logger = logging.getLogger(__name__)


class Embedder:
    """Wrapper around sentence-transformers for BGE-M3 embeddings.

    Usage:
        embedder = Embedder()
        vectors = embedder.encode(["text1", "text2"])
    """

    def __init__(
        self,
        model_name: str = EMBED_MODEL_NAME,
        device: str = EMBED_DEVICE,
        batch_size: int = EMBED_BATCH_SIZE,
        use_fp16: Optional[bool] = None,
    ):
        """Initialize the embedding model.

        Args:
            model_name: HuggingFace model identifier (default: BAAI/bge-m3).
            device: "cpu" or "cuda".
            batch_size: Number of texts to encode in one batch.
            use_fp16: Force FP16 mode (auto-detect if None: True when device is cuda).
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.use_fp16 = use_fp16 if use_fp16 is not None else (device == "cuda")
        self._model = None

        logger.info("Embedder configured: model=%s, device=%s, batch_size=%d, fp16=%s",
                     model_name, device, batch_size, self.use_fp16)

    @property
    def model(self):
        """Lazy-load the sentence-transformers model with FP16 support."""
        if self._model is None:
            logger.info("Loading embedding model: %s (fp16=%s) ...", self.model_name, self.use_fp16)
            try:
                from sentence_transformers import SentenceTransformer

                model_kwargs = {}
                if self.use_fp16:
                    import torch
                    model_kwargs = {"torch_dtype": torch.float16}

                # Force loading from local cache only
                model_kwargs["local_files_only"] = True

                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    trust_remote_code=True,
                    model_kwargs=model_kwargs,
                )
                # Warm up with a dummy encoding to load model fully
                _ = self._model.encode(["warmup"], show_progress_bar=False)
                logger.info("Model loaded successfully on %s", self.device)
            except Exception as e:
                logger.error("Failed to load embedding model: %s", e)
                raise
        return self._model

    @property
    def vector_dim(self) -> int:
        """Return the output vector dimension.

        BGE-M3 outputs 1024-dimensional vectors.
        """
        # BGE-M3 is known to output 1024-d vectors
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Encode a list of texts into embedding vectors.

        Args:
            texts: List of text strings to embed.
            show_progress: Whether to show a progress bar.

        Returns:
            NumPy array of shape (len(texts), vector_dim).
        """
        if not texts:
            logger.warning("encode() called with empty text list")
            return np.empty((0, self.vector_dim), dtype=np.float32)

        # BGE-M3 default prompt for retrieval tasks adds instruction prefix
        # (not strictly required but can improve quality for passage retrieval)
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # Cosine similarity → unit vectors
            convert_to_numpy=True,
        )

        return np.array(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string.

        Args:
            query: The query text.

        Returns:
            1-D NumPy array of shape (vector_dim,).
        """
        vec = self.encode([query], show_progress=False)
        return vec[0]

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode multiple query strings (convenience alias)."""
        return self.encode(queries, show_progress=False)
