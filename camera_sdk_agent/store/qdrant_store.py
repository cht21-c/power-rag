"""
Qdrant vector store operations.

Provides:
- create_collection(): Create the collection with the correct schema.
- insert_chunks(): Batch-insert chunks with payload (metadata).
- search(): Vector similarity search with optional metadata filtering.
- delete_collection(): Drop and recreate (for re-ingestion).
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_LOCAL_PATH,
    QDRANT_COLLECTION_NAME,
    QDRANT_VECTOR_DIM,
    QDRANT_DISTANCE_METRIC,
)

logger = logging.getLogger(__name__)


class QdrantStore:
    """Manages the Qdrant collection for camera SDK chunks.

    Supports two modes:
    - Local file mode (default): data stored in a local directory, no Docker needed.
    - Server mode: connect to a Qdrant server (Docker or remote).
    """

    def __init__(
        self,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        local_path: str = QDRANT_LOCAL_PATH,
        collection_name: str = QDRANT_COLLECTION_NAME,
        vector_dim: int = QDRANT_VECTOR_DIM,
        distance_metric: str = QDRANT_DISTANCE_METRIC,
    ):
        """Initialize the Qdrant client.

        If host is empty, uses local file mode (no server needed).
        Otherwise connects to a Qdrant server at host:port.

        Args:
            host: Qdrant server host. Empty string → local file mode.
            port: Qdrant server port (server mode only).
            local_path: Directory for local file storage.
            collection_name: Name of the collection.
            vector_dim: Dimension of embedding vectors (1024 for BGE-M3).
            distance_metric: Distance metric (Cosine, Dot, Euclid).
        """
        self.host = host
        self.port = port
        self.local_path = local_path
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.distance_metric = distance_metric
        self._mode = "local" if not host else "server"

        if not host:
            # Local file mode — no Docker required
            self.client = QdrantClient(path=local_path, timeout=30)
            logger.info("QdrantStore (local mode): %s (collection: %s)",
                         local_path, collection_name)
        else:
            # Server mode — connect to Docker or remote Qdrant
            self.client = QdrantClient(host=host, port=port, timeout=30)
            logger.info("QdrantStore (server mode): %s:%d (collection: %s)",
                         host, port, collection_name)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def collection_exists(self) -> bool:
        """Check whether the collection already exists."""
        try:
            collections = self.client.get_collections()
            return any(c.name == self.collection_name for c in collections.collections)
        except Exception:
            return False

    def create_collection(self, force_recreate: bool = False) -> None:
        """Create the Qdrant collection.

        Args:
            force_recreate: If True, drop and recreate the collection
                            if it already exists.
        """
        if self.collection_exists():
            if force_recreate:
                logger.info("Recreating collection '%s' ...", self.collection_name)
                self.client.delete_collection(self.collection_name)
            else:
                logger.info("Collection '%s' already exists.", self.collection_name)
                return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=rest.VectorParams(
                size=self.vector_dim,
                distance=self.distance_metric,
            ),
        )
        logger.info(
            "Collection '%s' created (dim=%d, distance=%s).",
            self.collection_name,
            self.vector_dim,
            self.distance_metric,
        )

    def delete_collection(self) -> None:
        """Delete the collection if it exists."""
        if self.collection_exists():
            self.client.delete_collection(self.collection_name)
            logger.info("Collection '%s' deleted.", self.collection_name)

    # ------------------------------------------------------------------
    # Data insertion
    # ------------------------------------------------------------------

    def insert_chunks(
        self,
        chunks: list,
        vectors: np.ndarray,
        batch_size: int = 100,
    ) -> int:
        """Insert chunks and their vectors into Qdrant.

        Args:
            chunks: List of Chunk objects (from chunker.py).
            vectors: NumPy array of shape (n_chunks, vector_dim).
            batch_size: Number of points to upsert per batch.

        Returns:
            Number of points inserted.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(vectors)} vectors"
            )

        total = 0
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i: i + batch_size]
            batch_vectors = vectors[i: i + batch_size]

            points = []

            for j, (chunk, vec) in enumerate(zip(batch_chunks, batch_vectors)):
                chunk_dict = chunk.to_dict()
                payload: Dict[str, Any] = {
                    "text": chunk_dict["text"],
                    **chunk_dict["metadata"],
                }
                point_id = str(uuid.uuid4())

                points.append(
                    rest.PointStruct(
                        id=point_id,
                        vector=vec.tolist(),
                        payload=payload,
                    )
                )

            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,
                )
                total += len(points)
                logger.debug("Upserted batch %d/%d", total, len(chunks))
            except Exception as e:
                logger.error("Failed to upsert batch starting at index %d: %s", i, e)
                raise

        logger.info("Inserted %d points into '%s'.", total, self.collection_name)
        return total

    # ------------------------------------------------------------------
    # Search / Query
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        camera_brand: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search with optional metadata filtering.

        Args:
            query_vector: 1-D embedding vector for the query.
            top_k: Number of results to return.
            camera_brand: If set, filter results to this brand.
            score_threshold: Minimum similarity score (cosine).

        Returns:
            List of results, each containing:
                {id, score, text, camera_brand, sdk_version, function_name, ...}
        """
        # Build filter
        query_filter = None
        if camera_brand:
            query_filter = rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="camera_brand",
                        match=rest.MatchValue(value=camera_brand),
                    )
                ]
            )

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist(),
                limit=top_k,
                query_filter=query_filter,
                score_threshold=score_threshold,
                with_payload=True,
            )
            results = response.points
        except Exception as e:
            logger.error("Search failed: %s", e)
            return []

        # Normalize results
        output: List[Dict[str, Any]] = []
        for hit in results:
            output.append({
                "id": str(hit.id),
                "score": float(hit.score),
                "text": str(hit.payload.get("text", "")),
                "camera_brand": str(hit.payload.get("camera_brand", "unknown")),
                "sdk_version": str(hit.payload.get("sdk_version", "unknown")),
                "function_name": str(hit.payload.get("function_name", "unknown")),
                "category": str(hit.payload.get("category", "general")),
                "source_file": str(hit.payload.get("source_file", "")),
                "chunk_index": int(hit.payload.get("chunk_index", 0)),
                "page_number": int(hit.payload.get("page_number", 0)),
            })
        return output

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_points(self) -> int:
        """Return the number of points in the collection."""
        try:
            import json, urllib.request
            url = f"http://{self.host}:{self.port}/collections/{self.collection_name}"
            with urllib.request.urlopen(url, timeout=5) as r:
                data = json.loads(r.read())
            return data.get("result", {}).get("points_count", 0)
        except Exception:
            return 0

    def get_all_texts(self) -> List[str]:
        """Retrieve all document texts from the collection (for BM25 indexing).

        Note: This scans the entire collection. For large collections,
        consider a separate lightweight store.

        Returns:
            List of all text payloads.
        """
        texts: List[str] = []
        offset = None
        batch_size = 100

        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break
            for pt in points:
                texts.append(str(pt.payload.get("text", "")))
            offset = next_offset
            if next_offset is None:
                break

        return texts

    def get_all_chunks_with_ids(self) -> List[Dict[str, Any]]:
        """Retrieve all chunks with their IDs and payloads (for BM25 cross-referencing).

        Returns:
            List of {id, text, ...metadata}.
        """
        results: List[Dict[str, Any]] = []
        offset = None
        batch_size = 100

        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break
            for pt in points:
                payload = pt.payload or {}
                results.append({
                    "id": str(pt.id),
                    "text": str(payload.get("text", "")),
                    "camera_brand": str(payload.get("camera_brand", "unknown")),
                    "sdk_version": str(payload.get("sdk_version", "unknown")),
                    "function_name": str(payload.get("function_name", "unknown")),
                    "category": str(payload.get("category", "general")),
                    "source_file": str(payload.get("source_file", "")),
                    "chunk_index": int(payload.get("chunk_index", 0)),
                    "page_number": int(payload.get("page_number", 0)),
                })
            offset = next_offset
            if next_offset is None:
                break

        return results
