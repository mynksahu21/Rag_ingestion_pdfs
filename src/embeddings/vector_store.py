"""
OpenAI Embedding Service + FAISS Vector Store with Hybrid Retrieval.
Replaces: Databricks Vector Search + databricks-bge-large-en.
Now uses: Azure OpenAI text-embedding-ada-002 + FAISS + BM25 (rank_bm25).

Hybrid search formula: score = 0.7 * semantic_score + 0.3 * keyword_score
Both scores are normalized to [0, 1] before combining.
"""
from __future__ import annotations

import pickle
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from openai import AzureOpenAI
from rank_bm25 import BM25Okapi
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import settings
from src.config.logging_cfg import logger
from src.utils.openai_client import get_openai_client


# ═══════════════════════════════════════════════════════════════
#  OpenAI Embedding Service
# ═══════════════════════════════════════════════════════════════


class OpenAIEmbeddingService:
    """Generate embeddings using Azure OpenAI text-embedding-ada-002."""

    def __init__(self):
        self._client: Optional[AzureOpenAI] = None

    def _get_client(self) -> AzureOpenAI:
        if self._client is None:
            if not settings.AZURE_OPENAI_API_KEY or not settings.AZURE_OPENAI_ENDPOINT:
                raise ValueError(
                    "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set in .env"
                )
            self._client = get_openai_client()
        return self._client

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts. Returns ndarray of shape (N, dim)."""
        if not texts:
            return np.empty((0, settings.EMBEDDING_DIMENSION), dtype="float32")

        client = self._get_client()

        # Azure OpenAI has a batch limit; chunk into batches of 16
        all_embeddings: List[List[float]] = []
        batch_size = 16

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = client.embeddings.create(
                model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                input=batch,
            )
            for item in response.data:
                all_embeddings.append(item.embedding)

        return np.array(all_embeddings, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query. Returns shape (1, dim)."""
        return self.embed_texts([query])

    @property
    def dimension(self) -> int:
        return settings.EMBEDDING_DIMENSION


embedding_service = OpenAIEmbeddingService()


# ═══════════════════════════════════════════════════════════════
#  FAISS Vector Store
# ═══════════════════════════════════════════════════════════════


class FAISSVectorStore:
    """
    FAISS-backed vector store — replaces Databricks Vector Search.
    Stores same columns notebook syncs: id, content, file_name, file_url, page, source, type
    """

    # Hybrid retrieval weights
    SEMANTIC_WEIGHT: float = 0.7
    KEYWORD_WEIGHT: float = 0.3

    def __init__(self, index_dir: Path | None = None):
        self.index_dir = index_dir or settings.VECTOR_INDEX_DIR
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._index: Optional[faiss.IndexIDMap] = None
        self._metadata: Dict[int, Dict[str, Any]] = {}
        self._next_id: int = 0
        # BM25 keyword index — rebuilt in-memory from metadata
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_ids: List[int] = []  # maps bm25 corpus index → faiss internal id
        self._load()
        self._rebuild_bm25()

    def _index_path(self) -> Path:
        return self.index_dir / "faiss.index"

    def _meta_path(self) -> Path:
        return self.index_dir / "metadata.pkl"

    def _load(self):
        if self._index_path().exists() and self._meta_path().exists():
            try:
                self._index = faiss.read_index(str(self._index_path()))
                with open(self._meta_path(), "rb") as f:
                    data = pickle.load(f)
                self._metadata = data.get("metadata", {})
                self._next_id = data.get("next_id", 0)
                logger.info("faiss.loaded", vectors=self._index.ntotal)
            except Exception as exc:
                logger.error("faiss.load_failed", error=str(exc))
                self._init_fresh()
        else:
            self._init_fresh()

    def _init_fresh(self):
        dim = embedding_service.dimension
        base = faiss.IndexFlatIP(dim)  # inner product (cosine after L2 norm)
        self._index = faiss.IndexIDMap(base)
        self._metadata = {}
        self._next_id = 0

    def _rebuild_bm25(self):
        """Rebuild the in-memory BM25 index from current metadata."""
        if not self._metadata:
            self._bm25 = None
            self._bm25_ids = []
            return

        self._bm25_ids = sorted(self._metadata.keys())
        corpus: List[List[str]] = []
        for fid in self._bm25_ids:
            meta = self._metadata[fid]
            # Combine content + keywords for richer keyword matching
            text = meta.get("content", "")
            keywords = meta.get("keywords", [])
            if keywords:
                text = text + " " + " ".join(keywords)
            corpus.append(text.lower().split())

        self._bm25 = BM25Okapi(corpus)
        logger.info("bm25.rebuilt", docs=len(corpus))

    def save(self):
        with self._lock:
            faiss.write_index(self._index, str(self._index_path()))
            with open(self._meta_path(), "wb") as f:
                pickle.dump(
                    {"metadata": self._metadata, "next_id": self._next_id}, f
                )

    @staticmethod
    def _build_embed_text(chunk: dict) -> str:
        parts: List[str] = []

        if chunk.get("file_name"):
            parts.append(f"Document: {chunk['file_name']}")
        if chunk.get("chunk_type"):
            parts.append(f"Type: {chunk['chunk_type']}")
        if chunk.get("page"):
            parts.append(f"Page: {chunk['page']}")
        if chunk.get("keywords"):
            parts.append(f"Keywords: {', '.join(chunk['keywords'])}")
        if chunk.get("context_before"):
            parts.append(f"Context Before: {chunk['context_before']}")

        parts.append(chunk.get("content", ""))

        if chunk.get("context_after"):
            parts.append(f"Context After: {chunk['context_after']}")

        return "\n".join(parts).strip()

    def add_chunks(self, chunks: List[dict]) -> int:
        """Embed and index chunks. Each must have: id, content, file_name, etc."""
        if not chunks:
            return 0

        texts = [self._build_embed_text(c) for c in chunks]
        embeddings = embedding_service.embed_texts(texts)
        faiss.normalize_L2(embeddings)

        with self._lock:
            ids = np.arange(
                self._next_id, self._next_id + len(chunks), dtype="int64"
            )
            for i, chunk in enumerate(chunks):
                fid = int(ids[i])
                # Write embedding vector back into the chunk dict
                chunk["embedding"] = embeddings[i].tolist()
                self._metadata[fid] = {
                    # Core identifiers
                    "chunk_id": chunk["id"],
                    "content": chunk["content"],
                    "embedding": chunk["embedding"],
                    # Summaries & keywords
                    "summary": chunk.get("summary", ""),
                    "keywords": chunk.get("keywords", []),
                    # Image summary fields
                    "image_description": chunk.get("image_description"),
                    "graph_data_table": chunk.get("graph_data_table"),
                    "graph_description": chunk.get("graph_description"),
                    # File references
                    "folder_meta": chunk.get("folder_meta", ""),
                    "document_id": chunk.get("document_id", ""),
                    "document_name": chunk.get("document_name", chunk.get("file_name", "")),
                    "file_name": chunk.get("file_name", ""),
                    "file_url": chunk.get("file_url", ""),
                    # Location
                    "section": chunk.get("section", ""),
                    "subsection": chunk.get("subsection", ""),
                    "page": chunk.get("page"),
                    "page_or_slide_number": chunk.get("page_or_slide_number", chunk.get("page")),
                    # Classification
                    "chunk_type": chunk.get("chunk_type", chunk.get("type", "text")),
                    "type": chunk.get("type", "text"),
                    # Context
                    "context_before": chunk.get("context_before", ""),
                    "context_after": chunk.get("context_after", ""),
                    # Quality
                    "confidence_score": chunk.get("confidence_score", 0.92),
                    # Source & image
                    "source": chunk.get("source", ""),
                    "image_path": chunk.get("image_path"),
                    # Timestamp
                    "created_at": chunk.get("created_at", ""),
                }
            self._index.add_with_ids(embeddings, ids)
            self._next_id += len(chunks)

        self.save()
        self._rebuild_bm25()
        logger.info("faiss.added", count=len(chunks), total=self._index.ntotal)
        return len(chunks)

    def search(
        self, query: str, top_k: int = 5, source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search: 0.7 * semantic + 0.3 * keyword (BM25).
        Both scores are min-max normalized to [0, 1] before combining.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        # ── 1. Semantic search (FAISS cosine similarity) ──
        qvec = embedding_service.embed_query(query)
        faiss.normalize_L2(qvec)

        # Fetch a wider candidate pool for re-ranking
        search_k = min(top_k * 10, self._index.ntotal)
        distances, ids = self._index.search(qvec, search_k)

        semantic_scores: Dict[int, float] = {}
        for score, fid in zip(distances[0], ids[0]):
            if fid != -1:
                semantic_scores[int(fid)] = float(score)

        # Normalize semantic scores to [0, 1]
        if semantic_scores:
            s_min = min(semantic_scores.values())
            s_max = max(semantic_scores.values())
            span = s_max - s_min
            if span > 0:
                semantic_scores = {k: (v - s_min) / span for k, v in semantic_scores.items()}
            else:
                semantic_scores = {k: 1.0 for k in semantic_scores}

        # ── 2. Keyword search (BM25) ──
        keyword_scores: Dict[int, float] = {}
        if self._bm25 is not None and self._bm25_ids:
            query_tokens = query.lower().split()
            raw_bm25 = self._bm25.get_scores(query_tokens)

            k_max = float(np.max(raw_bm25)) if raw_bm25.size > 0 else 0.0
            for idx, bm25_score in enumerate(raw_bm25):
                if idx < len(self._bm25_ids):
                    fid = self._bm25_ids[idx]
                    keyword_scores[fid] = float(bm25_score) / k_max if k_max > 0 else 0.0

        # ── 3. Hybrid fusion ──
        all_fids = set(semantic_scores.keys()) | set(keyword_scores.keys())
        hybrid: Dict[int, float] = {
            fid: (
                self.SEMANTIC_WEIGHT * semantic_scores.get(fid, 0.0)
                + self.KEYWORD_WEIGHT * keyword_scores.get(fid, 0.0)
            )
            for fid in all_fids
        }

        # ── 4. Rank, filter by source, return top_k ──
        results: List[Dict[str, Any]] = []
        for fid, score in sorted(hybrid.items(), key=lambda x: x[1], reverse=True):
            meta = self._metadata.get(fid)
            if meta is None:
                continue
            if source and meta.get("source") != source:
                continue
            results.append({**meta, "score": score})
            if len(results) >= top_k:
                break

        logger.info(
            "hybrid_search",
            query=query[:80],
            semantic_candidates=len(semantic_scores),
            keyword_candidates=len(keyword_scores),
            returned=len(results),
            weights=f"semantic={self.SEMANTIC_WEIGHT}, keyword={self.KEYWORD_WEIGHT}",
        )
        return results

    def delete_by_source(self, source: str) -> int:
        with self._lock:
            to_remove = [
                fid
                for fid, m in self._metadata.items()
                if m.get("source") == source
            ]
            if not to_remove:
                return 0
            self._index.remove_ids(np.array(to_remove, dtype="int64"))
            for fid in to_remove:
                del self._metadata[fid]
        self.save()
        self._rebuild_bm25()
        return len(to_remove)

    def stats(self) -> Dict[str, Any]:
        sources = set()
        type_counts: Dict[str, int] = {}
        for m in self._metadata.values():
            sources.add(m.get("source", ""))
            t = m.get("type", "text")
            type_counts[t] = type_counts.get(t, 0) + 1
        return {
            "total_vectors": self._index.ntotal if self._index else 0,
            "sources": list(sources),
            "type_distribution": type_counts,
            "dimension": embedding_service.dimension,
        }


vector_store = FAISSVectorStore()
