"""
RAG Ingestion — public API for use inside a larger pipeline.

Usage (async context):
    from ingest import ingest_files, ingest_directory, search, PipelineResult

    # Ingest a list of file paths
    result: PipelineResult = await ingest_files(
        file_paths=["/data/doc1.pdf", "/data/slides.pptx"],
        collection_name="Q1 Reports",
        collection_id="my-stable-id",   # omit to auto-generate
    )

    # Ingest every PDF/PPTX under a directory tree
    result = await ingest_directory(
        root_dir="/data/reports/",
        collection_name="All Reports",
    )

    # Vector search (returns raw dicts, does NOT save the query)
    hits = search(query="revenue forecast", top_k=5)

    # Full query — searches + saves the result to disk
    query_result = await query(query="revenue forecast", top_k=5)
"""
from __future__ import annotations

from typing import List, Optional

from src.config.models import PipelineResult, QueryResult
from src.processors.pipeline import pipeline
from src.embeddings.vector_store import vector_store
from src.retriever.retriever import retriever


async def ingest_files(
    file_paths: List[str],
    collection_name: str = "",
    description: str = "",
    created_by: str = "",
    collection_id: Optional[str] = None,
) -> PipelineResult:
    """
    Ingest an explicit list of PDF / PPTX file paths into the vector store.

    Args:
        file_paths:       Absolute paths to the files to ingest.
        collection_name:  Human-readable name for the collection.
        description:      Free-text description stored in collection metadata.
        created_by:       Owner / pipeline name for audit purposes.
        collection_id:    Stable ID — pass the same value on re-runs to resume
                          without re-processing files already marked "done".

    Returns:
        PipelineResult with stats (files processed, chunks embedded, timing …)
    """
    return await pipeline.ingest_files(
        file_paths=file_paths,
        collection_name=collection_name,
        description=description,
        created_by=created_by,
        collection_id=collection_id,
    )


async def ingest_directory(
    root_dir: str,
    collection_name: str = "",
    description: str = "",
    created_by: str = "",
    collection_id: Optional[str] = None,
) -> PipelineResult:
    """
    Recursively ingest all PDF / PPTX files found under *root_dir*.

    Args:
        root_dir:         Root directory to walk (any depth).
        collection_name:  Defaults to the directory's basename if omitted.
        description:      Free-text description stored in collection metadata.
        created_by:       Owner / pipeline name for audit purposes.
        collection_id:    Stable ID for resume support (see ingest_files).

    Returns:
        PipelineResult with stats (files processed, chunks embedded, timing …)
    """
    return await pipeline.ingest_directory(
        root_dir=root_dir,
        collection_name=collection_name,
        description=description,
        created_by=created_by,
        collection_id=collection_id,
    )


def search(
    query: str,
    top_k: int = 5,
    source: Optional[str] = None,
) -> list[dict]:
    """
    Pure vector search — results are NOT persisted.

    Args:
        query:   Natural-language query string.
        top_k:   Number of results to return.
        source:  Filter by collection_id (source field) — pass None to search all.

    Returns:
        List of chunk dicts ordered by relevance score.
    """
    return vector_store.search(query=query, top_k=top_k, source=source)


async def query(
    query_text: str,
    top_k: int = 5,
    collection_id: Optional[str] = None,
) -> QueryResult:
    """
    Search the vector store and persist the result to disk for later review.

    Args:
        query_text:     Natural-language query string.
        top_k:          Number of results to return.
        collection_id:  Limit search to a specific collection.

    Returns:
        QueryResult (query text, retrieved chunks, timing metadata).
    """
    return await retriever.search_and_save(query_text, top_k, collection_id)


# Re-export the result type so callers don't need to import from src.config.models
__all__ = [
    "ingest_files",
    "ingest_directory",
    "search",
    "query",
    "PipelineResult",
    "QueryResult",
]
