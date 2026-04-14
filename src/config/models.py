"""
Data Models — matches notebook's output_schema + task_df columns.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── File metadata (notebook's task_df) ──

class FileMetadata(BaseModel):
    file_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_name: str
    raw_file_name: str = ""
    file_type: str = ""
    file_size: str = ""
    file_url: str = ""
    storage_path: str = ""
    relative_path: str = ""              # subfolder path relative to root dir (e.g. "hr/policies/")

    # Processing state (notebook flags)
    is_downloaded: str = "pending"
    is_chunked: bool = False
    is_embedded: bool = False
    is_redacted: bool = False
    status: str = "pending"          # "pending" | "processing" | "done" | "failed"

    # Extraction stats
    total_pages: int = 0
    total_text_rows: int = 0
    total_table_rows: int = 0
    total_image_rows: int = 0
    total_chunks: int = 0
    total_errors: int = 0

    # Timestamps
    triggered_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None
    created_by: str = ""

    error: Optional[str] = None
    extra_metadata: Dict[str, Any] = {}


# ── Collection ──

class CollectionMetadata(BaseModel):
    collection_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    collection_name: str = ""
    description: str = ""
    created_by: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    file_count: int = 0
    total_size: int = 0
    files: List[FileMetadata] = []


# ── API response ──

class PipelineResult(BaseModel):
    collection_id: str
    files_processed: int
    total_rows: int
    text_rows: int
    table_rows: int
    image_rows: int
    error_rows: int
    chunks_embedded: int
    processing_time_sec: float


# ── Query / Retrieval models ──

class RetrievedChunk(BaseModel):
    """A single search result with full metadata."""
    chunk_id: str = ""
    content: str
    summary: str = ""
    keywords: List[str] = []
    folder_meta: str = ""
    document_id: str = ""
    document_name: str = ""
    file_name: str = ""
    file_url: str = ""
    section: str = ""
    subsection: str = ""
    page: Optional[int] = None
    page_or_slide_number: Optional[int] = None
    chunk_type: str = "text"
    type: str = "text"
    context_before: str = ""
    context_after: str = ""
    confidence_score: float = 0.92
    source: str = ""
    score: float = 0.0
    created_at: str = ""


class QueryResult(BaseModel):
    """Stored result of a single query."""
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    collection_id: Optional[str] = None
    top_k: int = 5
    results: List[RetrievedChunk] = []
    total_results: int = 0
    query_time_ms: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class QuerySession(BaseModel):
    """A session of multiple queries against a collection."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    collection_id: Optional[str] = None
    queries: List[QueryResult] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
