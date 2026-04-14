"""RAG Ingestion Pipeline — FastAPI Application."""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import settings
from src.config.models import PipelineResult
from src.processors.pipeline import pipeline
from src.embeddings.vector_store import vector_store
from src.storage.metadata_store import metadata_store
from src.retriever.retriever import retriever

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "healthy", "vector_store": vector_store.stats()}


@app.post("/api/v1/ingest", response_model=PipelineResult)
async def ingest_files(
    files: List[UploadFile] = File(...),
    collection_name: str = Form(""),
    description: str = Form(""),
    created_by: str = Form(""),
    collection_id: Optional[str] = Form(None),
):
    for f in files:
        ext = Path(f.filename or "").suffix.lower().lstrip(".")
        if ext not in settings.SUPPORTED_EXTENSIONS:
            raise HTTPException(400, f"Unsupported: .{ext}")

    temp_dir = settings.UPLOAD_DIR / str(uuid.uuid4())
    temp_dir.mkdir(parents=True, exist_ok=True)
    file_paths = []
    try:
        for f in files:
            dest = temp_dir / (f.filename or "unnamed")
            with open(dest, "wb") as out:
                out.write(await f.read())
            file_paths.append(str(dest))
        return await pipeline.ingest_files(
            file_paths=file_paths,
            collection_name=collection_name,
            description=description,
            created_by=created_by,
            collection_id=collection_id,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/api/v1/ingest/local", response_model=PipelineResult)
async def ingest_local(
    file_paths: List[str],
    collection_name: str = "",
    description: str = "",
    created_by: str = "",
    collection_id: Optional[str] = None,
):
    for fp in file_paths:
        if not os.path.exists(fp):
            raise HTTPException(400, f"File not found: {fp}")
    return await pipeline.ingest_files(
        file_paths=file_paths,
        collection_name=collection_name,
        description=description,
        created_by=created_by,
        collection_id=collection_id,
    )


@app.post("/api/v1/ingest/directory", response_model=PipelineResult)
async def ingest_directory(
    root_dir: str,
    collection_name: str = "",
    description: str = "",
    created_by: str = "",
    collection_id: Optional[str] = None,
):
    """Recursively ingest all PDFs/PPTXs from a directory tree (any depth)."""
    if not os.path.isdir(root_dir):
        raise HTTPException(400, f"Directory not found: {root_dir}")
    return await pipeline.ingest_directory(
        root_dir=root_dir,
        collection_name=collection_name,
        description=description,
        created_by=created_by,
        collection_id=collection_id,
    )


@app.post("/api/v1/search")
async def search(query: str = Form(...), top_k: int = Form(5), source: Optional[str] = Form(None)):
    """Pure vector search — results NOT saved."""
    results = vector_store.search(query=query, top_k=top_k, source=source)
    return {"query": query, "results": results, "total": len(results)}


@app.post("/api/v1/query")
async def query_and_save(
    query: str = Form(...),
    top_k: int = Form(5),
    collection_id: Optional[str] = Form(None),
):
    """Search and save the result to disk for later review."""
    result = await retriever.search_and_save(query, top_k, collection_id)
    return result.model_dump()


@app.post("/api/v1/query/batch")
async def batch_query(
    queries: List[str],
    top_k: int = 5,
    collection_id: Optional[str] = None,
):
    """Run multiple queries and save all results as a session."""
    session = await retriever.batch_search(queries, top_k, collection_id)
    return session.model_dump()


@app.get("/api/v1/query/sessions")
async def list_sessions():
    """List all saved query sessions."""
    return await retriever.list_sessions()


@app.get("/api/v1/query/sessions/{session_id}")
async def get_session(session_id: str):
    """Load a saved query session."""
    session = await retriever.load_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session.model_dump()


@app.get("/api/v1/collections")
async def list_collections():
    return [c.model_dump() for c in await metadata_store.list_collections()]


@app.get("/api/v1/collections/{cid}")
async def get_collection(cid: str):
    meta = await metadata_store.get_collection(cid)
    if not meta:
        raise HTTPException(404, "Collection not found")
    return meta.model_dump()


@app.get("/api/v1/collections/{cid}/files")
async def list_files(cid: str):
    return [f.model_dump() for f in await metadata_store.list_file_meta(cid)]


@app.get("/api/v1/collections/{cid}/files/{fid}")
async def get_file(cid: str, fid: str):
    meta = await metadata_store.get_file_meta(cid, fid)
    if not meta:
        raise HTTPException(404, "File not found")
    return meta.model_dump()


@app.delete("/api/v1/collections/{cid}")
async def delete_collection(cid: str):
    removed = vector_store.delete_by_source(cid)
    deleted = await metadata_store.delete_collection(cid)
    return {"deleted": deleted, "vectors_removed": removed}


@app.get("/api/v1/vector-store/stats")
async def stats():
    return vector_store.stats()
