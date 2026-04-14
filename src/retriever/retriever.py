"""
Retriever — search the vector store and persist query results.

Features:
  - Single query with result storage
  - Batch queries from a file (one query per line)
  - Query sessions (group multiple queries together)
  - All results saved to data/output/queries/ as JSON
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import aiofiles

from src.config.settings import settings
from src.config.logging_cfg import logger
from src.config.models import QueryResult, QuerySession, RetrievedChunk
from src.embeddings.vector_store import vector_store


class Retriever:
    """Search vector store and persist results."""

    def __init__(self):
        self.queries_dir = settings.OUTPUT_DIR / "queries"
        self.queries_dir.mkdir(parents=True, exist_ok=True)

    # ── Single query ──

    def search(
        self,
        query: str,
        top_k: int = 5,
        collection_id: Optional[str] = None,
    ) -> QueryResult:
        """
        Search the vector store and return a QueryResult with full metadata.
        """
        start = time.time()

        raw_results = vector_store.search(
            query=query, top_k=top_k, source=collection_id
        )

        chunks = [
            RetrievedChunk(
                chunk_id=r.get("chunk_id", ""),
                content=r["content"],
                summary=r.get("summary", ""),
                keywords=r.get("keywords", []),
                folder_meta=r.get("folder_meta", ""),
                document_id=r.get("document_id", ""),
                document_name=r.get("document_name", r.get("file_name", "")),
                file_name=r.get("file_name", ""),
                file_url=r.get("file_url", ""),
                section=r.get("section", ""),
                subsection=r.get("subsection", ""),
                page=r.get("page"),
                page_or_slide_number=r.get("page_or_slide_number", r.get("page")),
                chunk_type=r.get("chunk_type", r.get("type", "text")),
                type=r.get("type", "text"),
                context_before=r.get("context_before", ""),
                context_after=r.get("context_after", ""),
                confidence_score=r.get("confidence_score", 0.92),
                source=r.get("source", ""),
                score=r.get("score", 0.0),
                created_at=r.get("created_at", ""),
            )
            for r in raw_results
        ]

        elapsed_ms = (time.time() - start) * 1000

        result = QueryResult(
            query=query,
            collection_id=collection_id,
            top_k=top_k,
            results=chunks,
            total_results=len(chunks),
            query_time_ms=round(elapsed_ms, 2),
        )

        logger.info(
            "retriever.search",
            query=query[:80],
            results=len(chunks),
            elapsed_ms=round(elapsed_ms, 1),
        )
        return result

    # ── Save single query result ──

    async def search_and_save(
        self,
        query: str,
        top_k: int = 5,
        collection_id: Optional[str] = None,
    ) -> QueryResult:
        """Search and persist the result to disk."""
        result = self.search(query, top_k, collection_id)
        await self._save_query_result(result)
        return result

    # ── Batch queries ──

    async def batch_search(
        self,
        queries: List[str],
        top_k: int = 5,
        collection_id: Optional[str] = None,
    ) -> QuerySession:
        """
        Run multiple queries and save all results as a session.
        """
        session = QuerySession(collection_id=collection_id)

        for q in queries:
            q = q.strip()
            if not q:
                continue
            result = self.search(q, top_k, collection_id)
            session.queries.append(result)

        await self._save_session(session)

        logger.info(
            "retriever.batch_search",
            queries=len(session.queries),
            session_id=session.session_id,
        )
        return session

    async def batch_search_from_file(
        self,
        query_file: str,
        top_k: int = 5,
        collection_id: Optional[str] = None,
    ) -> QuerySession:
        """
        Read queries from a text file (one per line) and run batch search.
        """
        path = Path(query_file)
        if not path.exists():
            raise FileNotFoundError(f"Query file not found: {query_file}")

        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            content = await f.read()

        queries = [line.strip() for line in content.strip().splitlines() if line.strip()]
        if not queries:
            raise ValueError(f"No queries found in {query_file}")

        logger.info("retriever.batch_from_file", file=query_file, queries=len(queries))
        return await self.batch_search(queries, top_k, collection_id)

    # ── Persistence ──

    async def _save_query_result(self, result: QueryResult):
        """Save a single query result."""
        path = self.queries_dir / f"{result.query_id}.json"
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(result.model_dump_json(indent=2))

    async def _save_session(self, session: QuerySession):
        """Save a full query session."""
        session_dir = self.queries_dir / f"session_{session.session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)

        # Save session metadata
        async with aiofiles.open(session_dir / "session.json", "w", encoding="utf-8") as f:
            await f.write(session.model_dump_json(indent=2))

        # Save each query result separately for easy inspection
        for i, qr in enumerate(session.queries):
            async with aiofiles.open(session_dir / f"query_{i:03d}.json", "w", encoding="utf-8") as f:
                await f.write(qr.model_dump_json(indent=2))

        # Save a summary CSV for quick review
        lines = ["query,top_score,top_file,top_page,results_count,time_ms"]
        for qr in session.queries:
            top = qr.results[0] if qr.results else None
            top_score = f"{top.score:.3f}" if top else "0.000"
            top_file  = top.file_name if top else ""
            top_page  = top.page if (top and top.page is not None) else ""
            lines.append(
                f'"{qr.query}",'
                f'{top_score},'
                f'"{top_file}",'
                f'{top_page},'
                f'{qr.total_results},'
                f'{qr.query_time_ms}'
            )
        async with aiofiles.open(session_dir / "summary.csv", "w", encoding="utf-8") as f:
            await f.write("\n".join(lines))

    # ── Load saved results ──

    async def list_saved_queries(self) -> List[str]:
        """List all saved query result IDs."""
        return [
            p.stem
            for p in self.queries_dir.glob("*.json")
            if not p.name.startswith(".")
        ]

    async def list_sessions(self) -> List[str]:
        """List all saved session IDs."""
        return [
            p.name.replace("session_", "")
            for p in self.queries_dir.iterdir()
            if p.is_dir() and p.name.startswith("session_")
        ]

    async def load_query_result(self, query_id: str) -> Optional[QueryResult]:
        """Load a saved query result by ID."""
        path = self.queries_dir / f"{query_id}.json"
        if not path.exists():
            return None
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            return QueryResult(**json.loads(await f.read()))

    async def load_session(self, session_id: str) -> Optional[QuerySession]:
        """Load a saved session by ID."""
        path = self.queries_dir / f"session_{session_id}" / "session.json"
        if not path.exists():
            return None
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            return QuerySession(**json.loads(await f.read()))


retriever = Retriever()
