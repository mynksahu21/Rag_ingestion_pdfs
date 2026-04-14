"""
RAG Ingestion Pipeline — Main Orchestrator.

Processing model: one file at a time.
  extract → summarize images → chunk → table titles → chunk summaries
  → context fill → embed → save  — all completed before the next file starts.

Resume support:
  Pass the same collection_id on re-run.  Any file whose stored status is
  already "done" is skipped; failed / partial files are re-processed.
"""
from __future__ import annotations

import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.config.settings import settings
from src.config.logging_cfg import logger
from src.config.models import CollectionMetadata, FileMetadata, PipelineResult
from src.extractors.dispatcher import process_file
from src.processors.image_summarizer import summarize_images
from src.processors.chunker import chunk_rows, generate_chunk_summaries, generate_table_titles
from src.embeddings.vector_store import vector_store
from src.storage.metadata_store import metadata_store


def scan_directory(root_dir: str, extensions: list[str]) -> list[tuple[str, str]]:
    """
    Recursively walk a directory tree and return (abs_path, relative_path) pairs
    for all files matching the given extensions.
    """
    root = Path(root_dir).resolve()
    results: list[tuple[str, str]] = []
    for ext in extensions:
        for p in root.rglob(f"*.{ext}"):
            rel = str(p.relative_to(root))
            results.append((str(p), rel))
    results.sort(key=lambda x: x[1])
    return results


class IngestionPipeline:
    """End-to-end ingestion — one file at a time with resume support."""

    async def ingest_directory(
        self,
        root_dir: str,
        collection_name: str = "",
        description: str = "",
        created_by: str = "",
        collection_id: str | None = None,
    ) -> PipelineResult:
        root = Path(root_dir).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {root_dir}")

        file_pairs = scan_directory(str(root), settings.SUPPORTED_EXTENSIONS)
        if not file_pairs:
            raise ValueError(f"No supported files found in {root_dir}")

        logger.info("pipeline.scan_directory", root=str(root), files_found=len(file_pairs))

        abs_paths = [fp[0] for fp in file_pairs]
        rel_paths = {os.path.basename(fp[0]): fp[1] for fp in file_pairs}

        return await self.ingest_files(
            file_paths=abs_paths,
            collection_name=collection_name or root.name,
            description=description or f"Ingested from {root_dir}",
            created_by=created_by,
            collection_id=collection_id,
            _relative_paths=rel_paths,
        )

    async def ingest_files(
        self,
        file_paths: List[str],
        collection_name: str = "",
        description: str = "",
        created_by: str = "",
        collection_id: Optional[str] = None,
        _relative_paths: Optional[Dict[str, str]] = None,
    ) -> PipelineResult:
        start = time.time()
        col_id = collection_id or str(uuid.uuid4())

        # ── 1. Collection metadata ──
        col_meta = await metadata_store.get_collection(col_id)
        if col_meta is None:
            col_meta = CollectionMetadata(
                collection_id=col_id,
                collection_name=collection_name or f"Collection {col_id[:8]}",
                description=description,
                created_by=created_by,
            )
            await metadata_store.save_collection(col_meta)

        # ── 2. Load existing file metadata (for resume detection) ──
        existing_metas: Dict[str, FileMetadata] = {
            fm.file_name: fm
            for fm in await metadata_store.list_file_meta(col_id)
        }
        already_done = sum(1 for fm in existing_metas.values() if fm.status == "done")
        logger.info(
            "pipeline.start",
            collection_id=col_id,
            files=len(file_paths),
            already_done=already_done,
        )

        all_file_metas: List[FileMetadata] = []
        new_embedded = 0

        # ── 3. Process each file independently ──
        for fpath in file_paths:
            fpath = str(fpath)
            fname = os.path.basename(fpath)
            fext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
            fsize = os.path.getsize(fpath) if os.path.exists(fpath) else 0
            rel_path = (_relative_paths or {}).get(fname, "")

            # ── Resume: skip files already successfully ingested ──
            if fname in existing_metas and existing_metas[fname].status == "done":
                logger.info("pipeline.file_already_done", file=fname)
                all_file_metas.append(existing_metas[fname])
                continue

            # Reuse file_id from a previous partial run so saved rows are
            # overwritten cleanly rather than creating orphan files.
            prior = existing_metas.get(fname)
            file_id = prior.file_id if prior else str(uuid.uuid4())

            fmeta = FileMetadata(
                file_id=file_id,
                file_name=fname,
                raw_file_name=fname,
                file_type=fext,
                file_size=f"{fsize} bytes",
                storage_path=fpath,
                relative_path=rel_path,
                is_downloaded="downloaded",
                status="processing",
            )
            await metadata_store.save_file_meta(col_id, fmeta)
            logger.info("pipeline.file_start", file=fname)

            # ── A. Extract ──
            rows: List[dict] = []
            try:
                rows = process_file(fpath, fname, fext)
                fmeta.total_text_rows = sum(1 for r in rows if r.get("type") == "text")
                fmeta.total_table_rows = sum(1 for r in rows if r.get("type") == "table")
                fmeta.total_image_rows = sum(1 for r in rows if r.get("type") == "image")
                fmeta.total_errors = sum(1 for r in rows if r.get("error"))
                fmeta.total_pages = max((r.get("page") or 0 for r in rows), default=0)
                fmeta.status = "extracted"
            except Exception as exc:
                fmeta.status = "failed"
                fmeta.error = str(exc)
                logger.error("pipeline.extract_failed", file=fname, error=str(exc))
                all_file_metas.append(fmeta)
                await metadata_store.save_file_meta(col_id, fmeta)
                await metadata_store.save_rows(col_id, file_id, rows)
                continue  # skip to next file

            await metadata_store.save_rows(col_id, file_id, rows)
            await metadata_store.save_file_meta(col_id, fmeta)

            # ── B. Page text context (for image summarization) ──
            page_texts: Dict[str, Dict[int, str]] = {}
            for row in rows:
                if row.get("type") == "text" and row.get("content"):
                    pg = row.get("page", 0) or 0
                    page_texts.setdefault(fname, {}).setdefault(pg, "")
                    page_texts[fname][pg] += "\n" + row["content"]

            # ── C. Summarize images ──
            try:
                image_summaries = await summarize_images(rows, page_texts)
            except Exception as exc:
                logger.error("pipeline.summarize_failed", file=fname, error=str(exc))
                image_summaries = []

            # ── D. Chunk ──
            chunks = chunk_rows(
                rows=rows,
                image_summaries=image_summaries,
                file_urls={fname: fmeta.file_url},
                source_name=col_id,
                file_metas=[{
                    "file_name": fname,
                    "file_id": file_id,
                    "relative_path": rel_path,
                }],
            )

            # ── E. LLM table titles ──
            try:
                await generate_table_titles(chunks)
            except Exception as exc:
                logger.error("pipeline.table_titles_failed", file=fname, error=str(exc))

            # ── F. GPT chunk summaries ──
            try:
                await generate_chunk_summaries(chunks)
            except Exception as exc:
                logger.error("pipeline.chunk_summaries_failed", file=fname, error=str(exc))

            # ── G. Fill context_before / context_after with GPT summaries ──
            for i, chunk in enumerate(chunks):
                if i > 0:
                    chunk["context_before"] = chunks[i - 1].get("summary", "")
                if i < len(chunks) - 1:
                    chunk["context_after"] = chunks[i + 1].get("summary", "")

            # ── H. Embed ──
            embedded = vector_store.add_chunks(chunks)
            new_embedded += embedded

            # ── I. Finalise and persist ──
            fmeta.is_embedded = True
            fmeta.is_redacted = True
            fmeta.is_chunked = True
            fmeta.status = "done"
            fmeta.processed_at = datetime.utcnow()
            fmeta.total_chunks = len(chunks)
            await metadata_store.save_file_meta(col_id, fmeta)
            await metadata_store.save_chunks(col_id, file_id, chunks)

            all_file_metas.append(fmeta)
            logger.info(
                "pipeline.file_done",
                file=fname,
                chunks=len(chunks),
                embedded=embedded,
            )

        # ── 4. Update collection ──
        col_meta.file_count = len(all_file_metas)
        col_meta.files = all_file_metas
        await metadata_store.save_collection(col_meta)

        elapsed = time.time() - start

        result = PipelineResult(
            collection_id=col_id,
            files_processed=len(all_file_metas),
            total_rows=sum(
                fm.total_text_rows + fm.total_table_rows + fm.total_image_rows
                for fm in all_file_metas
            ),
            text_rows=sum(fm.total_text_rows for fm in all_file_metas),
            table_rows=sum(fm.total_table_rows for fm in all_file_metas),
            image_rows=sum(fm.total_image_rows for fm in all_file_metas),
            error_rows=sum(fm.total_errors for fm in all_file_metas),
            chunks_embedded=new_embedded,
            processing_time_sec=round(elapsed, 2),
        )

        logger.info(
            "pipeline.complete",
            collection=col_id,
            files=result.files_processed,
            skipped_done=already_done,
            new_embedded=new_embedded,
            elapsed=result.processing_time_sec,
        )
        return result


pipeline = IngestionPipeline()
