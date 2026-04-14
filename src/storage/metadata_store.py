"""
Metadata Store — JSON persistence on disk.
Replaces notebook's Delta table writes.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import aiofiles

from src.config.settings import settings
from src.config.models import CollectionMetadata, FileMetadata


class MetadataStore:
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or settings.OUTPUT_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _col_dir(self, collection_id: str) -> Path:
        p = self.base_dir / collection_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ── Collection ──

    async def save_collection(self, meta: CollectionMetadata):
        meta.updated_at = datetime.utcnow()
        path = self._col_dir(meta.collection_id) / ".collection_metadata.json"
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(meta.model_dump_json(indent=2))

    async def get_collection(self, cid: str) -> Optional[CollectionMetadata]:
        path = self._col_dir(cid) / ".collection_metadata.json"
        if not path.exists():
            return None
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            return CollectionMetadata(**json.loads(await f.read()))

    async def list_collections(self) -> List[CollectionMetadata]:
        results = []
        for child in self.base_dir.iterdir():
            if not child.is_dir():
                continue
            # Only descend into directories that contain the collection marker.
            if not (child / ".collection_metadata.json").exists():
                continue
            meta = await self.get_collection(child.name)
            if meta:
                results.append(meta)
        return results

    async def delete_collection(self, cid: str) -> bool:
        d = self.base_dir / cid
        if d.exists():
            shutil.rmtree(d)
            return True
        return False

    # ── File metadata ──

    async def save_file_meta(self, collection_id: str, meta: FileMetadata):
        path = self._col_dir(collection_id) / f"{meta.file_id}.metadata.json"
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(meta.model_dump_json(indent=2))

    async def get_file_meta(self, cid: str, fid: str) -> Optional[FileMetadata]:
        path = self._col_dir(cid) / f"{fid}.metadata.json"
        if not path.exists():
            return None
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            return FileMetadata(**json.loads(await f.read()))

    async def list_file_meta(self, cid: str) -> List[FileMetadata]:
        results = []
        for p in self._col_dir(cid).glob("*.metadata.json"):
            if p.name.startswith("."):
                continue
            async with aiofiles.open(p, "r", encoding="utf-8") as f:
                results.append(FileMetadata(**json.loads(await f.read())))
        return results

    # ── Raw data persistence ──

    async def save_rows(self, cid: str, fid: str, rows: list):
        path = self._col_dir(cid) / f"{fid}.rows.json"
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(rows, indent=2, default=str))

    async def save_chunks(self, cid: str, fid: str, chunks: list):
        path = self._col_dir(cid) / f"{fid}.chunks.json"
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(chunks, indent=2, default=str))


metadata_store = MetadataStore()
