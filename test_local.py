#!/usr/bin/env python3
"""
Local test — ingest files/directories and query with result persistence.

INGEST:
    python test_local.py ingest --file report.pdf deck.pptx
    python test_local.py ingest --dir /path/to/nested/folders
    python test_local.py ingest --dir /data/company_docs --name "Company Docs"

QUERY (results saved to data/output/queries/):
    python test_local.py query --q "What is the leave policy?"
    python test_local.py query --q "revenue trends" --collection <col-id>
    python test_local.py query --file queries.txt              # batch: one query per line
    python test_local.py query --q "topic1" --q "topic2"       # multiple inline

BOTH:
    python test_local.py ingest --dir ./docs query --q "What are the key points?"
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.settings import settings
from src.processors.pipeline import pipeline, scan_directory
from src.embeddings.vector_store import vector_store
from src.retriever.retriever import retriever
from src.extractors.azure_di_extractor import get_di_client, process_azure_di


def do_classify(args):
    """Show classifier result and PDF metadata for one or more files."""
    from src.extractors.classifier import classify_document_type

    print("\n" + "=" * 65)
    print("  🔎 Classifier")
    print("=" * 65)

    for fp in args.file:
        p = Path(fp)
        label_map = {
            "pptx":      "PPTX          (true PPTX / Open XML)",
            "slide_pdf": "SLIDE_PDF     (PDF exported from a presentation app)",
            "pdf":       "PDF           (regular document)",
            "unknown":   "UNKNOWN       (unrecognised format)",
        }
        if not p.exists():
            print(f"\n  ✗ {fp}  —  file not found")
            continue

        doc_type = classify_document_type(str(p))
        label = label_map.get(doc_type, doc_type)
        size_kb = round(p.stat().st_size / 1024, 1)

        print(f"\n  File   : {p.name}  ({size_kb:,} KB)")
        print(f"  Result : {label}")

        # Show PDF metadata when available
        if doc_type in ("pdf", "slide_pdf"):
            try:
                import fitz
                doc = fitz.open(str(p))
                meta = doc.metadata or {}
                doc.close()
                creator  = meta.get("creator",  "") or ""
                producer = meta.get("producer", "") or ""
                if creator or producer:
                    print(f"  Creator : {creator[:80]}")
                    print(f"  Producer: {producer[:80]}")
            except Exception:
                pass

    print("\n" + "=" * 65 + "\n")


async def do_dump_images(args):
    """Extract and save raw images from Azure DI to disk for inspection."""
    import base64

    file_path = Path(args.file).resolve()
    if not file_path.exists():
        print(f"  ✗ File not found: {args.file}")
        return

    out_dir = settings.OUTPUT_DIR / "images" / file_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  🖼  Dump Images")
    print("=" * 65)
    print(f"\n  File : {file_path.name}")
    print(f"  Out  : {out_dir}\n")

    di_client = get_di_client()
    rows = process_azure_di(str(file_path), file_path.name, di_client)

    image_rows = [r for r in rows if r.get("type") == "image"]
    print(f"  Total image rows extracted: {len(image_rows)}")

    saved, skipped = 0, 0
    for row in image_rows:
        b64 = row.get("image")
        page = row.get("page", 0)
        row_id = row.get("id", "")[:8]

        if not b64:
            err = row.get("error", "no image data")
            print(f"  ✗ page={page} id={row_id} — skipped ({err})")
            skipped += 1
            continue

        img_bytes = base64.b64decode(b64)
        out_file = out_dir / f"page{page:03d}_{row_id}.png"
        out_file.write_bytes(img_bytes)
        print(f"  ✓ page={page} id={row_id} — {len(img_bytes):,} bytes → {out_file.name}")
        saved += 1

    print(f"\n  Saved: {saved}  |  Skipped: {skipped}")
    print(f"  📁 {out_dir}")
    print("=" * 65 + "\n")


async def do_ingest(args) -> str | None:
    """Run ingestion and return collection_id."""
    print("\n" + "=" * 65)
    print("  📥 Ingestion")
    print("=" * 65)

    paths: list[str] = []

    # ── Recursive directory scan ──
    if args.dir:
        root = Path(args.dir).resolve()
        if not root.is_dir():
            print(f"  ✗ Directory not found: {args.dir}")
            return None

        file_pairs = scan_directory(str(root), settings.SUPPORTED_EXTENSIONS)
        print(f"\n  📂 Scanned: {root}")
        print(f"     Found {len(file_pairs)} file(s):\n")
        for abs_path, rel_path in file_pairs:
            size = Path(abs_path).stat().st_size
            print(f"     ✓ {rel_path} ({size:,} bytes)")
        paths = [fp[0] for fp in file_pairs]

    # ── Explicit files ──
    if args.file:
        for fp in args.file:
            p = Path(fp)
            if not p.exists():
                print(f"  ✗ Not found: {fp}")
                continue
            ext = p.suffix.lower().lstrip(".")
            if ext not in settings.SUPPORTED_EXTENSIONS:
                print(f"  ✗ Unsupported: {fp}")
                continue
            paths.append(str(p.resolve()))
            print(f"  ✓ {p.name} ({p.stat().st_size:,} bytes)")

    if not paths:
        print("\n  No valid files found.")
        return None

    col_id_arg = getattr(args, "collection", None) or None
    if col_id_arg:
        print(f"  ↩  Resuming collection: {col_id_arg}")
        print(f"     (files already marked 'done' will be skipped)\n")

    print(f"  ⏳ Processing {len(paths)} file(s)...\n")

    if args.dir:
        result = await pipeline.ingest_directory(
            root_dir=args.dir,
            collection_name=getattr(args, "name", "") or "",
            created_by="test-script",
            collection_id=col_id_arg,
        )
    else:
        result = await pipeline.ingest_files(
            file_paths=paths,
            collection_name=getattr(args, "name", "") or "CLI Test",
            created_by="test-script",
            collection_id=col_id_arg,
        )

    print(f"  {'─' * 50}")
    print(f"  ✓ Pipeline complete in {result.processing_time_sec:.1f}s")
    print(f"    Files:    {result.files_processed}")
    print(f"    Text:     {result.text_rows} rows")
    print(f"    Tables:   {result.table_rows} rows")
    print(f"    Images:   {result.image_rows} rows")
    print(f"    Errors:   {result.error_rows} rows")
    print(f"    Chunks:   {result.chunks_embedded} embedded")
    print(f"    Collection: {result.collection_id}")

    stats = vector_store.stats()
    print(f"\n  📊 Vector Store: {stats['total_vectors']} total vectors")
    print(f"     Types: {stats['type_distribution']}")

    return result.collection_id


async def do_query(args, collection_id: str | None = None):
    """Run queries and save results."""
    print("\n" + "=" * 65)
    print("  🔍 Query")
    print("=" * 65)

    col_id = getattr(args, "collection", None) or collection_id
    top_k = getattr(args, "top_k", 5)
    queries: list[str] = []

    # ── From --q flags ──
    if args.q:
        queries.extend(args.q)

    # ── From --file (batch) ──
    if getattr(args, "query_file", None):
        qf = Path(args.query_file)
        if qf.exists():
            queries.extend(
                line.strip()
                for line in qf.read_text().splitlines()
                if line.strip()
            )
            print(f"  Loaded {len(queries)} queries from {qf.name}")
        else:
            print(f"  ✗ Query file not found: {args.query_file}")

    if not queries:
        print("  No queries provided. Use --q or --query-file.")
        return

    if col_id:
        print(f"  Collection: {col_id}")
    print(f"  Queries: {len(queries)}  |  Top-K: {top_k}\n")

    # ── Run batch search (saves results automatically) ──
    session = await retriever.batch_search(queries, top_k=top_k, collection_id=col_id)

    # ── Show where results are saved ──
    session_dir = settings.OUTPUT_DIR / "queries" / f"session_{session.session_id}"
    print(f"  💾 Results saved to:")
    print(f"     {session_dir}/")
    print(f"     ├── session.json        (full session data)")
    print(f"     ├── query_000.json      (per-query results)")
    print(f"     └── summary.csv         (quick overview)")


def main():
    parser = argparse.ArgumentParser(
        description="RAG Pipeline — Ingest & Query",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # ── ingest ──
    p_ingest = sub.add_parser("ingest", help="Ingest files or directories")
    p_ingest.add_argument("--file", nargs="*", help="File path(s)")
    p_ingest.add_argument("--dir", type=str, help="Root directory (recursive)")
    p_ingest.add_argument("--name", type=str, default="", help="Collection name")
    p_ingest.add_argument("--collection", type=str, default=None,
                          help="Resume an existing collection (pass the collection ID printed by a previous run)")

    # ── classify ──
    p_cls = sub.add_parser("classify", help="Show how the classifier sees a file")
    p_cls.add_argument("--file", nargs="+", required=True, help="File path(s) to classify")

    # ── dump-images ──
    p_dump = sub.add_parser("dump-images", help="Extract and save raw images from a file")
    p_dump.add_argument("--file", type=str, required=True, help="Path to PDF or PPTX file")

    # ── query ──
    p_query = sub.add_parser("query", help="Search and save results")
    p_query.add_argument("--q", action="append", help="Query string(s)")
    p_query.add_argument("--query-file", type=str, help="Text file with one query per line")
    p_query.add_argument("--collection", type=str, help="Collection ID to search")
    p_query.add_argument("--top-k", type=int, default=5, help="Results per query")

    # ── both ──
    # Allow: python test_local.py ingest --dir ./docs query --q "question"
    # by parsing known args
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    asyncio.run(_run(args))


async def _run(args):
    col_id = None

    if args.command == "ingest":
        col_id = await do_ingest(args)
        print()

    elif args.command == "classify":
        do_classify(args)

    elif args.command == "dump-images":
        await do_dump_images(args)

    elif args.command == "query":
        await do_query(args)

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
