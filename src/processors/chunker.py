"""
Text Chunker — chunking strategy:

PPTX files (detected by .pptx extension in file_name, set by dispatcher):
  One chunk per slide — title + bullets + tables + graph data + image descriptions
  combined into a single rich chunk.  This preserves slide-level context.

PDF files:
  TEXT  : token-based 300-500 tokens, paragraph merging, 80-token overlap
  TABLE : never split; convert to structured Row format; LLM-generated title
          prepended using surrounding text context (generate_table_titles)
  CHART : Graph Data Table + Graph Description from image summariser
  IMAGE : description as single chunk

All chunk types share the same metadata schema (summary, keywords, section,
subsection, document_id, …).
"""
from __future__ import annotations

import asyncio
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from openai import BadRequestError
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import settings
from src.config.logging_cfg import logger
from src.utils.openai_client import get_async_openai_client


def _is_content_filter(exc: Exception) -> bool:
    """Return True for Azure content-policy rejections (HTTP 400, no point retrying)."""
    if not isinstance(exc, BadRequestError):
        return False
    body = getattr(exc, "body", None) or {}
    code = str((body.get("error") or {}).get("code", ""))
    return (
        "ResponsibleAIPolicyViolation" in code
        or "content_filter" in code.lower()
        or "ResponsibleAIPolicyViolation" in str(exc)
    )

# ── Token approximation (1 token ≈ 4 chars for English) ──
# Avoids adding tiktoken dependency while staying close to GPT tokenization
TARGET_MIN_TOKENS = 300
TARGET_MAX_TOKENS = 500
OVERLAP_TOKENS = 80

CHUNK_SIZE_CHARS = TARGET_MAX_TOKENS * 4    # 2000 chars  ≈ 500 tokens
OVERLAP_CHARS = OVERLAP_TOKENS * 4          # 320 chars   ≈ 80 tokens
MIN_CHUNK_CHARS = 100                        # ~25 tokens minimum


def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ── Stopwords for keyword extraction ──
_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'be', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'this', 'that', 'these',
    'those', 'it', 'its', 'as', 'not', 'no', 'so', 'if', 'than', 'then',
    'each', 'all', 'any', 'both', 'few', 'more', 'most', 'other', 'some',
    'such', 'only', 'own', 'same', 'too', 'very', 'just', 'also', 'into',
    'about', 'over', 'after', 'before', 'between', 'through', 'during',
}


def _extract_keywords(text: str, top_n: int = 8) -> List[str]:
    """Extract top N meaningful keywords from text."""
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b', text.lower())
    words = [w for w in words if w not in _STOPWORDS and len(w) > 3]
    return [w for w, _ in Counter(words).most_common(top_n)]


def _generate_summary(content: str, max_len: int = 150) -> str:
    """Return first 1-2 sentences of content, capped at max_len chars."""
    flat = re.sub(r'\s+', ' ', content).strip()
    sentences = re.split(r'(?<=[.!?])\s+', flat)
    summary = ''
    for s in sentences:
        candidate = (summary + ' ' + s).strip() if summary else s
        if len(candidate) <= max_len:
            summary = candidate
        else:
            if not summary:
                summary = candidate[:max_len]
            break
    return summary or flat[:max_len]


def _detect_section(content: str) -> Tuple[str, str]:
    """
    Detect section/subsection headings from content's leading lines.
    Returns (section, subsection).
    """
    lines = [ln.strip() for ln in content.strip().split('\n') if ln.strip()]
    if not lines:
        return '', ''

    first = lines[0]

    # Numbered heading: "1. Title", "1.2 Title", "Chapter 3 Title", etc.
    if re.match(r'^(?:chapter\s+)?\d+(?:\.\d+)*[\.\s]\s*\w', first, re.IGNORECASE):
        section = first
        subsection = lines[1] if len(lines) > 1 and len(lines[1]) < 100 else ''
        return section, subsection

    # All-caps short heading
    if first.isupper() and 3 < len(first) < 100 and len(first.split()) <= 12:
        section = first
        subsection = lines[1] if len(lines) > 1 and len(lines[1]) < 100 else ''
        return section, subsection

    # Title-case short standalone line (likely a heading)
    if first.istitle() and len(first) < 80 and len(first.split()) <= 8:
        return first, ''

    return '', ''


def _format_table(markdown_content: str) -> str:
    """
    Convert a markdown table to structured row format:

    Table Title: <title if present>
    Row1: Col1=Val1, Col2=Val2
    Row2: Col1=Val3, Col2=Val4
    """
    lines = [ln.strip() for ln in markdown_content.strip().split('\n') if ln.strip()]

    # Separate out non-table lines (title/header text before the table)
    title_lines = []
    table_lines = []
    for ln in lines:
        if '|' in ln:
            table_lines.append(ln)
        elif not table_lines:
            title_lines.append(ln)

    if not table_lines:
        return markdown_content  # Not a standard markdown table, keep as-is

    title = ' '.join(title_lines).strip()

    # Parse header — first non-separator row
    header_line = None
    data_lines = []
    for ln in table_lines:
        # Separator row (dashes/colons only between pipes)
        if re.match(r'^\|[-:\s|]+\|$', ln):
            continue
        if header_line is None:
            header_line = ln
        else:
            data_lines.append(ln)

    if not header_line:
        return markdown_content

    cols = [c.strip() for c in header_line.strip('|').split('|') if c.strip()]

    result_parts = []
    if title:
        result_parts.append(f"Table Title: {title}")

    for i, data_line in enumerate(data_lines, start=1):
        cells = [c.strip() for c in data_line.strip('|').split('|')]
        # Pad to match column count
        while len(cells) < len(cols):
            cells.append('')
        row_parts = [f"{col}={cell}" for col, cell in zip(cols, cells)]
        result_parts.append(f"Row{i}: {', '.join(row_parts)}")

    return '\n'.join(result_parts) if result_parts else markdown_content


def _make_chunk(
    content: str,
    file_name: str,
    page: Optional[int],
    chunk_type: str,
    source_name: str,
    file_meta: dict,
    file_urls: Dict[str, str],
    section: str = '',
    subsection: str = '',
    image_path: Optional[str] = None,
    now: str = '',
    image_description: Optional[str] = None,
    graph_data_table: Optional[str] = None,
    graph_description: Optional[str] = None,
) -> dict:
    """Build a chunk dict with the full metadata schema."""
    chunk_id = f"chunk_{uuid.uuid4().hex[:12]}"
    # Placeholder summary — will be replaced by GPT in generate_chunk_summaries()
    summary = _generate_summary(content)
    keywords = _extract_keywords(content)

    return {
        # Core
        "id": chunk_id,
        "content": content.strip(),
        # Embedding populated by add_chunks() and written back here
        "embedding": [],
        # Summaries & keywords
        "summary": summary,
        "keywords": keywords,
        # Image summary fields (populated for image/chart chunks)
        "image_description": image_description,
        "graph_data_table": graph_data_table,
        "graph_description": graph_description,
        # File references
        "folder_meta": file_meta.get("relative_path", ""),
        "document_id": file_meta.get("file_id", ""),
        "document_name": file_name,
        "file_name": file_name,
        "file_url": file_urls.get(file_name, ""),
        # Location
        "section": section,
        "subsection": subsection,
        "page": page,
        "page_or_slide_number": page,
        # Classification
        "chunk_type": chunk_type,
        "type": chunk_type,         # backward compat for vector store / retriever
        # Context (filled in post-processing pass)
        "context_before": "",
        "context_after": "",
        # Quality
        "confidence_score": 0.92,
        # Source
        "source": source_name,
        # Image data (for image/chart chunks)
        "image_path": image_path,
        # Timestamp
        "created_at": now,
    }


async def generate_chunk_summaries(chunks: List[dict]) -> None:
    """
    Use GPT to generate a proper summary for each chunk in-place.
    Replaces the simple text-slicing placeholder set by _generate_summary().
    """
    if not chunks:
        return

    if not settings.AZURE_OPENAI_API_KEY:
        logger.warning("chunker.gpt_summary_skipped", reason="AZURE_OPENAI_API_KEY not set")
        return

    client = get_async_openai_client()
    sem = asyncio.Semaphore(10)

    async def _summarize_one(chunk: dict) -> None:
        content = (chunk.get("content") or "").strip()
        if not content:
            return
        async with sem:
            try:
                response = await client.chat.completions.create(
                    model=settings.AZURE_OPENAI_DEPLOYMENT,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Summarize the following content in 1-2 concise sentences "
                                "(150 characters max):\n\n" + content[:2000]
                            ),
                        }
                    ],
                    temperature=0.2,
                    max_completion_tokens=100,
                )
                gpt_summary = (response.choices[0].message.content or "").strip()
                if gpt_summary:
                    chunk["summary"] = gpt_summary[:150]
            except Exception as exc:
                if _is_content_filter(exc):
                    logger.warning("chunker.gpt_summary_content_filtered",
                                   chunk_id=chunk.get("id"))
                else:
                    logger.warning("chunker.gpt_summary_error", error=str(exc)[:200])

    await asyncio.gather(*[_summarize_one(c) for c in chunks])
    logger.info("chunker.gpt_summaries_done", count=len(chunks))


async def generate_table_titles(chunks: List[dict]) -> None:
    """
    Use LLM to attach a descriptive title to every table chunk.

    For each chunk with chunk_type="table" the function uses the text that
    appears immediately before and after the table in the source document
    (stored in table_context_before / table_context_after by the extractor)
    to ask the LLM for a short, accurate title.

    The generated title is:
      • Prepended to chunk["content"] as "Table Title: <title>"
      • Stored in chunk["section"] when no section heading was detected
    """
    table_chunks = [c for c in chunks if c.get("chunk_type") == "table"]
    if not table_chunks:
        return

    if not settings.AZURE_OPENAI_API_KEY:
        logger.warning(
            "chunker.table_titles_skipped", reason="AZURE_OPENAI_API_KEY not set"
        )
        return

    client = get_async_openai_client()
    sem = asyncio.Semaphore(10)

    async def _title_one(chunk: dict) -> None:
        before = (chunk.get("table_context_before") or "").strip()
        after = (chunk.get("table_context_after") or "").strip()

        # Skip when there is no surrounding context to derive a title from.
        if not before and not after:
            return

        context_parts: List[str] = []
        if before:
            context_parts.append(f"Text before table:\n{before[:600]}")
        if after:
            context_parts.append(f"Text after table:\n{after[:300]}")

        table_preview = (chunk.get("content") or "")[:600]

        prompt = (
            "\n\n".join(context_parts)
            + f"\n\nTable content (preview):\n{table_preview}\n\n"
            "Based on the surrounding document context, generate a concise and "
            "accurate title for this table (maximum 10 words). "
            "Respond with ONLY the title — no quotes, no punctuation at the end."
        )

        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=settings.AZURE_OPENAI_DEPLOYMENT,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_completion_tokens=30,
                )
                title = (resp.choices[0].message.content or "").strip().strip("\"'")
                if title and len(title) <= 120:
                    chunk["content"] = f"Table Title: {title}\n\n{chunk['content']}"
                    if not chunk.get("section"):
                        chunk["section"] = title
            except Exception as exc:
                if _is_content_filter(exc):
                    logger.warning("chunker.table_title_content_filtered",
                                   chunk_id=chunk.get("id"))
                else:
                    logger.warning("chunker.table_title_error", error=str(exc)[:200])

    await asyncio.gather(*[_title_one(c) for c in table_chunks])
    titled = sum(1 for c in table_chunks if c.get("section"))
    logger.info(
        "chunker.table_titles_done",
        table_chunks=len(table_chunks),
        titled=titled,
    )


def chunk_rows(
    rows: List[dict],
    image_summaries: List[dict],
    file_urls: Dict[str, str],
    source_name: str = "",
    file_metas: Optional[List[dict]] = None,
) -> List[dict]:
    """
    Main chunking entry point.

    Parameters
    ----------
    rows : extracted rows (text/table/image) from all files
    image_summaries : results from image_summarizer.summarize_images()
    file_urls : mapping file_name → URL
    source_name : collection id
    file_metas : list of dicts with {file_name, file_id, relative_path}
    """
    now = datetime.now(timezone.utc).isoformat()

    # ── Build file_meta lookup ──
    file_meta_lookup: Dict[str, dict] = {}
    if file_metas:
        for fm in file_metas:
            file_meta_lookup[fm["file_name"]] = fm

    # ── Build image summary lookup: row_id → summary dict ──
    img_summary_map: Dict[str, dict] = {}
    for s in image_summaries:
        img_summary_map[s["id"]] = s

    # ── Filter rows: drop error-only rows ──
    valid_rows: List[dict] = []
    for row in rows:
        err = (row.get("error") or "").strip()
        if err and not row.get("content") and row.get("type") != "image":
            continue
        valid_rows.append(row)

    # ── Identify PPTX files ──
    pptx_files = {
        r["file_name"]
        for r in valid_rows
        if (r.get("file_name") or "").lower().endswith(".pptx")
    }

    # ── LangChain splitter for PDF text ──
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_CHARS,
        chunk_overlap=OVERLAP_CHARS,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: List[dict] = []

    # ═══════════════════════════════════════════════════════
    # PPTX: One chunk per slide
    # ═══════════════════════════════════════════════════════
    pptx_rows = [r for r in valid_rows if r.get("file_name") in pptx_files]
    pdf_rows = [r for r in valid_rows if r.get("file_name") not in pptx_files]

    # Group PPTX rows by (file_name, slide_number)
    slide_groups: Dict[Tuple[str, int], List[dict]] = {}
    for row in pptx_rows:
        fn = row.get("file_name", "")
        pg = row.get("page") or 0
        slide_groups.setdefault((fn, pg), []).append(row)

    for (fn, pg), slide_rows in sorted(slide_groups.items()):
        fm = file_meta_lookup.get(fn, {})
        content_parts: List[str] = []
        slide_section = ''

        for row in slide_rows:
            row_type = row.get("type", "text")

            if row_type == "text":
                text = (row.get("content") or "").strip()
                if not text:
                    continue
                # First non-empty text on a slide is the title/section
                if not slide_section:
                    first_line = text.split('\n')[0].strip()
                    if len(first_line) < 150:
                        slide_section = first_line
                content_parts.append(text)

            elif row_type == "table":
                table_text = _format_table(row.get("content") or "")
                if table_text.strip():
                    # For PPTX slides, tables are inlined into the slide chunk
                    # so we don't need separate table_context fields here.
                    content_parts.append(table_text)

            elif row_type == "image":
                sd = img_summary_map.get(row.get("id", ""), {})
                if not sd or sd.get("skipped") or sd.get("error"):
                    continue
                if sd.get("is_chart"):
                    parts = []
                    if sd.get("graph_data_table"):
                        parts.append(sd["graph_data_table"])
                    if sd.get("graph_description"):
                        parts.append(f"Graph Description:\n{sd['graph_description']}")
                    if parts:
                        content_parts.append("\n\n".join(parts))
                elif sd.get("image_description"):
                    content_parts.append(sd["image_description"])
                # Note: image summary fields are only stored on standalone image chunks (PDF path)

        if content_parts:
            combined = "\n\n".join(content_parts).strip()
            if len(combined) >= MIN_CHUNK_CHARS:
                chunk = _make_chunk(
                    content=combined,
                    file_name=fn,
                    page=pg,
                    chunk_type="text",
                    source_name=source_name,
                    file_meta=fm,
                    file_urls=file_urls,
                    section=slide_section,
                    subsection='',
                    now=now,
                )
                chunks.append(chunk)

    # ═══════════════════════════════════════════════════════
    # PDF rows: TEXT / TABLE / IMAGE (chart or non-chart)
    # ═══════════════════════════════════════════════════════
    for row in pdf_rows:
        fn = row.get("file_name", "")
        pg = row.get("page")
        row_type = row.get("type", "text")
        fm = file_meta_lookup.get(fn, {})

        if row_type == "table":
            # ── TABLE: never split, convert to structured rows ──
            raw = (row.get("content") or "").strip()
            if not raw:
                continue
            formatted = _format_table(raw)
            if len(formatted.strip()) < MIN_CHUNK_CHARS:
                continue
            section, subsection = _detect_section(raw)
            chunk = _make_chunk(
                content=formatted,
                file_name=fn,
                page=pg,
                chunk_type="table",
                source_name=source_name,
                file_meta=fm,
                file_urls=file_urls,
                section=section,
                subsection=subsection,
                now=now,
            )
            # Carry forward context so generate_table_titles can call the LLM.
            chunk["table_context_before"] = row.get("table_context_before") or ""
            chunk["table_context_after"] = row.get("table_context_after") or ""
            chunks.append(chunk)

        elif row_type == "image":
            # ── IMAGE: chart gets data table + description; others get description ──
            sd = img_summary_map.get(row.get("id", ""), {})
            if not sd or sd.get("skipped") or sd.get("error"):
                continue

            if sd.get("is_chart"):
                # Chart chunk: structured data table + description
                chart_parts = []
                if sd.get("graph_data_table"):
                    chart_parts.append(sd["graph_data_table"])
                if sd.get("graph_description"):
                    chart_parts.append(f"Graph Description:\n{sd['graph_description']}")
                if not chart_parts:
                    continue
                chart_content = "\n\n".join(chart_parts)
                if len(chart_content.strip()) >= MIN_CHUNK_CHARS:
                    chunk = _make_chunk(
                        content=chart_content,
                        file_name=fn,
                        page=pg,
                        chunk_type="chart",
                        source_name=source_name,
                        file_meta=fm,
                        file_urls=file_urls,
                        image_path=row.get("image_path"),
                        now=now,
                        image_description=sd.get("image_description"),
                        graph_data_table=sd.get("graph_data_table"),
                        graph_description=sd.get("graph_description"),
                    )
                    chunks.append(chunk)
            else:
                # Non-chart image: use description as content; also store it explicitly
                desc = sd.get("image_description") or (row.get("content") or "")
                desc = desc.strip()
                if len(desc) >= MIN_CHUNK_CHARS:
                    chunk = _make_chunk(
                        content=desc,
                        file_name=fn,
                        page=pg,
                        chunk_type="image",
                        source_name=source_name,
                        file_meta=fm,
                        file_urls=file_urls,
                        image_path=row.get("image_path"),
                        now=now,
                        image_description=sd.get("image_description"),
                    )
                    chunks.append(chunk)

        elif row_type == "text":
            # ── TEXT: token-based chunking with paragraph merging ──
            content = (row.get("content") or "").strip()
            if not content:
                continue

            section, subsection = _detect_section(content)
            token_count = _count_tokens(content)

            if token_count <= TARGET_MAX_TOKENS:
                # Within target size — single chunk
                if len(content) >= MIN_CHUNK_CHARS:
                    chunk = _make_chunk(
                        content=content,
                        file_name=fn,
                        page=pg,
                        chunk_type="text",
                        source_name=source_name,
                        file_meta=fm,
                        file_urls=file_urls,
                        section=section,
                        subsection=subsection,
                        now=now,
                    )
                    chunks.append(chunk)
            else:
                # Exceeds 500 tokens — split with 80-token overlap
                for chunk_text in splitter.split_text(content):
                    chunk_text = chunk_text.strip()
                    if len(chunk_text) >= MIN_CHUNK_CHARS:
                        chunk = _make_chunk(
                            content=chunk_text,
                            file_name=fn,
                            page=pg,
                            chunk_type="text",
                            source_name=source_name,
                            file_meta=fm,
                            file_urls=file_urls,
                            section=section,
                            subsection=subsection,
                            now=now,
                        )
                        chunks.append(chunk)

    # NOTE: context_before / context_after are filled by pipeline.py AFTER
    # generate_chunk_summaries() so they contain GPT-quality summaries, not
    # placeholder text.  Initialise to empty here so the schema is consistent.

    logger.info(
        "chunker.done",
        pptx_slides=len(slide_groups),
        pdf_rows=len(pdf_rows),
        output_chunks=len(chunks),
    )
    return chunks
