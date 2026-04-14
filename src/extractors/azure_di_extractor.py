"""
Azure Document Intelligence Extractor.

New additions vs the original notebook port
────────────────────────────────────────────
1. Figure-bbox index built BEFORE paragraphs are processed.
2. Paragraphs whose bounding box significantly overlaps a figure region are
   dropped — they are part of the image, not standalone text.  This prevents
   the same text appearing in both a text chunk and an image description.
3. Paragraphs are collected with their vertical (y) position so each table
   can record which paragraphs appear directly before/after it on the page.
   This context is stored on the row and used by the chunker to generate
   LLM-based table titles.
4. When is_pptx_source=True the extracted image bytes are passed through the
   PPTX image filter (classify_pdf_image) so background slides, logos and
   tiny decorative images are discarded.
"""
from __future__ import annotations

import base64
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

from src.config.settings import settings
from src.config.logging_cfg import logger
from src.utils.text_cleaner import clean_ocr_text
from src.utils.helpers import (
    table_to_markdown,
    get_page,
    extract_image,
    make_row,
    para_to_bbox,
    text_overlaps_figure,
    classify_pdf_image,
)


def get_di_client() -> DocumentIntelligenceClient:
    """Create Azure Document Intelligence client."""
    if not settings.AZURE_DI_ENDPOINT or not settings.AZURE_DI_KEY:
        raise ValueError(
            "AZURE_DI_ENDPOINT and AZURE_DI_KEY must be set in .env"
        )
    return DocumentIntelligenceClient(
        settings.AZURE_DI_ENDPOINT,
        AzureKeyCredential(settings.AZURE_DI_KEY),
    )


def process_azure_di(
    file_path: str,
    file_name: str,
    di_client: Optional[DocumentIntelligenceClient] = None,
    is_pptx_source: bool = False,
) -> List[dict]:
    """
    Extract text, tables and images from a PDF via Azure DI.

    Parameters
    ----------
    file_path      : path to the PDF file on disk (may be a converted PPTX)
    file_name      : display name stored in every row — pass the original
                     .pptx filename so downstream code (classifier, chunker)
                     can still detect PPTX-sourced rows via the extension
    di_client      : reuse an existing client; created on demand if None
    is_pptx_source : True when the PDF was produced by converting a PPTX.
                     Enables the PPTX image filter on extracted image bytes.
    """
    logger.info("azure_di.start", file=file_name, pptx_source=is_pptx_source)
    rows: List[dict] = []

    if di_client is None:
        di_client = get_di_client()

    try:
        with open(file_path, "rb") as f:
            poller = di_client.begin_analyze_document("prebuilt-layout", body=f)
            result = poller.result()

        page_map: Dict[int, object] = {p.page_number: p for p in (result.pages or [])}

        # ── Step 1: Build figure bounding-box index (per page) ────────────────
        # Must happen BEFORE paragraph processing so overlap checks work.
        figure_bboxes_per_page: Dict[int, List[dict]] = {}
        for fig in getattr(result, "figures", None) or []:
            for region in getattr(fig, "bounding_regions", None) or []:
                pg = region.page_number
                polygon = getattr(region, "polygon", None)
                if polygon and len(polygon) >= 4:
                    xs = polygon[0::2]
                    ys = polygon[1::2]
                    figure_bboxes_per_page.setdefault(pg, []).append(
                        {
                            "x_min": min(xs),
                            "y_min": min(ys),
                            "x_max": max(xs),
                            "y_max": max(ys),
                        }
                    )

        # ── Step 2: Collect paragraphs with vertical position ─────────────────
        # para_by_page: {pg: [(y_min, text)]} — sorted later, used for table context.
        # page_text_map: {pg: [text]} — joined for image summarization context.
        para_by_page: Dict[int, List[tuple]] = {}
        page_text_map: Dict[int, List[str]] = {pg: [] for pg in page_map}

        for para in result.paragraphs or []:
            pg = get_page(para, result.pages)
            content = clean_ocr_text(para.content or "")
            if not (pg and content):
                continue

            # Skip paragraphs embedded inside a figure (avoids duplicate text).
            para_bbox = para_to_bbox(para)
            fig_bboxes = figure_bboxes_per_page.get(pg, [])
            if para_bbox and fig_bboxes and text_overlaps_figure(
                para_bbox, fig_bboxes,
                threshold=settings.TEXT_FIGURE_OVERLAP_THRESHOLD,
            ):
                logger.debug(
                    "azure_di.para_in_figure_skipped",
                    page=pg,
                    preview=content[:60],
                )
                continue

            y_pos = para_bbox["y_min"] if para_bbox else float("inf")
            para_by_page.setdefault(pg, []).append((y_pos, content))
            page_text_map.setdefault(pg, []).append(content)

        # Sort paragraphs by vertical position within each page.
        for pg in para_by_page:
            para_by_page[pg].sort(key=lambda t: t[0])

        # ── Step 3: Tables with before/after context ───────────────────────────
        # For each table, record the paragraphs that appear immediately before
        # and after it (by y position) so the LLM title generator has context.
        page_tables: Dict[int, List[dict]] = {pg: [] for pg in page_map}

        for tbl in result.tables or []:
            tbl_markdown = table_to_markdown(tbl.cells)
            if not tbl_markdown.strip():
                continue

            tbl_pg: Optional[int] = None
            tbl_y_min = float("inf")
            tbl_y_max = float("-inf")

            for region in getattr(tbl, "bounding_regions", None) or []:
                tbl_pg = region.page_number
                polygon = getattr(region, "polygon", None)
                if polygon:
                    ys = polygon[1::2]
                    tbl_y_min = min(tbl_y_min, min(ys))
                    tbl_y_max = max(tbl_y_max, max(ys))

            if tbl_pg is None:
                continue

            paras_on_pg = para_by_page.get(tbl_pg, [])
            # Last ≤3 paragraphs before the table top edge
            before_ctx = "\n".join(
                text for y, text in paras_on_pg if y < tbl_y_min
            )[-1500:]  # cap at 1500 chars
            # First ≤2 paragraphs after the table bottom edge
            after_lines = [text for y, text in paras_on_pg if y > tbl_y_max][:2]
            after_ctx = "\n".join(after_lines)

            page_tables.setdefault(tbl_pg, []).append(
                {
                    "markdown": tbl_markdown,
                    "context_before": before_ctx,
                    "context_after": after_ctx,
                }
            )

        # ── Step 4: Images / Figures ───────────────────────────────────────────
        # file_path is always a PDF here — PPTX files are converted to PDF by
        # the dispatcher before reaching this function.
        page_images: Dict[int, List[dict]] = {pg: [] for pg in page_map}
        file_stem = Path(file_name).stem

        for fig in getattr(result, "figures", None) or []:
            for region in getattr(fig, "bounding_regions", None) or []:
                pg = region.page_number
                img_data = extract_image(fig, region, file_path, page_map.get(pg))

                if is_pptx_source and img_data.get("base64"):
                    # Apply PPTX image filter — drop backgrounds, logos, tiny icons.
                    img_bytes = base64.b64decode(img_data["base64"])
                    page_obj  = page_map.get(pg)
                    page_w_in = getattr(page_obj, "width",  8.5)
                    page_h_in = getattr(page_obj, "height", 11.0)
                    category  = classify_pdf_image(
                        img_bytes,
                        img_data.get("bounding_box"),
                        page_w_in,
                        page_h_in,
                    )
                    if category != "useful":
                        logger.debug(
                            "azure_di.pptx_image_filtered",
                            page=pg, reason="not_useful",
                        )
                        continue

                saved_path = _save_image(img_data, file_stem, pg)
                if saved_path:
                    img_data["saved_path"] = saved_path

                page_images.setdefault(pg, []).append(img_data)

        # ── Step 5: Assemble rows per page ─────────────────────────────────────
        for pg_num in sorted(page_map):
            # Text row (all surviving paragraphs on this page, preserving order)
            ordered_texts = [text for _, text in para_by_page.get(pg_num, [])]
            text_content = "\n\n".join(ordered_texts)
            if text_content.strip():
                rows.append(make_row(file_name, pg_num, "text", content=text_content))

            # Table rows (each with before/after context for title generation)
            for tbl_info in page_tables.get(pg_num, []):
                row = make_row(file_name, pg_num, "table", content=tbl_info["markdown"])
                row["table_context_before"] = tbl_info["context_before"]
                row["table_context_after"] = tbl_info["context_after"]
                rows.append(row)

            # Image rows
            for img in page_images.get(pg_num, []):
                if settings.IMAGE_MODE == "base64":
                    img_val = img.get("base64")
                else:
                    img_val = json.dumps(img.get("bounding_box"))

                if img_val:
                    row = make_row(file_name, pg_num, "image", image=img_val)
                else:
                    row = make_row(
                        file_name,
                        pg_num,
                        "image",
                        image=None,
                        error=img.get("extract_error") or "No image data extracted",
                    )
                # Always attach the saved disk path when available
                row["image_path"] = img.get("saved_path")
                rows.append(row)

        logger.info(
            "azure_di.done",
            file=file_name,
            rows=len(rows),
            pptx_source=is_pptx_source,
        )

    except Exception as e:
        rows.append(make_row(file_name, None, None, error=str(e)))
        logger.error("azure_di.failed", file=file_name, error=str(e))

    return rows


# ── Private helper ─────────────────────────────────────────────────────────────


def _save_image(img_data: dict, file_stem: str, page: int) -> Optional[str]:
    """
    Write the image bytes from *img_data* to
    ``<IMAGES_DIR>/<file_stem>/page<NNN>_<uuid8>.png``.

    Returns the absolute path string, or None when there are no bytes to save.
    """
    b64 = img_data.get("base64")
    if not b64:
        return None

    try:
        img_bytes = base64.b64decode(b64)
        out_dir = settings.IMAGES_DIR / file_stem
        out_dir.mkdir(parents=True, exist_ok=True)
        img_id = uuid.uuid4().hex[:8]
        out_path = out_dir / f"page{page:03d}_{img_id}.png"
        out_path.write_bytes(img_bytes)
        logger.debug(
            "azure_di.image_saved",
            path=str(out_path),
            size_kb=round(len(img_bytes) / 1024, 1),
        )
        return str(out_path)
    except Exception as exc:
        logger.warning("azure_di.image_save_failed", error=str(exc))
        return None


