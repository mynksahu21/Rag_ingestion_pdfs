"""
Shared Helper Functions — port from notebook CELL 4 & 5, extended.

Sections:
  CELL 4  — Azure DI helpers (table_to_markdown, get_page, extract_image, make_row)
  NEW     — PDF image classifier (classify_pdf_image)
  NEW     — Overlap detection (para_to_bbox, text_overlaps_figure)
"""
from __future__ import annotations

import io
import uuid
import base64
from typing import Optional

from PIL import Image, ImageStat

from src.config.settings import settings
from src.config.logging_cfg import logger


# ═══════════════════════════════════════════════════════════════
#  CELL 4 — Azure DI helpers
# ═══════════════════════════════════════════════════════════════


def table_to_markdown(cells) -> str:
    """Convert Azure DI table cells to a Markdown table string."""
    if not cells:
        return ""
    d: dict = {}
    for c in cells:
        d.setdefault(c.row_index, {})[c.column_index] = (
            c.content or ""
        ).strip().replace("\n", " ")
    if not d:
        return ""
    max_r = max(d)
    max_c = max(max(v) for v in d.values())
    grid = [
        [d.get(r, {}).get(c, "") for c in range(max_c + 1)]
        for r in range(max_r + 1)
    ]
    lines: list[str] = []
    for i, row in enumerate(grid):
        lines.append(
            "| " + " | ".join(str(x).replace("|", "\\|") for x in row) + " |"
        )
        if i == 0:
            lines.append("| " + " | ".join("---" for _ in row) + " |")
    return "\n".join(lines)


def get_page(para, pages) -> Optional[int]:
    """Resolve which page a paragraph span falls on."""
    if not para.spans or not pages:
        return None
    off = para.spans[0].offset
    for p in pages:
        for s in p.spans or []:
            if s.offset <= off < s.offset + s.length:
                return p.page_number
    return None


def extract_image(fig, region, file_path: str, page_obj) -> dict:
    """
    Crop and base64-encode a figure region from a PDF page using PyMuPDF.
    Falls back to bounding_box JSON when extraction fails.
    extract_error is always surfaced in logs and stored in the row.
    """
    polygon = getattr(region, "polygon", None)
    b64: Optional[str] = None
    bbox: Optional[dict] = None
    extract_error: Optional[str] = None

    if polygon:
        xs = polygon[0::2]
        ys = polygon[1::2]
        bbox = {
            "x_min": round(min(xs), 4),
            "y_min": round(min(ys), 4),
            "x_max": round(max(xs), 4),
            "y_max": round(max(ys), 4),
        }

    ext = (file_path or "").rsplit(".", 1)[-1].lower()

    if ext == "pdf" and settings.IMAGE_MODE == "base64" and bbox:
        try:
            import fitz  # PyMuPDF

            pg_num = region.page_number
            doc = fitz.open(file_path)
            try:
                if 1 <= pg_num <= len(doc):
                    page = doc[pg_num - 1]
                    page_width = page.rect.width
                    page_height = page.rect.height

                    iw = getattr(page_obj, "width", 8.5)
                    ih = getattr(page_obj, "height", 11.0)

                    clip = fitz.Rect(
                        bbox["x_min"] / iw * page_width,
                        bbox["y_min"] / ih * page_height,
                        bbox["x_max"] / iw * page_width,
                        bbox["y_max"] / ih * page_height,
                    )
                    mat = fitz.Matrix(150 / 72, 150 / 72)
                    pix = page.get_pixmap(matrix=mat, clip=clip)
                    b64 = base64.b64encode(pix.tobytes("png")).decode()
            finally:
                doc.close()

        except Exception as e:
            extract_error = f"PyMuPDF extraction error: {e}"
            logger.warning("helpers.extract_image_failed", error=extract_error)

    caption = getattr(getattr(fig, "caption", None), "content", "") or ""
    return {
        "figure_id": getattr(fig, "id", None),
        "caption": caption,
        "bounding_box": bbox,
        "base64": b64,
        "extract_error": extract_error,
    }


def make_row(
    file_name: str,
    page: Optional[int],
    row_type: Optional[str],
    content: Optional[str] = None,
    image: Optional[str] = None,
    error: Optional[str] = None,
) -> dict:
    """Canonical row factory — all processors must use this."""
    return {
        "file_name": file_name,
        "page": page,
        "type": row_type,
        "content": content,
        "image": image,
        "error": error,
        "id": str(uuid.uuid4()),
    }


def _ocr_text(image_bytes: bytes) -> Optional[str]:
    """
    Run Tesseract OCR on image bytes.

    Returns the extracted text string on success, or None when Tesseract is
    not installed / not on PATH / fails for any reason.  Callers must treat
    None as "OCR unavailable — skip the text check" rather than "no text".
    """
    try:
        import pytesseract
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return pytesseract.image_to_string(img)
    except ImportError:
        logger.debug("helpers.ocr_skipped", reason="pytesseract_not_installed")
        return None
    except Exception as exc:
        logger.debug("helpers.ocr_failed", error=str(exc))
        return None


# ═══════════════════════════════════════════════════════════════
#  Bounding-box overlap detection
#  Used by azure_di_extractor to exclude paragraph text that
#  falls inside a figure region (prevents duplicate content).
# ═══════════════════════════════════════════════════════════════


def para_to_bbox(para) -> Optional[dict]:
    """
    Extract an axis-aligned bounding box from an Azure DI paragraph object.

    Azure DI coordinates are in the document's unit system (inches for PDFs).
    Returns None when bounding-region data is absent or malformed.
    """
    brs = getattr(para, "bounding_regions", None)
    if not brs:
        return None
    polygon = getattr(brs[0], "polygon", None)
    if not polygon or len(polygon) < 4:
        return None
    xs = polygon[0::2]
    ys = polygon[1::2]
    return {
        "x_min": min(xs),
        "y_min": min(ys),
        "x_max": max(xs),
        "y_max": max(ys),
        "page": getattr(brs[0], "page_number", None),
    }


def text_overlaps_figure(
    para_bbox: dict,
    figure_bboxes: list,
    threshold: float = 0.5,
) -> bool:
    """
    Return True when more than *threshold* fraction of the paragraph's area
    is contained within any of the supplied figure bounding boxes.

    This is used to skip paragraphs whose text is embedded inside an image
    region — Azure DI sometimes surfaces such text as a separate paragraph,
    which would create duplicate/redundant content alongside the image chunk.

    Parameters
    ----------
    para_bbox     : bbox dict with x_min/y_min/x_max/y_max (same units as figures)
    figure_bboxes : list of bbox dicts for all figures on the same page
    threshold     : overlap fraction above which the paragraph is considered
                    part of the figure (default 0.50 = 50 %)
    """
    px1, py1 = para_bbox["x_min"], para_bbox["y_min"]
    px2, py2 = para_bbox["x_max"], para_bbox["y_max"]
    para_area = (px2 - px1) * (py2 - py1)
    if para_area <= 0:
        return False

    for fig in figure_bboxes:
        ix1 = max(px1, fig["x_min"])
        iy1 = max(py1, fig["y_min"])
        ix2 = min(px2, fig["x_max"])
        iy2 = min(py2, fig["y_max"])
        if ix2 > ix1 and iy2 > iy1:
            overlap_area = (ix2 - ix1) * (iy2 - iy1)
            if overlap_area / para_area >= threshold:
                return True

    return False


# ═══════════════════════════════════════════════════════════════
#  PDF image classifier
#  Applied to images extracted from PDFs (including PPTX-converted).
#  Uses page-coverage geometry from Azure DI for background detection.
#
#  Rules (in order):
#    1. Unreadable by Pillow                            → not_useful
#    2. Below MIN_PIXEL_AREA (100×100)                  → not_useful
#    3. bbox covers ≥ PPTX_BG_THRESHOLD of the page    → not_useful  (background)
#    4. Max channel stddev < 2.0  (blank / solid fill)  → not_useful
#    5. file < LOGO_MAX_KB AND pixels < LOGO_MAX_PIXEL_AREA → not_useful  (logo/icon)
#    6. OCR text length < MIN_TEXT_LENGTH               → not_useful  (empty image)
#    7. Everything else                                 → useful
# ═══════════════════════════════════════════════════════════════


def classify_pdf_image(
    image_bytes: bytes,
    bbox: Optional[dict],
    page_width_in: float,
    page_height_in: float,
) -> str:
    """
    Classify an image extracted from a PDF (including PPTX-converted PDFs)
    as "useful" or "not_useful".

    Parameters
    ----------
    image_bytes    : raw PNG/JPEG bytes of the extracted image region
    bbox           : bounding box dict (x_min/y_min/x_max/y_max) in inches,
                     as returned by extract_image(); may be None
    page_width_in  : page width in inches (from Azure DI page object)
    page_height_in : page height in inches (from Azure DI page object)
    """
    # Rule 1: (no file-size upper limit — all sizes are processed)

    # Rule 2 & initial Rule 3 setup: Pillow readability + image size
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode == "P" and isinstance(img.info.get("transparency"), bytes):
            img = img.convert("RGBA")
        pixel_width, pixel_height = img.size
    except Exception as exc:
        logger.debug("classify_pdf.skip", reason="pillow_unreadable", error=str(exc))
        return "not_useful"

    # Rule 3: pixel area too small
    if pixel_width * pixel_height < settings.MIN_PIXEL_AREA:
        logger.debug("classify_pdf.skip", reason="too_small",
                     pixels=pixel_width * pixel_height)
        return "not_useful"

    # Rule 4: background image — covers most of the page
    if bbox and page_width_in > 0 and page_height_in > 0:
        bbox_w = bbox["x_max"] - bbox["x_min"]
        bbox_h = bbox["y_max"] - bbox["y_min"]
        coverage = (bbox_w * bbox_h) / (page_width_in * page_height_in)
        if coverage >= settings.PPTX_BG_THRESHOLD:
            logger.debug("classify_pdf.skip", reason="background",
                         coverage=round(coverage, 3))
            return "not_useful"

    # Rule 5: blank image — all pixels are the same colour (solid fill / empty crop)
    try:
        stat = ImageStat.Stat(img)
        if max(stat.stddev) < 2.0:
            logger.debug("classify_pdf.skip", reason="blank_image")
            return "not_useful"
    except Exception:
        pass

    # Rule 6: logo/icon — small file AND small pixel area
    # Both conditions must be true: charts with color complexity have larger
    # file sizes even when physically small, so they won't be caught here.
    if (
        len(image_bytes) < settings.LOGO_MAX_KB * 1024
        and pixel_width * pixel_height < settings.LOGO_MAX_PIXEL_AREA
    ):
        logger.debug(
            "classify_pdf.skip", reason="logo",
            size_kb=round(len(image_bytes) / 1024, 1),
            pixels=pixel_width * pixel_height,
        )
        return "not_useful"

    # Rule 7: no OCR text — blank or decorative image
    # Only applied when Tesseract is available; None means OCR unavailable → skip.
    ocr = _ocr_text(image_bytes)
    if ocr is not None and len(ocr.strip()) < settings.MIN_TEXT_LENGTH:
        logger.debug("classify_pdf.skip", reason="no_text",
                     ocr_chars=len(ocr.strip()))
        return "not_useful"

    # Rule 8: passed all checks — keep it
    return "useful"
