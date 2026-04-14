"""
Document Type Classifier.

Handles four cases:
  1. True PPTX file (.pptx / renamed to .pdf) — ZIP with ppt/ directory.
  2. PDF exported from PowerPoint / Impress / Keynote / Google Slides —
     detected via PDF metadata (Creator/Producer) and slide-style page geometry.
  3. Regular PDF document.
  4. Unknown / unsupported format.

Detection priority (highest → lowest confidence):
  ① ZIP structure   — file is literally a PPTX, regardless of extension.
  ② PDF metadata    — Creator/Producer fields contain presentation-app strings.
  ③ Page geometry   — all pages share a consistent landscape aspect ratio
                      matching common slide dimensions (4:3, 16:9, 16:10).
  ④ File extension  — last resort when content inspection is inconclusive.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Literal

from src.config.logging_cfg import logger

DocumentType = Literal["pdf", "slide_pdf", "pptx", "unknown"]

# ── Strings that identify presentation-app PDF exporters ──────────────────────
_PRESENTATION_APP_PATTERNS = re.compile(
    r"microsoft[® ]*powerpoint|powerpoint"
    r"|libreoffice\s*impress|openoffice\s*impress"
    r"|apple\s*keynote|keynote"
    r"|google\s*slides"
    r"|impress",
    re.IGNORECASE,
)

# ── Common slide page-size ranges in PDF points (1 pt = 1/72 inch) ────────────
# We check width/height of each page because slide PDFs are *landscape* and
# consistently sized.  Regular documents are portrait or mixed.
#
# 4:3  standard  : 720 × 540 pt  (10 × 7.5 in)
# 16:9 widescreen: 960 × 540 pt  (13.33 × 7.5 in)
# 16:10           : 720 × 450 pt
# "Wide" (older) : 864 × 540 pt  (12 × 7.5 in)
# US Letter land. : 792 × 612 pt  — also used for some docs, so NOT sufficient alone
#
# We accept any landscape page whose aspect ratio is within ±8 % of 4:3 or 16:9.
_SLIDE_ASPECT_RATIOS = [4 / 3, 16 / 9, 16 / 10]
_ASPECT_TOLERANCE = 0.08   # ±8 %

# Minimum slide height in points (below this it's likely a banner/thumbnail)
_MIN_SLIDE_HEIGHT_PT = 400


def classify_document_type(file_path: str) -> DocumentType:
    """
    Return the document type for *file_path*.

    Return values
    -------------
    'pptx'      — true PPTX (Open XML), regardless of file extension
    'slide_pdf' — PDF that was exported from a presentation tool
    'pdf'       — regular PDF document
    'unknown'   — format not recognised
    """
    path = Path(file_path)
    ext = path.suffix.lower().lstrip(".")

    # ① True PPTX: ZIP archive containing ppt/ directory
    if _is_pptx_content(file_path):
        if ext not in ("pptx", "ppt"):
            logger.warning(
                "classifier.ext_mismatch",
                file=path.name,
                extension=ext,
                detected="pptx",
            )
        return "pptx"

    # For everything that starts with %PDF- we do deeper inspection.
    if _verify_pdf_magic(file_path):
        result = _inspect_pdf(file_path)
        logger.info(
            "classifier.pdf_classified",
            file=path.name,
            doc_type=result,
        )
        return result

    # Extension fallback
    if ext == "pdf":
        return "pdf"
    if ext in ("pptx", "ppt"):
        return "pptx"

    logger.warning("classifier.unknown_type", file=path.name, extension=ext)
    return "unknown"


def is_pptx(file_path: str) -> bool:
    """Return True for both true PPTX files and PDFs exported from presentations."""
    return classify_document_type(file_path) in ("pptx", "slide_pdf")


# ── Internal helpers ──────────────────────────────────────────────────────────


def _verify_pdf_magic(file_path: str) -> bool:
    """Check for the PDF magic header (%PDF-)."""
    try:
        with open(file_path, "rb") as fh:
            return fh.read(5) == b"%PDF-"
    except OSError:
        return False


def _is_pptx_content(file_path: str) -> bool:
    """
    A PPTX is a ZIP archive whose central directory contains a 'ppt/' entry.
    """
    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            return any(name.startswith("ppt/") for name in zf.namelist())
    except (zipfile.BadZipFile, Exception):
        return False


def _inspect_pdf(file_path: str) -> DocumentType:
    """
    Open the PDF with PyMuPDF and check:
      1. Creator / Producer metadata for known presentation-app strings.
      2. Page geometry — all pages landscape with a slide-like aspect ratio.

    Returns 'slide_pdf' or 'pdf'.
    """
    try:
        import fitz  # type: ignore[import]  # PyMuPDF — project dependency

        doc = fitz.open(file_path)
        try:
            # ── Check 1: metadata ──────────────────────────────────────────
            meta = doc.metadata or {}
            creator  = meta.get("creator", "")  or ""
            producer = meta.get("producer", "") or ""

            if _PRESENTATION_APP_PATTERNS.search(creator) or \
               _PRESENTATION_APP_PATTERNS.search(producer):
                logger.debug(
                    "classifier.slide_pdf_by_metadata",
                    creator=creator[:80],
                    producer=producer[:80],
                )
                return "slide_pdf"

            # ── Check 2: page geometry ─────────────────────────────────────
            if doc.page_count == 0:
                return "pdf"

            if _all_pages_are_slides(doc):
                logger.debug(
                    "classifier.slide_pdf_by_geometry",
                    pages=doc.page_count,
                )
                return "slide_pdf"

        finally:
            doc.close()

    except Exception as exc:
        logger.warning("classifier.pdf_inspect_error", error=str(exc)[:200])

    return "pdf"


def _all_pages_are_slides(doc) -> bool:
    """
    Return True when every page in the document has:
      • landscape orientation (width > height)
      • height >= _MIN_SLIDE_HEIGHT_PT  (rules out tiny thumbnails)
      • aspect ratio within ±_ASPECT_TOLERANCE of any known slide ratio
      • consistent dimensions (all pages same size — typical for slide exports)
    """
    if doc.page_count == 0:
        return False

    widths:  list[float] = []
    heights: list[float] = []

    for page in doc:
        rect = page.rect
        w, h = rect.width, rect.height

        # Must be landscape
        if w <= h:
            return False

        # Must be tall enough to be a real slide
        if h < _MIN_SLIDE_HEIGHT_PT:
            return False

        # Aspect ratio must match a known slide format
        aspect = w / h
        if not any(
            abs(aspect - target) / target <= _ASPECT_TOLERANCE
            for target in _SLIDE_ASPECT_RATIOS
        ):
            return False

        widths.append(w)
        heights.append(h)

    if not widths:
        return False

    # Consistent dimensions — max deviation ≤ 2 pt across all pages
    w_range = max(widths)  - min(widths)
    h_range = max(heights) - min(heights)
    return w_range <= 2 and h_range <= 2
