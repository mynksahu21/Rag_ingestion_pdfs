"""
File Dispatcher.

Routing strategy
────────────────
PPTX (true .pptx detected by content):
  1. Convert to PDF via PowerPoint COM (comtypes / win32com).
  2. Pass the resulting PDF to Azure DI with is_pptx_source=True so the
     PPTX image filter is applied (backgrounds, logos, tiny icons dropped).
  3. Clean up the temporary PDF after extraction.

slide_pdf (PDF exported from PowerPoint — metadata / geometry detected):
  → Azure DI directly with is_pptx_source=True (already a PDF).

PDF (regular document):
  → Azure DI with is_pptx_source=False.
"""
from __future__ import annotations

import shutil
import tempfile
from typing import List

from src.config.logging_cfg import logger
from src.extractors.classifier import classify_document_type
from src.extractors.pptx_to_pdf import convert_pptx_to_pdf
from src.extractors.azure_di_extractor import get_di_client, process_azure_di
from src.utils.helpers import make_row


def process_file(file_path: str, file_name: str, file_type: str) -> List[dict]:
    """
    Route a single file to the correct extraction path.

    Parameters
    ----------
    file_path : absolute path to the file on disk
    file_name : display name kept in all row metadata (the original PPTX name
                is preserved even after conversion so the chunker can detect
                PPTX-sourced rows via the .pptx extension)
    file_type : file extension hint (classifier uses content inspection first)

    Returns
    -------
    List of row dicts (text / table / image).
    """
    doc_type = classify_document_type(file_path)
    logger.info("dispatcher.process", file=file_name, doc_type=doc_type)

    rows: List[dict] = []

    try:
        if doc_type == "pptx":
            rows = _process_pptx(file_path, file_name)

        elif doc_type == "slide_pdf":
            # Already a PDF — send straight to Azure DI with PPTX image filtering.
            di_client = get_di_client()
            rows = process_azure_di(
                file_path, file_name, di_client, is_pptx_source=True
            )

        elif doc_type == "pdf":
            di_client = get_di_client()
            rows = process_azure_di(file_path, file_name, di_client)

        else:
            rows.append(
                make_row(
                    file_name, None, None,
                    error=f"Unsupported document type '{file_type}' — "
                          "only PDF and PPTX files are supported.",
                )
            )

    except Exception as exc:
        rows.append(make_row(file_name, None, None, error=str(exc)))
        logger.error("dispatcher.failed", file=file_name, error=str(exc))

    logger.info("dispatcher.done", file=file_name, total_rows=len(rows))
    return rows


# ── PPTX: convert to PDF then process as slide_pdf ────────────────────────────

def _process_pptx(file_path: str, file_name: str) -> List[dict]:
    """Convert PPTX → PDF via PowerPoint COM, then extract via Azure DI."""
    tmp_dir = tempfile.mkdtemp(prefix="pptx_pdf_")
    try:
        pdf_path = convert_pptx_to_pdf(file_path, output_dir=tmp_dir)
        logger.info("dispatcher.pptx_converted", file=file_name, pdf=pdf_path)
        di_client = get_di_client()
        return process_azure_di(
            pdf_path,
            file_name,       # keep original .pptx name in all metadata
            di_client,
            is_pptx_source=True,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
