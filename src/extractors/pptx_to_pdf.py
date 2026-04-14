"""
PPTX → PDF conversion via PowerPoint COM automation (Windows only).

Requires Microsoft Office (PowerPoint) to be installed.
Uses comtypes to drive the PowerPoint COM interface.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from src.config.logging_cfg import logger


def convert_pptx_to_pdf(pptx_path: str, output_dir: str) -> str:
    """
    Convert a PPTX file to PDF using PowerPoint COM automation.

    Parameters
    ----------
    pptx_path  : absolute path to the source .pptx file
    output_dir : directory where the output PDF will be written

    Returns
    -------
    Absolute path to the generated PDF file.

    Raises
    ------
    RuntimeError if not running on Windows or if comtypes is unavailable.
    Exception propagated from PowerPoint COM on conversion failure.
    """
    if sys.platform != "win32":
        raise RuntimeError(
            "PowerPoint COM automation is only available on Windows. "
            "PPTX conversion requires Microsoft Office to be installed."
        )

    try:
        import comtypes.client  # noqa: F401 — triggers comtypes availability check
    except ImportError:
        raise RuntimeError(
            "comtypes is not installed. "
            "Run: pip install comtypes"
        )

    pptx_abs = str(Path(pptx_path).resolve())
    stem = Path(pptx_path).stem
    pdf_path = str(Path(output_dir).resolve() / f"{stem}.pdf")

    powerpoint = None
    presentation = None
    try:
        import comtypes.client

        powerpoint = comtypes.client.CreateObject("PowerPoint.Application")
        powerpoint.Visible = 1  # required for RDP / VDI sessions

        presentation = powerpoint.Presentations.Open(
            pptx_abs,
            ReadOnly=1,
            Untitled=0,
            WithWindow=0,
        )

        ppSaveAsPDF = 32
        presentation.SaveAs(pdf_path, ppSaveAsPDF)
        logger.info("pptx_to_pdf.converted", src=pptx_abs, dst=pdf_path)
        return pdf_path

    finally:
        if presentation is not None:
            try:
                presentation.Close()
            except Exception:
                pass
        if powerpoint is not None:
            try:
                powerpoint.Quit()
            except Exception:
                pass
