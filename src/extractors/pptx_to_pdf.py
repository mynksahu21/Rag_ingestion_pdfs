"""
PPTX → PDF conversion via LibreOffice.

Requires LibreOffice to be installed and the `libreoffice-convert` Python
package (which shells out to the `soffice` binary).

On Windows, LibreOffice is typically NOT on PATH.  The converter probes the
standard installation directories automatically.
"""
from __future__ import annotations

import sys
from pathlib import Path

from src.config.logging_cfg import logger

# Common Windows installation paths for soffice.exe (tried in order).
_WINDOWS_SOFFICE_CANDIDATES = [
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
]


def _soffice_path() -> str | None:
    """Return the soffice binary path, or None to let the library search PATH."""
    if sys.platform == "win32":
        for candidate in _WINDOWS_SOFFICE_CANDIDATES:
            if Path(candidate).exists():
                return candidate
        raise RuntimeError(
            "LibreOffice not found. Install it from https://www.libreoffice.org/download/ "
            "or add soffice.exe to your PATH."
        )
    return None  # macOS / Linux: rely on PATH


def convert_pptx_to_pdf(pptx_path: str, output_dir: str) -> str:
    """
    Convert a PPTX file to PDF using LibreOffice.

    Parameters
    ----------
    pptx_path  : absolute path to the source .pptx / .ppt file
    output_dir : directory where the output PDF will be written

    Returns
    -------
    Absolute path to the generated PDF file.

    Raises
    ------
    RuntimeError if libreoffice-convert is unavailable or LibreOffice is not found.
    Exception propagated from LibreOffice on conversion failure.
    """
    try:
        import libreoffice_convert
    except ImportError:
        raise RuntimeError(
            "libreoffice-convert is not installed. "
            "Run: pip install libreoffice-convert"
        )

    pptx_abs = Path(pptx_path).resolve()
    pdf_path = Path(output_dir).resolve() / f"{pptx_abs.stem}.pdf"

    soffice = _soffice_path()  # None on non-Windows → library uses PATH

    with open(pptx_abs, "rb") as f:
        pptx_bytes = f.read()

    convert_kwargs: dict = {"unoconv": False}
    if soffice is not None:
        convert_kwargs["soffice_path"] = soffice

    pdf_bytes = libreoffice_convert.convert(pptx_bytes, ".pdf", **convert_kwargs)

    pdf_path.write_bytes(pdf_bytes)
    logger.info("pptx_to_pdf.converted", src=str(pptx_abs), dst=str(pdf_path))
    return str(pdf_path)
