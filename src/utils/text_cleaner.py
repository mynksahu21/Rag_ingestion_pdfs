"""
OCR Text Cleaner — exact port from notebook CELL 3.
Called on every extracted paragraph before storing.
"""
import re


def clean_ocr_text(text: str) -> str:
    """
    Normalise raw OCR / Document Intelligence output.
    - Fixes hyphenated line breaks
    - Standardises bullet symbols
    - Merges wrapped paragraph lines
    - Collapses whitespace
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\n(?=\w)", "", text)

    for b in ["«", "•", "▪", "◦", "‣", "*"]:
        text = text.replace(b, "•")

    text = re.sub(r"\s*>\s*", ": ", text)

    lines = text.split("\n")
    cleaned: list[str] = []
    buffer = ""

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if buffer:
                cleaned.append(buffer.strip())
                buffer = ""
            continue

        if stripped.endswith(":"):
            if buffer:
                cleaned.append(buffer.strip())
                buffer = ""
            cleaned.append(stripped)
            continue

        if stripped.startswith("•"):
            if buffer:
                cleaned.append(buffer.strip())
                buffer = ""
            cleaned.append(stripped)
            continue

        buffer = (buffer + " " + stripped) if buffer else stripped

    if buffer:
        cleaned.append(buffer.strip())

    result = "\n".join(cleaned)
    result = re.sub(r"[ \t]+", " ", result)
    return result.strip()
