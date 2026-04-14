"""
Image Summarizer — exact port of notebook's CELL 24.
Uses Azure OpenAI GPT-4o to summarize extracted images with page context.
For charts/graphs: returns BOTH a structured data table AND a description.
"""
from __future__ import annotations

import asyncio
import re
from typing import Dict, List

from openai import BadRequestError

from src.config.settings import settings
from src.config.logging_cfg import logger
from src.utils.openai_client import get_async_openai_client, AsyncAzureOpenAI


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

# Updated prompt that handles charts distinctly from other images
PROMPT_TEMPLATE = """PAGE CONTEXT:
{context}

Analyze this image carefully.

If this is a CHART or GRAPH (bar chart, line chart, pie chart, scatter plot, histogram, etc.):
Respond in EXACTLY this format:

Graph Data Table:
Row1: <x_label>=<value>, <y_label>=<value>
Row2: <x_label>=<value>, <y_label>=<value>
(List ALL visible data points)

Graph Description:
<A concise 1-3 sentence description of the chart type, title, axes, and key trends>

If this is any OTHER image (photo, diagram, schematic, table screenshot, etc.):
Describe: type of visual, key data, labels/titles, insights, and all readable text.
Length: 150-300 words.

If only logo/copyright/watermark → respond with exactly: SKIP
If blank/decorative/background → respond with exactly: SKIP
"""


def _parse_summary_response(text: str) -> dict:
    """
    Parse the GPT response into structured fields.
    Returns dict with: graph_data_table, graph_description, is_chart, image_description.
    """
    if not text:
        return {"is_chart": False, "graph_data_table": None, "graph_description": None, "image_description": None}

    # Detect chart response by presence of structured markers
    has_data_table = bool(re.search(r'Graph Data Table\s*:', text, re.IGNORECASE))
    has_description_section = bool(re.search(r'Graph Description\s*:', text, re.IGNORECASE))

    if has_data_table or has_description_section:
        # Extract data table block
        dt_match = re.search(
            r'Graph Data Table\s*:\s*\n((?:Row\d+:.*(?:\n|$))*)',
            text,
            re.IGNORECASE,
        )
        graph_data_table = None
        if dt_match:
            rows_text = dt_match.group(1).strip()
            graph_data_table = f"Graph Data Table:\n{rows_text}"

        # Extract description block (everything after "Graph Description:")
        desc_match = re.search(
            r'Graph Description\s*:\s*\n([\s\S]+?)(?:\n\n|\Z)',
            text,
            re.IGNORECASE,
        )
        graph_description = desc_match.group(1).strip() if desc_match else None

        # Full text kept in image_description for backward compat
        return {
            "is_chart": True,
            "graph_data_table": graph_data_table,
            "graph_description": graph_description,
            "image_description": text,
        }
    else:
        return {
            "is_chart": False,
            "graph_data_table": None,
            "graph_description": None,
            "image_description": text,
        }


async def summarize_image(row: dict, client: AsyncAzureOpenAI) -> dict:
    """
    Summarize one image row. Always returns a dict (never None).
    For charts: populates graph_data_table + graph_description.
    For other images: populates image_description.
    """
    base_result = {
        "id": str(row.get("id") or ""),
        "file_name": str(row.get("file_name") or ""),
        "page": int(row.get("page") or 0),
        "image": str(row.get("image") or ""),
        "image_description": None,
        "graph_data_table": None,
        "graph_description": None,
        "is_chart": False,
        "error": None,
        "skipped": False,
    }

    image_b64 = row.get("image")
    if not image_b64:
        base_result["error"] = "No base64 image data"
        return base_result

    prompt = PROMPT_TEMPLATE.format(context=row.get("page_context") or "")

    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=settings.AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                },
                            },
                        ],
                    }
                ],
                temperature=0.2,
                max_completion_tokens=600,
            )

            summary = (response.choices[0].message.content or "").strip()
            if "SKIP" in summary.upper() and len(summary) < 20:
                base_result["skipped"] = True
                return base_result

            parsed = _parse_summary_response(summary)
            base_result.update(parsed)
            return base_result

        except Exception as e:
            # Content-policy rejection — no point retrying, skip the image.
            if _is_content_filter(e):
                base_result["skipped"] = True
                logger.warning(
                    "image_summarizer.content_filtered",
                    file=base_result["file_name"],
                    page=base_result["page"],
                    image_id=base_result["id"],
                )
                return base_result

            if attempt < 2:
                await asyncio.sleep(2**attempt)
            else:
                base_result["error"] = str(e)[:500]
                logger.error(
                    "image_summarizer.api_error",
                    file=base_result["file_name"],
                    page=base_result["page"],
                    image_id=base_result["id"],
                    error=str(e),
                )
                return base_result

    base_result["error"] = "Max retries exhausted"
    return base_result


async def summarize_images(
    rows: List[dict], page_texts: Dict[str, Dict[int, str]]
) -> List[dict]:
    """
    Summarize all image rows using Azure OpenAI.
    Matches notebook's process_images with semaphore concurrency.
    """
    if not settings.AZURE_OPENAI_API_KEY:
        raise ValueError(
            "AZURE_OPENAI_API_KEY is not set — image summarization cannot proceed. "
            "Set it in your .env file or environment."
        )

    if settings.IMAGE_MODE != "base64":
        logger.warning(
            "image_summarizer.skipped",
            reason=f"IMAGE_MODE='{settings.IMAGE_MODE}' — only 'base64' mode provides "
                   "actual image data for the vision API",
        )
        return []

    image_rows = [
        r
        for r in rows
        if r.get("type") == "image"
        and r.get("image")
        and not r.get("error")
    ]

    if not image_rows:
        logger.info("image_summarizer.no_images")
        return []

    # Attach page context (notebook's page_text_df join logic)
    for row in image_rows:
        fn = row.get("file_name", "")
        pg = row.get("page", 0) or 0
        ctx = page_texts.get(fn, {}).get(pg, "")
        row["page_context"] = ctx[:1500]

    logger.info("image_summarizer.start", count=len(image_rows))

    client = get_async_openai_client()
    sem = asyncio.Semaphore(settings.IMAGE_SUMMARIZATION_CONCURRENCY)

    async def _bounded(r: dict) -> dict:
        async with sem:
            return await summarize_image(r, client)

    results = await asyncio.gather(*[_bounded(r) for r in image_rows])

    summarized = sum(1 for r in results if r.get("image_description"))
    charts = sum(1 for r in results if r.get("is_chart"))
    skipped = sum(1 for r in results if r.get("skipped"))
    errors = sum(1 for r in results if r.get("error"))
    logger.info(
        "image_summarizer.done",
        summarized=summarized,
        charts=charts,
        skipped=skipped,
        errors=errors,
    )
    return list(results)
