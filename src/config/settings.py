"""
RAG Ingestion Pipeline — Configuration.
Single source of truth. All values from .env file.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Central configuration — replaces notebook's config dict + Databricks widgets."""

    # ── App ──
    APP_NAME: str = "RAG Ingestion Pipeline"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── Paths (computed in model_post_init if not set) ──
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    UPLOAD_DIR: Optional[Path] = None
    OUTPUT_DIR: Optional[Path] = None
    VECTOR_INDEX_DIR: Optional[Path] = None
    IMAGES_DIR: Optional[Path] = None        # saved useful images: data/output/images/

    # ── Azure Document Intelligence ──
    AZURE_DI_ENDPOINT: str = ""
    AZURE_DI_KEY: str = ""

    # ── Azure OpenAI (image summarization + embeddings) ──
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_API_VERSION: str = "2024-02-01"
    AZURE_OPENAI_DEPLOYMENT: str = "gpt-5.1"

    # ── OpenAI Embeddings ──
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = "text-embedding-ada-002"
    EMBEDDING_DIMENSION: int = 1536  # text-embedding-ada-002 = 1536

    # ── Chunking (token-based: target 300-500 tokens, 80-token overlap) ──
    # 1 token ≈ 4 chars for English text
    CHUNK_SIZE: int = 2000          # chars ≈ 500 tokens (max)
    CHUNK_OVERLAP: int = 320        # chars ≈ 80 tokens
    MIN_CHUNK_LENGTH: int = 100     # chars ≈ 25 tokens minimum
    CHUNK_TARGET_MIN_TOKENS: int = 300
    CHUNK_TARGET_MAX_TOKENS: int = 500
    CHUNK_OVERLAP_TOKENS: int = 80

    # ── Image Processing ──
    IMAGE_MODE: str = "base64"          # "base64" | "bbox"
    MIN_PIXEL_AREA: int = 100 * 100     # minimum pixel area for useful image (10 000 px)
    PPTX_BG_THRESHOLD: float = 0.85    # bbox coverage ≥ this → background image (PPTX & PDF)
    MIN_TEXT_LENGTH: int = 1            # OCR char count below this → drop (0 = no text at all)
    LOGO_MAX_KB: int = 25              # logo filter: file must be below this size...
    LOGO_MAX_PIXEL_AREA: int = 200 * 200  # ...AND pixel area below this (40 000 px)

    # ── Text/Image overlap detection ──
    # Paragraph is dropped when this fraction of its area falls inside a figure bbox.
    TEXT_FIGURE_OVERLAP_THRESHOLD: float = 0.50

    # ── File Routing ──
    # Azure DI natively supports both PDF and PPTX — no conversion needed.
    SUPPORTED_EXTENSIONS: List[str] = ["pdf", "pptx"]

    # ── Concurrency ──
    MAX_CONCURRENT_PROCESSING: int = 5
    IMAGE_SUMMARIZATION_CONCURRENCY: int = 5

    # ── FAISS ──
    FAISS_TOP_K: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Allow extra vars in .env without crashing

    def model_post_init(self, __context):
        if self.UPLOAD_DIR is None:
            self.UPLOAD_DIR = self.BASE_DIR / "data" / "uploads"
        if self.OUTPUT_DIR is None:
            self.OUTPUT_DIR = self.BASE_DIR / "data" / "output"
        if self.VECTOR_INDEX_DIR is None:
            self.VECTOR_INDEX_DIR = self.BASE_DIR / "data" / "vector_index"
        if self.IMAGES_DIR is None:
            self.IMAGES_DIR = self.OUTPUT_DIR / "images"
        for d in (self.UPLOAD_DIR, self.OUTPUT_DIR, self.VECTOR_INDEX_DIR, self.IMAGES_DIR):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
