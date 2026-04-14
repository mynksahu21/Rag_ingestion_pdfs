"""
Shared Azure OpenAI client factory.

Every part of the codebase that needs to call the Azure OpenAI API should
obtain its client from here.  The clients are built with:
  • certifi CA bundle  — works in environments with corporate SSL inspection
  • check_hostname = False / CERT_NONE  — handles self-signed / proxy certs
  • httpx transport  — required by the openai SDK's async/sync HTTP layer
"""
from __future__ import annotations

import os
import ssl

import certifi
import httpx
from openai import AsyncAzureOpenAI, AzureOpenAI

from src.config.settings import settings


def get_ssl_context() -> ssl.SSLContext:
    """Return an SSL context that trusts certifi's CA bundle."""
    ctx = ssl.create_default_context(cafile=certifi.where())
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def get_openai_client() -> AzureOpenAI:
    """
    Synchronous Azure OpenAI client (used for embeddings, batch calls).
    Credentials are read from settings (which reads from .env).
    """
    return AzureOpenAI(
        api_key=settings.AZURE_OPENAI_API_KEY,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        http_client=httpx.Client(verify=get_ssl_context()),
    )


def get_async_openai_client() -> AsyncAzureOpenAI:
    """
    Async Azure OpenAI client (used for image summarisation, chunk summaries,
    table title generation).
    """
    return AsyncAzureOpenAI(
        api_key=settings.AZURE_OPENAI_API_KEY,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        http_client=httpx.AsyncClient(verify=get_ssl_context()),
    )
