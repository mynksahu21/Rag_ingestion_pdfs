"""
LLM-based Reranker using Azure OpenAI GPT-5.1.

Takes hybrid-retrieval candidates and re-orders them by relevance
to the query using a single GPT call that returns a ranked index list.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from src.config.settings import settings
from src.config.logging_cfg import logger
from src.utils.openai_client import get_openai_client

_RERANK_PROMPT = """\
You are a relevance ranking assistant.

Given a user query and a list of document chunks, return a JSON array of \
the chunk indices (0-based) ordered from MOST to LEAST relevant to the query.
Include ALL indices. Return ONLY the JSON array — no explanation, no markdown.

Query: {query}

Chunks:
{chunks}

Respond with a JSON array like: [2, 0, 4, 1, 3]
"""


class LLMReranker:
    """Rerank retrieval candidates using GPT-5.1 in a single API call."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = get_openai_client()
        return self._client

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Rerank `candidates` by relevance to `query` using GPT-5.1.
        Returns the top_k most relevant chunks with a `rerank_score` field added.
        Falls back to original order if the LLM call fails.
        """
        if not candidates:
            return []

        # Build numbered chunk list for the prompt (content only — keeps tokens low)
        chunk_lines = "\n\n".join(
            f"[{i}] {c.get('content', '')[:600]}"
            for i, c in enumerate(candidates)
        )

        prompt = _RERANK_PROMPT.format(query=query, chunks=chunk_lines)

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=settings.AZURE_OPENAI_DEPLOYMENT,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=256,
            )
            raw = response.choices[0].message.content.strip()

            # Parse the JSON index list returned by GPT
            ranked_indices: List[int] = json.loads(raw)

            # Validate — indices must be in range
            ranked_indices = [
                i for i in ranked_indices
                if isinstance(i, int) and 0 <= i < len(candidates)
            ]

            # Attach rerank_score (descending position → score)
            n = len(ranked_indices)
            reranked = []
            for rank, idx in enumerate(ranked_indices):
                chunk = dict(candidates[idx])
                chunk["rerank_score"] = round(1.0 - rank / n, 4)
                reranked.append(chunk)

            logger.info(
                "reranker.done",
                query=query[:80],
                candidates=len(candidates),
                returned=min(top_k, len(reranked)),
            )
            return reranked[:top_k]

        except Exception as exc:
            logger.warning(
                "reranker.failed",
                error=str(exc),
                fallback="original hybrid order",
            )
            # Graceful fallback — return original hybrid-ranked order
            return candidates[:top_k]


reranker = LLMReranker()