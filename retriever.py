"""
retriever.py — embedding retrieval over the best-practices KB.

Embeds each practice doc once (via the embedder client), then scores a query by
cosine similarity and returns the top-k. The KB is tiny (~7 docs), so cosine is
implemented in pure Python — no numpy — which keeps the whole pipeline runnable
offline (mock embeddings) with zero third-party dependencies.
"""
from __future__ import annotations

import json
import math

from llm import get_client


def _normalize(vec: list) -> list:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else list(vec)


def _dot(a: list, b: list) -> float:
    return sum(x * y for x, y in zip(a, b))


class Retriever:
    """Cosine top-k over the practices KB; documents embedded once at init."""

    def __init__(self, practices: "list | None" = None, overrides: "dict | None" = None, top_k: int = 3):
        if practices is None:
            with open("practices.json", encoding="utf-8") as fh:
                practices = json.load(fh)
        self.practices = practices
        self.top_k = top_k
        self._client = get_client("embedder", overrides)
        texts = [f"{p['title']}\n{p['content']}" for p in practices]
        self._doc_vecs = [_normalize(v) for v in self._client.embed(texts, is_query=False)]

    def retrieve(self, query: str, k: "int | None" = None) -> list:
        """Return the top-k practices for `query` as [{id, title, content, score}]."""
        k = self.top_k if k is None else k
        if not query or not query.strip() or not self.practices:
            return []
        qv = _normalize(self._client.embed([query], is_query=True)[0])
        scored = [
            {"id": p["id"], "title": p["title"], "content": p["content"], "score": _dot(dv, qv)}
            for p, dv in zip(self.practices, self._doc_vecs)
        ]
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:k]
