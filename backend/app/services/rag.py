"""Chunking, embeddings, hybrid scoring, and text extraction.

Two embedding backends share one interface:

* ``deterministic`` — a SHA-256 derived pseudo-vector. It is reproducible, needs
  no network, and keeps the offline benchmark and test suite hermetic. It is
  **not** semantic: near-synonymous text hashes to unrelated vectors.
* ``openai`` — real embeddings from OpenAI or any OpenAI-compatible gateway
  (set ``OPENAI_BASE_URL``). This is what should be used in production.

The hybrid scorer deliberately weights lexical overlap heavily. Payment
operations questions turn on exact tokens — ``401``, ``429``, ``refund ID``,
``trace ID``, ``p95`` — and a dense vector alone ranks those poorly. The
weights are part of the published evaluation result, so changing them requires
re-running ``evals/benchmark.py`` and updating the reported numbers.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings

try:
    import numpy as np
except ImportError:  # pragma: no cover - NumPy is a declared dependency
    np = None  # type: ignore[assignment]

#: Blend of dense and lexical signal. Tuned so exact operational identifiers
#: are not outranked by topical similarity. See module docstring.
VECTOR_WEIGHT = 0.15
LEXICAL_WEIGHT = 0.85

_CJK_RANGE = r"一-鿿"


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    content: str
    embedding: list[float]
    metadata: dict[str, object] = field(default_factory=dict)


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def chunk_text(text: str, *, size: int = 900, overlap: int = 120) -> list[str]:
    """Split text on whitespace boundaries with a fixed overlap.

    Overlap is what lets a runbook sentence that straddles a boundary still be
    retrieved whole.
    """
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + size)
        if end < len(normalized):
            boundary = normalized.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        chunks.append(normalized[start:end].strip())
        if end == len(normalized):
            break
        start = max(0, end - overlap)
    return chunks


def deterministic_embedding(text: str, dimensions: int = 1536) -> list[float]:
    """Stable, non-semantic embedding for offline mode and tests.

    The SHA-256 digest provides 32 bytes which are tiled across ``dimensions``
    and scaled to ``[-1, 1]``, then L2-normalised. The tiling is what makes this
    useless for semantic recall while remaining perfectly deterministic.
    """
    digest = hashlib.sha256(text.lower().encode("utf-8")).digest()
    values = [((digest[i % len(digest)] / 255.0) * 2.0) - 1.0 for i in range(dimensions)]
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using the configured provider."""
    if not texts:
        return []
    if settings.embedding_provider == "deterministic":
        return [deterministic_embedding(text, settings.embedding_dimensions) for text in texts]

    if settings.embedding_provider in {"openai", "openai_compatible"}:
        from openai import OpenAI

        client_kwargs: dict[str, object] = {}
        if settings.openai_api_key:
            client_kwargs["api_key"] = settings.openai_api_key
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url
        client = OpenAI(**client_kwargs)  # type: ignore[arg-type]
        response = client.embeddings.create(model=settings.embedding_model, input=texts)
        return [item.embedding for item in response.data]

    raise ValueError(f"unknown embedding provider: {settings.embedding_provider}")


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Both operands are expected to be L2-normalised, so this is a dot product."""
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    return sum(a * b for a, b in zip(left, right))


def cosine_similarity_batch(query: list[float], embeddings: list[list[float]]) -> list[float]:
    """Score many normalised vectors in one vectorized operation when available.

    This is the optimised path measured in ``evals/benchmark.py``: a single
    matrix-vector product replaces a Python loop over every candidate.
    """
    if not embeddings:
        return []
    if any(len(embedding) != len(query) for embedding in embeddings):
        raise ValueError("embedding dimensions do not match")
    if np is not None:
        query_array = np.asarray(query, dtype=np.float32)
        matrix = np.asarray(embeddings, dtype=np.float32)
        return (matrix @ query_array).tolist()
    return [cosine_similarity(query, embedding) for embedding in embeddings]


def lexical_terms(value: str) -> set[str]:
    """Tokenize for lexical overlap, adding CJK character bigrams.

    Chinese is typically returned as one long contiguous token, which makes a
    whole-sentence equality test useless. Bigrams give short questions a real
    chance of matching runbook phrases such as 退款 / 工作日.
    """
    tokens = set(re.findall(rf"[\w{_CJK_RANGE}]+", value.lower()))
    for token in list(tokens):
        if any("一" <= char <= "鿿" for char in token):
            tokens.update(token[index : index + 2] for index in range(len(token) - 1))
    return tokens


def lexical_overlap(query: str, content: str) -> float:
    """Fraction of query terms present in the content."""
    query_terms = lexical_terms(query)
    if not query_terms:
        return 0.0
    return len(query_terms & lexical_terms(content)) / len(query_terms)


def hybrid_score(query: str, content: str, vector_score: float) -> float:
    return VECTOR_WEIGHT * vector_score + LEXICAL_WEIGHT * lexical_overlap(query, content)


def extract_text(filename: str, raw: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(raw))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:  # pragma: no cover - depends on the PDF itself
            raise ValueError(f"PDF extraction failed: {exc}") from exc
    if suffix in {".html", ".htm"}:
        from bs4 import BeautifulSoup

        return BeautifulSoup(raw.decode("utf-8", errors="ignore"), "html.parser").get_text(" ")
    if suffix in {".md", ".markdown", ".txt", ".rst", ".log", ".csv", ""}:
        return raw.decode("utf-8", errors="ignore")
    # Unknown extension: attempt UTF-8 rather than silently indexing nothing.
    return raw.decode("utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# Metrics used by the retrieval evaluation
# ---------------------------------------------------------------------------


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for position, candidate in enumerate(ranked_ids, start=1):
        if candidate in relevant_ids:
            return 1.0 / position
    return 0.0
