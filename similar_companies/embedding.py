"""Turn a company into the embeded text, and that text into a unit-length 384-dim vector.
"""

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

from similar_companies.validation import UpsertEvent

DIMENSIONS = 384  # must match vector(384) in schema.sql
HASHING_MODEL = "hashing-bow-384"
WORD = re.compile(r"\w+")


def embedding_text(event: UpsertEvent) -> str:
    """The text that represents what a company does.

    The description is the similarity signal. 39 companies never have
    one, so they fall back to industry and name (names often say the trade, e.g. "Bakery");
    otherwise they could never appear in recommendations.
    """
    if event.nature_of_business:
        return event.nature_of_business
    return ". ".join(part for part in (event.industry, event.company_name) if part)


def l2_normalize(vector: Sequence[float]) -> list[float]:
    """Scale to length 1, so cosine distance, inner product and L2 all rank the same way."""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise ValueError("cannot normalize a zero vector")  # its cosine distance would be NaN
    return [value / norm for value in vector]


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Bag-of-words hashed into 384 buckets, then normalized.

    Uses blake2b rather than Python's hash(): hash() of a string changes every time the
    process starts, which would give a company a different vector after a restart.
    """

    model_name = HASHING_MODEL

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * DIMENSIONS
        tokens = WORD.findall(text.casefold()) or [text.casefold()]
        for token in tokens:
            value = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest())
            vector[value % DIMENSIONS] += 1.0
        return l2_normalize(vector)


class FastEmbedEmbedder:
    """A sentence-embedding model run locally on CPU through fastembed"""

    def __init__(self, model_name: str) -> None:
        # Imported here so the hashing placeholder still works if fastembed can't be installed.
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = [l2_normalize(vector.tolist()) for vector in self._model.embed(list(texts))]
        for vector in vectors:
            if len(vector) != DIMENSIONS:
                message = f"{self.model_name} gives {len(vector)} dimensions, not {DIMENSIONS}"
                raise ValueError(message)
        return vectors


def create_embedder(model_name: str) -> Embedder:
    """Build the embedder named in configuration."""
    if model_name == HASHING_MODEL:
        return HashingEmbedder()
    return FastEmbedEmbedder(model_name)
