"""Unit tests for embedding.py. 
"""

import math
import os
from datetime import UTC, datetime

import pytest

from similar_companies.embedding import (
    DIMENSIONS,
    HASHING_MODEL,
    FastEmbedEmbedder,
    HashingEmbedder,
    create_embedder,
    embedding_text,
    l2_normalize,
)
from similar_companies.validation import UpsertEvent


def event(nature_of_business: str | None, industry: str | None = "Food & Beverage") -> UpsertEvent:
    return UpsertEvent(
        registration_number="SYN-001550",
        updated_at=datetime(2026, 8, 14, tzinfo=UTC),
        company_name="Sinar Teguh Bakery Sdn. Bhd.",
        nature_of_business=nature_of_business,
        state="Selangor",
        industry=industry,
    )


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))  # inputs are unit length


def test_embedding_text_uses_the_description():
    assert embedding_text(event("Operating cafes")) == "Operating cafes"


def test_embedding_text_falls_back_to_industry_and_name():
    assert embedding_text(event(None)) == "Food & Beverage. Sinar Teguh Bakery Sdn. Bhd."


def test_embedding_text_falls_back_to_name_alone():
    assert embedding_text(event(None, industry=None)) == "Sinar Teguh Bakery Sdn. Bhd."


def test_l2_normalize_gives_unit_length():
    assert math.isclose(math.hypot(*l2_normalize([3.0, 4.0])), 1.0)


def test_l2_normalize_refuses_zero_vector():
    with pytest.raises(ValueError):
        l2_normalize([0.0, 0.0])


def test_hashing_vectors_are_unit_length_with_384_dimensions():
    (vector,) = HashingEmbedder().embed(["Operating cafes and coffee shops"])
    assert len(vector) == DIMENSIONS
    assert math.isclose(math.hypot(*vector), 1.0)


def test_hashing_is_deterministic_across_instances_and_ignores_case():
    first = HashingEmbedder().embed(["Kedai Kopi"])[0]
    assert HashingEmbedder().embed(["kedai kopi"])[0] == first


def test_hashing_ranks_shared_words_closer():
    query, near, far = HashingEmbedder().embed(
        ["Operating cafes and coffee shops", "Coffee shops and bakeries", "Freight forwarding"]
    )
    assert cosine(query, near) > cosine(query, far)


def test_hashing_handles_text_without_word_characters():
    (vector,) = HashingEmbedder().embed(["---"])
    assert math.isclose(math.hypot(*vector), 1.0)


def test_create_embedder_returns_placeholder_by_name():
    assert isinstance(create_embedder(HASHING_MODEL), HashingEmbedder)


class FakeVector:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def tolist(self) -> list[float]:
        return self.values


def test_fastembed_wrong_dimensions_fail_loudly(monkeypatch):
    fastembed = pytest.importorskip("fastembed")

    class FakeTextEmbedding:
        def __init__(self, model_name: str) -> None:
            pass

        def embed(self, texts):
            return [FakeVector([1.0] * 768) for _ in texts]

    monkeypatch.setattr(fastembed, "TextEmbedding", FakeTextEmbedding)
    with pytest.raises(ValueError, match="768 dimensions, not 384"):
        FastEmbedEmbedder("some-768-dim-model").embed(["cafes"])


@pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="downloads a model")
@pytest.mark.parametrize(
    "model_name",
    ["BAAI/bge-small-en-v1.5", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"],
)
def test_real_model_smoke(model_name):
    query, near, far = create_embedder(model_name).embed(
        [
            "Operating cafes and coffee shops",
            "A coffee shop and cafe",
            "Freight forwarding services",
        ]
    )
    assert len(query) == DIMENSIONS
    assert math.isclose(math.hypot(*query), 1.0, rel_tol=1e-6)
    assert cosine(query, near) > cosine(query, far)
