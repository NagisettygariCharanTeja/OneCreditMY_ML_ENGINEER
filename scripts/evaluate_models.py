"""Compare embedding set-ups on how well "similar companies" match by industry.

Builds the final company state from data/feed.jsonl in memory (same validation and
newer-wins rule as the consumer), embeds each company's text with every set-up, and
ranks neighbours exactly like the API (cosine, then registration number, never itself).

Metric: industry precision@5 = share of a company's top 5 that have its industry.
Industry is a coarse, weak label (11 values), so it is a proxy for "does similar things",
not ground truth. Only active companies with a known industry take part, as queries and as
candidates. Reported two ways:
- all candidates
- excluding candidates with exactly the same text as the query: many companies share an
  identical description, which every set-up (even word hashing) matches trivially; this
  version measures whether a model generalizes across wording and language.
A random ranking would score the "random" row.
"""

import argparse
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from similar_companies.embedding import HASHING_MODEL, create_embedder, embedding_text
from similar_companies.validation import StruckOffEvent, UpsertEvent, parse_event

FEED_PATH = Path("data/feed.jsonl")
K = 5
PLACES = "the Klang Valley|Ipoh|Johor Bahru|Kota Kinabalu|Kuantan|Kuching|Melaka|Penang|Seremban"
TRAILING_PLACE = re.compile(rf"\s+(?:in|di|across|ke seluruh)\s+(?:{PLACES})$")


def strip_location(text: str) -> str:
    """'Operating cafes in the Klang Valley' -> 'Operating cafes' (also di, across, ke seluruh)."""
    return TRAILING_PLACE.sub("", text)


@dataclass(frozen=True)
class Setup:
    label: str
    model: str
    text: Callable[[str], str] = lambda text: text
    pseudo_relevance_feedback: bool = False  # re-query with the mean of the first top 5


SETUPS = [
    Setup("hashing (baseline)", HASHING_MODEL),
    Setup("bge-small-en-v1.5", "BAAI/bge-small-en-v1.5"),
    Setup("multilingual", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    Setup(
        "multilingual, location stripped",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        text=strip_location,
    ),
    Setup(
        "multilingual + pseudo-relevance feedback",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        pseudo_relevance_feedback=True,
    ),
]


def final_companies() -> list[UpsertEvent]:
    """Active companies after applying the feed with the consumer's newer-wins rule."""
    latest: dict[str, UpsertEvent | StruckOffEvent] = {}
    for line in FEED_PATH.read_text(encoding="utf-8").splitlines():
        event = parse_event(line)
        if isinstance(event, UpsertEvent | StruckOffEvent):
            current = latest.get(event.registration_number)
            if current is None or event.updated_at > current.updated_at:
                latest[event.registration_number] = event
    active = [e for e in latest.values() if isinstance(e, UpsertEvent)]
    return sorted(active, key=lambda e: e.registration_number)


def top_k(scores: np.ndarray, excluded: np.ndarray) -> np.ndarray:
    """Indices of the K best candidates per row: highest score, then lowest index
    (rows are sorted by registration number, so this matches the API's tie-break)."""
    scores = np.where(excluded, -np.inf, scores)
    order = np.argsort(-scores, axis=1, kind="stable")
    return order[:, :K]


def precision(neighbours: np.ndarray, industries: np.ndarray) -> float:
    return float((industries[neighbours] == industries[:, None]).mean())


def evaluate(setup: Setup, companies: list[UpsertEvent]) -> tuple[float, float]:
    originals = [embedding_text(c) for c in companies]
    texts = [setup.text(original) for original in originals]
    unique = sorted(set(texts))
    vectors = dict(zip(unique, create_embedder(setup.model).embed(unique), strict=True))
    matrix = np.array([vectors[t] for t in texts])
    scores = matrix @ matrix.T  # cosine similarity: vectors are unit length
    industries = np.array([c.industry for c in companies])
    self_mask = np.eye(len(companies), dtype=bool)
    # "Identical" is judged on the original text, so every set-up faces the same candidates.
    same_text = np.array(originals)[:, None] == np.array(originals)[None, :]

    results = []
    for excluded in (self_mask, self_mask | same_text):
        neighbours = top_k(scores, excluded)
        if setup.pseudo_relevance_feedback:
            expanded = matrix + matrix[neighbours].mean(axis=1)
            expanded /= np.linalg.norm(expanded, axis=1, keepdims=True)
            neighbours = top_k(expanded @ matrix.T, excluded)
        results.append(precision(neighbours, industries))
    return results[0], results[1]


def random_precision(companies: list[UpsertEvent]) -> float:
    """Expected precision@K of a random ranking: the chance a random other company
    shares the query's industry, averaged over queries."""
    industries = np.array([c.industry for c in companies])
    counts = {i: int((industries == i).sum()) for i in set(industries)}
    return float(np.mean([(counts[i] - 1) / (len(industries) - 1) for i in industries]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", help="only set-ups whose label contains these words")
    args = parser.parse_args()
    setups = [s for s in SETUPS if not args.models or any(word in s.label for word in args.models)]

    companies = [c for c in final_companies() if c.industry]
    print(f"{len(companies)} active companies with a known industry (queries and candidates)\n")
    print(f"| Set-up | Industry P@{K} | P@{K}, identical text excluded |")
    print("|---|---|---|")
    chance = random_precision(companies)
    print(f"| random ranking | {chance:.3f} | {chance:.3f} |")
    results = {}
    for setup in setups:
        all_candidates, generalizing = evaluate(setup, companies)
        results[setup.label] = {"all": all_candidates, "identical_text_excluded": generalizing}
        print(f"| {setup.label} | {all_candidates:.3f} | {generalizing:.3f} |")
    Path("evaluation_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
