"""Print how similar each model thinks a few real feed descriptions are to a café.
"""

from similar_companies.embedding import HASHING_MODEL, create_embedder

MODELS = [
    HASHING_MODEL,
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]
QUERY = "Operating cafes and coffee shops in the Klang Valley"
CANDIDATES = {
    "café, Malay, same place": "Mengusahakan kedai kopi dan kafe di the Klang Valley",
    "café, English, other place": "Operating cafes and coffee shops in Penang",
    "parcels, English, same place": "Last-mile parcel delivery across the Klang Valley",
    "parcels, Malay, same place": "Penghantaran bungkusan ke seluruh the Klang Valley",
    "kindergarten, same place": "Private kindergarten and childcare in the Klang Valley",
}


def main() -> None:
    print(f"Query: {QUERY!r}\n")
    print(f"{'candidate':<30}" + "".join(f"{name.split('/')[-1][:22]:>24}" for name in MODELS))
    scores = {}
    for model_name in MODELS:
        query, *candidates = create_embedder(model_name).embed([QUERY, *CANDIDATES.values()])
        scores[model_name] = [
            sum(q * c for q, c in zip(query, vec, strict=True)) for vec in candidates
        ]
    for row, label in enumerate(CANDIDATES):
        print(f"{label:<30}" + "".join(f"{scores[m][row]:>24.3f}" for m in MODELS))


if __name__ == "__main__":
    main()
