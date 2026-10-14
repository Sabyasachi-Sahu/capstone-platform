"""Phase 2: retrieve the closest patient documents for a question."""

import sys

import chromadb
import ollama

from src.common.config import PROJECT_ROOT, get_settings

DB_DIR = PROJECT_ROOT / "data" / "vectordb"
COLLECTION = "patients"
TOP_K = 5


def search(question: str, top_k: int = TOP_K) -> list:
    s = get_settings()
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(COLLECTION)

    q_vec = ollama.embed(model=s.embed_model, input=question).embeddings[0]
    result = collection.query(query_embeddings=[q_vec], n_results=top_k)

    hits = []
    for doc_id, text, dist in zip(
        result["ids"][0], result["documents"][0], result["distances"][0]
    ):
        hits.append({"id": doc_id, "text": text, "similarity": 1 - dist})
    return hits


def main():
    question = " ".join(sys.argv[1:]) or "Which patients have diabetes?"
    print(f"Question: {question}\n")
    for rank, hit in enumerate(search(question), start=1):
        print(f"#{rank}  similarity={hit['similarity']:.3f}  id={hit['id']}")
        print(f"    {hit['text'].splitlines()[1]}\n")


if __name__ == "__main__":
    main()
