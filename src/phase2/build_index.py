"""Phase 2: embed patient documents and store them in ChromaDB."""

import json

import chromadb
import ollama

from src.common.config import PROJECT_ROOT, get_settings

DOCS_FILE = PROJECT_ROOT / "data" / "processed" / "patient_docs.jsonl"
DB_DIR = PROJECT_ROOT / "data" / "vectordb"
COLLECTION = "patients"
BATCH_SIZE = 64


def load_docs() -> list:
    with DOCS_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    s = get_settings()
    docs = load_docs()
    print(f"Loaded documents : {len(docs)}")

    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    for start in range(0, len(docs), BATCH_SIZE):
        batch = docs[start : start + BATCH_SIZE]
        texts = [d["text"] for d in batch]
        ids = [d["id"] for d in batch]

        vectors = ollama.embed(model=s.embed_model, input=texts).embeddings
        collection.upsert(ids=ids, documents=texts, embeddings=vectors)

        print(f"  indexed {start + len(batch):>5} / {len(docs)}")

    print(f"Collection count : {collection.count()}")
    print(f"Stored at        : {DB_DIR}")


if __name__ == "__main__":
    main()
