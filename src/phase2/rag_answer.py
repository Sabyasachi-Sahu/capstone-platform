"""Phase 2: retrieve patient documents and generate a grounded answer."""

import re
import sys

import ollama

from src.common.config import get_settings
from src.phase2.search_index import search

SYSTEM_PROMPT = (
    "You are a clinical data assistant. Answer ONLY from the patient records "
    "provided in the context. Cite the patient ID in square brackets for every "
    "fact you use. If the context does not contain the answer, say so. "
    "The context is a small sample of the most relevant records, not the full "
    "database, so never present counts or lists as complete."
)


def build_context(hits: list) -> str:
    blocks = [f"[{hit['id']}]\n{hit['text']}" for hit in hits]
    return "\n\n".join(blocks)


HYPHENS = "\u2010\u2011\u2012\u2013\u2014\u2212"
ID_PATTERN = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def check_citations(text: str, sources: list) -> dict:
    normalised = text.translate(str.maketrans(HYPHENS, "-" * len(HYPHENS)))
    cited = set(ID_PATTERN.findall(normalised.lower()))
    return {
        "verified": sorted(cited & set(sources)),
        "unverified": sorted(cited - set(sources)),
    }


def answer(question: str) -> dict:
    s = get_settings()
    hits = search(question)
    context = build_context(hits)

    response = ollama.chat(
        model=s.chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return {"answer": response.message.content, "sources": [h["id"] for h in hits]}


def main():
    question = (
        " ".join(sys.argv[1:])
        or "What conditions commonly occur alongside type 2 diabetes?"
    )
    result = answer(question)
    print(f"Question: {question}\n")
    print(result["answer"])
    print("\nRetrieved sources:")
    for source_id in result["sources"]:
        print(f"  - {source_id}")
    check = check_citations(result["answer"], result["sources"])
    print(
        f"\nCitation check: {len(check['verified'])} verified, "
        f"{len(check['unverified'])} unverified"
    )
    for bad in check["unverified"]:
        print(f"  ⚠ not retrieved: {bad}")


if __name__ == "__main__":
    main()
