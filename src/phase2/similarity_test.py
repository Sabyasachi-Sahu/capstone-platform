"""Phase 2: compare embeddings with cosine similarity."""

import math

import ollama

from src.common.config import get_settings


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def main():
    s = get_settings()
    question = "Which patients have high blood sugar?"
    sentences = [
        "Patient was diagnosed with type 2 diabetes.",
        "Elevated glucose levels were recorded at the visit.",
        "The claim was denied due to a missing policy number.",
        "The customer applied for a credit card last month.",
    ]

    response = ollama.embed(model=s.embed_model, input=[question] + sentences)
    q_vec, *sent_vecs = response.embeddings

    print(f"Question: {question}\n")
    scores = [(cosine(q_vec, v), text) for v, text in zip(sent_vecs, sentences)]
    for score, text in sorted(scores, reverse=True):
        print(f"{score:.3f}  {text}")


if __name__ == "__main__":
    main()
