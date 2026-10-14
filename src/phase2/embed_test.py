"""Phase 2: first embedding request to Ollama."""

import ollama

from src.common.config import get_settings


def main():
    s = get_settings()
    text = "Patient was diagnosed with type 2 diabetes."

    response = ollama.embed(model=s.embed_model, input=text)
    vector = response.embeddings[0]

    print(f"Model      : {s.embed_model}")
    print(f"Text       : {text}")
    print(f"Dimensions : {len(vector)}")
    print(f"First 5    : {[round(v, 4) for v in vector[:5]]}")


if __name__ == "__main__":
    main()
