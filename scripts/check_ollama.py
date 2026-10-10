import os, ollama
from dotenv import load_dotenv

load_dotenv()
chat_model  = os.getenv("CHAT_MODEL", "gpt-oss:20b-cloud")
embed_model = os.getenv("EMBED_MODEL", "nomic-embed-text")

reply = ollama.chat(model=chat_model,
                    messages=[{"role": "user", "content": "In one line, what is RAG?"}])
print(f"CHAT ({chat_model}):", reply["message"]["content"])

emb = ollama.embed(model=embed_model, input="adverse drug event")
print(f"EMBEDDING ({embed_model}) LENGTH:", len(emb["embeddings"][0]))
