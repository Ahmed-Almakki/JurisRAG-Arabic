from pathlib import Path

from llama_cpp import Llama

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "bge-m3" / "bge-m3-Q8_0.gguf"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = Llama(
    model_path=str(MODEL_PATH),
    embedding=True,
    verbose=False,
)

texts = [
    "What are the conditions of a contract?",
    "ما هي شروط العقد؟",
]

for text in texts:
    result = model.create_embedding(text)

    embedding = result["data"][0]["embedding"]

    print(f"\nText: {text}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First five values: {embedding[:5]}")

assert len(embedding) == 1024, f"Expected 1024 dimensions, got {len(embedding)}"
