from pathlib import Path

from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models" / "bge-m3"
MODEL_FILE = MODEL_DIR / "bge-m3-Q8_0.gguf"

if MODEL_FILE.exists():
    print("Model already exists. Skipping download.")
else:
    print("Downloading model from Hugging Face Hub...")

    # Create the directory and its parent directories if necessary.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = hf_hub_download(
        repo_id="gpustack/bge-m3-GGUF",
        filename="bge-m3-Q8_0.gguf",
        local_dir=MODEL_DIR,
    )

    print(f"Downloaded model to: {model_path}")
