import logging
import re
import threading
from pathlib import Path

from llama_cpp import Llama

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "bge-m3" / "bge-m3-Q8_0.gguf"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = None
_model_lock = threading.Lock()


def _get_model():
    """Load the embedding model on first use so importing helpers needs no model file."""
    global model
    if model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        model = Llama(model_path=str(MODEL_PATH), embedding=True, verbose=False)
    return model


def chunk_article(
    article_dict: dict, text_field: str = "text_ar", max_words: int = 300
) -> list[dict]:
    """Split one language's article text into chunks, preserving its metadata."""
    article_number = article_dict["article_number"]
    text = article_dict.get(text_field, "").strip()
    if not text:
        return []

    words = text.split()
    if len(words) <= max_words:
        pieces = [text]
    else:
        # Keep numbered paragraphs together where possible, then cap oversized ones.
        paragraphs = [
            p.strip()
            for p in re.split(r"\n+|\s(?=\d+[\-–.]|\([\d\w]+\))", text)
            if p.strip()
        ]
        pieces = []
        for paragraph in paragraphs:
            paragraph_words = paragraph.split()
            pieces.extend(
                " ".join(paragraph_words[i : i + max_words])
                for i in range(0, len(paragraph_words), max_words)
            )

    return [
        {
            "chunk_id": f"art_{article_number}_{text_field}_{index + 1}",
            "text": piece,
            "article_number": article_number,
            "metadata": {
                key: article_dict.get(key)
                for key in (
                    "book",
                    "chapter",
                    "section",
                    "section_name",
                    "topic",
                    "is_repealed",
                    "source_page",
                )
            },
        }
        for index, piece in enumerate(pieces)
    ]


def embedd_text(texts: list[str]) -> list[float]:
    """
    Embedding texts
    """
    logger.info("Creating embeddings for %d text(s)", len(texts))
    if not texts:
        return []

    with _model_lock:
        result = _get_model().create_embedding(texts)

    return [item["embedding"] for item in result["data"]]
