import hashlib
import logging
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_CONFIG_PATH = PROJECT_ROOT / "config" / "rag.yaml"
logger = logging.getLogger(__name__)


def load_model_config() -> dict:
    """Read and validate model download settings directly from rag.yaml."""
    logger.debug("Loading model download settings from %s", RAG_CONFIG_PATH)
    with RAG_CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    if not isinstance(config, dict):
        raise TypeError(f"RAG config must contain a YAML mapping: {RAG_CONFIG_PATH}")

    model_config = config.get("embedding_model")
    if not isinstance(model_config, dict):
        raise TypeError("rag.yaml must contain an 'embedding_model' mapping")

    for key in ("repo_id", "model_name", "revision", "sha256"):
        if not isinstance(model_config.get(key), str) or not model_config[key].strip():
            raise ValueError(f"embedding_model.{key} must be a non-empty string")

    logger.info(
        "Loaded model configuration for %s from repository %s",
        model_config["model_name"],
        model_config["repo_id"],
    )
    return model_config


def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 checksum without loading the whole file into RAM."""
    logger.debug("Calculating SHA-256 checksum for %s", file_path)
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def download_model() -> Path:
    """Download the configured model unless a verified copy already exists."""
    model_config = load_model_config()
    model_dir = PROJECT_ROOT / "models" / "bge-m3"
    model_file = model_dir / model_config["model_name"]
    model_dir.mkdir(parents=True, exist_ok=True)

    if model_file.is_file():
        logger.info(
            "Found existing model file at %s; checking its checksum", model_file
        )
        actual_sha256 = calculate_sha256(model_file)
        if actual_sha256 == model_config["sha256"]:
            logger.info("Existing model checksum is valid; skipping download")
            return model_file

        logger.warning(
            "Existing model checksum mismatch (expected %s, received %s); "
            "removing it before downloading a verified copy",
            model_config["sha256"],
            actual_sha256,
        )
        model_file.unlink()
    else:
        logger.info("Model file does not exist at %s", model_file)

    logger.info(
        "Downloading %s from %s at revision %s",
        model_config["model_name"],
        model_config["repo_id"],
        model_config["revision"],
    )
    model_path = hf_hub_download(
        repo_id=model_config["repo_id"],
        filename=model_config["model_name"],
        revision=model_config["revision"],
        local_dir=model_dir,
    )
    downloaded_file = Path(model_path)
    actual_sha256 = calculate_sha256(downloaded_file)

    if actual_sha256 != model_config["sha256"]:
        downloaded_file.unlink(missing_ok=True)
        logger.error(
            "Downloaded model checksum mismatch (expected %s, received %s)",
            model_config["sha256"],
            actual_sha256,
        )
        raise RuntimeError(
            "Model checksum verification failed. "
            f"Expected: {model_config['sha256']}, received: {actual_sha256}"
        )

    logger.info("Model downloaded and checksum verified: %s", downloaded_file)
    return downloaded_file


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        download_model()
    except Exception:
        logger.exception("Model download failed")
        raise
