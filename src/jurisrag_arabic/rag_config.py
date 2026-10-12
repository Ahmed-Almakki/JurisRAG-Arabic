"""Load and validate shared RAG settings from the project YAML file."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "rag.yaml"


def load_rag_config() -> dict:
    """Read RAG settings and validate chunking/retrieval defaults."""
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    if not isinstance(config, dict):
        raise TypeError(f"RAG config must contain a YAML mapping: {CONFIG_PATH}")

    for section, key in (("chunking", "chunk_size"), ("retrieval", "top_k")):
        section_values = config.get(section)
        if not isinstance(section_values, dict):
            raise TypeError(f"RAG config section '{section}' must be a mapping")
        value = section_values.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{section}.{key} must be a positive integer")

    return config
