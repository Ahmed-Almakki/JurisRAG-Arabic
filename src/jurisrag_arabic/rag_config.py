"""Load and validate settings from the project's RAG YAML configuration."""

import os
import re
from pathlib import Path
from string import Formatter

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "rag.yaml"
_ENV_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_environment_variables(value):
    """Expand ${NAME} placeholders recursively in YAML string values."""
    if isinstance(value, dict):
        return {key: _expand_environment_variables(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment_variables(item) for item in value]
    if isinstance(value, str):
        return _ENV_VARIABLE.sub(lambda match: os.getenv(match.group(1), ""), value)
    return value


def load_rag_config() -> dict:
    """Read, validate, and environment-expand all RAG settings."""
    load_dotenv()
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    if not isinstance(config, dict):
        raise TypeError(f"RAG config must contain a YAML mapping: {CONFIG_PATH}")

    config = _expand_environment_variables(config)
    for section, key in (("embedding", "chunk_size"), ("retrieval", "top_k")):
        section_values = config.get(section)
        if not isinstance(section_values, dict):
            raise TypeError(f"RAG config section '{section}' must be a mapping")
        value = section_values.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{section}.{key} must be a positive integer")

    llm = config.get("llm")
    if not isinstance(llm, dict):
        raise TypeError("RAG config section 'llm' must be a mapping")
    if not isinstance(llm.get("model_name"), str) or not llm["model_name"].strip():
        raise ValueError("llm.model_name must be a non-empty string")
    temperature = llm.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("llm.temperature must be a number")
    if temperature < 0:
        raise ValueError("llm.temperature must be non-negative")
    if not isinstance(llm.get("api_key"), str):
        raise TypeError("llm.api_key must be a string or environment placeholder")

    system_prompt = config.get("system_prompt")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("system_prompt must be a non-empty string")

    user_template = config.get("user_message_template")
    if not isinstance(user_template, str) or not user_template.strip():
        raise ValueError("user_message_template must be a non-empty string")
    try:
        template_fields = {
            field for _, field, _, _ in Formatter().parse(user_template) if field
        }
    except ValueError as error:
        raise ValueError(
            "user_message_template contains invalid format syntax"
        ) from error
    required_fields = {"context_block", "query"}
    if not required_fields.issubset(template_fields):
        raise ValueError(
            "user_message_template must include {context_block} and {query}"
        )

    return config
