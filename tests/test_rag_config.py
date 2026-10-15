from jurisrag_arabic import rag_config


def test_load_rag_config_returns_every_setting_and_expands_env(monkeypatch, tmp_path):
    config_path = tmp_path / "rag.yaml"
    config_path.write_text(
        """
embedding:
  chunk_size: 240
retrieval:
  top_k: 7
llm:
  model_name: example/model
  temperature: 0.4
  api_key: ${TEST_RAG_API_KEY}
system_prompt: "Answer from the supplied sources."
user_message_template: |
  Sources: {context_block}
  Question: {query}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(rag_config, "CONFIG_PATH", config_path)
    monkeypatch.setattr(rag_config, "load_dotenv", lambda: None)
    monkeypatch.setenv("TEST_RAG_API_KEY", "test-secret")

    config = rag_config.load_rag_config()

    assert config == {
        "embedding": {"chunk_size": 240},
        "retrieval": {"top_k": 7},
        "llm": {
            "model_name": "example/model",
            "temperature": 0.4,
            "api_key": "test-secret",
        },
        "system_prompt": "Answer from the supplied sources.",
        "user_message_template": "Sources: {context_block}\nQuestion: {query}\n",
    }
