"""Integration test for ingestion, PostgreSQL retrieval, and RAG answering."""

import asyncio
import os
import uuid

import pytest
from psycopg2 import OperationalError

from jurisrag_arabic.ingestion import process_docs
from jurisrag_arabic.ingestion.add_to_db import (
    get_connection,
    init_db,
    insert_processed_articles,
)
from jurisrag_arabic.rag import RAG, QueryRequest
from jurisrag_arabic.retrival.retrive import Retriever

pytestmark = pytest.mark.integration


@pytest.fixture
def postgres_connection():
    if not os.getenv("DB_HOST"):
        pytest.skip(
            "Set DB_HOST and the DB_* variables to run system integration tests"
        )

    try:
        init_db()
    except OperationalError:
        if os.getenv("CI"):
            raise
        pytest.skip("PostgreSQL is not available for integration tests")

    connection = get_connection()
    try:
        yield connection
    finally:
        connection.close()


def _test_embedding(text: str) -> list[float]:
    """Return a 1024-dimensional vector that makes the marker searchable."""
    vector = [0.0] * 1024
    vector[0 if "jurisrag-integration" in text else 1] = 1.0
    return vector


def test_ingestion_retrieval_and_rag_flow(postgres_connection, monkeypatch):
    marker = f"jurisrag-integration-{uuid.uuid4()}"
    full_english_text = f"{marker} " + "legal words " * 150
    article = {
        "article_number": 987654,
        "book": "Integration test book",
        "chapter": "Integration test chapter",
        "section": "Integration test section",
        "section_name": "Integration test section name",
        "topic": "Integration test topic",
        "text_ar": "نص قانوني لاختبار التكامل",
        "text_en": full_english_text,
        "is_repealed": False,
        "source_page": 1,
    }

    # Exercise the real chunking and ingestion orchestration without loading the
    # local model. Only vector generation is replaced with a deterministic stub.
    monkeypatch.setattr(
        process_docs,
        "embedd_text",
        lambda texts: [_test_embedding(text) for text in texts],
    )
    enriched_articles = process_docs.enrich_articles_with_embeddings([article])

    class FakeLLM:
        def __init__(self):
            self.messages = None

        async def astream(self, messages):
            self.messages = messages
            yield type("LLMChunk", (), {"content": "Answer based on "})()
            yield type("LLMChunk", (), {"content": "article 987654."})()

    fake_llm = FakeLLM()

    try:
        assert len(enriched_articles[0]["chunks_en"]) == 2
        insert_processed_articles(enriched_articles)

        retriever = Retriever(
            db_connection=postgres_connection,
            embedding_function=lambda _texts: [_test_embedding(marker)],
        )
        retrieved = retriever.retrieve(marker, language="en", top_k=1)

        assert len(retrieved) == 1
        assert retrieved[0]["text"].startswith(marker)
        assert retrieved[0]["text_en"] == full_english_text
        assert retrieved[0]["article_number"] == 987654
        assert retrieved[0]["book"] == "Integration test book"

        # Call the service's actual ask implementation with a stub LLM. The
        # retriever above remains real and uses PostgreSQL/pgvector.
        service_class = RAG.inner
        service = service_class.__new__(service_class)
        service.retriever = retriever
        service.llm = fake_llm

        async def collect_answer():
            return "".join(
                [
                    chunk
                    async for chunk in service_class.ask.func(
                        service, QueryRequest(query=marker)
                    )
                ]
            )

        answer = asyncio.run(collect_answer())

        assert answer == "Answer based on article 987654."
        assert fake_llm.messages is not None
        assert marker in fake_llm.messages[1].content
        assert "987654" in fake_llm.messages[1].content
        assert "Integration test book" in fake_llm.messages[1].content
    finally:
        with postgres_connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM metadat_and_docs WHERE text_en = %s", (full_english_text,)
            )
        postgres_connection.commit()
