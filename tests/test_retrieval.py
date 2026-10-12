import pytest

from jurisrag_arabic.retrival import retrive


class FakeCursor:
    def __init__(self, rows=(), columns=()):
        self.rows = rows
        self.description = [(column,) for column in columns]
        self.query = None
        self.params = None
        self.closed = False

    def execute(self, query, params):
        self.query = query
        self.params = params

    def fetchall(self):
        return self.rows

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, cursor):
        self.fake_cursor = cursor

    def cursor(self):
        return self.fake_cursor


def test_retrieve_returns_ranked_chunks_joined_with_article_metadata(monkeypatch):
    columns = [
        "chunk_id",
        "language",
        "chunk_text",
        "article_id",
        "article_number",
        "book",
        "chapter",
        "section",
        "section_name",
        "topic",
        "text_ar",
        "text_en",
        "is_repealed",
        "source_page",
        "distance",
    ]
    row = (
        11,
        "ar",
        "النص القانوني",
        3,
        17,
        "BOOK I",
        "CHAPTER II",
        "SECTION A",
        "General rules",
        "Topic",
        "النص القانوني الكامل",
        "Full legal text",
        False,
        4,
        0.15,
    )
    cursor = FakeCursor(rows=[row], columns=columns)
    connection = FakeConnection(cursor)
    monkeypatch.setattr(retrive, "load_rag_config", lambda: {"retrieval": {"top_k": 5}})
    retriever = retrive.Retriever(
        db_connection=connection,
        embedding_function=lambda texts: [[0.1, 0.2, 0.3]],
    )

    results = retriever.retrieve("query", top_k=2, language="ar")

    assert results == [
        {
            "chunk_id": 11,
            "language": "ar",
            "article_id": 3,
            "article_number": 17,
            "book": "BOOK I",
            "chapter": "CHAPTER II",
            "section": "SECTION A",
            "section_name": "General rules",
            "topic": "Topic",
            "text_ar": "النص القانوني الكامل",
            "text_en": "Full legal text",
            "is_repealed": False,
            "source_page": 4,
            "distance": 0.15,
            "text": "النص القانوني",
            "similarity": 0.85,
        }
    ]
    assert "FROM embedding_vector AS ev" in cursor.query
    assert "INNER JOIN metadat_and_docs AS md ON md.id = ev.article_id" in cursor.query
    assert "ORDER BY ev.embedding <=> %s::vector" in cursor.query
    assert cursor.params == ("[0.1,0.2,0.3]", "ar", "ar", "[0.1,0.2,0.3]", 2)
    assert cursor.closed is True


def test_retrieve_uses_configured_top_k_when_not_provided(monkeypatch):
    cursor = FakeCursor(columns=["chunk_id"])
    retriever = retrive.Retriever(
        db_connection=FakeConnection(cursor),
        embedding_function=lambda texts: [[0.5]],
    )
    monkeypatch.setattr(retrive, "load_rag_config", lambda: {"retrieval": {"top_k": 8}})

    assert retriever.retrieve("query") == []
    assert cursor.params[-1] == 8
    assert cursor.closed is True


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_retrieve_rejects_invalid_top_k(top_k):
    retriever = retrive.Retriever(
        db_connection=FakeConnection(FakeCursor()),
        embedding_function=lambda texts: [[0.5]],
    )

    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("query", top_k=top_k)


@pytest.mark.parametrize("language", ["fr", "Arabic", ""])
def test_retrieve_rejects_unsupported_language(language):
    retriever = retrive.Retriever(
        db_connection=FakeConnection(FakeCursor()),
        embedding_function=lambda texts: [[0.5]],
    )

    with pytest.raises(ValueError, match="language"):
        retriever.retrieve("query", language=language)


def test_retrieve_returns_empty_when_query_embedding_is_empty():
    cursor = FakeCursor()
    retriever = retrive.Retriever(
        db_connection=FakeConnection(cursor), embedding_function=lambda texts: []
    )

    assert retriever.retrieve("query") == []
    assert cursor.query is None
    assert cursor.closed is False
