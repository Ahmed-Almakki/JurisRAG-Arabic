import json

import pytest

from jurisrag_arabic.ingestion import add_to_db


class FakeCursor:
    def __init__(self):
        self.executed = []
        self.closed = False

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchone(self):
        return (42,)

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self):
        self.fake_cursor = FakeCursor()
        self.committed = False
        self.rolled_back = False
        self.closed = False
        self.autocommit = False

    def cursor(self):
        return self.fake_cursor

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def test_insert_processed_articles_inserts_article_and_bilingual_chunks(monkeypatch):
    connection = FakeConnection()
    inserted_chunks = []
    monkeypatch.setattr(add_to_db, "get_connection", lambda: connection)
    monkeypatch.setattr(
        add_to_db,
        "execute_values",
        lambda cursor, query, rows, template: inserted_chunks.extend(rows),
    )
    article = {
        "article_number": 9,
        "text_ar": "عربي",
        "text_en": "English",
        "chunks_ar": [{"text": "عربي", "embedding": [0.1, 0.2]}],
        "chunks_en": [{"text": "English", "embedding": [0.3, 0.4]}],
    }

    add_to_db.insert_processed_articles([article])

    assert connection.committed is True
    assert connection.rolled_back is False
    assert inserted_chunks == [
        (42, "ar", "عربي", "[0.1,0.2]"),
        (42, "en", "English", "[0.3,0.4]"),
    ]
    assert connection.closed is True
    assert connection.fake_cursor.closed is True


def test_insert_processed_articles_rolls_back_on_database_error(monkeypatch):
    connection = FakeConnection()
    monkeypatch.setattr(add_to_db, "get_connection", lambda: connection)

    def fail_insert(*args, **kwargs):
        raise RuntimeError("insert failed")

    monkeypatch.setattr(add_to_db, "execute_values", fail_insert)
    article = {
        "article_number": 1,
        "chunks_ar": [{"text": "text", "embedding": [0.1]}],
    }

    with pytest.raises(RuntimeError, match="insert failed"):
        add_to_db.insert_processed_articles([article])

    assert connection.rolled_back is True
    assert connection.committed is False
    assert connection.closed is True


def test_insert_processed_articles_empty_input_skips_connection(monkeypatch):
    monkeypatch.setattr(
        add_to_db,
        "get_connection",
        lambda: pytest.fail("Should not connect for empty input"),
    )

    add_to_db.insert_processed_articles([])


def test_insert_processed_json_reads_and_delegates(tmp_path, monkeypatch):
    path = tmp_path / "articles.json"
    articles = [{"article_number": 12}]
    path.write_text(json.dumps(articles), encoding="utf-8")
    received = []
    monkeypatch.setattr(add_to_db, "insert_processed_articles", received.append)

    add_to_db.insert_processed_json(path)

    assert received == [articles]


def test_insert_processed_json_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        add_to_db.insert_processed_json(tmp_path / "missing.json")
