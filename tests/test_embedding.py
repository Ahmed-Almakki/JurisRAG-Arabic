from jurisrag_arabic.ingestion import embedding


def test_chunk_article_keeps_short_text_and_metadata():
    article = {
        "article_number": 7,
        "text_ar": "نص المادة",
        "book": "BOOK I",
        "section": "SECTION II",
        "source_page": 3,
    }

    chunks = embedding.chunk_article(article)

    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "art_7_text_ar_1"
    assert chunks[0]["text"] == "نص المادة"
    assert chunks[0]["article_number"] == 7
    assert chunks[0]["metadata"]["book"] == "BOOK I"
    assert chunks[0]["metadata"]["source_page"] == 3


def test_chunk_article_splits_long_text_and_supports_english_field():
    article = {"article_number": 2, "text_en": "one two three four five"}

    chunks = embedding.chunk_article(article, text_field="text_en", max_words=2)

    assert [chunk["text"] for chunk in chunks] == ["one two", "three four", "five"]
    assert [chunk["chunk_id"] for chunk in chunks] == [
        "art_2_text_en_1",
        "art_2_text_en_2",
        "art_2_text_en_3",
    ]


def test_chunk_article_empty_text_returns_no_chunks():
    assert embedding.chunk_article({"article_number": 1, "text_ar": "  "}) == []


def test_embedd_text_batches_inputs_and_returns_vectors(monkeypatch):
    class FakeModel:
        def create_embedding(self, texts):
            assert texts == ["first", "second"]
            return {"data": [{"embedding": [1.0]}, {"embedding": [2.0]}]}

    monkeypatch.setattr(embedding, "model", FakeModel())

    assert embedding.embedd_text(["first", "second"]) == [[1.0], [2.0]]


def test_embedd_text_empty_input_does_not_load_model(monkeypatch):
    def unexpected_model_load():
        raise AssertionError("Model should not be loaded for empty input")

    monkeypatch.setattr(embedding, "_get_model", unexpected_model_load)

    assert embedding.embedd_text([]) == []
