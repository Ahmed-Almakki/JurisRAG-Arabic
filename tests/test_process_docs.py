from jurisrag_arabic.ingestion import process_docs


def test_chunk_and_embed_handles_both_language_fields(monkeypatch):
    calls = []

    def fake_embed(texts):
        calls.append(texts)
        return [[float(index)] for index, _ in enumerate(texts)]

    monkeypatch.setattr(process_docs, "embedd_text", fake_embed)
    article = {
        "article_number": 4,
        "text_ar": "نص عربي",
        "text_en": "English text",
        "topic": "Topic",
    }

    enriched = process_docs.enrich_articles_with_embeddings([article])[0]

    assert enriched["topic"] == "Topic"
    assert enriched["chunks_ar"][0]["text"] == "نص عربي"
    assert enriched["chunks_en"][0]["text"] == "English text"
    assert enriched["chunks_ar"][0]["embedding"] == [0.0]
    assert enriched["chunks_en"][0]["embedding"] == [0.0]
    assert calls == [["نص عربي"], ["English text"]]
    assert "chunks_ar" not in article


def test_enrich_articles_keeps_empty_language_without_embedding(monkeypatch):
    calls = []

    def fake_embed(texts):
        calls.append(texts)
        return [[0.5] for _ in texts]

    monkeypatch.setattr(process_docs, "embedd_text", fake_embed)
    article = {"article_number": 1, "text_ar": "عربي", "text_en": ""}

    enriched = process_docs.enrich_articles_with_embeddings([article])[0]

    assert len(enriched["chunks_ar"]) == 1
    assert enriched["chunks_ar"][0]["embedding"] == [0.5]
    assert enriched["chunks_en"] == []
    assert calls == [["عربي"]]


def test_add_missing_articles_fills_number_gaps():
    articles = [
        {"article_number": 1, "text_ar": "one", "text_en": "one"},
        {"article_number": 3, "text_ar": "three", "text_en": "three"},
    ]

    result = process_docs.Converter.add_missing_articles(articles)

    assert [article["article_number"] for article in result] == [1, 2, 3]
    assert result[1]["is_repealed"] is True
    assert result[1]["source_page"] is None


def test_fill_empty_article_text_fills_both_languages():
    articles = [{"article_number": 8, "text_ar": " ", "text_en": ""}]

    result = process_docs.Converter.fill_empty_article_text(articles)

    assert result[0]["text_ar"] == "هذه المادة فارغة"
    assert result[0]["text_en"] == "this is an empty article"
