import json

from jurisrag_arabic.rag import format_context_block


def test_format_context_block_returns_bilingual_text_and_citation_dicts():
    sources = [
        {
            "article_number": 12,
            "book": "BOOK I",
            "chapter": "CHAPTER II",
            "section": "SECTION III",
            "text_ar": "النص العربي",
            "text_en": "English text",
        }
    ]

    context = format_context_block(sources)

    assert context == [
        {
            "text": {"text_ar": "النص العربي", "text_en": "English text"},
            "citation": {
                "article_number": 12,
                "chapter": "CHAPTER II",
                "book": "BOOK I",
                "section": "SECTION III",
            },
        }
    ]
    assert "النص العربي" in json.dumps(context, ensure_ascii=False)


def test_format_context_block_returns_empty_list_without_sources():
    assert format_context_block([]) == []
