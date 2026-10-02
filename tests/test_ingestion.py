import json

import pytest

from jurisrag_arabic.ingestion.process_docs import PROJECT_ROOT, Converter

# ==============================================================================
# 1. UNIT TESTS: Transformation Functions
# ==============================================================================


def test_convert_to_eastern_arabic_digits():
    """Verify Western digits (0-9) correctly map to Eastern Arabic digits (٠-٩)."""
    assert Converter.convert_to_eastern_arabic_digits("1234567890") == "١٢٣٤٥٦٧٨٩٠"
    assert Converter.convert_to_eastern_arabic_digits("Article 147") == "Article ١٤٧"


def test_extract_english_only():
    """Verify Arabic characters are stripped while leaving English text intact."""
    raw_input = "BOOK ONE الكتاب الأول - Obligations"
    expected = "BOOK ONE - Obligations"
    assert Converter.extract_english_only(raw_input) == expected


def test_fix_visual_arabic_word():
    """Verify visual character reversal logic for Arabic words and embedded digits."""
    # Reverse of 'ةدام' should be 'مادة'
    assert Converter.fix_visual_arabic_word("ةدام") == "مادة"
    # Pure numbers should remain intact and convert to Eastern digits
    assert Converter.fix_visual_arabic_word("147") == "١٤٧"


def test_process_arabic_line_ordering():
    """Verify words sorted by descending x0 coordinate construct Right-To-Left sequence."""
    ar_word_objs = [
        {"text": "بل", "x0": 100.0},
        {"text": "يجب", "x0": 200.0},
        {"text": "أن", "x0": 300.0},
    ]
    # Highest x0 (300.0 -> 'أن') comes first in Arabic Right-To-Left sentence flow
    processed_line = Converter.process_arabic_line(ar_word_objs)
    words = processed_line.split()
    assert words[0] == Converter.fix_visual_arabic_word("أن")


# ==============================================================================
# 2. DATA QUALITY & INTEGRATION TESTS: Extracted JSON Corpus
# ==============================================================================


@pytest.fixture
def corpus_data():
    """Fixture to load extracted JSON file."""
    json_path = PROJECT_ROOT / "data" / "processed" / "legal_doc.json"
    assert json_path.exists(), (
        f"Corpus output file not found at '{json_path}'. Run 'dvc repro' first."
    )

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_corpus_not_empty(corpus_data):
    """Verify the extracted corpus contains records."""
    assert len(corpus_data) > 0, "Validation Error: The extracted JSON corpus is empty!"


def test_schema_integrity(corpus_data):
    """Verify every record contains all required fields from the target schema."""
    required_keys = {
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
    }
    for record in corpus_data:
        missing_keys = required_keys - set(record.keys())
        assert not missing_keys, (
            f"Article {record.get('article_number')} is missing keys: {missing_keys}"
        )


def test_text_ar_not_empty(corpus_data):
    """Verify no record has an empty Arabic text field."""
    for record in corpus_data:
        art_num = record["article_number"]
        assert record["text_ar"] and record["text_ar"].strip(), (
            f"Validation Error: Article {art_num} has empty 'text_ar'!"
        )


def test_sane_length(corpus_data):
    """Detect boundary splitting failures where multiple articles were merged."""
    MAX_CHAR_LIMIT = 5000  # Sane upper bound for an individual civil code article
    for record in corpus_data:
        art_num = record["article_number"]
        char_length = len(record["text_ar"])
        assert char_length < MAX_CHAR_LIMIT, (
            f"Validation Error: Article {art_num} text length ({char_length} chars) "
            f"exceeds upper threshold. Possible header detection split failure."
        )


def test_article_contiguity_and_sorting(corpus_data):
    """Verify article numbers are sorted integers and evaluate sequence gaps."""
    article_numbers = [record["article_number"] for record in corpus_data]

    # Must be sorted in ascending order
    assert article_numbers == sorted(article_numbers), (
        "Articles are not properly sorted by article_number!"
    )

    # Evaluate sequence contiguity
    min_art, max_art = min(article_numbers), max(article_numbers)
    expected_set = set(range(min_art, max_art + 1))
    actual_set = set(article_numbers)
    missing = sorted(expected_set - actual_set)

    # Log any sequence gaps for analysis
    if missing:
        print(
            f"\n[INFO] Detected {len(missing)} non-contiguous gaps in sequence: {missing[:10]}..."
        )


def test_repealed_flag_detection(corpus_data):
    """Verify that repealed articles (such as Articles 54–80) exist and have 'is_repealed: True'."""
    repealed_articles = [r for r in corpus_data if r["is_repealed"]]

    assert len(repealed_articles) > 0, (
        "Validation Error: No repealed articles detected. Check regex patterns for 'ملغاة' or 'repealed'."
    )

    repealed_nums = {r["article_number"] for r in repealed_articles}
    # Check known Egyptian Civil Code repealed range (Articles 54 to 80)
    known_repealed_subset = set(range(54, 81))
    found_intersection = known_repealed_subset.intersection(repealed_nums)

    assert len(found_intersection) > 0, (
        "Validation Error: Known repealed range (54-80) missing 'is_repealed: True' flag."
    )
