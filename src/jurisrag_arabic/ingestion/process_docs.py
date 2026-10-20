import json
import logging
import os
import re
from pathlib import Path

import pdfplumber
from dotenv import load_dotenv

from .add_to_db import init_db, insert_processed_articles
from .embedding import chunk_article, embedd_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
logger = logging.getLogger(__name__)


def _chunk_and_embed(article: dict, text_field: str) -> list[dict]:
    """Chunk and embed one language for an article."""
    article_number = article.get("article_number", "unknown")
    language = text_field.removeprefix("text_")
    try:
        chunks = chunk_article(article, text_field=text_field)
        if not chunks:
            return []

        logger.info(
            "Article %s (%s): created %d chunk(s)",
            article_number,
            language,
            len(chunks),
        )
        # for index, chunk in enumerate(chunks, start=1):
        #     logger.debug(
        #         "Article %s (%s): embedding chunk %d/%d (%d characters)",
        #         article_number, language, index, len(chunks), len(chunk["text"]),
        #     )
        #     chunk["embedding"] = embedd_text(chunk["text"])
        texts = [chunk["text"] for chunk in chunks]
        embeddings = embedd_text(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb

        logger.info("Article %s (%s): embeddings complete", article_number, language)
        return chunks
    except Exception:
        logger.exception(
            "Article %s (%s): chunking or embedding failed", article_number, language
        )
        raise


def enrich_articles_with_embeddings(articles: list[dict]) -> list[dict]:
    """Return article records with independently chunked Arabic and English text."""
    enriched = []
    logger.info("Starting chunking and embedding for %d articles", len(articles))
    article_number = "unknown"
    try:
        for index, article in enumerate(articles, start=1):
            article_number = article.get("article_number", "unknown")
            logger.info(
                "Processing article %d/%d (number %s)",
                index,
                len(articles),
                article_number,
            )
            # with ThreadPoolExecutor(max_workers=2) as executor:
            #     ar_future = executor.submit(_chunk_and_embed, article, "text_ar")
            #     en_future = executor.submit(_chunk_and_embed, article, "text_en")
            #     result = dict(article)
            # result["chunks_ar"] = ar_future.result()
            # result["chunks_en"] = en_future.result()
            chunks_ar = _chunk_and_embed(article, "text_ar")
            chunks_en = _chunk_and_embed(article, "text_en")

            result = dict(article)
            result["chunks_ar"] = chunks_ar
            result["chunks_en"] = chunks_en

            enriched.append(result)
    except Exception:
        logger.exception(
            "Embedding stopped while processing article %s", article_number
        )
        raise
    logger.info("Chunking and embedding complete for %d articles", len(enriched))
    return enriched


class Converter:
    """Converting PDF doc to json formate"""

    def __init__(self, doc, path):
        self.doc = doc
        self.path = path

    @staticmethod
    def add_missing_articles(articles: list) -> list:
        by_number = {article["article_number"]: article for article in articles}
        first_number = min(by_number)
        last_number = max(by_number)
        missing_numbers = [
            number
            for number in range(first_number, last_number + 1)
            if number not in by_number
        ]

        for number in missing_numbers:
            missing_article = {
                "article_number": number,
                "book": "",
                "chapter": "",
                "section": "",
                "section_name": "",
                "topic": "",
                "text_ar": "هذه المادة ملغاة",
                "text_en": "This article is repealed",
                "is_repealed": True,
                "source_page": None,
            }
            by_number[number] = missing_article

        logger.info(
            "Filled %d missing article numbers (%d–%d)",
            len(missing_numbers),
            first_number,
            last_number,
        )
        return [by_number[number] for number in sorted(by_number)]

    @staticmethod
    def fill_empty_article_text(articles: list) -> list:
        """Fill missing article text with language-specific fallback messages."""
        missing_ar = 0
        missing_en = 0
        for article in articles:
            text_ar = article.get("text_ar")
            if not text_ar or not text_ar.strip():
                article["text_ar"] = "هذه المادة فارغة"
                missing_ar += 1

            text_en = article.get("text_en")
            if not text_en or not text_en.strip():
                article["text_en"] = "this is an empty article"
                missing_en += 1

        logger.info(
            "Filled empty text for %d Arabic and %d English article(s)",
            missing_ar,
            missing_en,
        )
        return articles

    @staticmethod
    def convert_to_eastern_arabic_digits(num_str: str) -> str:
        """Converts Western digits (0-9) to Eastern Arabic digits (٠-٩)."""
        western_to_eastern = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
        return num_str.translate(western_to_eastern)

    @staticmethod
    def convert_to_western_digits(num_str: str) -> str:
        """Converts Eastern Arabic digits (٠-٩) to Western digits (0-9)."""
        eastern_to_western = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
        return num_str.translate(eastern_to_western)

    @staticmethod
    def extract_english_only(text: str) -> str:
        """Removes Arabic characters and extra spaces, returning only English metadata."""
        text_western = Converter.convert_to_western_digits(text)
        cleaned = re.sub(r"[\u0600-\u06FF]+", "", text_western)
        return " ".join(cleaned.split()).strip()

    @classmethod
    def fix_visual_arabic_word(cls, word: str) -> str:
        """
        Reverses character order for visual Arabic words (e.g., 'ةدام' -> 'مادة'),
        while preserving digit sequences and formatting.
        """
        if not word:
            return ""

        # Preserve standalone digits / punctuation direction
        if re.match(r"^[0-9\u0660-\u0669\.,\(\)\-]+$", word):
            return cls.convert_to_eastern_arabic_digits(word)

        # Reverse character sequence for visual Arabic words
        if re.search(r"[\u0600-\u06FF]", word):
            reversed_word = word[::-1]

            # Restore embedded multi-digit numbers to LTR reading order
            def restore_digits(match):
                return match.group(0)[::-1]

            corrected = re.sub(r"[\d\u0660-\u0669]+", restore_digits, reversed_word)
            return cls.convert_to_eastern_arabic_digits(corrected)

        return word

    @classmethod
    def process_arabic_line(cls, ar_words: list) -> str:
        """
        Sorts Arabic words from RIGHT to LEFT (descending x0 coordinate)
        to restore true Arabic sentence word order, then fixes letter sequence.
        """
        if not ar_words:
            return ""

        # Sort Right-to-Left (highest x0 coordinate to lowest)
        ar_words_rtl = sorted(ar_words, key=lambda w: w["x0"], reverse=True)

        fixed_words = [cls.fix_visual_arabic_word(w["text"]) for w in ar_words_rtl]
        return " ".join(fixed_words).strip()

    @classmethod
    def process_english_line(cls, en_words: list) -> str:
        """Sorts English words Left-to-Right (ascending x0 coordinate)."""
        if not en_words:
            return ""
        en_words_ltr = sorted(en_words, key=lambda w: w["x0"])
        raw_text = " ".join(w["text"] for w in en_words_ltr).strip()
        return cls.extract_english_only(raw_text)

    def extract_legal_document(self):
        logger.info("Starting extraction from %s", self.doc)

        current_book = ""
        current_chapter = ""
        current_section = ""
        current_section_name = ""
        current_topic = ""
        expecting_section_name = False

        extracted_articles = []
        current_article = None

        # Flexible Regex Patterns
        re_book = re.compile(r"^(BOOK|الكتاب)\b", re.IGNORECASE)
        re_chapter = re.compile(r"^(CHAPTER|الباب)\b", re.IGNORECASE)
        re_section = re.compile(r"^(SECTION|الفرع|القسم)\b", re.IGNORECASE)

        # Article header matches
        re_article_en = re.compile(r"\b(?:Article|ARTICLE)\s*\(?\s*(\d+)\s*\)?")
        re_article_ar = re.compile(
            r"\b(?:مادة|المادة|مادّة)\s*\(?\s*([\d\u0660-\u0669]+)\s*\)?"
        )

        # Topic index pattern (e.g., "1. ", "2. ", "1 -", "(1)")
        re_topic_number = re.compile(r"^\s*\d+[\.\s\-]", re.IGNORECASE)

        with pdfplumber.open(self.doc) as pdf:
            logger.info("Opened PDF with %d pages", len(pdf.pages))

            for page_num, page in enumerate(pdf.pages, start=1):
                words = page.extract_words(extra_attrs=["fontname"])
                if not words:
                    continue

                # Sort top-to-bottom
                words_sorted = sorted(words, key=lambda w: w["top"])

                # Group words into visual lines (3.5pt height tolerance)
                lines = []
                for w in words_sorted:
                    placed = False
                    for line in lines:
                        if abs(w["top"] - line[0]["top"]) < 3.5:
                            line.append(w)
                            placed = True
                            break
                    if not placed:
                        lines.append([w])

                for line_words in lines:
                    line_words_ltr = sorted(line_words, key=lambda w: w["x0"])
                    line_text_raw = " ".join(w["text"] for w in line_words_ltr).strip()

                    if not line_text_raw:
                        continue

                    is_bold = any(
                        "bold" in str(w.get("fontname", "")).lower() for w in line_words
                    )

                    # Separate language components
                    ar_word_objs = [
                        w
                        for w in line_words
                        if re.search(r"[\u0600-\u06FF]", w["text"])
                    ]
                    en_word_objs = [
                        w
                        for w in line_words
                        if not re.search(r"[\u0600-\u06FF]", w["text"])
                    ]

                    line_ar_str = self.process_arabic_line(ar_word_objs)
                    line_en_str = self.process_english_line(en_word_objs)

                    # 1. BOOK DETECTION (English Only)
                    if re_book.search(line_text_raw):
                        if current_article:
                            extracted_articles.append(current_article)
                            current_article = None
                        current_book = self.extract_english_only(line_text_raw)
                        current_chapter = ""
                        current_section = ""
                        current_section_name = ""
                        current_topic = ""
                        expecting_section_name = False
                        continue

                    # 2. CHAPTER DETECTION (English Only)
                    if re_chapter.search(line_text_raw):
                        if current_article:
                            extracted_articles.append(current_article)
                            current_article = None
                        current_chapter = self.extract_english_only(line_text_raw)
                        current_section = ""
                        current_section_name = ""
                        current_topic = ""
                        expecting_section_name = False
                        continue

                    # 3. SECTION DETECTION (English Only)
                    if re_section.search(line_text_raw):
                        if current_article:
                            extracted_articles.append(current_article)
                            current_article = None
                        current_section = self.extract_english_only(line_text_raw)
                        current_section_name = ""
                        current_topic = ""
                        expecting_section_name = True
                        continue

                    # 4. CAPTURE SECTION NAME OR TOPIC RIGHT AFTER SECTION
                    if expecting_section_name:
                        match_en = re_article_en.search(line_text_raw)
                        match_ar = re_article_ar.search(line_text_raw)
                        if not (match_en or match_ar):
                            if re_topic_number.match(
                                line_en_str
                            ) or re_topic_number.match(line_text_raw):
                                current_topic = self.extract_english_only(line_text_raw)
                            else:
                                current_section_name = self.extract_english_only(
                                    line_text_raw
                                )
                            expecting_section_name = False
                            continue
                        expecting_section_name = False

                    # 5. ARTICLE START DETECTION
                    match_en = re_article_en.search(line_text_raw)
                    match_ar = re_article_ar.search(line_text_raw)

                    if match_en or match_ar:
                        if current_article:
                            extracted_articles.append(current_article)

                        art_num_str = (
                            match_en.group(1) if match_en else match_ar.group(1)
                        )
                        art_num = int(
                            art_num_str.translate(
                                str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
                            )
                        )

                        current_article = {
                            "article_number": art_num,
                            "book": current_book,
                            "chapter": current_chapter,
                            "section": current_section,
                            "section_name": current_section_name,
                            "topic": current_topic,
                            "text_ar": line_ar_str,
                            "text_en": line_en_str,
                            "is_repealed": "ملغاة" in line_text_raw
                            or "repealed" in line_text_raw.lower(),
                            "source_page": page_num,
                        }
                        continue

                    # 6. TOPIC DETECTION
                    is_numbered_topic = bool(
                        re_topic_number.match(line_en_str)
                        or re_topic_number.match(line_text_raw)
                    )

                    if current_article is None or is_bold or is_numbered_topic:
                        if current_article:
                            extracted_articles.append(current_article)
                            current_article = None

                        topic_cand = self.extract_english_only(line_text_raw)
                        if topic_cand:
                            current_topic = topic_cand
                        continue

                    # 7. CONTINUATION TEXT APPENDING TO ACTIVE ARTICLE
                    if current_article is not None:
                        if line_ar_str:
                            current_article["text_ar"] = (
                                current_article["text_ar"] + " " + line_ar_str
                            ).strip()
                        if line_en_str:
                            current_article["text_en"] = (
                                current_article["text_en"] + " " + line_en_str
                            ).strip()

            if current_article:
                extracted_articles.append(current_article)

        # Deduplicate and sort articles
        unique_articles = {}
        for art in extracted_articles:
            num = art["article_number"]
            if num not in unique_articles:
                unique_articles[num] = art
            else:
                if art["text_ar"] and not unique_articles[num]["text_ar"]:
                    unique_articles[num]["text_ar"] = art["text_ar"]
                if art["text_en"] and not unique_articles[num]["text_en"]:
                    unique_articles[num]["text_en"] = art["text_en"]

        final_list = self.add_missing_articles(
            [unique_articles[k] for k in sorted(unique_articles.keys())]
        )
        logger.info(
            "Parsed %d unique articles; %d records after filling article number gaps",
            len(unique_articles),
            len(final_list),
        )
        final_list = self.fill_empty_article_text(final_list)

        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(final_list, f, ensure_ascii=False, indent=2)

        logger.info("Extraction complete: %d articles", len(final_list))
        return final_list


def ingest_documents() -> list[dict]:
    """Extract, embed, and store the configured legal PDF."""
    load_dotenv()
    pdf_filename = os.getenv("PDF_FILENAME", "legal_doc.pdf")
    pdf_output_filename = os.getenv("PDF_OUTPUT_FILENAME", "legal_doc.json")

    pdf_input = PROJECT_ROOT / "data" / "raw" / pdf_filename
    json_output = PROJECT_ROOT / "data" / "processed" / pdf_output_filename

    logger.info("Ingestion started for PDF %s", pdf_input)
    converter = Converter(doc=pdf_input, path=json_output)
    result = converter.extract_legal_document()
    result = enrich_articles_with_embeddings(result)
    logger.info("Initializing database")
    init_db()
    logger.info("Saving %d enriched articles to database", len(result))
    insert_processed_articles(result)
    logger.info("Ingestion completed successfully")
    return result


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        ingest_documents()
    except Exception:
        logger.exception("Ingestion failed")
        raise
