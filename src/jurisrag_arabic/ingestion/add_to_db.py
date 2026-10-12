import json
import logging

os = __import__("os")
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


def get_connection():
    """Establish connection to PostgreSQL database."""
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
    )


def init_db():
    """Create database tables and vector extension if they do not exist."""
    conn = get_connection()
    conn.autocommit = True
    cursor = conn.cursor()

    try:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadat_and_docs (
                id SERIAL PRIMARY KEY,
                article_number INTEGER NOT NULL,
                book TEXT,
                chapter TEXT,
                section TEXT,
                section_name TEXT,
                topic TEXT,
                text_ar TEXT,
                text_en TEXT,
                is_repealed BOOLEAN DEFAULT FALSE,
                source_page INTEGER
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embedding_vector (
                id SERIAL PRIMARY KEY,
                article_id INTEGER REFERENCES metadat_and_docs(id) ON DELETE CASCADE,
                language VARCHAR(10) NOT NULL, -- 'ar' or 'en'
                chunk_text TEXT NOT NULL,
                embedding vector(1024) NOT NULL -- bge-m3 dimension is 1024
            );
        """)
        logger.info(
            "Database tables 'metadat_and_docs' and 'embedding_vector' verified/created successfully."
        )

    except Exception:
        logger.exception("Failed to initialize database schema.")
        raise
    finally:
        cursor.close()
        conn.close()


def insert_processed_articles(articles: list[dict]):
    """Insert enriched article records and their chunk embeddings into PostgreSQL."""
    if not articles:
        logger.info("No articles to insert.")
        return

    conn = get_connection()
    cursor = conn.cursor()

    try:
        for article in articles:
            cursor.execute(
                """
                INSERT INTO metadat_and_docs (
                    article_number, book, chapter, section, section_name,
                    topic, text_ar, text_en, is_repealed, source_page
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """,
                (
                    article.get("article_number"),
                    article.get("book", ""),
                    article.get("chapter", ""),
                    article.get("section", ""),
                    article.get("section_name", ""),
                    article.get("topic", ""),
                    article.get("text_ar", ""),
                    article.get("text_en", ""),
                    article.get("is_repealed", False),
                    article.get("source_page"),
                ),
            )
            article_id = cursor.fetchone()[0]

            chunk_rows = []
            for language, field in (("ar", "chunks_ar"), ("en", "chunks_en")):
                for chunk in article.get(field, []):
                    # PostgreSQL's vector type accepts its textual literal format.
                    vector = "[" + ",".join(map(str, chunk["embedding"])) + "]"
                    chunk_rows.append((article_id, language, chunk["text"], vector))

            if chunk_rows:
                execute_values(
                    cursor,
                    """INSERT INTO embedding_vector
                       (article_id, language, chunk_text, embedding) VALUES %s""",
                    chunk_rows,
                    template="(%s, %s, %s, %s::vector)",
                )

        conn.commit()
        logger.info(
            "Inserted %d articles and their chunks into the database.", len(articles)
        )
    except Exception:
        conn.rollback()
        logger.exception("Database insertion failed; transaction rolled back.")
        raise
    finally:
        cursor.close()
        conn.close()


def insert_processed_json(json_path: Path):
    """Read enriched JSON and insert it using the in-memory database writer."""
    if not json_path.is_file():
        raise FileNotFoundError(f"Processed JSON file not found at: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    insert_processed_articles(articles)
