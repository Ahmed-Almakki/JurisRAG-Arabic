from ..ingestion.add_to_db import get_connection
from ..ingestion.embedding import embedd_text


class Retriever:
    """Retrieve the nearest embedded chunks with their article metadata."""

    def __init__(self, db_connection=None, embedding_function=embedd_text):
        self.db_connection = (
            db_connection if db_connection is not None else get_connection()
        )
        self.embedding_function = embedding_function

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        language: str | None = None,
    ) -> list[dict]:
        """Return nearest chunks joined to their source article records.

        ``language`` can be ``"ar"`` or ``"en"`` to limit retrieval to one
        language. The pgvector cosine distance is returned as ``distance`` and
        its inverse ranking score as ``similarity``.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if language not in (None, "ar", "en"):
            raise ValueError("language must be 'ar', 'en', or None")

        embeddings = self.embedding_function([query])
        if not embeddings:
            return []
        query_vector = "[" + ",".join(map(str, embeddings[0])) + "]"

        cursor = self.db_connection.cursor()
        try:
            cursor.execute(
                """
                SELECT
                    ev.id AS chunk_id, ev.language, ev.chunk_text, md.id AS article_id,
                    md.article_number, md.book, md.chapter, md.section, md.section_name,
                    md.topic, md.text_ar, md.text_en, md.is_repealed, md.source_page,
                    ev.embedding <=> %s::vector AS distance
                FROM embedding_vector AS ev
                INNER JOIN metadat_and_docs AS md ON md.id = ev.article_id
                WHERE (%s IS NULL OR ev.language = %s)
                ORDER BY ev.embedding <=> %s::vector
                LIMIT %s;
                """,
                (query_vector, language, language, query_vector, top_k),
            )
            rows = cursor.fetchall()
            columns = [column[0] for column in cursor.description]
            results = []
            for row in rows:
                result = dict(zip(columns, row))
                result["text"] = result.pop("chunk_text")
                result["similarity"] = 1 - result["distance"]
                results.append(result)
            return results
        finally:
            cursor.close()
