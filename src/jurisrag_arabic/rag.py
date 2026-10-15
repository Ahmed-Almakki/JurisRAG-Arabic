import json
import logging
from typing import Any

import bentoml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, field_validator

from .rag_config import load_rag_config
from .retrival.retrive import Retriever

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("jusrig.rag")

config = load_rag_config()


class BilingualText(BaseModel):
    text_ar: str
    text_en: str


class CitationDetails(BaseModel):
    article_number: int | None = None
    chapter: str | None = None
    book: str | None = None
    section: str | None = None


class ContextBlock(BaseModel):
    """One retrieved source: bilingual text plus its article citation."""

    text: BilingualText
    citation: CitationDetails


def format_context_block(
    retrieved_docs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Format each retrieved source as bilingual text and citation dictionaries."""
    if not retrieved_docs:
        logger.warning("Retriever returned 0 documents for the given query.")
        return []

    logger.info(
        "Formatting %d retrieved documents into context block.", len(retrieved_docs)
    )
    return [
        ContextBlock(
            text=BilingualText(
                text_ar=doc.get("text_ar") or "",
                text_en=doc.get("text_en") or "",
            ),
            citation=CitationDetails(
                article_number=doc.get("article_number"),
                chapter=doc.get("chapter"),
                book=doc.get("book"),
                section=doc.get("section"),
            ),
        ).model_dump()
        for doc in retrieved_docs
    ]


class QueryRequest(BaseModel):
    """Validated request schema for RAG retrieval."""

    query: str = Field(strict=True, min_length=1, description="The user's search query")

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


@bentoml.service
class RAG:
    """BentoML service returning relevant legal passages and article metadata."""

    def __init__(self) -> None:
        logger.info("Initializing RAG service and connecting to Retriever...")
        self.retriever = Retriever()
        self.llm = ChatGroq(
            model_name=config["llm"]["model_name"],
            temperature=config["llm"]["temperature"],
            api_key=config["llm"]["api_key"],
        )

    @bentoml.api
    def ask(self, request: QueryRequest, /) -> str:
        """Retrieve relevant chunks, format prompt, invoke LLM, and return raw text string."""
        logger.info(f"Received incoming query: '{request.query}'")

        # 1. Retrieve top-k articles
        retrieved_docs = self.retriever.retrieve(request.query)

        # 2. Format articles into context string
        context_block = json.dumps(
            format_context_block(retrieved_docs), ensure_ascii=False, indent=2
        )
        print("context_block", context_block)
        # 3. Format user message template
        user_content = config["user_message_template"].format(
            context_block=context_block, query=request.query
        )

        messages = [
            SystemMessage(content=config["system_prompt"]),
            HumanMessage(content=user_content),
        ]
        print("messages", messages)
        # 4. Invoke LLM and log completion
        logger.info("Invoking LLM via Groq API...")
        response = self.llm.invoke(messages)
        print("response", response)
        logger.info("LLM generation complete. Returning raw string response.")

        return response.content
