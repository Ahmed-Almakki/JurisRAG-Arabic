# JurisRAG Arabic

JurisRAG Arabic is a question-answering project for legal documents in Arabic and English. It finds relevant legal articles and uses them as sources for answers. Answers are generated from the retrieved sources and should include article citations.

## How it works

The project has three main steps:

1. **Read legal documents.** The ingestion pipeline extracts articles and their details from PDF files. It handles Arabic and English text and records information such as the book, chapter, section, and article number.
2. **Store and search the text.** Article details and full text are stored in PostgreSQL. The text is split into smaller pieces, converted into embeddings, and stored in a second table for similarity search.
3. **Answer questions.** When someone asks a question, the system searches for related text and sends the matching legal articles to a language model. The model uses those sources to prepare an answer.

## My development setup

I am developing this project on a computer with an Intel Core i7 processor and an NVIDIA GTX 1050 graphics card. The amount of RAM has not been added yet. This computer has limited resources for AI development, so I use different tools for embeddings and answer generation:

- **Embeddings:** I run the quantized BGE-M3 model locally with `llama.cpp`. The model creates a vector for each text chunk so the database can find related legal text.
- **Answers:** I use the Groq API with its free tier to run the language model. This keeps the large language model off my computer, but the free API has request limits.

## Ingestion and database

The ingestion pipeline reads a PDF, extracts its legal articles, and creates Arabic and English text records. It uses a chunk limit of **300 words**, set in `config/rag.yaml`. This keeps each piece small enough to embed while allowing long articles to be searched in parts.

The database has two main tables:

| Table | What it stores |
| --- | --- |
| `metadat_and_docs` | One record for each article, including its text and metadata such as article number, book, chapter, section, and section name. |
| `embedding_vector` | The text chunks, their language, their embeddings, and a link to the article they came from. |

An article may be split into several chunks. All of those chunks point to the same article record. This means that if search finds any chunk, the system can also retrieve the full article text and its metadata.

The local embedding model is protected by a lock so that embedding calls from threads in the same Python process do not run at the same time. This helps reduce pressure on the computer during ingestion and question answering. The lock does not coordinate separate Python processes.

## Retrieval and answers

The system turns a user's question into an embedding using the same local model. PostgreSQL with pgvector compares it with the stored chunk embeddings and returns the closest matches. The current default is to retrieve **5** chunks. The RAG service sends the matching articles and citation details to the language model along with the question.

## Evaluation

The evaluation script uses a golden dataset of questions and expected answers. It currently evaluates **5 examples** to reduce the number of API requests and the time needed to run an evaluation. This is useful because the free Gemini API used as the judge has request limits, and evaluations can be slow when the models are busy.

The current script has **5 evaluation metrics**: Correctness, Relevance to Query, Faithfulness, Context Precision, and Context Recall. I plan to reduce the number of metrics to 2 to use fewer API requests. The script currently runs with one evaluation worker and pauses briefly between answers.

## Configuration and running

General RAG settings, including the chunk limit, number of retrieved chunks, model name, and prompt, are in `config/rag.yaml`. Database credentials and API keys are read from environment variables, usually set in a local `.env` file. The database requires PostgreSQL with the pgvector extension. The project requires Python 3.13 or newer.

After setting up the database, environment variables, dependencies, and embedding model file, run ingestion with:

```bash
python -m jurisrag_arabic.ingestion.process_docs
```

Run evaluation with:

```bash
python evaluation/rag_eval.py
```

## Main project files

- `src/jurisrag_arabic/ingestion/process_docs.py` — reads PDFs and prepares articles for storage.
- `src/jurisrag_arabic/ingestion/embedding.py` — splits text into chunks and creates embeddings.
- `src/jurisrag_arabic/ingestion/add_to_db.py` — creates database tables and saves articles and embeddings.
- `src/jurisrag_arabic/retrival/retrive.py` — searches for relevant text chunks.
- `src/jurisrag_arabic/rag.py` — retrieves sources and sends them to the language model.
- `evaluation/rag_eval.py` — runs the RAG evaluation.
