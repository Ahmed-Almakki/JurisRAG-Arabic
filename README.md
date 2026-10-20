# JurisRAG Arabic

JurisRAG Arabic is a question-answering system for legal documents in Arabic and English. It retrieves relevant legal articles from a PostgreSQL database and uses a language model to generate answers grounded in the retrieved sources. Answers should include citations to the relevant legal articles.

## How It Works

The project has three main steps:

1. **Document Ingestion:** Extracts legal articles from PDF documents, processes Arabic and English text, and records metadata such as article numbers, books, chapters, and sections.
2. **Embedding and Storage:** Splits legal text into smaller chunks, generates embeddings, and stores the text, metadata, and embeddings in PostgreSQL with pgvector.
3. **Retrieval and Answer Generation:** Converts a user's question into an embedding, retrieves the most similar text chunks, and sends the relevant legal context to a language model to generate an answer with article citations.

## Development Environment

I develop this project on a computer with an Intel Core i7 processor and an NVIDIA GTX 1050 graphics card. Because the hardware has limited resources for running AI models, I use a combination of local inference and cloud APIs.

- **Embedding Model:** Quantized BGE-M3, run locally using `llama.cpp`.
- **Language Model:** Groq API with a free-tier API key for answer generation.
- **Database:** PostgreSQL with the pgvector extension.
- **Python:** 3.13 or newer.

Using `llama.cpp` allows me to generate embeddings locally without relying on a cloud embedding API. Groq handles answer generation, reducing the computational requirements on my computer. However, free API tiers have rate limits that can affect evaluation and testing.

## Ingestion and Database

The ingestion pipeline extracts legal articles from PDF documents and prepares Arabic and English text for storage. It also records the legal structure and metadata associated with each article.

The maximum chunk size is configured to **300 words** in `config/rag.yaml`. This allows longer articles to be divided into smaller pieces for embedding and retrieval.

The database uses two main tables:

| Table | Description |
|---|---|
| `metadat_and_docs` | Stores legal articles, including their full text and metadata such as article number, book, chapter, section, and section name. |
| `embedding_vector` | Stores text chunks, their languages, embeddings, and references to their original article records. |

An article can be split into multiple chunks, with each chunk linked to the same article record. When a chunk is retrieved, the system can use this relationship to obtain the complete article text and its metadata for answer generation and citations.

### Embedding Concurrency

The local embedding model is protected by a thread lock and a shared file lock. This prevents concurrent embedding calls both within a Python process and between the ingestion and API containers. Docker mounts the processed-data directory into both containers so they share the lock file.

This helps control resource usage on a computer with limited hardware. However, the lock does not coordinate embedding operations across separate Python processes.

## Retrieval and Answer Generation

When a user submits a question, the system generates an embedding using the same local BGE-M3 model used during ingestion.

PostgreSQL with pgvector compares the query embedding with the stored chunk embeddings and retrieves the most similar text chunks.

The default retrieval setting is **5 chunks per query**, configurable in `config/rag.yaml`.

The RAG service then sends the question and the retrieved legal context to the language model through the Groq API. The model uses the provided context to generate an answer with relevant legal article citations.

The goal is to keep answers grounded in the retrieved legal documents rather than relying solely on the language model's existing knowledge.

## Evaluation

The evaluation script uses a golden dataset containing **20 questions and their expected answers**.

Due to the rate limits of free API tiers, I have currently been able to evaluate **2 out of the 20 examples**. Evaluating the full dataset requires additional API requests and may take longer when the services are busy.

The evaluation script currently supports two metrics:

- **Correctness:** Evaluates the factual accuracy of the generated answer.
- **Relevance to Query:** Measures how relevant the answer is to the user's question.


The script also uses a single evaluation worker and short pauses between requests to help manage rate limits.

### API Requirements for Evaluation and RAG

**You need a valid Groq API key to run the RAG system and the evaluation.**

- **Groq API key:** Used to generate answers.
- Groq is also used by the evaluation judges through its OpenAI-compatible API.

Groq's free tier is subject to rate limits, which may affect the number of examples that can be evaluated in a single run.

To run the evaluation, use:

```bash
python evaluation/rag_eval.py
```

### Performance & Load Testing Reports

When load-testing the RAG pipeline using Locust, system performance and stability are heavily tied to external cloud API limits:

* **Rate Limit Constraints & Concurrency (`-u 2 -r 1`):** 
  Because the system relies on Groq's free-tier API, sending high request volumes in parallel quickly exhausts the Tokens Per Minute (TPM) quota. To evaluate baseline system performance without triggering rate-limit errors, testing with low concurrency (e.g., `-u 2 -r 1` or 2 users with a spawn rate of 1) is ideal. Higher user counts immediately breach the free tier limit, causing HTTP `429 Too Many Requests` errors.
* **Observed Performance:** 
  Under controlled, low-concurrency testing, the system demonstrates strong efficiency:
  * **Time to First Token (TTFT):** A median of **~1.4 seconds**, indicating fast local vector retrieval and quick initial generation startup.
  * **End-to-End Latency:** Average completion times hover around **2.1 to 2.5 seconds**.
  * **Reliability:** When kept within rate limits, error rates drop to **0%**.

## Configuration

General RAG settings are managed in `config/rag.yaml`, including:

- Maximum chunk size.
- Number of chunks retrieved per query.
- Model configuration.
- Prompt settings.

Database credentials and API keys are read from environment variables, typically configured in a local `.env` file.


### Requirements

Before running the project, ensure you have:

- Python 3.13 or newer.
- PostgreSQL with the pgvector extension.
- The required Python dependencies.
- The local BGE-M3 quantized model file.
- A valid Groq API key for answer generation.

## Running the Project

### 1. Configure the Environment

Install Python 3.13 or newer and the project's required dependencies.

Configure the necessary environment variables for PostgreSQL and the Groq API. The evaluation uses the same Groq API key.

### 2. Set Up the Database

Start PostgreSQL and ensure the pgvector extension is installed and enabled in the target database.

Configure the database connection details using your environment variables.

### 3. Prepare the Embedding Model

Download or place the quantized BGE-M3 model in the expected local model directory. Ensure that the configured model path matches the actual location of the model file.

### 4. Run Document Ingestion

To process the legal documents and store their articles and embeddings in the database, run:

```bash
python -m jurisrag_arabic.ingestion.process_docs
```

Ensure that the source PDFs are available in the locations expected by the ingestion pipeline.

### 5. Run Evaluation

After configuring the Groq API key, run:

```bash
make eval
# Runs: python evaluation/rag_eval.py
```

The script evaluates the generated answers against the golden dataset. Due to free-tier API limits, the number of examples evaluated in one run may be limited.

## Run with Docker Compose

Docker Compose starts PostgreSQL (`jurisage-db`), MLflow (`mlflow`), and the RAG API. The ingestion container uses the same `jurisrag-arabic:local` image as the API and runs when requested. Existing PostgreSQL and MLflow data are kept in the `pg_data` and `mlflow-artifacts` volumes.

Prepare the environment file, model, and first source PDF:

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY and the database/MLflow settings.
# Place the BGE-M3 GGUF file in models/bge-m3/ and the source PDF in data/raw/.
make run
```

`make run` starts PostgreSQL, ingests the configured PDF, and then starts MLflow and the RAG API. Use `make up` to start the services later without re-ingesting:

```bash
make up
# Runs: docker compose up --build -d
```

The model and data directories are mounted from the host; the large GGUF model is not copied into the image.

Check service status and logs:

```bash
docker compose ps
docker compose logs -f db mlflow rag
```

The RAG API exposes liveness at `/livez`, readiness at `/readyz`, and Prometheus metrics at `/metrics`. BentoML provides these monitoring endpoints and metrics by default. [BentoML metrics documentation](https://docs.bentoml.com/en/latest/build-with-bentoml/observability/metrics.html) · [BentoML health endpoint documentation](https://docs.bentoml.com/en/latest/build-with-bentoml/observability/monitoring-and-data-collection.html)

MLflow tracks evaluation runs. RAG serving and retrieval do not import or call MLflow.

Send a question with curl:

```bash
curl -X POST http://localhost:3000/ask/ \
  -H 'Content-Type: application/json' \
  -d '{"query":"ما هي مدة تقادم الضرائب المستحقة للدولة؟"}'
```

Check metrics with:

```bash
curl http://localhost:3000/metrics
```

If ingestion fails, resolve the error and run `make ingest` to retry the configured PDF. Use `make add-doc DOC=/path/to/document.pdf` to ingest another PDF.

### Load Test the RAG API

Start the services with `make up`, then run the configured Locust load test:

```bash
uv sync --group dev
make locst
```

`make locst` runs a headless test with two users for one minute against `http://localhost:3000`. It writes an HTML report to `reports/locust_report_<timestamp>.html`.

Each simulated user sends a legal question every 1–3 seconds. Requests invoke retrieval and the Groq LLM, so the measured latency includes both; start with a low user count and account for provider rate limits and API costs before increasing load.

## Tests and Continuous Integration

Run the tests locally with:

```bash
make test
# Runs: uv run pytest
```

GitHub Actions runs the test suite on pushes and pull requests. The workflow starts a temporary PostgreSQL database with pgvector and runs a system integration test through article chunking and embedding orchestration, database storage, vector retrieval, and RAG prompt creation. It uses a fixed test embedding and a stub language model, so it does not need the local embedding model or API keys.

## Main Project Files

```text
.
├── config/
│   └── rag.yaml
├── evaluation/
│   └── rag_eval.py
└── src/
    └── jurisrag_arabic/
        ├── ingestion/
        │   ├── process_docs.py
        │   ├── embedding.py
        │   └── add_to_db.py
        ├── retrival/
        │   └── retrive.py
        └── rag.py
```

| File | Responsibility |
|---|---|
| `config/rag.yaml` | Stores general RAG settings, including chunk size, retrieval count, model configuration, and prompts. |
| `src/jurisrag_arabic/ingestion/process_docs.py` | Extracts and processes legal articles from PDF documents. |
| `src/jurisrag_arabic/ingestion/embedding.py` | Splits article text into chunks and generates embeddings using the local model. |
| `src/jurisrag_arabic/ingestion/add_to_db.py` | Creates the database tables and stores articles and embeddings. |
| `src/jurisrag_arabic/retrival/retrive.py` | Retrieves relevant text chunks from PostgreSQL using vector similarity search. |
| `src/jurisrag_arabic/rag.py` | Coordinates retrieval and answer generation using the Groq API. |
| `evaluation/rag_eval.py` | Evaluates answer quality using the golden dataset and Gemini as an LLM judge. |
