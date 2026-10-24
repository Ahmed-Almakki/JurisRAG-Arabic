# JurisRAG Arabic

JurisRAG Arabic is a question-answering system for legal documents in Arabic and English. It retrieves relevant legal articles from a PostgreSQL database and uses a language model to generate answers grounded in the retrieved sources. Answers should include citations to the relevant legal articles.

## How It Works

The project has three main steps:

1. **Document Ingestion:** Extracts legal articles from PDF documents, processes Arabic and English text, and records metadata such as article numbers, books, chapters, and sections.
2. **Embedding and Storage:** Splits legal text into smaller chunks, generates embeddings, and stores the text, metadata, and embeddings in PostgreSQL with pgvector.
3. **Retrieval and Answer Generation:** Converts a user's question into an embedding, retrieves the most similar text chunks, and sends the relevant legal context to a language model to generate an answer with article citations.

## Models and API Keys

The project uses one local model and one cloud model. This setup keeps the embedding work on the project computer, while a cloud service handles the larger answer-generation model.

- **Local embedding model:** Quantized BGE-M3 (`bge-m3-Q8_0.gguf`) runs on the project computer through `llama.cpp`. It turns legal text and questions into vectors for search. Running it locally avoids sending embedding requests to a cloud embedding service. The quantized model is smaller than the full model, which helps it run on computers with limited resources.
- **Cloud answer model:** The RAG service sends the question and retrieved legal text to `openai/gpt-oss-120b` through the Groq API. Answer generation needs more compute than embedding, and the development computer has limited GPU resources. Using Groq means the project does not need to run this large model locally. It does require internet access, a Groq API key, and is subject to Groq rate limits.
- **Database:** PostgreSQL with the pgvector extension runs in Docker Compose.
- **Python:** 3.13 or newer.

Create two Groq API keys so RAG requests and evaluation requests can be managed separately. Put them in the local `.env` file:

| Variable | Used for |
|---|---|
| `GROQ_API_KEY_RAG` | The application uses this key to generate answers through Groq. |
| `GROQ_API_KEY_EVAL` | The evaluation script uses this key for its answer-quality judges. |

The evaluation key is only needed when running evaluation. Never commit `.env` or share either key.

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

The RAG application and evaluation each use a Groq API key. Set `GROQ_API_KEY_RAG` for the application and `GROQ_API_KEY_EVAL` for evaluation in `.env`.

Groq's free tier has rate limits, which may limit the number of evaluation examples that can run at one time.

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

- Python 3.13 or newer and `uv` for the project tools.
- Docker with the Docker Compose plugin.
- An internet connection to download the documents, embedding model, and call Groq.

## Instructor Quick Start

Run these steps from a terminal. Replace `https://github.com/Ahmed-Almakki/JurisRAG-Arabic.git` with the URL provided by the instructor.

### 1. Clone the Project and Install Dependencies

Clone the repository, enter its folder, and install the Python dependencies. `uv sync` also installs DVC, which is needed to download the project data.

```bash
git clone https://github.com/Ahmed-Almakki/JurisRAG-Arabic.git
cd JurisRAG-Arabic
uv sync
```

### 2. Configure the Environment

Create your local environment file from the example:

```bash
cp .env.example .env
```

Edit `.env` and set `DB_PASSWORD`, `MLFLOW_BACKEND_STORE_URI`, and both Groq keys. Create two keys in Groq: put the key for application answers in `GROQ_API_KEY_RAG`, and the key for evaluation in `GROQ_API_KEY_EVAL`. The evaluation key is only needed if you will run `make eval`. Keep `.env` private and do not commit it.

### 3. Download the Documents and Embedding Model

Run:

```bash
make pull
make download_model
```

`make pull` downloads `service-account.json` and then runs `dvc pull`. The service-account file lets DVC access the Google Drive remote that stores the project documents. Keep this file private and do not commit it. `dvc pull` downloads the tracked data into the repository. `make download_model` downloads the local BGE-M3 model used by ingestion and the API.

### 4. Ingest and Start the Application

Choose one of these two options:

**Option A: Run ingestion, then start the app.** Choose this when you want to run or retry ingestion separately. Run:

```bash
make ingest
make up
```

The first command ingests the documents into the database. After it succeeds, the second command starts the API and other services without repeating ingestion.

**Option B: Run everything with one command.** Choose this for the normal full setup. It builds the Docker image, runs ingestion, and starts the application:

```bash
make run
```

Use Option A if you need to control or retry ingestion separately. Use Option B to build, ingest, and start the app together. The DVC pipeline (`dvc.yaml`) tracks ingestion output; it does not build the Docker image or start the application.

### 5. Try Example Questions

With the API running, test it with the Arabic or English example:

```bash
make ask_example_ar
make ask_example_en
```

The RAG API is available at `http://localhost:3000`. To run the evaluation after configuring the Groq API key:

```bash
make eval
```

Free-tier API limits may restrict the number of evaluation examples that can be completed in one run.

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


## Tests and Continuous Integration

Run the tests locally with:

```bash
make test
# Runs: uv run pytest
```

GitHub Actions runs the test suite on pushes and pull requests. The workflow starts a temporary PostgreSQL database with pgvector and runs a system integration test through article chunking and embedding orchestration, database storage, vector retrieval, and RAG prompt creation. It uses a fixed test embedding and a stub language model, so it does not need the local embedding model or API keys.

## Project Directories

Here is what each main project directory contains:

| Directory | What it contains |
|---|---|
| `.github/` | GitHub Actions workflows, including automated tests. |
| `.dvc/` | DVC settings for tracking and downloading project data. |
| `config/` | RAG settings such as the model names, chunk size, retrieval count, and prompts. |
| `data/raw/` | Original legal documents used as ingestion input. DVC manages these files. |
| `data/processed/` | Processed legal documents and shared ingestion files. |
| `evaluation/` | The evaluation script and test questions with expected answers. |
| `grafana/` | Grafana dashboard and provisioning files. |
| `models/bge-m3/` | The downloaded local BGE-M3 embedding model. |
| `monitoring/` | Prometheus and Grafana configuration for monitoring the services. |
| `notebook/` | Jupyter notebooks used to explore or test ingestion. |
| `reports/` | Generated Locust load-test reports. |
| `scripts/` | Helper scripts, including the embedding-model downloader. |
| `src/jurisrag_arabic/` | The application code: document ingestion, embeddings, retrieval, and the RAG API. |
| `tests/` | Automated unit and integration tests. |
