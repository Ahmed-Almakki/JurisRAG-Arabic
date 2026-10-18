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

The local embedding model is protected by a thread lock to prevent multiple threads within the same Python process from running embedding operations simultaneously.

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

**You need valid API keys for both Groq and Gemini to run the evaluation and rag retrival successfully.**

- **Groq API key:** Used to generate answers.
- **Gemini API key:** Used by the LLM judge to evaluate the results.

Both APIs are subject to their respective rate limits, which may affect the number of examples that can be evaluated in a single run.

To run the evaluation, use:

```bash
python evaluation/rag_eval.py
```

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
- A valid Gemini API key for evaluation.

## Running the Project

### 1. Configure the Environment

Install Python 3.13 or newer and the project's required dependencies.

Configure the necessary environment variables for PostgreSQL and the Groq API. For evaluation, configure the Gemini API key as well.

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

After configuring both the Groq and Gemini API keys, run:

```bash
python evaluation/rag_eval.py
```

The script evaluates the generated answers against the golden dataset. Due to free-tier API limits, the number of examples evaluated in one run may be limited.

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