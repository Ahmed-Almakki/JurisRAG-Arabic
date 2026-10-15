ingestion:
	python3 -m src.jurisrag_arabic.ingestion.process_docs

test:
	python -m pytest tests

download_model:
	HF_HUB_DISABLE_XET=1 uv run python -m scripts.download_model

db:
	docker rm -f jurisage-db || true
	docker run -d \
	  --name jurisage-db \
	  -p 5432:5432 \
	  -e POSTGRES_DB=jurisage_db \
	  -e POSTGRES_USER=postgres \
	  -e POSTGRES_PASSWORD=postgres \
	  -v pg_data:/var/lib/postgresql/data \
	  pgvector/pgvector:pg17

run:
	bentoml serve jurisrag_arabic.rag:RAG --reload