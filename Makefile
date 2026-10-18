
.PHONY: mlf

ingestion:
	python3 -m src.jurisrag_arabic.ingestion.process_docs

test:
	python -m pytest tests

download_model:
	HF_HUB_DISABLE_XET=1 uv run python -m scripts.download_model

db:
	docker rm -f jurisage-db || true
	docker network inspect jurisage-network >/dev/null 2>&1 || docker network create jurisage-network

	docker run -d \
	  --network jurisage-network \
	  --name jurisage-db \
	  -p 5432:5432 \
	  -e POSTGRES_DB=jurisage_db \
	  -e POSTGRES_USER=postgres \
	  -e POSTGRES_PASSWORD=postgres \
	  -v pg_data:/var/lib/postgresql/data \
	  pgvector/pgvector:pg17
mlf:
	set -a; . ./.env; set +a; \
	docker rm -f mlflow 2>/dev/null || true; \
	docker run -d \
	  --network jurisage-network \
	  --name mlflow \
	  -p 5000:5000 \
	  -e MLFLOW_BACKEND_STORE_URI="$$MLFLOW_BACKEND_STORE_URI" \
	  -v mlflow-artifacts:/mlflow/artifacts \
	  ghcr.io/mlflow/mlflow:latest \
	  mlflow server \
	  --backend-store-uri "$$MLFLOW_BACKEND_STORE_URI" \
	  --default-artifact-root /mlflow/artifacts \
	  --artifacts-destination /mlflow/artifacts \
	  --serve-artifacts \
	  --host 0.0.0.0 \
	  --port 5000 \
	  --allowed-hosts "*"
run:
	bentoml serve jurisrag_arabic.rag:RAG --reload

eval:
	python3 -m evaluation.rag_eval