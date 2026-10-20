.PHONY: run up ingest add-doc test download_model eval ask_example

run:
	docker compose build
	docker compose run --rm ingestion
	docker compose up -d

up:
	docker compose up --build -d

ingest:
	docker compose run --build --rm ingestion

test:
	python -m pytest tests

download_model:
	HF_HUB_DISABLE_XET=1 uv run python -m scripts.download_model

eval:
	python3 -m evaluation.rag_eval

locst:
# 	uv run locust -f locust.py --host http://localhost:3000
	uv run locust -f locust.py --host http://localhost:3000 \
	--headless -u 2 -r 1 --run-time 1m \
	--html reports/locust_report_$(shell date +%s).html

ask_example:
	curl -N -X POST 'http://localhost:3000/ask/' \
     -H 'accept: application/json' \
     -H 'Content-Type: application/json' \
     -d '{"query": "ما هي مدة تقادم الضرائب المستحقة للدولة؟"}'