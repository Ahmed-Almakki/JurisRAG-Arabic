.PHONY: run up ingest add-doc test download_model eval ask_example

run:
	docker compose build
	docker compose run --rm ingestion
	docker compose up -d

up:
	docker compose up --build -d

ingest:
	docker compose run --build --rm ingestion

pull:
	curl -L -o service-account.json "https://www.dropbox.com/scl/fi/6p5fdnhlix0rp8z7qvu6x/service-account.json?rlkey=52x2thyogerdknghidwn8nht8&st=j6spbjhf&dl=1" && dvc pull

test:
	uv run pytest

download_model:
	HF_HUB_DISABLE_XET=1 uv run python -m scripts.download_model

eval:
	python3 -m evaluation.rag_eval

locst:
# 	uv run locust -f locust.py --host http://localhost:3000
	uv run locust -f locust.py --host http://localhost:3000 \
	--headless -u 2 -r 1 --run-time 1m \
	--html reports/locust_report_$(shell date +%s).html

ask_example_ar:
	curl -N -X POST 'http://localhost:3000/ask/' \
     -H 'accept: application/json' \
     -H 'Content-Type: application/json' \
     -d '{"query": "ما هي مدة تقادم الضرائب المستحقة للدولة؟"}'

ask_example_en:
	curl -N -X POST 'http://localhost:3000/ask/' \
     -H 'accept: application/json' \
     -H 'Content-Type: application/json' \
     -d '{"query": "What is the statute of limitations for taxes owed to the state?"}'


	 