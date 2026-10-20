FROM ghcr.io/astral-sh/uv:0.12.7-python3.13-trixie-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_HTTP_TIMEOUT=120 \
    UV_HTTP_RETRIES=8 \
    BENTOML_DO_NOT_TRACK=true \
    BENTOML_HOME=/tmp/bentoml

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY config/ ./config/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev \
    && uv pip install --python /app/.venv/bin/python \
        --index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-best-match \
        --only-binary llama-cpp-python \
        llama-cpp-python==0.3.36 \
    && mkdir -p /app/data/raw /app/data/processed /app/models/bge-m3 /tmp/bentoml \
    && chmod 1777 /tmp/bentoml

EXPOSE 3000

CMD ["bentoml", "serve", "jurisrag_arabic.rag:RAG", "--host", "0.0.0.0", "--port", "3000"]
