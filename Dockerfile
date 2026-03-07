FROM python:3.12-slim AS base

WORKDIR /app
COPY pyproject.toml .
COPY sentinel/ sentinel/

RUN pip install --no-cache-dir ".[api]"

EXPOSE 8329

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8329/health').raise_for_status()"

CMD ["uvicorn", "sentinel.api:app", "--host", "0.0.0.0", "--port", "8329"]
