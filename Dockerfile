FROM python:3.12-slim AS base

WORKDIR /app
COPY pyproject.toml .
COPY sentinel/ sentinel/

RUN pip install --no-cache-dir .

# Default: run as MCP server (stdio transport)
# For REST API mode, override with:
#   docker run -p 8329:8329 sentinel-ai uvicorn sentinel.api:app --host 0.0.0.0 --port 8329
CMD ["python", "-m", "sentinel.mcp_server"]
