# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS base

# Core env + make Gradio skip localhost prelaunch check
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SKIP_PRELAUNCH_CHECK=true \
    PYTHONPATH=/app \
    MCP_SERVER_CWD=/app \
    MCP_SERVER_PY=/app/server/pbm_server.py

# System deps (curl for healthcheck)
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Upgrade pip first, then install once
RUN python -m pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Copy app code
COPY server /app/server
COPY client /app/client
COPY data /app/data
# Optional: copy .env if you want default values baked in (compose will still override)
# COPY .env /app/.env

EXPOSE 7860

# Default command: run the Gradio client (which will spawn the MCP server via stdio)
CMD ["python", "-u", "client/app.py"]
