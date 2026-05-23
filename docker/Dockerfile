FROM python:3.11-slim

LABEL maintainer="RAG Team"
LABEL description="RAG Python AI Service"

WORKDIR /app

# System dependencies for pypdf / sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy source code
COPY src/ src/
COPY config/ config/
COPY proto/ proto/
COPY scripts/ scripts/

# Pre-compile proto files
RUN python scripts/compile_proto.py

# Non-root user
RUN useradd -m -s /bin/bash rag && chown -R rag:rag /app
USER rag

EXPOSE 50051 50052

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import grpc; channel=grpc.insecure_channel('localhost:50051'); grpc.channel_ready_future(channel).result(timeout=3)" || exit 1

ENV PYTHONPATH=/app/src
CMD ["python", "-m", "main"]
