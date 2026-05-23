FROM python:3.11-alpine

LABEL maintainer="RAG Team"
LABEL description="RAG Python AI Service"

WORKDIR /app

# System dependencies for pypdf / sentence-transformers
RUN apk add --no-cache gcc g++ musl-dev

# Copy source code and config first (needed for setuptools package discovery)
COPY src/ src/
COPY config/ config/
COPY proto/ proto/
COPY scripts/ scripts/
COPY pyproject.toml .

# Install PyTorch CPU first (from PyTorch wheel index; musl wheels available)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
RUN pip install --no-cache-dir .

# Pre-compile proto files
RUN python scripts/compile_proto.py

# Non-root user
RUN adduser -D -s /bin/sh rag && chown -R rag:rag /app
USER rag

EXPOSE 50051 50052

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import grpc; channel=grpc.insecure_channel('localhost:50051'); grpc.channel_ready_future(channel).result(timeout=3)" || exit 1

ENV PYTHONPATH=/app/src
CMD ["python", "-m", "main"]
