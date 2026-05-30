FROM python:3.11-slim

LABEL maintainer="RAG Team"
LABEL description="RAG Python AI Service"

WORKDIR /app

# System dependencies for pypdf / sentence-transformers / PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    tesseract-ocr tesseract-ocr-chi-sim poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy source code and config first (needed for setuptools package discovery)
COPY src/ src/
COPY config/ config/
COPY proto/ proto/
COPY scripts/ scripts/
COPY pyproject.toml .

# Install PyTorch CPU first (from PyTorch wheel index)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
RUN pip install --no-cache-dir .

# Pre-compile proto files
RUN python scripts/compile_proto.py

# Use HuggingFace mirror for model downloads (HF官方在中国大陆不可用)
ENV HF_ENDPOINT=https://hf-mirror.com

# Pre-download the local embedding model (avoids download timeout at startup)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-zh-v1.5')"

# Non-root user
RUN useradd -m -s /bin/sh rag && chown -R rag:rag /app
USER rag

EXPOSE 50051 50052

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import grpc; channel=grpc.insecure_channel('localhost:50051'); grpc.channel_ready_future(channel).result(timeout=3)" || exit 1

ENV PYTHONPATH=/app/src
CMD ["python", "-m", "main"]
