FROM python:3.11-slim

# Install system audio, C++ build utilities, and ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and knowledge base
COPY src/ ./src/
COPY knowledge_base/ ./knowledge_base/
COPY data/ ./data/
COPY prompts/ ./prompts/

# Set environment defaults
ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    APP_ENV=production

EXPOSE 8000

# Start Uvicorn with WebSocket support
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT} --ws websockets --timeout-keep-alive 65"]