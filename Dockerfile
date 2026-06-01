# Use an optimized, official Python footprint image as our runtime foundation
FROM python:3.11-slim

WORKDIR /app

# Install operating system dependencies required for performance scaling extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy python packaging manifests over first to exploit docker build layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy over our core development source layout layers and serialization caches
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

ENV PYTHONPATH=/app/src

# Trigger the production application instance securely using performance-tuned uvicorn threading flags
CMD ["uvicorn", "src.api.main.py:app", "--host", "0.0.0.0", "--port", "8000"] 
