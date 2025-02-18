FROM python:3.11-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CHROMA_API_ENDPOINT=http://embeddings-chroma:8000

# Expose port
EXPOSE 8001

# Run the application
CMD ["uvicorn", "face:app", "--host", "0.0.0.0", "--port", "8001", "--log-level", "debug"]