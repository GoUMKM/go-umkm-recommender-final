FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app only
COPY app/ ./app/
COPY download_artifacts.sh ./download_artifacts.sh

# Make script executable
RUN chmod +x download_artifacts.sh

EXPOSE 8000
ENV PYTHONUNBUFFERED=1 \
    TF_ONEDNN_OPTS=0

# Run download script first, then launch the app
CMD ["bash", "-c", "./download_artifacts.sh && uvicorn app.app:app --host 0.0.0.0 --port 8000"]
