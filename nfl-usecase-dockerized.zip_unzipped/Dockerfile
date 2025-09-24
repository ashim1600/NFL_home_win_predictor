# syntax=docker/dockerfile:1

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x scripts/docker-entry.sh

EXPOSE 8501

ENV RUN_PIPELINE_ON_START=true \
    PORT=8501

ENTRYPOINT ["/bin/bash", "scripts/docker-entry.sh"]
