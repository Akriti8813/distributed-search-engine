# Single image reused for every shard AND the gateway - which service
# runs is chosen by the container's command (see docker-compose.yml),
# so we build once and cache layers across all N+1 containers.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common/ common/
COPY shard_service/ shard_service/
COPY gateway_service/ gateway_service/
COPY scripts/ scripts/

# Prebuilt shard indexes are mounted in via docker-compose volumes in
# dev; for a real image you'd COPY data/shards/ here at build time.
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
