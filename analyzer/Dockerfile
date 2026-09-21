# Builds the SyncSentry API with the full ML stack (torch + SyncNet), for
# hosts with enough CPU/RAM to actually run it, e.g. a Hugging Face Space
# (Docker SDK). Build context is this directory; see ../SETUP.md for
# running the API without Docker.
FROM python:3.11-slim

# ffmpeg: required by the pipeline for all media I/O.
# git + curl: required by scripts/fetch_syncnet.sh to vendor SyncNet and
# download its pretrained weights at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-core.txt requirements-api.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements-core.txt -r requirements-api.txt -r requirements-ml.txt

COPY . .
RUN pip install --no-cache-dir -e .

# Fetches SyncNet + S3FD pretrained weights into third_party/, so
# use_syncnet actually runs the learned detector instead of silently
# falling back to the coarse one.
RUN bash scripts/fetch_syncnet.sh

# Hugging Face Spaces (Docker SDK) sends traffic to port 7860 by default.
EXPOSE 7860

# Spaces run containers as a non-root user; match that here too.
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["uvicorn", "syncsentry.api:app", "--host", "0.0.0.0", "--port", "7860"]
