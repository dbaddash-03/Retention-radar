FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# non-root user with uid 1000 (required by Hugging Face Spaces, good practice everywhere)
RUN useradd -m -u 1000 user
WORKDIR /app
# dependencies first so code edits don't invalidate this cached layer
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY --chown=user:user . .
USER user
# train inside the image so the served model always matches the pinned library versions
RUN python -m churn.train
EXPOSE 7860
# PORT is set by Render/Railway; Hugging Face Spaces uses 7860 (the default here).
# 1 worker + threads keeps memory low enough for free tiers.
CMD ["sh", "-c", "exec gunicorn -w 1 --threads 4 --timeout 60 -b 0.0.0.0:${PORT:-7860} app:app"]
