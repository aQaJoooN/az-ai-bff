FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY pyproject.toml .
COPY src/ ./src/

RUN pip install --no-cache-dir .

COPY tests/ ./tests/

RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["az-ai-bff"]