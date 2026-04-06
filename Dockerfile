FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md
COPY openenv.yaml /app/openenv.yaml
COPY uv.lock /app/uv.lock
COPY server /app/server
COPY src /app/src

RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "supportdesk_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]

