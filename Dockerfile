FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY backend ./backend
COPY evals ./evals
RUN pip install --no-cache-dir .

ENV PYTHONPATH=/app/backend

# Managed platforms (Render, Cloud Run, Railway) assign the port at runtime via
# PORT. Default to 8080 so the same image still runs unchanged under docker
# compose and the Makefile targets.
ENV PORT=8080

# Bind 0.0.0.0 rather than 127.0.0.1: inside a container, loopback is reachable
# only from the container itself, so the platform's router could not connect.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
