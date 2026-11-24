
FROM python:3.11.18-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /uvx /bin/

WORKDIR /app

COPY uv.lock /app
COPY pyproject.toml /app


ENV UV_NO_DEV=1

RUN uv venv && uv sync --locked && uv pip install waitress flask

COPY . /app

EXPOSE 8000

CMD ["uv", "run", "waitress-serve", "--host", "0.0.0.0", "--port", "8000", "website:app"]
