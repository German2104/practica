FROM python:3.12-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==0.11.14
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts ./scripts
RUN uv sync --locked --extra ml --no-dev
COPY data/portfolio_data.csv ./data/portfolio_data.csv
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "ec_analysis.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
