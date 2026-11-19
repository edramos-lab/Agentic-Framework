FROM python:3.12-slim

WORKDIR /app

# Install system deps for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ openssh-client && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY web/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project (backend needs paper_orchestra core + web backend)
COPY paper_orchestra/ ./paper_orchestra/
COPY web/backend/ ./web/backend/
COPY pyproject.toml ./
COPY examples/ ./examples/

# Install paper-orchestra as a package
RUN pip install --no-cache-dir -e .

# Data dirs
RUN mkdir -p /app/data /app/data/uploads /app/data/runs

ENV PYTHONUNBUFFERED=1

# The Jetson SSH key is bind-mounted read-only (see docker-compose.yml). ssh refuses keys with
# loose permissions, so copy it to ~/.ssh with mode 600 before starting.
CMD ["sh", "-c", "if [ -s /run/secrets/jetson_key ]; then mkdir -p /root/.ssh && cp /run/secrets/jetson_key /root/.ssh/id_ed25519 && chmod 600 /root/.ssh/id_ed25519; fi; exec uvicorn web.backend.app:app --host 0.0.0.0 --port 8000"]
