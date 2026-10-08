# syntax=docker/dockerfile:1
# Fraud Risk Agent: A2A / REST service (server.py). Template wording by default;
# set FRA_USE_LLM=1 and OLLAMA_HOST to reword explanations with a local Ollama model.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements-server.txt .
# Behind a TLS-intercepting proxy, pass its CA without baking it into the image:
#   docker build --secret id=pip_cert,src=/path/to/proxy-ca.pem -t fraud-risk-agent:1.0.0 .
RUN --mount=type=secret,id=pip_cert \
    if [ -s /run/secrets/pip_cert ]; then export PIP_CERT=/run/secrets/pip_cert; fi; \
    pip install -r requirements-server.txt
COPY fra_agent.py server.py ./

# Run as an unprivileged user; the app writes nothing to disk.
RUN useradd --system --uid 10001 --no-create-home fra
USER 10001

EXPOSE 8000
# /health is public and does not depend on Ollama (the template always answers).
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

# Exec form so uvicorn gets SIGTERM directly; in-flight calls get 20 s to finish
# (Pega's per-attempt timeout is 15 s).
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "20"]
