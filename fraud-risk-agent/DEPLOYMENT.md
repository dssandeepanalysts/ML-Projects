# Deploying the Fraud Risk Agent

The agent ships as one Docker image (`server.py` + `fra_agent.py`, about 290 MB) and a
`docker-compose.yml` that can also run a local Ollama model. It keeps no state, so you can
run as many copies as you like behind a load balancer.

| File | Purpose |
|---|---|
| `Dockerfile` | Python 3.11-slim image, non-root user, built-in health check, graceful shutdown |
| `docker-compose.yml` | `fra` service (loopback port, read-only, 25 s stop grace); `ollama` + one-off `ollama-pull` services behind the `llm` profile |
| `.env.example` | All settings; copy to `.env` |
| `requirements-server.txt` | Runtime dependencies only (what the image installs) |

## 1. Run it

Needs Docker with Compose v2.

```bash
cd fraud-risk-agent
cp .env.example .env                       # then set FRA_API_TOKEN (compose refuses to start without it)
docker compose up -d --build --wait        # returns once the container reports healthy
curl -s localhost:8000/health              # {"status":"ok","version":"1.0.0","llm_wording":"off"}
curl -s -X POST localhost:8000/v1/assess -H "Authorization: Bearer <your token>" \
     -H "Content-Type: application/json" -d @examples/request_duplicate.json
```

This is the recommended production mode. Explanations use the fixed template, so a reply takes
about 1 ms. The port is published on `127.0.0.1` only (`FRA_BIND`); see section 4 for HTTPS.
Docker gives the container 25 s to finish in-flight calls on stop or update. If you run the image
without Compose, use `docker run --stop-timeout 25` (or `docker stop -t 25`).

## 2. Optional: local LLM wording (Ollama)

```bash
# in .env: FRA_USE_LLM=1
docker compose --profile llm up -d --build --wait fra ollama
docker compose --profile llm run --rm ollama-pull     # downloads llama3.2 (~2 GB) once, into a volume
curl -s localhost:8000/health                          # "llm_wording":"on"
```

* Until the model has downloaded, and whenever Ollama is slow or down, replies fall back to the
  template. Scores and flags never depend on the model.
* On CPU each LLM explanation takes about 3–9 s. The agent stops waiting after `FRA_LLM_TIMEOUT`
  (default 8 s) and uses the template. A request that already queued for a free worker for more
  than `14 − FRA_LLM_TIMEOUT` seconds skips the LLM. Together these keep replies inside Pega's
  15 s limit. Keep `FRA_LLM_TIMEOUT` at 10 s or less: a larger value leaves almost no queueing
  room, so most replies would use the template (the server logs a warning). A GPU makes the LLM
  step sub-second. With the NVIDIA Container Toolkit, add a `docker-compose.override.yml`:

  ```yaml
  services:
    ollama:
      deploy:
        resources:
          reservations:
            devices: [{ driver: nvidia, count: all, capabilities: [gpu] }]
  ```

## 3. Health checks

| Check | What it does |
|---|---|
| `GET /health` | Public, no token. Returns `status`, `version` and `llm_wording` (`on`/`off`). It stays `ok` when Ollama is down, because the template still answers. |
| Docker `HEALTHCHECK` | Calls `/health` every 30 s (15 s start period, 3 retries). `docker compose ps` shows `(healthy)`, and `--wait` blocks until it is. |
| Ollama (`llm` profile) | `ollama list` every 15 s. |
| Kubernetes | Use `httpGet: {path: /health, port: 8000}` for both liveness and readiness probes (a 2–3 s `timeoutSeconds` is plenty; `/health` never waits behind assessments). Keep `terminationGracePeriodSeconds` at 25 or more (the default 30 is fine). |

Watch these in the logs (one line per claim; the claimant's description and the hashed IDs are never logged):

* `status=FAILED`, or `contract violation`: Pega sent a request that breaks the contract.
* `narrative=template-fallback`, `LLM unavailable` or `LLM exceeded`: only matters in LLM mode.
* `ms=`: processing time. It should stay far below Pega's 15 s.

## 4. Before production: steps that need your infrastructure details

| Item | What you provide | What to change |
|---|---|---|
| **OAuth 2.0** (spec s4) | Identity provider: issuer, audience, JWKS URL, scope (e.g. `fraud.assess`) | Replace the static-token comparison in `require_token()` (`server.py`) with JWT validation against your JWKS, e.g. with PyJWT. Keep `FRA_API_TOKEN` for local use only. |
| **HTTPS** | A domain and a TLS certificate | Terminate TLS at your load balancer, ingress or reverse proxy, and forward to the container. A proxy on the same host uses `127.0.0.1:8000` (the default `FRA_BIND`); a proxy in Docker joins the Compose network; a remote load balancer needs `FRA_BIND` set to a private-network IP. Never publish port 8000 on a public interface: Docker-published ports bypass host firewalls such as ufw. Set `FRA_PUBLIC_URL=https://<your-domain>` so the Agent Card advertises the HTTPS endpoint. |
| **Pega connection** | Your Pega environment | Create the Connect Agent rule from `https://<your-domain>/.well-known/agent.json` (skill `assess_fraud_risk`). Give it an authentication profile with the OAuth client credentials. Confirm the A2A version and DataPart shape (spec open items O-1, O-3). `POST /v1/assess` is the Connect-REST fallback. |

Also for production:
* Keep the token in a secret manager, not in `.env`.
* Pin the image by digest.
* Ship container logs to your log store.
* Scale by adding replicas. In LLM mode, run Ollama on a GPU host.

## 5. Day-to-day operations

```bash
docker compose logs -f fra                      # follow the logs
git pull && docker compose up -d --build --wait # update in place
docker compose down                             # stop (add -v to also delete the downloaded model)
```

To roll back, tag each release image (`fraud-risk-agent:<version>`) and start the previous tag.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Set FRA_API_TOKEN in .env` when starting | No `.env` file, or `FRA_API_TOKEN` is empty: `cp .env.example .env` and set the token. |
| `401` on every call | The token Pega sends does not match `FRA_API_TOKEN`. |
| `llm_wording` is `on` but replies use the template | The model is not downloaded yet (run `ollama-pull`), or replies take longer than `FRA_LLM_TIMEOUT`. Check `docker compose logs fra`. |
| `pip` TLS errors during `docker build` | Your network intercepts TLS. Build with your proxy's CA as a build secret, which does not end up in the image: `docker build --secret id=pip_cert,src=/path/to/proxy-ca.pem -t fraud-risk-agent:1.0.0 .`, then `docker compose up -d --wait` (without `--build`). |
