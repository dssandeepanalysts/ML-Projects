# Deploying the Fraud Risk Agent

The agent ships as one Docker image (`server.py` + `fra_agent.py`, about 290 MB) and a
`docker-compose.yml` that can also run a local Ollama model. It keeps no state, so you can
run as many copies as you like behind a load balancer.

| File | Purpose |
|---|---|
| `Dockerfile` | Python 3.11-slim image, non-root user, built-in health check, graceful shutdown |
| `docker-compose.yml` | `fra` service (loopback port, read-only, 25 s stop grace); `ollama` + one-off `ollama-pull` services behind the `llm` profile; a test `keycloak` behind the `oauth` profile |
| `.env.example` | All settings; copy to `.env` |
| `keycloak/` | Test realm (`claims-realm.json`) and `get-token.sh` for trying OAuth locally |
| `requirements-server.txt` | Runtime dependencies only (what the image installs) |

## 1. Run it

Needs Docker with Compose v2.

```bash
cd fraud-risk-agent
cp .env.example .env                       # then set FRA_API_TOKEN, or the OAuth settings (section 4)
docker compose up -d --build --wait        # returns once the container reports healthy
curl -s localhost:8000/health              # {"status":"ok","version":"1.0.0","llm_wording":"off"}
curl -s -X POST localhost:8000/v1/assess -H "Authorization: Bearer <your token>" \
     -H "Content-Type: application/json" -d @examples/request_duplicate.json
```

This is the recommended production mode. Explanations use the fixed template, so a reply takes
about 1 ms. The port is published on `127.0.0.1` only (`FRA_BIND`); see section 5 for HTTPS.
`FRA_API_TOKEN` is a stand-in for local use; production uses OAuth 2.0 (section 4). With neither set,
the agent refuses to start and `docker compose logs fra` says why.
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

## 4. OAuth 2.0 login

Pega gets an access token from your identity provider (OAuth 2.0 client credentials) and sends it as
`Authorization: Bearer <token>`. On every call the agent checks:

* the signature, against the provider's public keys from `FRA_OAUTH_JWKS_URL`, never from a URL
  named in the token;
* the algorithm: RS, PS or ES only (never `none` or HS256);
* `exp` (required), and `nbf` and `iat` if present, each with 30 s of clock-skew allowance;
* `iss` equals `FRA_OAUTH_ISSUER` exactly, and `aud` contains `FRA_OAUTH_AUDIENCE`;
* the scope `FRA_OAUTH_SCOPE` (default `fraud.assess`) in the `scope`, `scp` or `roles` claim.

A missing, invalid or expired token gets `401`, and a valid token without the scope gets `403`. If
the agent cannot get the provider's keys it fails closed with `503` and `Retry-After: 10`, so Pega
may retry. The log line says why (`token rejected: InvalidAudienceError`) and never contains the
token. The agent itself holds no secret in this mode, only public keys.

How the agent fetches the provider's keys:

* It fetches them on first use and keeps them for 10 minutes; after that it refreshes them in the
  background while still answering with the cached keys.
* A token signed with a key it has not seen (the provider rotated its keys) makes it fetch again,
  so key rotation needs no restart. Fetches run one at a time and at most once every 10 s, so forged
  tokens cannot make the agent flood your provider. A token with a brand-new key that arrives within
  10 s of the previous fetch is refused until the next fetch, and forged tokens can use up that
  window too. Providers that publish new keys before using them (Entra ID, Okta) avoid this; in
  Keycloak, add a new key at a lower priority than the current one, then raise its priority after
  10 minutes.
* If the provider cannot be reached, the agent keeps using the last keys it fetched for up to an hour.
* A request waits at most 3 s for a fetch, without tying up a worker thread, and a download is
  abandoned after 10 s. The URL must answer directly: redirects are not followed.
* The URL must use https. Plain http is refused unless `FRA_OAUTH_ALLOW_HTTP=1`, which exists only for
  the local Keycloak below: over http, anyone on the network path could swap in their own keys.

| Setting | Local Keycloak value | Notes |
|---|---|---|
| `FRA_OAUTH_ISSUER` | `http://localhost:8180/realms/claims` | Exactly the tokens' `iss` claim |
| `FRA_OAUTH_AUDIENCE` | `fraud-risk-agent` | A value in the tokens' `aud` claim |
| `FRA_OAUTH_JWKS_URL` | `http://keycloak:8080/realms/claims/protocol/openid-connect/certs` | Must be reachable from inside the container |
| `FRA_OAUTH_SCOPE` | `fraud.assess` | The default |
| `FRA_OAUTH_TOKEN_URL` | `http://localhost:8180/realms/claims/protocol/openid-connect/token` | Optional; only shown in the Agent Card |
| `FRA_OAUTH_ALLOW_HTTP` | `1` | Local testing only: allows the plain-http key URL above. Leave it unset in production |

The first three turn OAuth on and must be set together; set only some and the agent refuses to start.
Once they are set, `FRA_API_TOKEN` is ignored.

### 4.1 Try it locally with Keycloak

```bash
# in .env: set KC_ADMIN_PASSWORD, and remove the "# " in front of the five FRA_OAUTH_* lines
docker compose --profile oauth up -d --build --wait     # agent + Keycloak, about a minute
TOKEN=$(keycloak/get-token.sh)                           # a client-credentials token, as Pega gets one
curl -s -X POST localhost:8000/v1/assess -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" -d @examples/request_duplicate.json
```

`keycloak/claims-realm.json` creates the realm `claims` with a client scope `fraud.assess` (its
audience mapper adds `fraud-risk-agent` to `aud`) and a confidential client `pega-claim-assist`
that may use client credentials and gets that scope by default. Keycloak generates the client
secret: see it in the admin console at `http://localhost:8180` (user `admin`) under realm
`claims` → Clients → `pega-claim-assist` → Credentials. Recreating the container makes a new one.
Tokens last 5 minutes. This Keycloak runs in dev mode over plain HTTP: use it for testing only. If
you change `KC_PORT`, change the port in `FRA_OAUTH_ISSUER` and `FRA_OAUTH_TOKEN_URL` too.

### 4.2 Your identity provider

| | Keycloak | Microsoft Entra ID | Okta |
|---|---|---|---|
| `FRA_OAUTH_ISSUER` | `https://<host>/realms/<realm>` | `https://login.microsoftonline.com/<tenant-id>/v2.0` | `https://<org>.okta.com/oauth2/<server-id>` |
| `FRA_OAUTH_JWKS_URL` | `https://<host>/realms/<realm>/protocol/openid-connect/certs` | `https://login.microsoftonline.com/<tenant-id>/discovery/v2.0/keys` | `https://<org>.okta.com/oauth2/<server-id>/v1/keys` |
| `FRA_OAUTH_AUDIENCE` | The audience mapper's value, e.g. `fraud-risk-agent` | The agent's app registration: Application (client) ID | Your dedicated authorization server's audience, e.g. `api://fraud-risk-agent` |
| `fraud.assess` is | A client scope, assigned to Pega's client only, sent in `scope` | An app role, granted to Pega's app registration as an application permission (admin consent), sent in `roles` | A custom scope, allowed for Pega's client only by an access policy, sent in `scp` |
| Token URL | `https://<host>/realms/<realm>/protocol/openid-connect/token` | `https://login.microsoftonline.com/<tenant-id>/oauth2/v2.0/token` | `https://<org>.okta.com/oauth2/<server-id>/v1/token` |
| Scope Pega requests | `fraud.assess` | `<agent's Application ID URI>/.default` | `fraud.assess` |

* **Entra ID:** set the access token version to 2 in the agent app's manifest
  (`requestedAccessTokenVersion`, or `accessTokenAcceptedVersion` in older manifests). Version 1
  tokens carry `iss` `https://sts.windows.net/<tenant-id>/` and the Application ID URI as `aud`.
* **Okta:** create a dedicated custom authorization server for the agent (audience
  `api://fraud-risk-agent`). Don't use the built-in `default` server: its audience `api://default` is
  shared by every app that uses it, and its default access policy lets every client request any
  scope. Give the new server one access policy, assigned only to Pega's client, with a rule for the
  client credentials grant and the scope `fraud.assess`.
* **Whichever provider:** grant `fraud.assess` to Pega's client and nothing else (in Keycloak, don't
  make it a realm-wide default client scope). The agent accepts any token from your provider that has
  the right issuer, audience and scope, so the provider decides who may call it.
* **Private CA or outbound proxy:** the image trusts the usual public CAs. If your provider's
  certificate comes from an internal CA, or the agent must go through a proxy, add a
  `docker-compose.override.yml`:

  ```yaml
  services:
    fra:
      environment:
        SSL_CERT_FILE: /etc/fra/ca-bundle.pem   # replaces the default CA list, so include public roots if needed
        HTTPS_PROXY: http://proxy.example:3128
        NO_PROXY: keycloak,ollama
      volumes:
        - ./ca-bundle.pem:/etc/fra/ca-bundle.pem:ro
  ```
* To check the values, decode a test token locally (never paste production tokens into websites):
  `python -c "import jwt,sys; print(jwt.decode(sys.argv[1], options={'verify_signature': False}))" "$TOKEN"`
  and copy `iss` and `aud` exactly, including any trailing slash.

### 4.3 Pega

Create an OAuth 2.0 authentication profile with grant type client credentials, using the token URL,
client ID, client secret and scope from your provider. Attach it to the Connect Agent rule, and to the
Connect-REST rule if you use the `/v1/assess` fallback. Pega fetches and renews the token itself.
Screen names vary between Pega versions.

**How Pega sends the claim.** In Pega Infinity 25 an external A2A agent is called by a Pega AI agent
(its *External agents* list), whose language model writes the request as **text**. In testing it sent
one `name: value` line per field, with the CSV's flat names (`loss_cause`, `late_reported: false`) and
day-first dates (`30-06-2024`). The agent reads that, JSON (as text or as a data part), the older A2A
`"type": "data"` form and a claim wrapped in one outer object. This reading is plain code, and every
value is still validated; personal data is refused in every format (a `Claimant Name:` line fails
the request, it is not skipped). Dates such as `30-06-2024` are read day-first unless another date in the same
claim proves month-first; `YYYY-MM-DD` avoids any doubt. In the Pega AI agent's instructions, ask it
to pass the 15 fields exactly as stored in the case and to write dates as `YYYY-MM-DD`; the Agent
Card's skill has a complete example.

**Check the case data, not only the call.** The agent scores what it receives. In testing, Pega's
copy of a sample claim differed from the CSV: the policy hash was cut to 32 characters, its other half
had moved into the VIN hash, and `days_since_policy_start` read 300 instead of 3 (the next columns'
zeros appended), so the claim scored 0 instead of 20. If a score looks wrong, compare the values in
the request (the notebook's step 9, or the agent log's claim line) with the source data. A message without a claim (for example, a question in plain words) gets a normal
`FAILED` answer that lists the fields to send, so the calling agent can try again. Every answer
carries the result twice: as a data part, and as JSON text for the calling agent's language model.

## 5. Before production: steps that need your infrastructure details

| Item | What you provide | What to change |
|---|---|---|
| **OAuth 2.0** (spec s4) | Identity provider: issuer, audience, JWKS URL, scope (e.g. `fraud.assess`) | Set the `FRA_OAUTH_*` settings (section 4.2) and remove `FRA_API_TOKEN` and `FRA_OAUTH_ALLOW_HTTP`. |
| **HTTPS** | A domain and a TLS certificate | Terminate TLS at your load balancer, ingress or reverse proxy, and forward to the container. A proxy on the same host uses `127.0.0.1:8000` (the default `FRA_BIND`); a proxy in Docker joins the Compose network; a remote load balancer needs `FRA_BIND` set to a private-network IP. Never publish port 8000 on a public interface: Docker-published ports bypass host firewalls such as ufw. Set `FRA_PUBLIC_URL=https://<your-domain>` so the Agent Card advertises the HTTPS endpoint. |
| **Pega connection** | Your Pega environment | Create the Connect Agent rule from `https://<your-domain>/.well-known/agent.json` (skill `assess_fraud_risk`). Give it the OAuth 2.0 authentication profile (section 4.3). Confirm the A2A version and DataPart shape (spec open items O-1, O-3). `POST /v1/assess` is the Connect-REST fallback. |

Also for production:
* Keep secrets in a secret manager, not in `.env`. In OAuth mode the agent needs none (Pega holds the client secret).
* Pin the image by digest.
* Ship container logs to your log store.
* Scale by adding replicas. In LLM mode, run Ollama on a GPU host.

## 6. Day-to-day operations

```bash
docker compose logs -f fra                      # follow the logs
git pull && docker compose up -d --build --wait # update in place
docker compose down                             # stop (add -v to also delete the downloaded model)
```

To roll back, tag each release image (`fraud-risk-agent:<version>`) and start the previous tag.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `up --wait` fails; the log says `Refusing to start: no authentication configured` | No `.env` file, or neither `FRA_API_TOKEN` nor the OAuth settings are set: `cp .env.example .env` and set one. |
| `Refusing to start: OAuth is half configured` | Set `FRA_OAUTH_ISSUER`, `FRA_OAUTH_AUDIENCE` and `FRA_OAUTH_JWKS_URL` together. |
| `401` on every call (static token) | The token Pega sends does not match `FRA_API_TOKEN`. |
| `Refusing to start: FRA_OAUTH_JWKS_URL uses plain http` | Use the provider's https key URL. Only for the local Keycloak, set `FRA_OAUTH_ALLOW_HTTP=1`. |
| `401`, log `token rejected: <reason>` (OAuth) | `InvalidIssuerError`: `FRA_OAUTH_ISSUER` differs from the token's `iss` (a trailing slash, or `localhost` vs another host name). `InvalidAudienceError`: wrong `FRA_OAUTH_AUDIENCE`, or the provider does not add it. `ExpiredSignatureError`: an expired token, or the agent's clock is ahead of the provider's by more than the token's lifetime. `ImmatureSignatureError`: the token's `iat` or `nbf` is in the future, usually because the agent's clock is more than 30 s behind the provider's: sync the host clock (NTP). `UnknownSigningKey`: no key with the token's `kid`, because `FRA_OAUTH_JWKS_URL` belongs to another realm or tenant, or (locally) Keycloak was recreated with new keys after the token was issued: get a new token. Right after the provider starts signing with a new key it can also last a few seconds and clear by itself. `InvalidSignatureError`: the token was altered, or the key URL is wrong. `DecodeError`: not a JWT at all (some providers issue opaque tokens unless an API audience is requested). |
| `FAILED`: "No claim was found in the message" | Pega's message had no readable claim (no JSON object, and no `name: value` lines with a `claim_id`), typically a Pega AI agent asking in plain words. Tell that agent, in its instructions, to pass the 15 fields (see section 4.3); the log line `a2a: no claim found in the message (parts: ...)` shows which kinds of parts it sent. With the notebook, step 9 shows the full message. |
| `-32602 Invalid params` | The JSON-RPC request has no `params.message.parts` list at all: check the A2A version your Pega uses (spec open item O-1). |
| `403` | The token is valid but lacks `FRA_OAUTH_SCOPE` in `scope`, `scp` or `roles`: grant the scope (or app role) to Pega's client. |
| `503 Cannot verify tokens right now`, log `cannot fetch signing keys from FRA_OAUTH_JWKS_URL: <reason>` | `HTTP 404`: wrong path. `HTTP 301`/`302`: the URL redirects; use the provider's `jwks_uri` exactly as its discovery document gives it. `CERTIFICATE_VERIFY_FAILED`: the provider's certificate is from a CA the image does not trust (see "Private CA" in section 4.2). `Name or service not known`, `Connection refused`, `timed out`, `did not answer within 3 s` or `took longer than 10 s`: DNS, firewall or proxy (see section 4.2). `not a JSON Web Key Set`: the URL returns a web page, not the keys. `no usable signing keys`: the key set has no signing key with a key ID that the agent can use. |
| `llm_wording` is `on` but replies use the template | The model is not downloaded yet (run `ollama-pull`), or replies take longer than `FRA_LLM_TIMEOUT`. Check `docker compose logs fra`. |
| `pip` TLS errors during `docker build` | Your network intercepts TLS. Build with your proxy's CA as a build secret, which does not end up in the image: `docker build --secret id=pip_cert,src=/path/to/proxy-ca.pem -t fraud-risk-agent:1.0.0 .`, then `docker compose up -d --wait` (without `--build`). |
