#!/usr/bin/env bash
# Local testing only. Prints an access token for the "pega-claim-assist" client, issued by the
# Keycloak that `docker compose --profile oauth up` starts. Usage: TOKEN=$(keycloak/get-token.sh)
# It reads the client secret with Keycloak's admin CLI inside the container, then asks for a token
# the way Pega will (OAuth 2.0 client credentials). Neither the client secret nor the admin password
# appears on a command line, where other users of the machine could read it.
set -euo pipefail
cd "$(dirname "$0")/.."

secret=$(docker compose --profile oauth exec -T keycloak bash -c '
  kc=/opt/keycloak/bin/kcadm.sh; cfg=--config=/tmp/kcadm.config
  KC_CLI_PASSWORD="$KC_BOOTSTRAP_ADMIN_PASSWORD" \
      $kc config credentials $cfg --server http://localhost:8080 --realm master --user admin > /dev/null
  id=$($kc get clients $cfg -r claims -q clientId=pega-claim-assist --fields id --format csv --noquotes)
  $kc get "clients/$id/client-secret" $cfg -r claims --fields value --format csv --noquotes')
port=$(docker compose --profile oauth port keycloak 8080)  # e.g. 127.0.0.1:8180

printf 'user = "pega-claim-assist:%s"\n' "$secret" \
  | curl -fsS -K - -d grant_type=client_credentials "http://$port/realms/claims/protocol/openid-connect/token" \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["access_token"])'
