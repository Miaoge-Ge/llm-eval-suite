#!/bin/sh
set -eu
[ "${RENEWED_LINEAGE:-}" = /etc/letsencrypt/live/llm-eval-ip ] || exit 0
install -d -m 0750 -o root -g caddy /etc/caddy/certs/llm-eval-ip
install -m 0640 -o root -g caddy "$RENEWED_LINEAGE/fullchain.pem" /etc/caddy/certs/llm-eval-ip/fullchain.pem
install -m 0640 -o root -g caddy "$RENEWED_LINEAGE/privkey.pem" /etc/caddy/certs/llm-eval-ip/privkey.pem
systemctl reload caddy
