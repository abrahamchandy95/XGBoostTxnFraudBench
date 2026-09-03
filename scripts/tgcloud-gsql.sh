#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "missing ${ENV_FILE}" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

: "${TG_HOST:?TG_HOST is required}"
: "${TG_SECRET:?TG_SECRET is required}"
: "${TG_GSQL_CLIENT_JAR:?TG_GSQL_CLIENT_JAR is required}"

JAR="${TG_GSQL_CLIENT_JAR/#\~/${HOME}}"

if [[ "${JAR}" != /* ]]; then
  JAR="${ROOT}/${JAR}"
fi

if [[ ! -f "${JAR}" ]]; then
  echo "missing GSQL client jar: ${JAR}" >&2
  exit 2
fi

DOMAIN="${TG_HOST#https://}"
DOMAIN="${DOMAIN#http://}"
DOMAIN="${DOMAIN%%/*}"
DOMAIN="${DOMAIN%%:*}"

exec java -jar "${JAR}" \
  --ssl \
  --ip "${DOMAIN}:443" \
  -u __GSQL__secret \
  -p "${TG_SECRET}" \
  "$@"
