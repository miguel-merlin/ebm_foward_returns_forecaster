#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${MLFLOW_DB_CONTAINER:-returns-ebm-mlflow-db}"
IMAGE="${MLFLOW_DB_IMAGE:-postgres:16}"
PORT="${MLFLOW_DB_PORT:-5432}"
DB_NAME="${MLFLOW_DB_NAME:-mlflow}"
DB_USER="${MLFLOW_DB_USER:-mlflow}"
DB_PASSWORD="${MLFLOW_DB_PASSWORD:-mlflow}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is not installed or not in PATH."
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  if [ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" = "true" ]; then
    echo "Container '${CONTAINER_NAME}' is already running."
  else
    echo "Starting existing container '${CONTAINER_NAME}'..."
    docker start "${CONTAINER_NAME}" >/dev/null
  fi
else
  echo "Creating and starting container '${CONTAINER_NAME}'..."
  docker run -d \
    --name "${CONTAINER_NAME}" \
    -e POSTGRES_DB="${DB_NAME}" \
    -e POSTGRES_USER="${DB_USER}" \
    -e POSTGRES_PASSWORD="${DB_PASSWORD}" \
    -p "${PORT}:5432" \
    --health-cmd "pg_isready -U ${DB_USER} -d ${DB_NAME}" \
    --health-interval 2s \
    --health-timeout 3s \
    --health-retries 30 \
    "${IMAGE}" >/dev/null
fi

echo "Waiting for database readiness..."
for _ in $(seq 1 60); do
  health_status="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}unknown{{end}}' "${CONTAINER_NAME}")"
  if [ "${health_status}" = "healthy" ]; then
    break
  fi
  sleep 1
done

final_status="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}unknown{{end}}' "${CONTAINER_NAME}")"
if [ "${final_status}" != "healthy" ]; then
  echo "Database did not become healthy. Current status: ${final_status}"
  echo "Recent container logs:"
  docker logs --tail 50 "${CONTAINER_NAME}" || true
  exit 1
fi

echo "Database is ready."
echo "MLflow tracking URI:"
echo "postgresql+psycopg2://${DB_USER}:${DB_PASSWORD}@localhost:${PORT}/${DB_NAME}"
